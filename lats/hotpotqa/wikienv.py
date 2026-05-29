"""A Wikipedia environment with the ReAct action set: Search / Lookup / Finish.

Mirrors the `wikienv` used by ReAct/LATS for HotPotQA:
- Search[entity]  -> load the page and return its first paragraph; if there is no
                     exact page, return up to 5 similar titles.
- Lookup[keyword] -> return sentences on the current page containing `keyword`,
                     one at a time ("(Result k/N) ...").
- Finish[answer]  -> end the episode with `answer`.

The HTTP layer is injected (`get_page`, `search_titles`) so the env is fully
testable offline; the defaults call the MediaWiki API using only stdlib urllib
(no extra dependency).
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Callable, List, Optional, Tuple

_API = "https://en.wikipedia.org/w/api.php"
_UA = "lats-hotpotqa-repro/0.1 (https://github.com/weill-labs/lats)"


def _http_get_json(params: dict, timeout: float = 10.0) -> Optional[dict]:
    url = f"{_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (https only)
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _mediawiki_get_page(title: str) -> Optional[str]:
    """Plain-text extract of an exact page title, or None if it doesn't exist."""
    data = _http_get_json(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
            "format": "json",
        }
    )
    if not data:
        return None
    pages = data.get("query", {}).get("pages", {})
    for _pid, page in pages.items():
        if "missing" in page:
            return None
        extract = page.get("extract")
        if extract:
            return extract
    return None


def _mediawiki_search(query: str) -> List[str]:
    data = _http_get_json(
        {"action": "query", "list": "search", "srsearch": query, "format": "json"}
    )
    if not data:
        return []
    return [hit["title"] for hit in data.get("query", {}).get("search", [])]


def _first_paragraph(text: str) -> str:
    for para in text.split("\n"):
        para = para.strip()
        if para:
            return para
    return text.strip()


def _sentences(text: str) -> List[str]:
    # Naive splitter: good enough for keyword lookup over a page.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class WikipediaEnv:
    def __init__(
        self,
        get_page: Optional[Callable[[str], Optional[str]]] = None,
        search_titles: Optional[Callable[[str], List[str]]] = None,
    ):
        self.get_page = get_page or _mediawiki_get_page
        self.search_titles = search_titles or _mediawiki_search
        self.reset()

    def reset(self) -> None:
        self.page: Optional[str] = None
        self.lookup_keyword: Optional[str] = None
        self.lookup_results: List[str] = []
        self.lookup_index = 0
        self.answer: Optional[str] = None
        self.done = False
        self.steps = 0

    # --- individual actions ---
    def search(self, entity: str) -> str:
        text = self.get_page(entity)
        if text:
            self.page = text
            self.lookup_keyword = None
            return _first_paragraph(text)
        similar = self.search_titles(entity)[:5]
        if similar:
            return f"Could not find {entity}. Similar: {similar}."
        return f"Could not find {entity}. No similar results."

    def lookup(self, keyword: str) -> str:
        if self.page is None:
            return "No page loaded — use Search[...] first."
        if keyword != self.lookup_keyword:
            self.lookup_keyword = keyword
            self.lookup_results = [
                s for s in _sentences(self.page) if keyword.lower() in s.lower()
            ]
            self.lookup_index = 0
        if self.lookup_index >= len(self.lookup_results):
            return f"No more results for '{keyword}'."
        result = self.lookup_results[self.lookup_index]
        self.lookup_index += 1
        return f"(Result {self.lookup_index}/{len(self.lookup_results)}) {result}"

    def finish(self, answer: str) -> str:
        self.answer = answer
        self.done = True
        return f"Episode finished. Answer: {answer}"

    # --- dispatch ---
    def act(self, action: str, argument: str) -> str:
        """Run one action; returns the observation. Increments the step counter."""
        self.steps += 1
        verb = action.strip().lower()
        if verb == "search":
            return self.search(argument)
        if verb == "lookup":
            return self.lookup(argument)
        if verb == "finish":
            return self.finish(argument)
        return (
            f"Invalid action '{action}'. Use Search[...], Lookup[...], or Finish[...]."
        )


_ACTION_RE = re.compile(r"^\s*(\w+)\s*\[(.*)\]\s*$", re.DOTALL)


def parse_action(action_str: str) -> Optional[Tuple[str, str]]:
    """Parse a ReAct action like 'Search[Eiffel Tower]' -> ('Search', 'Eiffel Tower')."""
    m = _ACTION_RE.match(action_str)
    if not m:
        return None
    return m.group(1), m.group(2).strip()
