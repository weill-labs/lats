# Project conventions — weill-labs/lats

- **Commit directly to `master`.** No feature branches or pull requests for this
  repo (it's a personal research/reproduction project; PR overhead isn't worth it).
- Before committing: run `uv run pytest -q` and `uvx ruff check lats tests`. The
  pre-commit hook enforces `ruff format`.
- **Keep secrets out of git.** `OPENAI_API_KEY` lives only in the gitignored
  `.env`. Never commit it.
- Use `uv` for everything (`uv sync`, `uv run …`).
