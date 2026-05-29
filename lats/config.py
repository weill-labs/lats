"""Configuration for a LATS run.

All knobs that the paper exposes live here so the search and the CLI share one
source of truth. Defaults mirror the official GPT-3.5 run script
(`run_lats_gpt3.sh`): max_iters=4, expansion_factor=3, internal tests=4.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LATSConfig:
    # --- LLM ---
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.8  # generation temperature (paper uses 0.8)
    gen_max_tokens: int = 1024

    # --- MCTS ---
    max_iters: int = 4  # number of MCTS iterations per problem
    expansion_factor: int = 3  # children sampled per expansion (`n`)
    exploration_weight: float = 1.0  # UCT exploration constant `c`

    # --- Self-testing / evaluation ---
    number_of_tests: int = 4  # internal (self-generated) unit tests
    eval_timeout: int = 10  # seconds for the hidden-test final check
    test_timeout: int = 5  # seconds per internal test execution

    @classmethod
    def gpt35(cls) -> "LATSConfig":
        """Paper's GPT-3.5-turbo configuration."""
        return cls(
            model="gpt-3.5-turbo", max_iters=4, expansion_factor=3, number_of_tests=4
        )

    @classmethod
    def gpt4(cls) -> "LATSConfig":
        """Paper's GPT-4 configuration."""
        return cls(model="gpt-4", max_iters=8, expansion_factor=5, number_of_tests=2)
