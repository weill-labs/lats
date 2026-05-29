"""HotPotQA domain for LATS: ReAct over a Wikipedia search environment.

Reuses the domain-agnostic core (`lats.node`, `lats.llm`, `lats.result`) and adds
the QA-specific pieces: a Wikipedia environment, dataset + EM/F1 scoring, ReAct
prompts, an LM value function, and an MCTS search over reasoning+acting
trajectories.
"""
