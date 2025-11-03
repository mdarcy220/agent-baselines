"""Test script to run a single eval directly (without subprocess) for debugging.

Usage:
    python agent_baselines/solvers/react/test_eval_direct.py
"""

import logging

logging.basicConfig(level=logging.INFO)

# Use the first sample from sqa_dev
from astabench.evals.sqa import sqa_dev
from astabench.tools import ToolsetConfig
from inspect_ai import eval as inspect_eval

from agent_baselines.solvers.react.basic_agent import DEFAULT_SYSTEM_MESSAGE, DEFAULT_CONTINUE_MESSAGE
from agent_baselines.solvers.react.dspy_agent import create_agent_with_dspy_prompts

# Get first sample
base_task = sqa_dev()
first_sample = list(base_task.dataset)[0]
sample_id = first_sample.id

print(f"Testing eval on sample: {sample_id}")

# Create tool config
tool_config = ToolsetConfig(
    with_search_tools=True,
    with_report_editor=True,
)

# Create solver with default prompts
agent_solver = create_agent_with_dspy_prompts(
    system_message_text=DEFAULT_SYSTEM_MESSAGE.format(submit="submit"),
    continue_message_text=DEFAULT_CONTINUE_MESSAGE,
    max_steps=10,
    tools=tool_config.create_tools(),
    add_submit_tool=not tool_config.with_editor_submit,
)

print("Running eval...")

# Run eval
logs = inspect_eval(
    tasks=[base_task],
    model="openai/gpt-4o-mini",
    solver=agent_solver,
    sample_id=sample_id,
    log_dir=".dspy_cache",
    log_level="info",
    display="plain",
)

print(f"Eval completed! Got {len(logs)} logs")

# Extract score
eval_log = logs[0]
if eval_log.samples:
    sample = eval_log.samples[0]
    if sample.scores:
        for scorer_name, score in sample.scores.items():
            print(f"Scorer {scorer_name}: {score.value}")
