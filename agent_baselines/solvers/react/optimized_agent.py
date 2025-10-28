"""Optimized ReAct agent that uses DSPy-optimized prompts.

This module provides a drop-in replacement for the basic_agent that uses
prompts optimized via DSPy GEPA/MIPROv2.

Usage:
    astabench eval --solver agent_baselines/solvers/react/optimized_agent.py@instantiated_optimized_agent ...
"""

import json
import logging
from pathlib import Path

from astabench.tools import ToolsetConfig
from inspect_ai.model import Model
from inspect_ai.solver import solver

from agent_baselines.solvers.react.basic_agent import (
    DEFAULT_CONTINUE_MESSAGE,
    DEFAULT_SYSTEM_MESSAGE,
)
from agent_baselines.solvers.react.dspy_agent import create_agent_with_dspy_prompts

logger = logging.getLogger(__name__)


def load_optimized_prompts(prompts_file: str = "optimized_prompts.json") -> dict:
    """Load optimized prompts from a JSON file.

    Args:
        prompts_file: Path to the JSON file containing optimized prompts.
                     Can be absolute or relative to this file's directory.

    Returns:
        Dictionary with 'system_message' and 'continue_message' keys
    """
    # Try to find the file
    prompts_path = Path(prompts_file)

    if not prompts_path.is_absolute():
        # Try relative to this file's directory
        prompts_path = Path(__file__).parent / prompts_file

    if not prompts_path.exists():
        logger.warning(
            f"Optimized prompts file not found at {prompts_path}. "
            f"Falling back to default prompts. "
            f"Run optimization first: python agent_baselines/solvers/react/optimize.py"
        )
        return {
            "system_message": DEFAULT_SYSTEM_MESSAGE,
            "continue_message": DEFAULT_CONTINUE_MESSAGE,
        }

    try:
        with open(prompts_path, "r") as f:
            prompts = json.load(f)

        logger.info(f"Loaded optimized prompts from {prompts_path}")
        if "metadata" in prompts:
            logger.info(f"Optimization metadata: {prompts['metadata']}")

        return prompts

    except Exception as e:
        logger.error(f"Error loading optimized prompts: {e}. Using defaults.")
        return {
            "system_message": DEFAULT_SYSTEM_MESSAGE,
            "continue_message": DEFAULT_CONTINUE_MESSAGE,
        }


@solver
def instantiated_optimized_agent(
    max_steps: int = 10,
    model_override: str | Model | None = None,
    prompts_file: str = "optimized_prompts.json",
    **tool_options,
):
    """ReAct agent with DSPy-optimized prompts.

    This is a drop-in replacement for instantiated_basic_agent that uses
    prompts optimized via DSPy.

    Args:
        max_steps: Maximum number of reasoning steps before terminating.
        model_override: Optional model override. If provided, will use this
            model instead of the default one (prefer `--model` instead of this,
            unless this agent is being used in a multi-agent system).
        prompts_file: Path to the JSON file containing optimized prompts.
        **tool_options: Tool configuration options. See ToolsetConfig in
            astabench.tools for available options (with_search_tools,
            with_stateful_python, with_report_editor, with_table_editor,
            with_thinking_tool, with_editor_submit).
    """
    # Load optimized prompts
    prompts = load_optimized_prompts(prompts_file)

    # Create tool configuration
    config = ToolsetConfig.model_validate(tool_options)
    tools = config.create_tools()

    logger.info("Tool configuration: %s", config.pretty_format())
    logger.info("Using optimized prompts from: %s", prompts_file)

    # Create agent with optimized prompts
    return create_agent_with_dspy_prompts(
        system_message_text=prompts["system_message"],
        continue_message_text=prompts["continue_message"],
        max_steps=max_steps,
        tools=tools,
        add_submit_tool=not config.with_editor_submit,
        model_override=model_override,
    )
