"""DSPy-compatible ReAct agent for prompt optimization.

This module provides DSPy signatures and a module for optimizing the ReAct agent's
prompts using DSPy optimizers like GEPA or MIPROv2.
"""

import dspy
from inspect_ai.solver import Solver, solver, system_message

from agent_baselines.solvers.react.basic_agent import (
    DEFAULT_SUBMIT_NAME,
    basic_agent,
)


class SystemMessageSignature(dspy.Signature):
    """Generate a system message for a ReAct agent that uses tools to answer questions.

    The agent needs to understand:
    - How to use available functions/tools
    - When to submit an answer
    - How to reason before taking actions
    """

    task_description = dspy.InputField(
        desc="Description of what the agent needs to accomplish"
    )
    submit_function_name = dspy.InputField(
        desc="Name of the function to call when submitting the final answer"
    )
    system_message = dspy.OutputField(
        desc="Clear, effective instructions for the agent"
    )


class ContinueMessageSignature(dspy.Signature):
    """Generate a message to urge the agent to continue when it doesn't make a tool call."""

    continue_message = dspy.OutputField(
        desc="A brief message encouraging the agent to proceed and reminding it to submit when done"
    )


class DSPyReActPrompts(dspy.Module):
    """DSPy module that generates optimized prompts for the ReAct agent."""

    def __init__(self):
        super().__init__()
        self.system_message_generator = dspy.ChainOfThought(SystemMessageSignature)
        self.continue_message_generator = dspy.ChainOfThought(ContinueMessageSignature)

    def forward(
        self, task_description: str, submit_function_name: str = DEFAULT_SUBMIT_NAME
    ):
        """Generate optimized prompts for the ReAct agent.

        Args:
            task_description: Description of the task the agent needs to accomplish
            submit_function_name: Name of the submission function

        Returns:
            dict with 'system_message' and 'continue_message' keys
        """
        system_result = self.system_message_generator(
            task_description=task_description,
            submit_function_name=submit_function_name,
        )
        continue_result = self.continue_message_generator()

        return dspy.Prediction(
            system_message=system_result.system_message,
            continue_message=continue_result.continue_message,
        )


def create_agent_with_dspy_prompts(
    system_message_text: str,
    continue_message_text: str,
    max_steps: int = 10,
    **kwargs,
) -> Solver:
    """Create a basic_agent solver with custom DSPy-optimized prompts.

    Args:
        system_message_text: The optimized system message
        continue_message_text: The optimized continue message
        max_steps: Maximum number of agent steps
        **kwargs: Additional arguments to pass to basic_agent

    Returns:
        A configured basic_agent solver
    """

    @solver
    def custom_system_message() -> Solver:
        return system_message(system_message_text, submit=DEFAULT_SUBMIT_NAME)

    return basic_agent(
        init=custom_system_message(),
        continue_message=continue_message_text,
        max_steps=max_steps,
        **kwargs,
    )
