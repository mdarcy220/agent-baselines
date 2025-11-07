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
    """Generate a system message for a ReAct agent solving diverse research and analysis tasks.

    The agent will receive a specific task (which may be a research question, data analysis problem,
    math calculation, coding challenge, or document analysis task) and must use available tools
    to solve it. Different task types require different approaches:

    - Research questions: Use search tools to find and cite relevant sources
    - Math/data problems: Use calculation or code execution tools
    - Multi-step problems: Break down into subtasks and use tools iteratively
    - Document analysis: Read and extract information from provided documents

    The system message should:
    - Clearly explain the task goal from task_description (which includes the actual question)
    - Describe when and how to use available tools (tools are provided by the task environment)
    - Emphasize the importance of calling submit_function_name when done with the final answer
    - Encourage step-by-step reasoning before taking actions
    - Be concise but complete - avoid unnecessary verbosity
    """

    task_description = dspy.InputField(
        desc="Complete description of the specific task to solve, including the actual question or problem statement. Format: 'Task: [task_name]\\n\\nQuestion: [actual question]'"
    )
    submit_function_name = dspy.InputField(
        desc="Name of the function the agent must call to submit the final answer (e.g., 'submit_answer', 'submit_code'). This function will be available as a tool."
    )
    system_message = dspy.OutputField(
        desc="Complete system message that prepares the agent to solve this specific task. Should be clear, actionable, and tailored to the task type."
    )


class ContinueMessageSignature(dspy.Signature):
    """Generate a brief message to encourage the agent to continue working when it hasn't made a tool call.

    This message helps the agent:
    - Stay on track toward completing the task
    - Remember to use tools when needed to make progress
    - Know when it's time to submit the final answer
    - Avoid getting stuck or giving up prematurely

    The continue message should be:
    - Brief (1-2 sentences)
    - Motivating but not pushy
    - Remind the agent of its goal without being repetitive
    - Generic enough to work across different task types
    """

    continue_message = dspy.OutputField(
        desc="Brief, motivating message (1-2 sentences) that reminds the agent to proceed with the task and use tools or submit when ready"
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
            task_description: The specific task/question the agent needs to solve.
                             Format: "Task: [task_name]\\n\\nQuestion: [actual question]"
            submit_function_name: Function name for submitting the final answer

        Returns:
            Prediction with 'system_message' and 'continue_message' fields
        """
        # Generate task-specific system message
        # The system message sets up the agent's understanding of:
        # - What the specific task is asking for (from task_description)
        # - What tools are available (provided by the task environment, not specified here)
        # - When to call submit_function_name with the final answer
        # - How to approach the problem (reasoning, tool use, etc.)
        #
        # This is generated using ChainOfThought, so DSPy optimizers can see
        # the reasoning process during bootstrap/optimization
        system_result = self.system_message_generator(
            task_description=task_description,
            submit_function_name=submit_function_name,
        )

        # Generate continuation prompt for when agent stalls
        # This helps the agent recover when it fails to make progress after
        # using a tool or when it seems stuck. The continue message should:
        # - Not be overly repetitive (agent sees it multiple times)
        # - Encourage forward progress without being pushy
        # - Be generic enough to work across diverse task types
        continue_result = self.continue_message_generator()

        return dspy.Prediction(
            system_message=system_result.system_message,
            continue_message=continue_result.continue_message,
        )


@solver
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
