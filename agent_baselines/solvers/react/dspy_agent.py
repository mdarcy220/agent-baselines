"""DSPy-compatible ReAct agent with optimizable prompts.

This module provides DSPy signatures and a module for optimizing the ReAct agent's
prompts using DSPy optimizers like MIPRO or GEPA.

The key design is that signature instructions (docstrings) ARE the agent prompts.
DSPy optimizers modify these instructions based on agent performance, allowing
direct optimization of the prompts that the agent sees.

Architecture:
1. Signature instructions define the agent prompts
2. DSPy optimizers modify the instructions based on agent performance
3. After optimization, extract the optimized instructions as final prompts
4. Use them directly as the agent's system_message and continue_message
"""

import dspy
from inspect_ai.solver import Solver, solver, system_message

from agent_baselines.solvers.react.basic_agent import (
    DEFAULT_SUBMIT_NAME,
    basic_agent,
)

# Note: docstrings of these signatures are seeded from the original
# basic_agent.py prompts (which in turn are seeded from inspect_ai)


class AgentSystemPromptSignature(dspy.Signature):
    """
    You are a helpful assistant attempting to submit the correct answer. You have
    several functions available to help with finding the answer. Each message may
    may perform one function call. You will see the result of the function right
    after sending the message. If you need to perform multiple actions, you can
    always send more messages with subsequent function calls. Do some reasoning
    before your actions, describing what function calls you are going to use and
    how they fit into your plan.

    When you have completed the task and have an answer, call the {submit}()
    function to report it."""

    # These fields satisfy DSPy's signature requirements
    # The signature instructions (docstring above) are what get optimized
    task = dspy.InputField(desc="The specific task instructions")
    response = dspy.OutputField(desc="The agent's solution to the task")


class AgentContinuePromptSignature(dspy.Signature):
    """Please proceed to the next step using your best judgement.  Remember to submit when the task is complete."""

    # No input or output fields here; this is just a message to encourage the
    # agent to keep working if it doesn't call a tool for one of its steps.


class DSPyReActPrompts(dspy.Module):
    """DSPy module that provides optimizable agent prompts via signature instructions.

    This module:
    1. Defines prompts as signature instructions (docstrings)
    2. Creates predictors so DSPy optimizers can modify the signatures
    3. Returns the current signature instructions in forward()
    4. After optimization, provides optimized instructions for use as agent prompts

    DSPy optimizers (like MIPRO) modify the signature instructions based on
    agent performance metrics, directly optimizing the prompts that the agent sees.
    """

    def __init__(self):
        super().__init__()

        # Create predictors so DSPy optimizers can access and modify their signatures
        # The signature instructions (docstrings) are what get optimized
        # Note: These predictors are never actually called - we just extract their instructions
        self.system_prompt = dspy.Predict(AgentSystemPromptSignature)
        self.continue_prompt = dspy.Predict(AgentContinuePromptSignature)

    def forward(
        self,
        task_description: str = "",
        submit_function_name: str = DEFAULT_SUBMIT_NAME,
    ):
        """Return current signature instructions as agent prompts.

        This method extracts the signature instructions (which DSPy optimizers modify)
        and returns them as the agent's system_message and continue_message.

        Note on Bootstrap Demonstrations:
        ---------------------------------
        This implementation does NOT support DSPy's bootstrap demonstrations because
        forward() returns the same output (signature instructions) for every input.
        The actual agent execution happens in eval_in_subprocess() within the metric
        function, disconnected from this module's forward().

        To make bootstrap work, forward() would need to:
        1. Call eval_in_subprocess() to actually run the agent
        2. Return the eval file path (not just the answer) along with the extracted answer
        3. Have the metric read the score from the eval file instead of re-running eval

        This would avoid redundant evals, but is architecturally awkward (forward()
        would see eval file paths, metric would need to handle both fresh evals and
        cached results). Instead, we default max_bootstrapped_demos=0 and rely on
        MIPRO's instruction optimization based on final scores alone.

        Args:
            task_description: Not used in direct optimization (kept for API compatibility
                with DSPy's example format from task_loader.py)
            submit_function_name: Name of submit function to inject into prompt

        Returns:
            Prediction with 'system_message' and 'continue_message' fields containing
            the current signature instructions
        """
        # Extract current instructions from signatures
        # After optimization, these will contain the optimized instructions
        system_instructions = self.system_prompt.signature.instructions
        continue_instructions = self.continue_prompt.signature.instructions

        # Inject submit_function_name into system message to support different submit names
        system_message_text = system_instructions.replace(
            "submit function", f"{submit_function_name}() function"
        ).replace("call the submit", f"call the {submit_function_name}()")

        return dspy.Prediction(
            system_message=system_message_text,
            continue_message=continue_instructions,
        )


@solver
def create_agent_with_dspy_prompts(
    system_message_text: str,
    continue_message_text: str,
    max_steps: int = 10,
    **kwargs,
) -> Solver:
    """Create a basic_agent solver with DSPy-optimized prompts.

    Args:
        system_message_text: The system message prompt
        continue_message_text: The continue message prompt
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
