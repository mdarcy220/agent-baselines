"""DSPy-compatible ReAct agent with optimizable prompts.

This module provides DSPy signatures and a module for optimizing the ReAct agent's
prompts using DSPy optimizers like MIPRO or GEPA.

The key design is that signature instructions (docstrings) ARE the agent prompts.
DSPy optimizers modify these instructions based on agent performance, allowing
direct optimization of the prompts that the agent sees.

Architecture:
1. Signature instructions define the agent prompts (seeded from basic_agent defaults)
2. DSPy optimizers modify the instructions based on agent performance
3. After optimization, extract the optimized instructions as final prompts
4. Use them with system_message() which handles {submit} placeholder replacement

Caching Mechanism:
- forward() runs actual agent evals when sample data is provided (during bootstrap)
- Results are cached in the prediction (eval_path, answer)
- The metric function checks for cached results and reuses them instead of re-running
- This allows bootstrap to create useful demonstrations without redundant evals
"""

import logging

import dspy
from inspect_ai.log import read_eval_log
from inspect_ai.solver import Solver, solver, system_message

from agent_baselines.solvers.react.basic_agent import (
    DEFAULT_CONTINUE_MESSAGE,
    DEFAULT_SUBMIT_NAME,
    DEFAULT_SYSTEM_MESSAGE,
    basic_agent,
)

logger = logging.getLogger(__name__)

# Create signature classes with instructions from basic_agent defaults
# These will be the starting point for DSPy optimization
AgentSystemPromptSignature = dspy.Signature("task -> response", DEFAULT_SYSTEM_MESSAGE)

AgentContinuePromptSignature = dspy.Signature(" -> ", DEFAULT_CONTINUE_MESSAGE)


def extract_answer_from_eval(eval_path: str) -> str | None:
    """Extract the answer from an .eval file.

    .eval files are zip archives containing JSON data about the evaluation.
    This function reads the samples.json file and extracts the answer from
    the sample's output completion.

    Args:
        eval_path: Path to the .eval file

    Returns:
        The answer string, or None if not found
    """
    try:
        # Read the eval log using inspect_ai's reader
        eval_log = read_eval_log(eval_path)

        # Get the first (and only) sample's output
        if eval_log.samples and len(eval_log.samples) > 0:
            sample = eval_log.samples[0]
            if sample.output and sample.output.completion:
                # The completion contains the agent's final answer
                return sample.output.completion

        logger.warning(f"No answer found in eval file: {eval_path}")
        return None

    except Exception as e:
        logger.error(f"Error extracting answer from {eval_path}: {e}")
        return None


def extract_score_from_eval(eval_path: str, primary_metric: str) -> float:
    """Extract the score from an .eval file.

    Args:
        eval_path: Path to the .eval file
        primary_metric: Primary metric to extract (format: "scorer_name/metric_name")

    Returns:
        The score value

    Raises:
        ValueError: If the metric is not found in the eval file
    """
    try:
        # Parse primary_metric format: "scorer_name/metric_name"
        metric_parts = primary_metric.split("/")
        if len(metric_parts) != 2:
            raise ValueError(f"Invalid primary_metric format: {primary_metric}")
        scorer_name, metric_name = metric_parts

        # Read the eval log
        eval_log = read_eval_log(eval_path, header_only=True)

        # Extract the score
        if not eval_log.results or not eval_log.results.scores:
            raise ValueError(f"No scores found in eval file: {eval_path}")

        # Find the scorer
        scorer = None
        for score in eval_log.results.scores:
            if score.name == scorer_name:
                scorer = score
                break

        if not scorer:
            available = [s.name for s in eval_log.results.scores]
            raise ValueError(
                f"Scorer '{scorer_name}' not found. Available: {available}"
            )

        if metric_name not in scorer.metrics:
            available = list(scorer.metrics.keys())
            raise ValueError(
                f"Metric '{metric_name}' not found in scorer '{scorer_name}'. Available: {available}"
            )

        return float(scorer.metrics[metric_name].value)

    except Exception as e:
        logger.error(f"Error extracting score from {eval_path}: {e}")
        raise


class DSPyReActPrompts(dspy.Module):
    """DSPy module that provides optimizable agent prompts via signature instructions.

    This module:
    1. Defines prompts as signature instructions (docstrings)
    2. Creates predictors so DSPy optimizers can modify the signatures
    3. Returns the current signature instructions in forward()
    4. Runs actual agent evals when sample data is provided (enables bootstrap)
    5. After optimization, provides optimized instructions for use as agent prompts

    DSPy optimizers (like MIPRO) modify the signature instructions based on
    agent performance metrics, directly optimizing the prompts that the agent sees.

    The forward() method now supports running actual agent evaluations during
    bootstrap, caching the results so the metric function can reuse them without
    redundant evals.
    """

    def __init__(
        self,
        eval_models: list[str] | None = None,
        agent_kwargs: dict | None = None,
        eval_timeout: int = 1200,
        log_dir: str | None = None,
        solver_path: str | None = None,
    ):
        """Initialize DSPyReActPrompts module.

        Args:
            eval_models: List of models to evaluate on (for running actual evals)
            agent_kwargs: Agent configuration dict (e.g., {"max_steps": 10})
            eval_timeout: Timeout per sample evaluation in seconds
            log_dir: Directory for evaluation logs
            solver_path: Path to solver for eval_in_subprocess
        """
        super().__init__()

        # Create predictors so DSPy optimizers can access and modify their signatures
        # The signature instructions (docstrings) are what get optimized
        # Note: These predictors are never actually called - we just extract their instructions
        self.system_prompt = dspy.Predict(AgentSystemPromptSignature)
        self.continue_prompt = dspy.Predict(AgentContinuePromptSignature)

        # Store config for running actual evals
        self.eval_models = eval_models
        self.agent_kwargs = agent_kwargs or {}
        self.eval_timeout = eval_timeout
        self.log_dir = log_dir
        self.solver_path = solver_path

    def forward(
        self,
        sample_id: str | None = None,
        task_path: str | None = None,
        primary_metric: str | None = None,
        task: str | None = None,
    ):
        """Return current signature instructions as agent prompts, optionally running eval.

        This method extracts the signature instructions (which DSPy optimizers modify)
        and returns them as the agent's system_message and continue_message.

        When sample data is provided (sample_id, task_path, primary_metric), this
        method will run an actual agent evaluation and cache the results. This enables
        DSPy's bootstrap to create useful demonstrations based on actual agent behavior.

        The metric function checks for the presence of eval_path in the prediction and
        reuses the cached results instead of re-running the evaluation, avoiding
        redundant computation.

        Note: The signature instructions contain {submit} placeholders which are NOT
        replaced here. They will be replaced by inspect_ai's system_message() via
        str.format() when the prompts are used in create_agent_with_dspy_prompts().

        Args:
            sample_id: Sample ID to evaluate (if provided, runs actual eval)
            task_path: Task path (e.g., "astabench/sqa_dev")
            primary_metric: Primary metric for this task (e.g., "global_avg/mean")
            task: The task/question content (aligns with signature field for MIPRO)

        Returns:
            Prediction with:
            - 'system_message': Current system prompt signature instructions
            - 'continue_message': Current continue prompt signature instructions
            - 'eval_path': Path to .eval file (if eval was run)
            - 'answer': Extracted answer from eval (if eval was run)
        """
        # Extract current instructions from signatures
        # After optimization, these will contain the optimized instructions
        system_instructions = self.system_prompt.signature.instructions
        continue_instructions = self.continue_prompt.signature.instructions

        # Create base prediction
        # Note: {submit} placeholder is left intact for system_message() to replace
        prediction_fields = {
            "system_message": system_instructions,
            "continue_message": continue_instructions,
        }

        # If sample data is provided, run actual agent evaluation
        if (
            sample_id is not None
            and task_path is not None
            and primary_metric is not None
        ):
            # Check if we have the necessary config to run evals
            if (
                self.eval_models is None
                or self.log_dir is None
                or self.solver_path is None
            ):
                logger.warning(
                    f"Sample data provided but eval config incomplete. "
                    f"Skipping eval for sample {sample_id}. "
                    f"Set eval_models, log_dir, and solver_path in __init__ to enable eval caching."
                )
            else:
                try:
                    # Import here to avoid circular dependency
                    from agent_baselines.solvers.react.parallel_eval import (
                        eval_in_subprocess,
                    )

                    # Build agent_params dict with current prompts
                    agent_params = {
                        "system_message_text": system_instructions,
                        "continue_message_text": continue_instructions,
                        **self.agent_kwargs,
                    }

                    # Run evaluation in subprocess
                    logger.info(
                        f"Running eval for sample {sample_id} (caching for bootstrap)"
                    )

                    # Run the evaluation - returns both score and eval_path
                    score, eval_path = eval_in_subprocess(
                        sample_id=sample_id,
                        model_names=self.eval_models,
                        task_path=task_path,
                        primary_metric=primary_metric,
                        solver_path=self.solver_path,
                        agent_params=agent_params,
                        timeout=self.eval_timeout,
                        inspect_log_dir=self.log_dir,
                    )

                    if eval_path:
                        # Extract answer from the eval file
                        answer = extract_answer_from_eval(eval_path)

                        # Add to prediction for caching
                        prediction_fields["eval_path"] = eval_path
                        prediction_fields["answer"] = answer or ""

                        logger.info(
                            f"Cached eval result for sample {sample_id}: "
                            f"score={score:.4f}, eval_path={eval_path}"
                        )
                    else:
                        logger.warning(
                            f"No eval file path returned for sample {sample_id}"
                        )

                except Exception as e:
                    logger.error(
                        f"Error running eval for sample {sample_id} in forward(): {e}"
                    )
                    # Continue without caching - metric will run eval normally

        return dspy.Prediction(**prediction_fields)


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
