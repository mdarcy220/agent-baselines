"""DSPy-compatible ReAct agent with optimizable prompts.

This module provides ReactInspectAgent, a dspy.Module that encapsulates the complete
optimization pipeline for ReAct agent prompts using DSPy optimizers like MIPRO or GEPA.

Key Design:
- ReactInspectAgent IS the dspy.Module (not a wrapper)
- Signature instructions ARE the agent prompts (seeded from basic_agent defaults)
- forward() and metric() work together via cached eval results

Optimization Flow:
1. DSPy optimizer calls agent.forward() which runs eval and caches result
2. DSPy optimizer calls agent.metric() which extracts score from cached eval
3. Optimizer modifies signature instructions based on scores
4. After optimization, extract optimized instructions as final prompts
5. Prompts flow through create_agent_with_dspy_prompts() where {submit} is replaced
"""

import logging
from dataclasses import dataclass

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


@dataclass
class ReactAgentConfig:
    """Fixed configuration for ReactInspectAgent.

    These parameters stay constant during optimization.
    Tunable parameters (what DSPy optimizes) are the signature instructions.
    """

    eval_models: list[str]  # Models to evaluate prompts on
    agent_kwargs: dict  # Agent config (e.g., {"max_steps": 10})
    eval_timeout: int  # Timeout per sample evaluation
    log_dir: str  # Directory for evaluation logs


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


class ReactInspectAgent(dspy.Module):
    """DSPy module for optimizing ReAct agent prompts.

    This module encapsulates the complete ReAct agent optimization:
    1. Tunable parameters - signature instructions that DSPy optimizes
    2. Fixed parameters - models, timeouts, agent config (stays constant)
    3. forward() - extracts prompts and runs agent evaluation
    4. metric() - extracts scores from cached eval results

    The evaluation flow:
    - forward() runs agent evals via eval_in_subprocess() and caches results
    - metric() extracts scores from the cached eval files

    This design makes it clear that forward() and metric() work together.
    To optimize a different agent, create a new dspy.Module subclass with
    its own forward() and metric() methods.
    """

    # Solver path for ReAct agent with DSPy-optimizable prompts
    solver_path = (
        "agent_baselines/solvers/react/dspy_agent.py@create_agent_with_dspy_prompts"
    )

    def __init__(self, config: ReactAgentConfig):
        """Initialize ReAct agent module with fixed configuration.

        Args:
            config: Fixed agent configuration (models, timeouts, etc.)
        """
        super().__init__()

        self.config = config

        # Create predictors so DSPy optimizers can access and modify their signatures
        # The signature instructions are what get optimized
        # Note: These predictors are never actually called - we just extract their instructions
        self.system_prompt = dspy.Predict(AgentSystemPromptSignature)
        self.continue_prompt = dspy.Predict(AgentContinuePromptSignature)

    def forward(
        self,
        sample_id: str | None = None,
        task_path: str | None = None,
        primary_metric: str | None = None,
        task: str | None = None,
    ):
        """Extract signature instructions and run agent evaluation.

        This method:
        1. Extracts signature instructions (modified by DSPy optimizers)
        2. Runs eval_in_subprocess() to evaluate the agent with current prompts
        3. Caches the eval result in prediction.eval_path for metric() to use

        The signature instructions contain {submit} placeholders which are NOT
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
            try:
                # Import here to avoid circular dependency
                from agent_baselines.solvers.react.parallel_eval import (
                    eval_in_subprocess,
                )

                # Build agent_params dict with current prompts
                agent_params = {
                    "system_message_text": system_instructions,
                    "continue_message_text": continue_instructions,
                    **self.config.agent_kwargs,
                }

                # Run evaluation in subprocess
                logger.info(
                    f"Running eval for sample {sample_id} (caching for optimization)"
                )

                # Run the evaluation - returns both score and eval_path
                score, eval_path = eval_in_subprocess(
                    sample_id=sample_id,
                    model_names=self.config.eval_models,
                    task_path=task_path,
                    primary_metric=primary_metric,
                    solver_path=self.solver_path,
                    agent_params=agent_params,
                    timeout=self.config.eval_timeout,
                    inspect_log_dir=self.config.log_dir,
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
                    logger.warning(f"No eval file path returned for sample {sample_id}")

            except Exception as e:
                logger.error(
                    f"Error running eval for sample {sample_id} in forward(): {e}"
                )
                # Continue without caching - metric will raise error

        return dspy.Prediction(**prediction_fields)

    def metric(self, example, prediction, trace=None):
        """Metric function for DSPy optimization that extracts cached eval scores.

        DSPy optimizers call forward() to get prompts, then call this metric to score
        them. Since forward() always runs eval_in_subprocess() and caches the result
        in prediction.eval_path, this function simply extracts the score from that
        cached eval file.

        Args:
            example: DSPy Example with sample_id, task_path, primary_metric
            prediction: DSPy Prediction with eval_path (cached result from forward())
            trace: Optional trace (unused)

        Returns:
            Score extracted from cached eval file

        Raises:
            ValueError: If eval_path is missing or score extraction fails
        """
        sample_id = example.sample_id
        primary_metric = example.primary_metric

        # Extract score from cached .eval file
        # forward() always caches eval results, so eval_path should always be present
        if not hasattr(prediction, "eval_path") or not prediction.eval_path:
            raise ValueError(
                f"Missing eval_path in prediction for sample {sample_id}. "
                f"forward() should have cached the eval result."
            )

        logger.info(
            f"Extracting score for {sample_id} from cached eval: {prediction.eval_path}"
        )
        score_value = extract_score_from_eval(prediction.eval_path, primary_metric)
        return score_value

    def logging_metric(self, example, prediction, trace=None):
        """Metric with logging wrapper for visibility during optimization.

        This wraps the base metric with logging behavior for user visibility.

        Args:
            example: DSPy Example with sample_id, task_name, primary_metric
            prediction: DSPy Prediction with eval_path (cached result from forward())
            trace: Optional trace (unused)

        Returns:
            Score extracted from cached eval file
        """
        sample_id = example.sample_id
        task_name = example.task_name

        try:
            # Call base metric for evaluation
            score_value = self.metric(example, prediction, trace)

            # Log result concisely
            logger.info(f"✓ {sample_id} ({task_name}): {score_value:.4f}")
            return score_value

        except Exception as e:
            # Log failure
            logger.info(f"✗ {sample_id} ({task_name}): {e}")
            raise
