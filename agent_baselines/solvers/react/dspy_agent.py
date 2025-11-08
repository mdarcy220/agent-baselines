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
import random
from dataclasses import dataclass

import dspy
from inspect_ai.solver import Solver, solver, system_message

from agent_baselines.solvers.react.basic_agent import (
    DEFAULT_CONTINUE_MESSAGE,
    DEFAULT_SUBMIT_NAME,
    DEFAULT_SYSTEM_MESSAGE,
    basic_agent,
)
from agent_baselines.solvers.react.parallel_eval import (
    eval_in_subprocess,
    load_summary_file,
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


def extract_answer_from_summary(
    summary_path: str, seed: int | None = None
) -> str | None:
    """Extract a random answer from a multi-model evaluation summary file.

    During optimization, we want to expose DSPy to variety in model answers
    (similar to dropout during training). This randomly selects one model's
    answer using a deterministic hash-based approach that's thread-safe.

    The selection is deterministic based on:
    - sample_id (from summary) - ensures different samples get different selections
    - seed (optional) - ensures different optimization runs get different selections

    This approach is thread-safe because each call creates its own RNG instance
    seeded from deterministic data.

    Args:
        summary_path: Path to the summary JSON file
        seed: Optional seed to combine with sample_id for deterministic variety

    Returns:
        The answer string from a randomly selected model, or None if not found
    """

    try:
        summary = load_summary_file(summary_path)

        if not summary.model_names or not summary.models:
            logger.warning(f"No models/answers found in summary file: {summary_path}")
            return None

        if seed is not None:
            # Deterministic: combine user seed with sample_id for reproducible
            # variety (con: will repeat if we eval this sample many times with
            # the same seed)
            local_seed = hash((seed, summary.sample_id)) % (2**32)
            rng = random.Random(local_seed)
        else:
            rng = random.Random()

        selected_model = rng.choice(summary.model_names)
        answer = summary.models[selected_model].answer

        logger.debug(
            f"Selected answer from model: {selected_model} "
            f"(out of {len(summary.model_names)} models, seed={seed})"
        )
        return answer

    except Exception as e:
        logger.error(f"Error extracting answer from {summary_path}: {e}")
        return None


def extract_score_from_summary(summary_path: str) -> float:
    """Extract the average score from a multi-model evaluation summary file.

    Summary files are JSON files created by parallel_eval.py that contain
    averaged scores across all evaluated models.

    Args:
        summary_path: Path to the summary JSON file

    Returns:
        The average score value

    Raises:
        KeyError: If required fields are missing from summary file
        FileNotFoundError: If summary file doesn't exist
    """

    try:
        summary = load_summary_file(summary_path)
        return summary.average_score

    except Exception:
        logger.exception(f"Error extracting score from {summary_path}:")
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

    def __init__(self, config: ReactAgentConfig, seed: int | None = None):
        """Initialize ReAct agent module with fixed configuration.

        Args:
            config: Fixed agent configuration (models, timeouts, etc.)
            seed: Optional seed for deterministic answer selection during optimization
        """
        super().__init__()

        self.config = config
        self.seed = seed

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
        """Run agent evaluation during optimization.

        Called by DSPy optimizers to evaluate candidate prompts. Runs eval_in_subprocess()
        and caches results for metric() to extract scores.

        Args:
            sample_id: Sample ID to evaluate
            task_path: Task path (e.g., "astabench/sqa_dev")
            primary_metric: Primary metric for this task (e.g., "global_avg/mean")
            task: The task/question content (aligns with signature field for MIPRO)

        Returns:
            Prediction with:
            - 'system_message': Current system prompt signature instructions
            - 'continue_message': Current continue prompt signature instructions
            - 'summary_path': Path to summary JSON file (if eval was run)
            - 'answer': Extracted answer from summary (if eval was run)
        """
        system_instructions = self.system_prompt.signature.instructions
        continue_instructions = self.continue_prompt.signature.instructions

        # {submit} placeholder is left intact for system_message() to replace later
        prediction_fields = {
            "system_message": system_instructions,
            "continue_message": continue_instructions,
        }

        if (
            sample_id is not None
            and task_path is not None
            and primary_metric is not None
        ):
            try:

                agent_params = {
                    "system_message_text": system_instructions,
                    "continue_message_text": continue_instructions,
                    **self.config.agent_kwargs,
                }

                logger.info(
                    f"Running eval for sample {sample_id} (caching for optimization)"
                )

                score, summary_path = eval_in_subprocess(
                    sample_id=sample_id,
                    model_names=self.config.eval_models,
                    task_path=task_path,
                    primary_metric=primary_metric,
                    solver_path=self.solver_path,
                    agent_params=agent_params,
                    timeout=self.config.eval_timeout,
                    inspect_log_dir=self.config.log_dir,
                )

                if summary_path:
                    # Use seed for deterministic but varied answer selection
                    answer = extract_answer_from_summary(summary_path, seed=self.seed)

                    prediction_fields["summary_path"] = summary_path
                    prediction_fields["answer"] = answer or ""

                    logger.info(
                        f"Cached eval result for sample {sample_id}: "
                        f"score={score:.4f}, summary_path={summary_path}"
                    )
                else:
                    logger.warning(f"No summary path returned for sample {sample_id}")

            except Exception:
                logger.exception(
                    f"Error running eval for sample {sample_id} in forward():"
                )
                # Continue without caching - metric() will fail with clear error

        return dspy.Prediction(**prediction_fields)

    def metric(self, example, prediction, trace=None):
        """Metric function for DSPy optimization that extracts cached eval scores.

        DSPy optimizers call forward() to get prompts, then call this metric to score
        them. Since forward() always runs eval_in_subprocess() and caches the result
        in prediction.summary_path, this function simply extracts the average score
        from that cached summary file.

        Args:
            example: DSPy Example with sample_id, task_path, primary_metric
            prediction: DSPy Prediction with summary_path (cached result from forward())
            trace: Optional trace (unused)

        Returns:
            Average score extracted from cached summary file

        Raises:
            ValueError: If summary_path is missing or score extraction fails
        """
        sample_id = example.sample_id

        # forward() always caches eval results, so summary_path should always be present
        if not hasattr(prediction, "summary_path") or not prediction.summary_path:
            raise ValueError(
                f"Missing summary_path in prediction for sample {sample_id}. "
                f"forward() should have cached the eval result."
            )

        logger.info(
            f"Extracting score for {sample_id} from cached summary: {prediction.summary_path}"
        )
        score_value = extract_score_from_summary(prediction.summary_path)
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
            score_value = self.metric(example, prediction, trace)
            logger.info(f"✓ {sample_id} ({task_name}): {score_value:.4f}")
            return score_value

        except Exception as e:
            logger.info(f"✗ {sample_id} ({task_name}): {e}")
            raise
