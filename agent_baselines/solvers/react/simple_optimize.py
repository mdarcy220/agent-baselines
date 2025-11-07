"""DSPy optimization pipeline for ReAct agent prompts.

Architecture:
1. Task Data - Load and prepare samples from multiple tasks
2. ReactInspectAgent - dspy.Module with forward() and metric() methods
3. Optimizer - Configure DSPy optimizer (MIPRO, GEPA, Bootstrap)
4. CLI - Wire everything together

Notes:
- forward() runs eval_in_subprocess() and caches results
- metric() extracts scores from cached eval files
- Signature instructions are the tunable parameters

Optimization Flow:
1. DSPy optimizer calls agent.forward() → runs eval, caches result
2. DSPy optimizer calls agent.metric() → extracts score from cache
3. Optimizer modifies signature instructions based on scores
4. After optimization, extract optimized instructions as final prompts

Extension Pattern:
To optimize a different agent, create a new dspy.Module subclass with
custom forward() and metric() methods.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import dspy
import numpy as np
from dspy.utils.callback import BaseCallback

from agent_baselines.solvers.react.dspy_agent import (
    ReactAgentConfig,
    ReactInspectAgent,
)
from agent_baselines.solvers.react.task_loader import (
    create_mixed_dspy_examples,
    interleave_tasks,
    load_samples_from_tasks,
    load_tasks_from_config,
)

logger = logging.getLogger(__name__)

# Target samples per task for MIPRO minibatch sizing
SAMPLES_PER_TASK_FOR_MINIBATCH = 5


# ============================================================================
# Agent Wrapper Abstraction
# ============================================================================


@dataclass
class TaskLoadingConfig:
    """Configuration for loading task data."""

    config_path: str | None
    task_split: str | None
    tasks: list[str] | None
    samples_per_task: int
    train_ratio: float


@dataclass
class OptimizerConfig:
    """Configuration for DSPy optimizer."""

    optimizer_type: str
    optimizer_model: str | None
    num_candidates: int
    num_trials: int | None
    max_bootstrapped_demos: int
    max_labeled_demos: int
    temperature: float
    seed: int | None = None
    minibatch_full_eval_steps: int = 5


@dataclass
class RunConfig:
    """Runtime configuration."""

    run_dir: str
    output_file: str
    verbose_llm: bool
    base_dir: str = ".dspy_cache"


# ============================================================================
# LLM Logging Infrastructure (Optional)
# ============================================================================


class LLMCallLogger(BaseCallback):
    """Callback to log all LLM calls during DSPy optimization.

    This is optional infrastructure for debugging optimizer behavior.
    Separate from evaluation logic to maintain separation of concerns.
    """

    def __init__(self, log_file: str):
        """Initialize logger with output file path.

        Args:
            log_file: Path to file where LLM calls will be logged
        """
        self.log_file = log_file
        self.call_count = 0

        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        with open(log_file, "w") as f:
            f.write(f"DSPy LLM Call Log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def on_lm_start(self, call_id, instance, inputs):
        """Log LLM call inputs."""
        self.call_count += 1

        with open(self.log_file, "a") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(
                f"LLM CALL #{self.call_count} - {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Model: {instance.model}\n")
            f.write(f"Call ID: {call_id}\n")
            f.write(f"{'=' * 80}\n\n")

            # Log messages or prompt
            if inputs.get("messages"):
                f.write("MESSAGES:\n")
                for msg in inputs["messages"]:
                    f.write(f"\n[{msg['role'].upper()}]\n")
                    f.write(f"{msg['content']}\n")
            elif inputs.get("prompt"):
                f.write("PROMPT:\n")
                f.write(f"{inputs['prompt']}\n")

            # Log other parameters
            other_params = {
                k: v for k, v in inputs.items() if k not in ["prompt", "messages"]
            }
            if other_params:
                f.write(f"\nPARAMETERS:\n{json.dumps(other_params, indent=2)}\n")

    def on_lm_end(self, call_id, outputs, exception):
        """Log LLM call outputs."""
        with open(self.log_file, "a") as f:
            f.write(f"\nOUTPUT (Call ID: {call_id}):\n")

            if exception:
                f.write(f"ERROR: {exception}\n")
            else:
                for output in outputs:
                    if isinstance(output, dict):
                        f.write(f"{output.get('text', str(output))}\n")
                    else:
                        f.write(f"{output}\n")

            f.write(f"\n{'=' * 80}\n\n")


def setup_dspy_with_logging(
    optimizer_model: str,
    run_dir: str,
    verbose_llm: bool,
    temperature: float = 1.0,
) -> None:
    """Configure DSPy language model with optional logging.

    Args:
        optimizer_model: Model to use for DSPy optimization
        run_dir: Run directory for logs
        verbose_llm: Whether to enable LLM call logging
        temperature: Temperature for LLM calls (default: 1.0)
    """
    # For reasoning models, we need temperature=1.0 and max_tokens >= 16000
    lm = dspy.LM(model=optimizer_model, temperature=temperature, max_tokens=16000)

    if verbose_llm:
        llm_log_file = f"{run_dir}/llm_calls.log"
        llm_logger = LLMCallLogger(llm_log_file)
        dspy.settings.configure(lm=lm, callbacks=[llm_logger])
        logger.info(f"LLM call logging enabled: {llm_log_file}")
    else:
        dspy.settings.configure(lm=lm)


# ============================================================================
# Optimization Pipeline Functions
# ============================================================================


def setup_optimization_run(
    optimizer_config: OptimizerConfig,
    agent_config: ReactAgentConfig,
    run_config: RunConfig,
) -> None:
    """Setup run directory and configure DSPy."""
    os.makedirs(run_config.run_dir, exist_ok=True)
    os.makedirs(agent_config.log_dir, exist_ok=True)

    # Configure logging to console and file
    log_file = f"{run_config.run_dir}/optimization.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )

    # Configure DSPy's logger to propagate to root logger
    # DSPy sets propagate=False by default, which prevents logs from reaching our file handler
    dspy_logger = logging.getLogger("dspy")
    dspy_logger.handlers.clear()
    dspy_logger.propagate = True

    logger.info(f"Run directory: {run_config.run_dir}")
    logger.info(f"Log file: {log_file}")

    optimizer_model = optimizer_config.optimizer_model or agent_config.eval_models[0]
    setup_dspy_with_logging(
        optimizer_model,
        run_config.run_dir,
        run_config.verbose_llm,
        optimizer_config.temperature,
    )

    logger.info(f"Eval models: {', '.join(agent_config.eval_models)}")
    logger.info(f"Optimizer model: {optimizer_model}")


def load_and_prepare_data(
    task_config: TaskLoadingConfig,
) -> tuple[list, list, list]:
    """Load task data and create train/val splits.

    Returns:
        Tuple of (train_examples, val_examples, task_configs)
    """
    task_configs = load_tasks_from_config(
        config_path=task_config.config_path,
        split=task_config.task_split,
        task_paths=task_config.tasks,
    )

    logger.info(
        f"Loading {len(task_configs)} tasks: {', '.join(tc.name for tc in task_configs)}"
    )

    sample_tuples = load_samples_from_tasks(
        task_configs=task_configs,
        samples_per_task=task_config.samples_per_task,
    )

    train_examples, val_examples = create_mixed_dspy_examples(
        sample_tuples=sample_tuples,
        train_ratio=task_config.train_ratio,
    )
    logger.info(
        f"Loaded {len(sample_tuples)} samples, split into {len(train_examples)} train / {len(val_examples)} val"
    )

    # Interleave training examples (critical for multi-task optimization)
    # MIPRO's dataset observation looks at first ~10 examples, so we need
    # to ensure all tasks appear early to signal multi-task nature
    import random

    rng = random.Random(42)
    train_examples = interleave_tasks(train_examples, rng=rng)

    logger.info(f"Train sample IDs: {[ex.sample_id for ex in train_examples]}")
    logger.info(f"Val sample IDs: {[ex.sample_id for ex in val_examples]}")

    return train_examples, val_examples, task_configs


def run_optimization(
    agent: ReactInspectAgent,
    train_examples: list,
    val_examples: list,
    optimizer_config: OptimizerConfig,
    num_tasks: int,
    run_dir: str,
) -> tuple[Any, str]:
    """Configure optimizer and run optimization.

    Args:
        agent: ReactInspectAgent module (dspy.Module subclass)
        train_examples: Training examples
        val_examples: Validation examples
        optimizer_config: Optimizer configuration
        num_tasks: Number of tasks
        run_dir: Run directory

    Returns:
        Tuple of (optimized_agent, optimizer_name)
    """
    metric = agent.logging_metric

    compile_kwargs = {
        "trainset": train_examples,
        "valset": val_examples,
        "max_bootstrapped_demos": optimizer_config.max_bootstrapped_demos,
        "max_labeled_demos": optimizer_config.max_labeled_demos,
    }

    # Add seed if provided (all optimizers support this)
    if optimizer_config.seed is not None:
        compile_kwargs["seed"] = optimizer_config.seed

    optimizer_type = optimizer_config.optimizer_type

    if optimizer_type == "gepa":
        try:
            from dspy.propose import GEPA

            optimizer = GEPA(
                metric=metric,
                breadth=optimizer_config.num_candidates,
                depth=3,
                init_temperature=optimizer_config.temperature,
            )
            optimizer_name = "GEPA"
        except ImportError:
            raise ImportError(
                "GEPA optimizer not available. Install with: pip install dspy-ai[gepa]"
            )

    elif optimizer_type == "mipro":
        optimizer = dspy.MIPROv2(
            metric=metric,
            auto=None,
            num_candidates=optimizer_config.num_candidates,
            init_temperature=optimizer_config.temperature,
            log_dir=run_dir,
        )
        optimizer_name = "MIPROv2"

        # Calculate num_trials using DSPy's formula if not provided
        # Formula: max(2 * num_vars * log2(N), 1.5 * N)
        # where num_vars = num_predictors * 2 (system + continue message)
        num_trials = optimizer_config.num_trials
        if num_trials is None:
            num_predictors = len(agent.predictors())
            num_vars = num_predictors * 2
            num_trials = int(
                max(
                    2 * num_vars * np.log2(optimizer_config.num_candidates),
                    1.5 * optimizer_config.num_candidates,
                )
            )

        compile_kwargs["num_trials"] = num_trials
        compile_kwargs["minibatch_full_eval_steps"] = (
            optimizer_config.minibatch_full_eval_steps
        )

        # Calculate adaptive minibatch size
        # Ensures adequate coverage across all tasks during optimization
        desired_minibatch = num_tasks * SAMPLES_PER_TASK_FOR_MINIBATCH
        compile_kwargs["minibatch_size"] = min(
            len(train_examples), len(val_examples), desired_minibatch
        )
        compile_kwargs["minibatch"] = True

        if len(val_examples) < desired_minibatch:
            logger.warning(
                f"Validation set size ({len(val_examples)}) limits minibatch size. "
                f"Consider increasing samples_per_task for better multi-task coverage."
            )

    elif optimizer_type == "bootstrap":
        optimizer = dspy.BootstrapFewShot(metric=metric)
        optimizer_name = "BootstrapFewShot"

    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Choose from: 'gepa', 'mipro', 'bootstrap'"
        )

    logger.info(
        f"Optimizer config: {json.dumps(asdict(optimizer_config), default=str)}"
    )
    logger.info(
        f"Compile kwargs: {json.dumps({k: v for k, v in compile_kwargs.items() if k not in ['trainset', 'valset']}, default=str)}"
    )
    logger.info(
        f"Starting {optimizer_name} optimization with {len(train_examples)} train samples..."
    )

    optimized_agent = optimizer.compile(agent, **compile_kwargs)

    logger.info("Optimization complete")

    return optimized_agent, optimizer_name


# ============================================================================
# Main Optimization Function
# ============================================================================


def optimize_react_prompts(
    task_config: TaskLoadingConfig,
    agent_config: ReactAgentConfig,
    optimizer_config: OptimizerConfig,
    run_config: RunConfig,
) -> dict:
    """Optimize ReAct agent prompts using DSPy across multiple tasks and models.

    This is the main entry point that orchestrates the optimization pipeline.

    Args:
        task_config: Configuration for loading task data
        agent_config: Configuration for ReactInspectAgent
        optimizer_config: Configuration for DSPy optimizer
        run_config: Runtime configuration

    Returns:
        Dictionary with optimized prompts and metadata
    """
    setup_optimization_run(optimizer_config, agent_config, run_config)
    train_examples, val_examples, task_configs = load_and_prepare_data(task_config)

    agent = ReactInspectAgent(agent_config)
    logger.info(f"Agent config: {json.dumps(asdict(agent_config))}")

    optimized_agent, optimizer_name = run_optimization(
        agent=agent,
        train_examples=train_examples,
        val_examples=val_examples,
        optimizer_config=optimizer_config,
        num_tasks=len(task_configs),
        run_dir=run_config.run_dir,
    )

    # Extract optimized prompts
    # After optimization, the agent's forward() method returns the optimized
    # signature instructions, which become our final agent prompts
    # Note: We don't pass sample data here, so forward() just returns the prompts
    # without running any evals (sample_id, task_path, primary_metric are optional)
    optimized_prediction = optimized_agent()

    optimizer_model = optimizer_config.optimizer_model or agent_config.eval_models[0]

    optimized_prompts = {
        "system_message": optimized_prediction.system_message,
        "continue_message": optimized_prediction.continue_message,
        "metadata": {
            "eval_models": agent_config.eval_models,
            "optimizer_model": optimizer_model,
            "optimizer": optimizer_name,
            "num_candidates": (
                optimizer_config.num_candidates
                if optimizer_config.optimizer_type != "bootstrap"
                else None
            ),
            "train_samples": len(train_examples),
            "val_samples": len(val_examples),
            "tasks": [tc.name for tc in task_configs],
            "task_paths": [tc.path for tc in task_configs],
            "task_split": task_config.task_split,
            "samples_per_task": task_config.samples_per_task,
            "train_sample_ids": [ex.sample_id for ex in train_examples],
            "val_sample_ids": [ex.sample_id for ex in val_examples],
        },
    }

    output_path = Path(__file__).parent / run_config.output_file
    with open(output_path, "w") as f:
        json.dump(optimized_prompts, f, indent=2)

    logger.info(f"Saved optimized prompts to: {output_path}")

    # Also save sample IDs to run directory for easy reference
    sample_ids_file = f"{run_config.run_dir}/sample_ids.json"
    with open(sample_ids_file, "w") as f:
        json.dump(
            {
                "train_sample_ids": [ex.sample_id for ex in train_examples],
                "val_sample_ids": [ex.sample_id for ex in val_examples],
            },
            f,
            indent=2,
        )
    logger.info(f"Saved sample IDs to: {sample_ids_file}")

    if val_examples:
        val_score = agent.logging_metric(val_examples[0], optimized_prediction)
        logger.info(f"Validation score: {val_score:.4f}")

    logger.info(f"Run directory: {run_config.run_dir}")

    return optimized_prompts


# ============================================================================
# CLI Interface
# ============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize ReAct agent prompts using DSPy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data loading
    data_group = parser.add_argument_group("Data Loading")
    data_group.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to astabench config YAML (defaults to astabench v1.0.0)",
    )
    data_group.add_argument(
        "--task-split",
        type=str,
        default="validation",
        help="Which split to use from config. Mutually exclusive with --tasks.",
    )
    data_group.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task paths (e.g., 'astabench/sqa_dev'). Mutually exclusive with --task-split.",
    )
    data_group.add_argument(
        "--samples-per-task",
        type=int,
        default=5,
        help="Max samples per task",
    )
    data_group.add_argument(
        "--train-ratio",
        type=float,
        default=0.33,
        help="Train/val split ratio (0.8 would be 80% train 20% val).  Note: MIPRO evalutes on the validation set and uses the train set for fewshot examples and dataset analysis.  This means it's generally desirable to have at least 1 train example per task, but additional ones have diminishing returns, while it's crucial to have sufficient validation examples for reliable evaluation.",
    )

    # Agent configuration
    agent_group = parser.add_argument_group("Agent Configuration")
    agent_group.add_argument(
        "--models",
        type=str,
        default="openai/gpt-4o",
        help="Comma-separated models for evaluation (scores averaged)",
    )
    agent_group.add_argument(
        "--agent-max-steps",
        type=int,
        default=10,
        help="Maximum agent steps",
    )

    # Optimizer configuration
    opt_group = parser.add_argument_group("Optimizer Configuration")
    opt_group.add_argument(
        "--optimizer",
        type=str,
        default="mipro",
        choices=["gepa", "mipro", "bootstrap"],
        help="DSPy optimizer to use",
    )
    opt_group.add_argument(
        "--optimizer-model",
        type=str,
        default=None,
        help="Model for DSPy optimization (defaults to first eval model)",
    )
    opt_group.add_argument(
        "--num-candidates",
        type=int,
        default=5,
        help="Number of candidate prompts",
    )
    opt_group.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="MIPRO trials (defaults to DSPy formula)",
    )
    opt_group.add_argument(
        "--optimizer-temperature",
        type=float,
        default=1.0,
        help="Temperature for prompt generation",
    )
    opt_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    opt_group.add_argument(
        "--minibatch-full-eval-steps",
        type=int,
        default=5,
        help="MIPRO: Steps between full validation evals (default: 5)",
    )
    opt_group.add_argument(
        "--max-bootstrapped-demos",
        type=int,
        default=0,
        help="Max bootstrapped demonstrations (default: 0 to minimize bootstrap compute cost. Forward() does run actual evals and cache results, so bootstrap works but requires running ~3-12 evals per candidate set)",
    )
    opt_group.add_argument(
        "--max-labeled-demos",
        type=int,
        default=0,
        help="Max labeled demonstrations (default: 0 since few-shot prompting typically doesn't help with instruction optimization)",
    )

    # Evaluation configuration
    eval_group = parser.add_argument_group("Evaluation Configuration")
    eval_group.add_argument(
        "--eval-timeout",
        type=int,
        default=1200,
        help="Timeout per sample evaluation (seconds)",
    )

    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output",
        type=str,
        default="optimized_prompts.json",
        help="Output file for optimized prompts",
    )
    output_group.add_argument(
        "--base-dir",
        type=str,
        default=".dspy_cache",
        help="Base directory for optimization runs",
    )
    output_group.add_argument(
        "--verbose-llm",
        action="store_true",
        help="Enable LLM call logging",
    )

    args = parser.parse_args()

    if args.task_split != "validation" and args.tasks:
        parser.error("Cannot specify both --task-split and --tasks")

    models = [m.strip() for m in args.models.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    task_split = None if tasks else args.task_split

    agent_kwargs = {"max_steps": args.agent_max_steps}

    # Create run directory upfront so we can configure log paths
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = f"{args.base_dir}/run_{timestamp}"

    task_config = TaskLoadingConfig(
        config_path=args.config_path,
        task_split=task_split,
        tasks=tasks,
        samples_per_task=args.samples_per_task,
        train_ratio=args.train_ratio,
    )

    agent_config = ReactAgentConfig(
        eval_models=models,
        agent_kwargs=agent_kwargs,
        eval_timeout=args.eval_timeout,
        log_dir=f"{run_dir}/eval_logs",
    )

    optimizer_config = OptimizerConfig(
        optimizer_type=args.optimizer,
        optimizer_model=args.optimizer_model,
        num_candidates=args.num_candidates,
        num_trials=args.num_trials,
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        max_labeled_demos=args.max_labeled_demos,
        temperature=args.optimizer_temperature,
        seed=args.seed,
        minibatch_full_eval_steps=args.minibatch_full_eval_steps,
    )

    run_config = RunConfig(
        run_dir=run_dir,
        output_file=args.output,
        verbose_llm=args.verbose_llm,
        base_dir=args.base_dir,
    )

    optimize_react_prompts(
        task_config=task_config,
        agent_config=agent_config,
        optimizer_config=optimizer_config,
        run_config=run_config,
    )
