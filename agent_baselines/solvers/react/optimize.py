"""Script to optimize ReAct agent prompts using DSPy GEPA optimizer.

This script:
1. Loads the sqa_dev dataset
2. Splits it into train (80%) and validation (20%) sets
3. Uses GEPA to optimize system and continue message prompts
4. Saves the optimized prompts to a JSON file

Key architecture:
- Each DSPy Example represents one sample from sqa_dev
- The metric function runs inspect_ai.eval() on that specific sample
- DSPy optimizes prompts based on per-sample scores
"""

import json
import logging
import os
import textwrap
import time
from pathlib import Path

import dspy
import numpy as np
from dspy.utils.callback import BaseCallback

from agent_baselines.solvers.react.basic_agent import (
    DEFAULT_SUBMIT_NAME,
)
from agent_baselines.solvers.react.dspy_agent import (
    DSPyReActPrompts,
)
from agent_baselines.solvers.react.parallel_eval import eval_in_subprocess

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Base directory for all DSPy optimization runs
DSPY_CACHE_BASE = ".dspy_cache"


def log_block(
    text: str,
    char: str = "=",
    width: int = 80,
    trailing_separator: bool = True,
):
    """Log a formatted block with leading and optional trailing separators.

    Args:
        text: Text to log (multiline text will be dedented automatically)
        char: Character for separator lines (default: "=")
        width: Width of separator lines (default: 80)
        trailing_separator: Add trailing separator after text (default: True)
    """
    separator = char * width

    # Log leading separator
    logger.info(separator)

    # Dedent and log text (handling multiline blocks)
    dedented_text = textwrap.dedent(text).strip()
    for line in dedented_text.split("\n"):
        logger.info(line)

    # Log trailing separator if requested
    if trailing_separator:
        logger.info(separator)


# Generic task description for generating final optimized prompts (multi-task optimization)
GENERIC_TASK_DESCRIPTION = """You will be given a task to complete. Use the available tools to help you solve the task, doing reasoning before each action to explain your approach."""

# Target number of samples per task when calculating adaptive minibatch size.
# This ensures adequate coverage across all task types during MIPRO optimization.
SAMPLES_PER_TASK_FOR_MINIBATCH = 5


def create_run_directory(base_dir: str = DSPY_CACHE_BASE) -> str:
    """Create a timestamped run directory for this optimization run.

    Args:
        base_dir: Base directory for all runs (default: .dspy_cache)

    Returns:
        Path to the created run directory (e.g., .dspy_cache/run_20251106_123045)
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = f"{base_dir}/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


class LLMCallLogger(BaseCallback):
    """Callback to log all LLM calls during DSPy optimization to a file."""

    def __init__(self, log_file: str):
        """Initialize logger with output file path.

        Args:
            log_file: Path to file where LLM calls will be logged
        """
        self.log_file = log_file
        self.call_count = 0

        # Create log directory if needed
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Clear/create the log file
        with open(log_file, "w") as f:
            f.write(
                f"DSPy LLM Call Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write("=" * 80 + "\n\n")

    def on_lm_start(self, call_id, instance, inputs):
        """Log LLM call inputs."""
        self.call_count += 1

        with open(self.log_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(
                f"LLM CALL #{self.call_count} - {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Model: {instance.model}\n")
            f.write(f"Call ID: {call_id}\n")
            f.write(f"{'='*80}\n\n")

            # Log messages or prompt
            if inputs.get("messages"):
                f.write("MESSAGES:\n")
                for msg in inputs["messages"]:
                    f.write(f"\n[{msg['role'].upper()}]\n")
                    f.write(f"{msg['content']}\n")
            elif inputs.get("prompt"):
                f.write("PROMPT:\n")
                f.write(f"{inputs['prompt']}\n")

            # Log any additional kwargs (temperature, max_tokens, etc.)
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
                for i, output in enumerate(outputs):
                    if isinstance(output, dict):
                        f.write(f"{output.get('text', str(output))}\n")
                    else:
                        f.write(f"{output}\n")

            f.write(f"\n{'='*80}\n\n")


def load_tasks_from_config(
    config_path: str | None = None,
    split: str | None = "validation",
    task_paths: list[str] | None = None,
) -> list[dict]:
    """Load task configurations from astabench config.

    Args:
        config_path: Path to astabench config YAML (default: uses astabench's default)
        split: Which split to load tasks from (default: "validation"). Mutually exclusive with task_paths.
        task_paths: Specific task paths to load (e.g., ["astabench/sqa_dev"]). Mutually exclusive with split.

    Returns:
        List of task configs with name, path, primary_metric, tags

    Raises:
        ValueError: If both split and task_paths are specified, or if a task path is not found in config
    """
    import importlib.resources
    import os

    import yaml
    from agenteval.models import SuiteConfig

    # Validate mutually exclusive args
    if split is not None and task_paths is not None:
        raise ValueError(
            "Cannot specify both 'split' and 'task_paths' - they are mutually exclusive"
        )

    if split is None and task_paths is None:
        raise ValueError("Must specify either 'split' or 'task_paths'")

    # Use astabench's default config if not specified
    if config_path is None:
        with importlib.resources.path("astabench.config", "v1.0.0.yml") as path:
            config_path = os.path.abspath(path)

    # Load and parse config
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    suite_config = SuiteConfig.model_validate(config_data)

    # Get all tasks from all splits (we need to search across splits for specific paths)
    all_tasks = []
    for split_data in config_data.get("splits", []):
        split_name = split_data.get("name")
        tasks_in_split = suite_config.get_tasks(split_name)
        all_tasks.extend(tasks_in_split)

    # If specific task paths requested, filter to those
    if task_paths:
        task_configs = []
        for task_path in task_paths:
            # Find task with matching path
            matching_task = None
            for task in all_tasks:
                if task.path == task_path:
                    matching_task = task
                    break

            if matching_task is None:
                raise ValueError(
                    f"Task path '{task_path}' not found in config. "
                    f"Available paths: {[t.path for t in all_tasks]}"
                )

            task_configs.append(matching_task)

        return task_configs

    # Otherwise return all tasks from the specified split
    return suite_config.get_tasks(split)


def create_metric_function(
    model_names: list[str],
    optimizer_type: str = "mipro",
    eval_timeout: int = 600,
    agent_kwargs: dict | None = None,
    eval_log_dir: str | None = None,
):
    """Create a metric function that evaluates agent performance across tasks and models.

    This metric function:
    1. Takes a DSPy prediction (containing candidate prompts)
    2. Runs inspect_ai.eval() in a subprocess (for parallelization)
    3. Evaluates on all specified models and averages the scores
    4. Returns the averaged score (or score + feedback for GEPA)

    Args:
        model_names: List of model names to evaluate on (scores will be averaged)
        optimizer_type: Type of optimizer ("mipro" or "gepa") for feedback support
        eval_timeout: Timeout in seconds for each sample evaluation
        agent_kwargs: Additional keyword arguments to pass to the agent solver (e.g., {"max_steps": 10})
        eval_log_dir: Directory for eval logs (default: uses global EVAL_STD_LOG_DIR)

    Returns:
        A function that takes (example, prediction, trace, pred_name, pred_trace) and returns:
        - float score for MIPROv2
        - dict with {"score": float, "feedback": str} for GEPA
    """
    if agent_kwargs is None:
        agent_kwargs = {}

    def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        """DSPy metric function for multi-task, multi-model evaluation.

        Args:
            example: DSPy Example containing sample_id, task_path, and primary_metric
            prediction: DSPy Prediction with system_message and continue_message
            trace: Optional trace information

        Returns:
            Score averaged across models for this sample
        """
        # Get the prompts from the prediction
        system_message = prediction.system_message
        continue_message = prediction.continue_message

        # Extract task info from example
        sample_id = example.sample_id
        task_path = example.task_path
        primary_metric = example.primary_metric
        task_name = example.task_name

        # Use provided eval_log_dir or fall back to default
        log_dir = (
            eval_log_dir if eval_log_dir is not None else f"{DSPY_CACHE_BASE}/eval_logs"
        )

        # Create log directory for this evaluation
        os.makedirs(log_dir, exist_ok=True)

        # Create unique log file for this sample evaluation
        # Use timestamp to avoid collisions if same sample is evaluated multiple times
        import time

        timestamp = int(time.time() * 1000)
        std_log_file = f"{log_dir}/{sample_id}_{timestamp}.log"

        # Print clearer DSPy-level logging
        log_block(
            f"""
            DSPy Metric Evaluation:
              Sample: {sample_id}
              Task: {task_name} ({task_path})
              Models: {', '.join(model_names)}
              Candidate prompts being tested...
              (Detailed eval output → {std_log_file})
            """,
        )

        # Run eval in subprocess (enables parallelization)
        try:
            score_value = eval_in_subprocess(
                sample_id=sample_id,
                system_message=system_message,
                continue_message=continue_message,
                model_names=model_names,
                task_path=task_path,
                primary_metric=primary_metric,
                timeout=eval_timeout,
                std_log_file=std_log_file,
                agent_kwargs=agent_kwargs,
            )

            # Print result
            logger.info(
                f"✓ Sample {sample_id} completed: avg score = {score_value:.4f}\n"
            )

            return score_value

        except Exception as e:
            logger.info(f"✗ Sample {sample_id} failed: {e}\n")
            logger.error(
                f"Failed to evaluate sample {sample_id} on task {task_path}: {e}"
            )
            raise

    return metric


def load_samples_from_tasks(
    task_configs: list[dict],
    samples_per_task: int | None = None,
) -> list[tuple]:
    """Load samples from multiple tasks.

    Args:
        task_configs: List of task configs from load_tasks_from_config()
        samples_per_task: Optional limit on samples per task

    Returns:
        List of (task_path, primary_metric, task_name, sample) tuples
    """
    from inspect_ai._eval.loader import load_tasks

    all_sample_tuples = []

    for task_config in task_configs:
        task_path = task_config.path
        primary_metric = task_config.primary_metric
        task_name = task_config.name

        logger.info(f"Loading task: {task_name} ({task_path})")

        # Load task using inspect_ai's loader
        tasks = load_tasks([task_path], task_args={})
        assert len(tasks) == 1, f"Expected 1 task, got {len(tasks)}"
        task = tasks[0]

        # Get samples from dataset
        samples = list(task.dataset)

        # Limit samples if specified
        if samples_per_task:
            samples = samples[:samples_per_task]

        logger.info(f"  Loaded {len(samples)} samples from {task_name}")

        # Create tuples with task metadata
        for sample in samples:
            all_sample_tuples.append((task_path, primary_metric, task_name, sample))

    return all_sample_tuples


def create_mixed_dspy_examples(
    sample_tuples: list[tuple],
    train_ratio: float = 0.8,
) -> tuple[list, list]:
    """Convert multi-task samples to DSPy examples and split into train/val.

    Each DSPy example contains:
    - task_description: Actual task content including the question/problem from sample.input
    - submit_function_name: Name of the submit function
    - sample: Full Sample object (for metric evaluation)
    - sample_id: The ID of the sample (for running eval on just this sample)
    - task_path: Path to the task (e.g., "astabench/sqa_dev")
    - primary_metric: Primary metric for this task (e.g., "global_avg/mean")
    - task_name: Human-readable task name

    This provides optimizers with actual task content they need for bootstrapping
    and instruction generation, while still using sample_id for evaluation.

    Args:
        sample_tuples: List of (task_path, primary_metric, task_name, sample) tuples
        train_ratio: Ratio of data to use for training (default 0.8)

    Returns:
        Tuple of (train_examples, val_examples)
    """
    # Split into train/val
    split_idx = int(len(sample_tuples) * train_ratio)
    train_tuples = sample_tuples[:split_idx]
    val_tuples = sample_tuples[split_idx:]

    logger.info(
        f"Creating DSPy examples: {len(train_tuples)} train, {len(val_tuples)} val"
    )

    def create_examples(tuples):
        examples = []
        for task_path, primary_metric, task_name, sample in tuples:
            # Use generic task description (same for all) to ensure universal prompts
            # Real questions stored as metadata for dataset summary observation
            example = dspy.Example(
                task_description=GENERIC_TASK_DESCRIPTION,
                submit_function_name=DEFAULT_SUBMIT_NAME,
                question=sample.input,
                task_name=task_name,
                sample_id=sample.id,
                task_path=task_path,
                primary_metric=primary_metric,
                sample=sample,
            ).with_inputs("task_description", "submit_function_name")

            examples.append(example)
        return examples

    train_examples = create_examples(train_tuples)
    val_examples = create_examples(val_tuples)

    return train_examples, val_examples


def interleave_tasks(examples: list, rng=None) -> list:
    """Interleave examples from different tasks to ensure early diversity.

    For multi-task optimization, MIPRO's dataset observation views the first ~10
    examples sequentially. If all early examples are from one task, MIPRO will
    incorrectly believe it's single-task optimization.

    This function shuffles within each task, then interleaves tasks in round-robin
    fashion (e.g., [task_a1, task_b1, task_c1, task_a2, task_b2, task_c2, ...]).

    Args:
        examples: List of DSPy examples with task_name attribute
        rng: Random number generator for reproducibility

    Returns:
        Interleaved list ensuring all tasks appear in the first N examples
    """
    import random

    rng = rng or random

    # Group examples by task
    by_task = {}
    for ex in examples:
        task = ex.task_name
        by_task.setdefault(task, []).append(ex)

    # Shuffle within each task
    for task in sorted(by_task.keys()):  # Sort for reproducibility
        rng.shuffle(by_task[task])

    # Interleave tasks in round-robin fashion
    interleaved = []
    task_names = sorted(by_task.keys())
    max_task_size = max(len(group) for group in by_task.values())

    for i in range(max_task_size):
        for task in task_names:
            if i < len(by_task[task]):
                interleaved.append(by_task[task][i])

    # Log distribution for verification
    logger.info("Interleaved task distribution (first 20 examples):")
    for i, ex in enumerate(interleaved[:20]):
        logger.info(f"  [{i:2d}] {ex.task_name}")

    return interleaved


def optimize_prompts(
    models: list[str] | str = "openai/gpt-4o",
    optimizer_model: str | None = None,
    optimizer_type: str = "mipro",
    samples_per_task: int = 5,
    train_ratio: float = 0.8,
    num_candidates: int = 5,
    num_trials: int | None = None,
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 3,
    output_file: str = "optimized_prompts.json",
    config_path: str | None = None,
    task_split: str | None = "validation",
    tasks: list[str] | None = None,
    eval_timeout: int = 600,
    optimizer_temperature: float = 1.0,
    agent_kwargs: dict | None = None,
    verbose_llm: bool = False,
    base_cache_dir: str = DSPY_CACHE_BASE,
):
    """Run DSPy optimization on ReAct agent prompts across multiple tasks and models.

    Args:
        models: Model(s) to use for agent evaluation (can be single string or list)
        optimizer_model: Model to use for DSPy optimization (defaults to first model)
        optimizer_type: Which optimizer to use: "gepa", "mipro", or "bootstrap"
        samples_per_task: Maximum number of samples to use per task
        train_ratio: Ratio of data to use for training
        num_candidates: Number of candidate prompts (for GEPA/MIPRO)
        num_trials: Number of trials for MIPRO (defaults to DSPy's formula: max(2*num_vars*log2(N), 1.5*N))
        max_bootstrapped_demos: Max bootstrapped demonstrations
        max_labeled_demos: Max labeled demonstrations
        output_file: Where to save the optimized prompts
        config_path: Path to astabench config (default: uses astabench default)
        task_split: Which split to use from config (default: "validation"). Mutually exclusive with tasks.
        tasks: Specific task paths to use (e.g., ["astabench/sqa_dev"]). Mutually exclusive with task_split.
        eval_timeout: Timeout in seconds for each sample evaluation
        optimizer_temperature: Temperature for DSPy optimizer prompt generation
        agent_kwargs: Additional keyword arguments to pass to the agent solver (e.g., {"max_steps": 10}).
                     Defaults to {"max_steps": 10} for backward compatibility.
        verbose_llm: If True, log all LLM calls during optimization to a timestamped file
        base_cache_dir: Base directory for all caching (default: .dspy_cache)
    """
    if agent_kwargs is None:
        agent_kwargs = {"max_steps": 10}

    # Create run directory for this optimization run
    run_dir = create_run_directory(base_cache_dir)

    log_block(
        f"""
        RUN DIRECTORY CREATED

        All outputs for this optimization run will be saved to:
          {run_dir}

        This includes:
          - LLM call logs (if --verbose-llm)
          - Evaluation logs (per-sample)
          - Evaluated programs (MIPRO candidates)
          - Inspect AI eval outputs
        """,
    )

    # Convert single model to list
    if isinstance(models, str):
        models = [models]

    # Set up DSPy language model for generating candidate prompts
    optimizer_model = optimizer_model or models[0]
    lm = dspy.LM(model=optimizer_model)

    # Configure LLM call logging if requested
    if verbose_llm:
        llm_log_file = f"{run_dir}/llm_calls.log"
        llm_logger = LLMCallLogger(llm_log_file)
        dspy.settings.configure(lm=lm, callbacks=[llm_logger])
        logger.info(f"LLM call logging enabled: {llm_log_file}")
    else:
        # Allow DSPy to use default parallelization (we use subprocesses for eval isolation)
        dspy.settings.configure(lm=lm)

    logger.info(f"Agent models: {', '.join(models)}")
    logger.info(f"Optimizer model: {optimizer_model}")
    logger.info("Loading tasks from config...")
    logger.info(
        "Running evaluations in parallel using subprocess isolation (bypasses inspect_ai limitation)"
    )

    # Load task configs from astabench
    task_configs = load_tasks_from_config(
        config_path=config_path,
        split=task_split,
        task_paths=tasks,
    )

    if tasks:
        logger.info(f"Loaded {len(task_configs)} specified tasks")
    else:
        logger.info(f"Loaded {len(task_configs)} tasks from {task_split} split")

    # Load samples from all tasks
    sample_tuples = load_samples_from_tasks(
        task_configs=task_configs,
        samples_per_task=samples_per_task,
    )
    logger.info(f"Loaded {len(sample_tuples)} total samples across all tasks")

    # Create mixed DSPy examples and split into train/val
    train_examples, val_examples = create_mixed_dspy_examples(
        sample_tuples=sample_tuples,
        train_ratio=train_ratio,
    )

    logger.info(f"Using {len(train_examples)} train, {len(val_examples)} val examples")

    # Create the DSPy module
    react_prompts = DSPyReActPrompts()

    # Create metric function that uses subprocess-based eval
    logger.info("Creating metric function (subprocess-based for parallelization)...")
    eval_log_dir = f"{run_dir}/eval_logs"
    metric = create_metric_function(
        models,
        eval_timeout=eval_timeout,
        agent_kwargs=agent_kwargs,
        eval_log_dir=eval_log_dir,
    )

    # Set up optimizer based on type
    logger.info(f"Setting up {optimizer_type} optimizer...")

    # Initialize mipro_num_trials (only used for MIPRO)
    mipro_num_trials = None

    if optimizer_type == "gepa":
        try:
            from dspy.propose import GEPA

            optimizer = GEPA(
                metric=metric,
                breadth=num_candidates,
                depth=3,
                init_temperature=optimizer_temperature,
            )
            optimizer_name = "GEPA"
        except ImportError:
            raise ImportError(
                "GEPA optimizer not available. Install with: pip install dspy-ai[gepa]"
            )

    elif optimizer_type == "mipro":
        # MIPRO has two modes:
        # 1. Auto mode: MIPROv2 decides num_candidates/num_trials
        # 2. Manual mode (auto=None): must provide num_candidates and num_trials
        # We use manual mode to allow user control
        optimizer = dspy.MIPROv2(
            metric=metric,
            auto=None,  # Enable manual mode
            num_candidates=num_candidates,
            init_temperature=optimizer_temperature,
            log_dir=run_dir,  # Save all candidate programs during optimization
        )
        optimizer_name = "MIPROv2"
        # Calculate num_trials for compile() call (MIPRO needs it there, not in __init__)
        # Use DSPy's official formula: max(2 * num_vars * log₂(N), 1.5 * N)
        # where num_vars = num_predictors * 2 for few-shot mode (bootstrap demos)
        # Source: DSPy MIPROv2 documentation
        if num_trials is None:
            num_predictors = len(react_prompts.predictors())
            num_vars = num_predictors * 2  # Assuming few-shot mode (bootstrap demos)
            mipro_num_trials = int(
                max(2 * num_vars * np.log2(num_candidates), 1.5 * num_candidates)
            )
        else:
            mipro_num_trials = num_trials
        logger.info(
            f"Using manual mode with num_candidates={num_candidates}, num_trials={mipro_num_trials} (DSPy formula)"
        )

    elif optimizer_type == "bootstrap":
        # BootstrapFewShot: max_bootstrapped_demos and max_labeled_demos
        # are passed to compile(), not __init__()
        optimizer = dspy.BootstrapFewShot(metric=metric)
        optimizer_name = "BootstrapFewShot"

    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Choose from: 'gepa', 'mipro', 'bootstrap'"
        )

    logger.info(f"Using {optimizer_name} optimizer")

    # Run optimization
    # Interleave tasks to ensure MIPRO's dataset observation sees all task types
    # MIPRO views first ~10 examples sequentially for dataset summarization
    # Interleaving ensures all tasks appear in those first examples
    import random

    rng = random.Random(42)  # Reproducible interleaving
    train_examples_interleaved = interleave_tasks(train_examples, rng=rng)

    log_block(
        f"""
        STARTING DSPY OPTIMIZATION

        Optimizer: {optimizer_name}
        Training samples: {len(train_examples_interleaved)}
        Models being optimized for: {', '.join(models)}
    """,
    )

    # Prepare compile() arguments based on optimizer type
    compile_kwargs = {
        "trainset": train_examples_interleaved,
        "max_bootstrapped_demos": max_bootstrapped_demos,
        "max_labeled_demos": max_labeled_demos,
    }

    # MIPRO needs num_trials and minibatch settings in compile() call
    if optimizer_type == "mipro":
        compile_kwargs["num_trials"] = mipro_num_trials
        # Adaptive minibatch sizing for multi-task optimization
        # Ensure adequate coverage across all task types
        # Must not exceed validation set size (DSPy internal requirement)
        num_tasks = len(task_configs)
        desired_minibatch = num_tasks * SAMPLES_PER_TASK_FOR_MINIBATCH
        minibatch_size = min(
            len(train_examples_interleaved), len(val_examples), desired_minibatch
        )
        compile_kwargs["minibatch_size"] = minibatch_size
        compile_kwargs["minibatch"] = True
        logger.info(
            f"Minibatch size: {minibatch_size} "
            f"(adaptive: min(train={len(train_examples_interleaved)}, val={len(val_examples)}, "
            f"{num_tasks} tasks × {SAMPLES_PER_TASK_FOR_MINIBATCH}))"
        )

        # Warn if validation set size is limiting the adaptive sizing
        if len(val_examples) < desired_minibatch and len(val_examples) < len(
            train_examples_interleaved
        ):
            logger.warning(
                f"Validation set size ({len(val_examples)}) is limiting minibatch size. "
                f"Desired {desired_minibatch} samples "
                f"({num_tasks} tasks × {SAMPLES_PER_TASK_FOR_MINIBATCH}) for multi-task coverage. "
                f"Consider increasing --samples-per-task or adjusting train_ratio."
            )
        logger.info(f"Number of trials: {mipro_num_trials}")
        logger.info(
            f"Each trial will evaluate {minibatch_size} sample(s) on {len(models)} model(s)"
        )
        logger.info(
            f"Estimated total evaluations: ~{mipro_num_trials * minibatch_size * len(models)}"
        )

    logger.info("Detailed per-sample logs are being written to: %s", eval_log_dir)
    logger.info("Candidate programs will be saved to: %s/evaluated_programs/", run_dir)

    optimized_module = optimizer.compile(react_prompts, **compile_kwargs)

    logger.info("OPTIMIZATION COMPLETE!")

    # Generate optimized prompts
    # Use generic task description since we want universal prompts for multi-task optimization
    logger.info("\nGenerating optimized prompts...")
    optimized_prediction = optimized_module(
        task_description=GENERIC_TASK_DESCRIPTION,
        submit_function_name=DEFAULT_SUBMIT_NAME,
    )

    # Collect task information for metadata
    task_names = [config.name for config in task_configs]
    task_paths = [config.path for config in task_configs]

    optimized_prompts = {
        "system_message": optimized_prediction.system_message,
        "continue_message": optimized_prediction.continue_message,
        "metadata": {
            "agent_models": models,
            "optimizer_model": optimizer_model,
            "train_samples": len(train_examples),
            "val_samples": len(val_examples),
            "optimizer": optimizer_name,
            "num_candidates": num_candidates if optimizer_type != "bootstrap" else None,
            "tasks": task_names,
            "task_paths": task_paths,
            "task_split": task_split,
            "samples_per_task": samples_per_task,
        },
    }

    # Save to file
    output_path = Path(__file__).parent / output_file
    with open(output_path, "w") as f:
        json.dump(optimized_prompts, f, indent=2)

    logger.info("📁 Optimized prompts saved to: %s", output_path)
    logger.info("📝 Optimized System Message: %s", optimized_prompts["system_message"])
    logger.info(
        "📝 Optimized Continue Message: %s", optimized_prompts["continue_message"]
    )

    # Evaluate on validation set
    if val_examples:
        logger.info("\n📊 Evaluating first validation example...")
        val_score = metric(val_examples[0], optimized_prediction)
        logger.info(f"✓ Validation score (first sample): {val_score:.4f}")

    logger.info(f"\n💡 Optimization Details:")
    logger.info(f"   Agent models (for evaluation): {', '.join(models)}")
    logger.info(f"   Optimizer model (for prompt generation): {optimizer_model}")
    logger.info(f"   Tasks optimized: {len(task_configs)} tasks ({task_split} split)")
    logger.info(f"   Total training samples: {len(train_examples)} across all tasks")

    logger.info(f"\n✓ To use the optimized agent on any task, run:")
    logger.info(f"  uv run astabench eval <task_path> \\")
    logger.info(
        f"    --solver agent_baselines/solvers/react/optimized_agent.py@instantiated_optimized_agent \\"
    )
    logger.info(f"    --model <model_name>")
    logger.info(
        f"\n  (The optimized agent will automatically load prompts from {output_path})"
    )

    logger.info(
        f"\n✓ Multi-task optimization completed successfully using subprocess isolation!"
    )

    log_block(
        f"""
        ALL OUTPUTS SAVED TO: {run_dir}

        Run directory contents:
          - llm_calls.log (if --verbose-llm was used)
          - eval_logs/ (per-sample evaluation logs)
          - evaluated_programs/ (MIPRO candidate programs)
          - <inspect_ai_eval_outputs>

        You can review this run's outputs anytime by examining: {run_dir}
        """,
    )

    return optimized_prompts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize ReAct agent prompts across multiple tasks and models using DSPy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        default="openai/gpt-4o",
        help="Comma-separated list of models to evaluate on (scores will be averaged)",
    )
    parser.add_argument(
        "--optimizer-model",
        type=str,
        default=None,
        help="Model to use for DSPy optimization (defaults to first model)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="mipro",
        choices=["gepa", "mipro", "bootstrap"],
        help="Which DSPy optimizer to use",
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=5,
        help="Max samples to use per task",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of samples to use for training vs validation (default: 0.8 = 80/20 split)",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=5,
        help="Number of candidate prompts (for GEPA/MIPRO)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="Number of trials for MIPRO (defaults to DSPy's formula: max(2*num_vars*log2(N), 1.5*N))",
    )
    parser.add_argument(
        "--eval-timeout",
        type=int,
        default=1200,
        help="Timeout in seconds for each sample evaluation",
    )
    parser.add_argument(
        "--optimizer-temperature",
        type=float,
        default=1.0,
        help="Temperature for DSPy optimizer prompt generation (default: 1.0 = neutral)",
    )
    parser.add_argument(
        "--output", type=str, default="optimized_prompts.json", help="Output file"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to astabench config YAML (defaults to astabench's v1.0.0.yml)",
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default="validation",
        help="Which split to use from config (default: validation). Mutually exclusive with --tasks.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of specific task paths (e.g., 'astabench/sqa_dev,astabench/litqa2_validation'). Mutually exclusive with --task-split.",
    )
    parser.add_argument(
        "--agent-max-steps",
        type=int,
        default=10,
        help="Maximum number of steps for the agent (default: 10)",
    )
    parser.add_argument(
        "--verbose-llm",
        action="store_true",
        help="Log all LLM calls during optimization to .dspy_cache/llm_calls_<timestamp>.log",
    )

    args = parser.parse_args()

    # Validate mutually exclusive args
    if args.task_split != "validation" and args.tasks:
        parser.error("Cannot specify both --task-split and --tasks")

    # Parse models list
    models = [m.strip() for m in args.models.split(",")]

    # Parse tasks list if provided
    tasks = None
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]

    # Set task_split to None if using --tasks
    task_split = None if tasks else args.task_split

    # Build agent_kwargs from CLI arguments
    agent_kwargs = {"max_steps": args.agent_max_steps}

    optimize_prompts(
        models=models,
        optimizer_model=args.optimizer_model,
        optimizer_type=args.optimizer,
        samples_per_task=args.samples_per_task,
        train_ratio=args.train_ratio,
        num_candidates=args.num_candidates,
        num_trials=args.num_trials,
        output_file=args.output,
        config_path=args.config_path,
        task_split=task_split,
        tasks=tasks,
        eval_timeout=args.eval_timeout,
        optimizer_temperature=args.optimizer_temperature,
        agent_kwargs=agent_kwargs,
        verbose_llm=args.verbose_llm,
    )
