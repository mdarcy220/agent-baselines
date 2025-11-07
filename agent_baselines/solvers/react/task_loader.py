"""Task data loading utilities for DSPy optimization.

Utilities for loading task configurations and samples from astabench,
converting them to DSPy examples, and preparing train/val splits.
"""

import logging

import dspy

logger = logging.getLogger(__name__)


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
    task_description: str = "You will be given a task to complete. Use the available tools to help you solve the task, doing reasoning before each action to explain your approach.",
    submit_function_name: str = "submit",
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
        task_description: Generic task description for all samples
        submit_function_name: Name of the submit function (default: "submit")

    Returns:
        Tuple of (train_examples, val_examples)
    """
    # Group samples by task to ensure each task contributes to both train and val
    by_task = {}
    for task_path, primary_metric, task_name, sample in sample_tuples:
        by_task.setdefault(task_name, []).append(
            (task_path, primary_metric, task_name, sample)
        )

    # Split each task into train/val
    train_tuples = []
    val_tuples = []
    for task_name in sorted(by_task.keys()):
        task_samples = by_task[task_name]
        split_idx = int(len(task_samples) * train_ratio)
        train_tuples.extend(task_samples[:split_idx])
        val_tuples.extend(task_samples[split_idx:])

    logger.info(
        f"Creating DSPy examples: {len(train_tuples)} train, {len(val_tuples)} val"
    )

    def create_examples(tuples):
        examples = []
        for task_path, primary_metric, task_name, sample in tuples:
            # Use generic task description (same for all) to ensure universal prompts
            # Real questions stored as metadata for dataset summary observation
            example = dspy.Example(
                task_description=task_description,
                submit_function_name=submit_function_name,
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
