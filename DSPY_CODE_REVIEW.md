# DSPy Optimization Code Review - Thorough Mode

**Review Date**: 2025-10-24
**Reviewer**: Claude Code (Code Quality Engineer)
**Mode**: Thorough
**Files Reviewed**:
- `/home/miked/grampro/ai2/nora/other/agent-baselines/agent_baselines/solvers/react/optimize.py`
- `/home/miked/grampro/ai2/nora/other/agent-baselines/agent_baselines/solvers/react/dspy_agent.py`
- `/home/miked/grampro/ai2/nora/other/agent-baselines/agent_baselines/solvers/react/optimized_agent.py`

---

## Executive Summary

The DSPy optimization code is **well-structured and functional**, with good integration between DSPy optimizers and inspect_ai. However, there are **7 critical issues** and **several medium-priority concerns** that should be addressed before production use. The most significant issues relate to:

1. **CRITICAL**: Missing validation in the metric function that could cause silent failures
2. **CRITICAL**: Incorrect parameter passing for BootstrapFewShot optimizer
3. **HIGH**: Potential data leakage in train/val split
4. **HIGH**: Score extraction using fail-silent pattern (violates CLAUDE.md)
5. **MEDIUM**: Missing error handling in several critical paths

---

## Issue Summary by Severity

### Critical Issues (Must Fix)
1. Missing validation for inspect_ai evaluation results
2. Incorrect optimizer parameter passing for BootstrapFewShot
3. Data split lacks shuffling - potential ordering bias
4. Missing tool_call_format parameter in create_agent_with_dspy_prompts

### High Priority Issues (Should Fix)
5. Score extraction uses fail-silent pattern (dict.get equivalent)
6. Potential KeyError when accessing score.value dict
7. Missing validation of DSPy prediction structure

### Medium Priority Issues (Consider Fixing)
8. Inconsistent error handling across functions
9. Missing type hints in several functions
10. Hard-coded constants that should be configurable
11. Limited logging for debugging optimization failures

---

## Detailed Issue Analysis

### CRITICAL ISSUE #1: Missing Validation in Metric Function

**Location**: `optimize.py`, lines 100-130

**Issue**: The metric function does not validate that the inspect_ai evaluation actually succeeded or produced valid results. It checks for empty logs/samples/scores but doesn't verify that the evaluation didn't error out.

**Code**:
```python
# Run eval on just this sample
logs = inspect_eval(
    tasks=[task],
    model=model_name,
    sample_id=sample_id,
    log_dir=".dspy_cache",
    log_level="warning",
)

# Extract score from the specific sample
if not logs or len(logs) == 0:
    logger.warning(f"No logs returned for sample {sample_id}")
    return 0.0
```

**Problem**: `inspect_eval()` could:
- Return logs with errors in them
- Return logs where the agent hit token limits
- Return logs where tools failed to initialize
- Return logs where the sample was skipped

All of these cases would result in returning `0.0`, which tells DSPy "this prompt is bad" when in reality it might be an infrastructure issue.

**Impact**:
- DSPy may incorrectly penalize good prompts due to transient failures
- No distinction between "prompt failed" vs "infrastructure failed"
- Silent failures make debugging optimization issues very difficult

**Recommendation**:
```python
# Check if eval completed successfully
eval_log = logs[0]
if eval_log.status != "success":
    raise RuntimeError(
        f"Evaluation failed for sample {sample_id}: {eval_log.status}. "
        f"Error: {getattr(eval_log, 'error', 'Unknown error')}"
    )

# Also check if the sample itself completed
sample = eval_log.samples[0]
if not sample.completed:
    raise RuntimeError(
        f"Sample {sample_id} did not complete. "
        f"This may indicate agent failure or timeout."
    )
```

**Why this matters**: DSPy optimization is expensive. If half your evaluations are silently failing and returning 0.0, you're optimizing on corrupted data and wasting compute/money.

---

### CRITICAL ISSUE #2: Incorrect BootstrapFewShot Parameters

**Location**: `optimize.py`, lines 293-299

**Issue**: The `max_bootstrapped_demos` and `max_labeled_demos` parameters are passed to BOTH the BootstrapFewShot constructor AND to the compile() call. This is incorrect - they should only go to compile().

**Code**:
```python
elif optimizer_type == "bootstrap":
    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=max_bootstrapped_demos,  # WRONG
        max_labeled_demos=max_labeled_demos,            # WRONG
    )
    optimizer_name = "BootstrapFewShot"
```

Then later:
```python
compile_kwargs = {
    "trainset": train_examples,
    "max_bootstrapped_demos": max_bootstrapped_demos,  # Correct
    "max_labeled_demos": max_labeled_demos,            # Correct
}
```

**Problem**: According to DSPy documentation, `BootstrapFewShot.__init__()` only takes:
- `metric`: The scoring function
- `max_rounds`: Optional number of bootstrapping rounds
- `teacher_settings`: Optional settings for teacher model

The `max_bootstrapped_demos` and `max_labeled_demos` are parameters to `compile()`, not `__init__()`.

**Impact**: This will raise a `TypeError` at runtime when attempting to use the bootstrap optimizer.

**Fix**:
```python
elif optimizer_type == "bootstrap":
    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        # max_bootstrapped_demos and max_labeled_demos go in compile(), not here
    )
    optimizer_name = "BootstrapFewShot"
```

---

### CRITICAL ISSUE #3: Data Split Lacks Shuffling

**Location**: `optimize.py`, lines 135-158

**Issue**: The train/val split is done on the dataset as-is, without shuffling. This could introduce ordering bias.

**Code**:
```python
def load_and_split_data(base_task, train_ratio: float = 0.8, limit: int = None):
    all_samples = list(base_task.dataset)

    if limit:
        all_samples = all_samples[:limit]

    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
```

**Problem**:
- If the dataset is ordered (e.g., by difficulty, topic, or any other attribute), this split will be biased
- Training on "easy" samples and validating on "hard" samples (or vice versa) will give misleading results
- The sqa_dev dataset may have implicit ordering that affects generalization

**Impact**:
- Optimized prompts may not generalize well
- Validation scores may not accurately reflect performance
- Potential data leakage if samples are related/similar in sequence

**Fix**:
```python
import random

def load_and_split_data(base_task, train_ratio: float = 0.8, limit: int = None, seed: int = 42):
    all_samples = list(base_task.dataset)

    # Shuffle to avoid ordering bias
    random.seed(seed)
    random.shuffle(all_samples)

    if limit:
        all_samples = all_samples[:limit]

    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    return train_samples, val_samples
```

---

### CRITICAL ISSUE #4: Missing tool_call_format Parameter

**Location**: `dspy_agent.py`, lines 78-105

**Issue**: The `create_agent_with_dspy_prompts()` function doesn't pass `tool_call_format` to `basic_agent()`, but `basic_agent()` defaults to "native" format. This inconsistency could cause issues.

**Code**:
```python
def create_agent_with_dspy_prompts(
    system_message_text: str,
    continue_message_text: str,
    max_steps: int = 10,
    **kwargs,
) -> Solver:
    # ...
    return basic_agent(
        init=custom_system_message(),
        continue_message=continue_message_text,
        max_steps=max_steps,
        **kwargs,  # tool_call_format could be in kwargs, but not explicit
    )
```

**Problem**:
- If a caller expects to control `tool_call_format`, they need to pass it via `**kwargs`
- This is implicit and easy to miss
- The function signature doesn't document this capability
- Different tool call formats may require different prompt styles

**Impact**:
- Optimized prompts might be optimized for "native" format but used with "text" format (or vice versa)
- Inconsistent behavior between optimization and deployment

**Recommendation**:
```python
def create_agent_with_dspy_prompts(
    system_message_text: str,
    continue_message_text: str,
    max_steps: int = 10,
    tool_call_format: Literal["text", "native"] = "native",  # Explicit parameter
    **kwargs,
) -> Solver:
    """Create a basic_agent solver with custom DSPy-optimized prompts.

    Args:
        system_message_text: The optimized system message
        continue_message_text: The optimized continue message
        max_steps: Maximum number of agent steps
        tool_call_format: Format for tool calls ("text" or "native")
        **kwargs: Additional arguments to pass to basic_agent
    """
    # ...
    return basic_agent(
        init=custom_system_message(),
        continue_message=continue_message_text,
        max_steps=max_steps,
        tool_call_format=tool_call_format,
        **kwargs,
    )
```

---

### HIGH ISSUE #5: Score Extraction Uses Fail-Silent Pattern

**Location**: `optimize.py`, lines 118-126

**Issue**: Violates CLAUDE.md principle of preferring fail-loud code. The score extraction uses an iteration pattern that silently returns 0.0 if the expected structure isn't found.

**Code**:
```python
# Extract global_avg from the sample's scores
if not sample.scores:
    logger.warning(f"No scores for sample {sample_id}")
    return 0.0

# The scores should have global_avg in the value dict
for score in sample.scores:
    if isinstance(score.value, dict) and "global_avg" in score.value:
        score_value = float(score.value["global_avg"])
        logger.info(...)
        return score_value

# Fallback: if no global_avg, return 0
logger.warning(f"No global_avg found in scores for sample {sample_id}")
return 0.0
```

**Problem**:
- Using `"global_avg" in score.value` is equivalent to fail-silent dict.get()
- If the score structure changes, this silently returns 0.0 instead of raising an error
- Impossible to distinguish between "actually scored 0.0" and "failed to extract score"

**CLAUDE.md Principle**: "Prefer code that fails loudly instead of silently. For example, when a key is expected to always be present in a dict, use dict[key] instead of dict.get(key)."

**Fix**:
```python
# Extract global_avg from the sample's scores
if not sample.scores:
    raise ValueError(f"No scores for sample {sample_id}")

# Find the score with global_avg
score_obj = None
for score in sample.scores:
    if isinstance(score.value, dict) and "global_avg" in score.value:
        score_obj = score
        break

if score_obj is None:
    # Log available scores for debugging
    available_scores = [
        (type(s.value), s.value if not isinstance(s.value, dict) else list(s.value.keys()))
        for s in sample.scores
    ]
    raise ValueError(
        f"No global_avg found in scores for sample {sample_id}. "
        f"Available scores: {available_scores}"
    )

# Use fail-loud dict access
score_value = float(score_obj.value["global_avg"])
logger.info(
    f"Sample {sample_id} score: {score_value:.4f} "
    f"(system_msg: {system_message[:50]}...)"
)
return score_value
```

**Why this matters**: During optimization, you want to know immediately if something is broken, not after wasting hours/dollars optimizing on bad data.

---

### HIGH ISSUE #6: Potential KeyError in Score Access

**Location**: `optimize.py`, line 121

**Issue**: After checking `"global_avg" in score.value`, the code uses `score.value["global_avg"]`, which is correct. However, there's a race condition if score.value is modified between check and access.

**Code**:
```python
if isinstance(score.value, dict) and "global_avg" in score.value:
    score_value = float(score.value["global_avg"])  # Could raise KeyError
```

**Problem**:
- While unlikely, if `score.value` is a mutable object shared across threads/processes, it could be modified
- More realistically: if score.value is a property that returns a new dict each time, the check and access see different dicts

**Impact**: Rare but possible runtime error during optimization

**Fix**: Store the dict once:
```python
if isinstance(score.value, dict):
    value_dict = score.value  # Access once
    if "global_avg" not in value_dict:
        raise ValueError(f"Missing global_avg in score for {sample_id}")
    score_value = float(value_dict["global_avg"])
```

---

### HIGH ISSUE #7: Missing Validation of DSPy Prediction Structure

**Location**: `optimize.py`, lines 72-74

**Issue**: The metric function assumes the DSPy prediction has `system_message` and `continue_message` attributes without validation.

**Code**:
```python
def metric(example, prediction, trace=None) -> float:
    # Get the prompts from the prediction
    system_message = prediction.system_message
    continue_message = prediction.continue_message
```

**Problem**:
- If DSPy returns a malformed prediction (e.g., due to LLM output parsing failure), this will raise AttributeError
- No way to distinguish between DSPy failure and metric function failure

**Impact**: Cryptic error messages during optimization

**Fix**:
```python
def metric(example, prediction, trace=None) -> float:
    # Validate prediction structure
    if not hasattr(prediction, 'system_message'):
        raise ValueError(
            f"DSPy prediction missing 'system_message' attribute. "
            f"Prediction: {prediction}"
        )
    if not hasattr(prediction, 'continue_message'):
        raise ValueError(
            f"DSPy prediction missing 'continue_message' attribute. "
            f"Prediction: {prediction}"
        )

    system_message = prediction.system_message
    continue_message = prediction.continue_message

    # Also validate they're not None/empty
    if not system_message or not isinstance(system_message, str):
        raise ValueError(f"Invalid system_message: {system_message}")
    if not continue_message or not isinstance(continue_message, str):
        raise ValueError(f"Invalid continue_message: {continue_message}")
```

---

### MEDIUM ISSUE #8: Inconsistent Error Handling

**Location**: `optimize.py`, multiple locations

**Issue**: Some functions log warnings and return default values, while others might raise exceptions. This inconsistency makes it hard to reason about error propagation.

**Examples**:
```python
# Pattern 1: Log and return default
if not logs or len(logs) == 0:
    logger.warning(f"No logs returned for sample {sample_id}")
    return 0.0

# Pattern 2: Raise exception (GEPA import)
except ImportError:
    raise ImportError(
        "GEPA optimizer not available. Install with: pip install dspy-ai[gepa]"
    )
```

**Recommendation**: Establish consistent error handling strategy:
- **Transient errors** (network issues, rate limits): Log and retry or skip
- **Configuration errors** (wrong parameters): Raise immediately
- **Data errors** (malformed results): Raise with details for debugging
- **Infrastructure errors** (missing dependencies): Raise with actionable message

---

### MEDIUM ISSUE #9: Missing Type Hints

**Location**: `optimize.py`, multiple functions

**Issue**: Several functions lack complete type hints, making it harder to catch type errors.

**Examples**:
```python
def create_metric_function(model_name: str, base_task, tool_config: ToolsetConfig):
    # base_task has no type hint

def create_dspy_examples(samples):
    # samples has no type hint
```

**Recommendation**: Add full type hints:
```python
from inspect_ai.dataset import Sample
from inspect_ai.task import Task

def create_metric_function(
    model_name: str,
    base_task: Task,
    tool_config: ToolsetConfig
) -> Callable[[Any, Any, Any], float]:
    ...

def create_dspy_examples(samples: list[Sample]) -> list[dspy.Example]:
    ...
```

---

### MEDIUM ISSUE #10: Hard-coded Constants

**Location**: Multiple locations

**Issue**: Several constants are hard-coded that should be configurable:

```python
# optimize.py, line 82
max_steps=10,  # Hard-coded in metric function

# optimize.py, line 95
log_dir=".dspy_cache",  # Hard-coded cache directory
log_level="warning",  # Hard-coded log level

# optimize.py, line 268
depth=3,  # GEPA depth hard-coded
```

**Impact**:
- Can't easily test with different max_steps during optimization
- Can't change cache directory without modifying code
- Limited flexibility for experimentation

**Recommendation**: Make these parameters of optimize_prompts():
```python
def optimize_prompts(
    model_name: str = "openai/gpt-4o",
    optimizer_model: str | None = None,
    optimizer_type: str = "mipro",
    max_steps: int = 10,  # Add this
    cache_dir: str = ".dspy_cache",  # Add this
    log_level: str = "warning",  # Add this
    gepa_depth: int = 3,  # Add this
    # ... other params
):
```

---

### MEDIUM ISSUE #11: Limited Logging for Debugging

**Location**: `optimize.py`, metric function

**Issue**: When optimization goes wrong, there's limited information to debug what happened.

**Missing information**:
- Which prompts were tried (only shows first 50 chars)
- How many evaluations succeeded vs failed
- Performance trends over iterations
- Detailed error traces when evaluations fail

**Recommendation**: Add structured logging:
```python
def metric(example, prediction, trace=None) -> float:
    sample_id = example.sample_id

    # Log the full prompt (not just first 50 chars) at debug level
    logger.debug(
        f"Evaluating sample {sample_id}\n"
        f"System message:\n{prediction.system_message}\n"
        f"Continue message:\n{prediction.continue_message}"
    )

    try:
        # ... evaluation logic ...

        logger.info(
            f"Sample {sample_id} completed successfully. Score: {score_value:.4f}"
        )
        return score_value

    except Exception as e:
        logger.error(
            f"Evaluation failed for sample {sample_id}: {e}",
            exc_info=True
        )
        raise
```

---

## Positive Observations

Despite the issues above, there are many **excellent design decisions**:

1. **Clean separation of concerns**: DSPy only generates prompts; inspect_ai handles all execution
2. **Per-sample evaluation**: Using `sample_id` parameter is clever and correct
3. **Flexible optimizer selection**: Supporting multiple optimizers (GEPA, MIPRO, Bootstrap) is good
4. **Graceful fallbacks**: The GEPA import try/except is well-handled
5. **Good documentation**: README_DSPY.md is comprehensive and helpful
6. **Appropriate use of model_copy()**: Line 88 correctly uses Pydantic's model_copy for task modification
7. **Proper caching**: Using log_dir for caching evaluations is smart
8. **Good parameter defaults**: The default values (train_ratio=0.8, etc.) are reasonable

---

## Additional Concerns

### Concern 1: MIPRO minibatch_size Calculation

**Location**: `optimize.py`, lines 325-328

**Code**:
```python
minibatch_size = min(25, max(1, len(train_examples) // 2))
```

**Question**: Why `// 2`? This means:
- 2 samples -> minibatch_size = 1
- 10 samples -> minibatch_size = 5
- 50 samples -> minibatch_size = 25 (capped)

**Potential issue**: With very small training sets (5-10 samples), this creates tiny minibatches (2-5 samples). MIPROv2 may not work well with such small minibatches.

**Recommendation**: Consider a different formula:
```python
# Use at least 5 samples per minibatch, or all samples if fewer than 5
minibatch_size = min(25, max(5, len(train_examples)))
```

Or make it configurable:
```python
def optimize_prompts(
    # ...
    minibatch_size: int | None = None,  # Auto-calculate if None
    # ...
):
    if minibatch_size is None:
        minibatch_size = min(25, max(5, len(train_examples)))
```

### Concern 2: Evaluation Cost Explosion

**Location**: Entire optimization pipeline

**Issue**: Each candidate prompt runs a FULL inspect_ai.eval(), which:
- Runs the entire agent loop (up to max_steps)
- Makes multiple LLM calls per sample
- Executes all tools
- Can take 30-60 seconds per sample

**Math**:
- 5 candidates × 20 training samples = 100 evaluations
- At 30 seconds each = 50 minutes
- At 10 LLM calls per sample × $0.01 per call = $10

With MIPROv2's num_trials (default 18 for 5 candidates):
- 18 trials × 20 samples = 360 evaluations
- At 30 seconds each = 3 hours
- At 10 LLM calls per sample × $0.01 per call = $36

**Recommendation**: Document this clearly in the README (which you do!) and consider:
1. Adding a cost estimation function:
```python
def estimate_optimization_cost(
    num_candidates: int,
    num_trials: int,
    num_train_samples: int,
    avg_eval_time_seconds: float = 30,
    avg_cost_per_eval: float = 0.10,
):
    total_evals = num_trials * num_train_samples
    total_time_hours = (total_evals * avg_eval_time_seconds) / 3600
    total_cost = total_evals * avg_cost_per_eval

    print(f"Estimated optimization cost:")
    print(f"  Total evaluations: {total_evals}")
    print(f"  Estimated time: {total_time_hours:.1f} hours")
    print(f"  Estimated cost: ${total_cost:.2f}")

    return total_evals, total_time_hours, total_cost
```

2. Adding a confirmation prompt:
```python
# In optimize_prompts(), before starting
total_evals = mipro_num_trials * len(train_examples) if optimizer_type == "mipro" else num_candidates * len(train_examples)
print(f"\nThis will run approximately {total_evals} evaluations.")
print(f"Continue? [y/N] ", end="")
response = input().strip().lower()
if response != 'y':
    print("Optimization cancelled.")
    return
```

### Concern 3: Validation Evaluation Only Runs One Sample

**Location**: `optimize.py`, lines 369-372

**Code**:
```python
# Evaluate on validation set
if val_examples:
    logger.info("\nEvaluating first validation example...")
    val_score = metric(val_examples[0], optimized_prediction)
    logger.info(f"Validation score (first sample): {val_score:.4f}")
```

**Issue**: This only evaluates the FIRST validation sample, not the entire validation set.

**Impact**:
- Not a true validation score
- Could be lucky/unlucky with that one sample
- Doesn't give a good sense of generalization

**Recommendation**: Evaluate on all validation samples:
```python
# Evaluate on validation set
if val_examples:
    logger.info(f"\nEvaluating on {len(val_examples)} validation examples...")
    val_scores = []
    for val_example in val_examples:
        try:
            score = metric(val_example, optimized_prediction)
            val_scores.append(score)
        except Exception as e:
            logger.error(f"Validation failed for {val_example.sample_id}: {e}")

    if val_scores:
        avg_val_score = sum(val_scores) / len(val_scores)
        logger.info(f"Validation score (average): {avg_val_score:.4f}")
        logger.info(f"Validation scores: {val_scores}")
```

---

## Testing Recommendations

Since there are no tests currently, here are the most critical tests to write:

### Unit Tests

1. **Test data splitting**:
```python
def test_load_and_split_data_shuffles():
    # Verify that data is shuffled
    # Verify that split ratio is correct
    # Verify that limit works correctly
```

2. **Test metric function error handling**:
```python
def test_metric_handles_failed_eval():
    # Mock inspect_eval to return failed log
    # Verify metric raises appropriate error

def test_metric_handles_missing_scores():
    # Mock inspect_eval to return log without global_avg
    # Verify metric raises helpful error
```

3. **Test DSPy example creation**:
```python
def test_create_dspy_examples_structure():
    # Verify each example has required fields
    # Verify with_inputs is called correctly
```

4. **Test optimizer configuration**:
```python
def test_mipro_configuration():
    # Verify auto=None is set
    # Verify num_trials is passed to compile()

def test_bootstrap_configuration():
    # Verify correct parameters are passed
```

### Integration Tests

1. **Test end-to-end on small dataset**:
```python
def test_optimize_prompts_small_dataset():
    # Run optimization with 2 train samples, 1 val sample, 2 candidates
    # Verify it completes without errors
    # Verify output file is created with correct structure
```

2. **Test optimized agent loading**:
```python
def test_load_optimized_prompts():
    # Create a test prompts.json file
    # Verify it loads correctly
    # Verify fallback to defaults works
```

---

## Recommended Fixes Priority

### Must Fix Before Production Use
1. Add validation in metric function for eval success (Issue #1)
2. Fix BootstrapFewShot parameter passing (Issue #2)
3. Add shuffling to data split (Issue #3)
4. Use fail-loud score extraction (Issue #5)

### Should Fix Soon
5. Add tool_call_format parameter explicitly (Issue #4)
6. Fix potential KeyError in score access (Issue #6)
7. Add DSPy prediction validation (Issue #7)
8. Evaluate full validation set, not just first sample (Concern #3)

### Nice to Have
9. Add consistent error handling strategy (Issue #8)
10. Add complete type hints (Issue #9)
11. Make hard-coded constants configurable (Issue #10)
12. Add better logging for debugging (Issue #11)
13. Add cost estimation and confirmation (Concern #2)
14. Review minibatch_size calculation (Concern #1)

---

## Conclusion

The code demonstrates **solid understanding** of both DSPy and inspect_ai frameworks, with good architectural decisions. However, the **critical issues around error handling and validation** must be addressed before this can be used reliably in production.

The most important principle from CLAUDE.md that's being violated is **fail-loud over fail-silent**. The current implementation silently returns 0.0 in many error cases, which can corrupt the optimization process and waste resources.

**Recommended next steps**:
1. Fix the 4 must-fix issues
2. Add basic unit tests for critical functions
3. Run a small-scale test optimization (2-3 samples) to verify fixes
4. Add integration test for end-to-end flow
5. Document known limitations and costs

Overall assessment: **Good foundation, needs hardening before production use.**
