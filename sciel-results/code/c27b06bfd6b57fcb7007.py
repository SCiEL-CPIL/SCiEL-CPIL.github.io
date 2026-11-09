"""Improved job-shop dispatch rule: combine machine load, remaining work, and critical path."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard: evaluator kills after 150s; use 140s as safety margin
_DEADLINE = _time.time() + 140.0


def _compute_score(ready_operations, time):
    """Compute a composite score for each ready operation. Higher = better to pick."""
    n = len(ready_operations)
    if n == 0:
        return 0

    scores = np.zeros(n, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        # 1. Remaining work (total processing time left for this job)
        remaining_work = op.get("remaining_work", 0)

        # 2. Remaining operations count (jobs with more ops left are more urgent)
        remaining_ops = op.get("remaining_ops", 0)

        # 3. Duration stats with soft-min (smooth surrogate for min)
        machine_options = op.get("machine_options", {})
        if machine_options:
            durations = np.array(list(machine_options.values()), dtype=np.float64)
            min_dur = float(np.min(durations))
            mean_dur = float(np.mean(durations))
            # Soft-min via log-sum-exp: smooth approximation to min
            # Using negative log-sum-exp of negative durations
            # Temperature parameter: lower = closer to hard min, higher = smoother
            temperature = 2.0
            neg_durs = -durations / temperature
            max_neg = np.max(neg_durs)
            soft_min = -(max_neg + np.log(np.sum(np.exp(neg_durs - max_neg)))) * temperature
            # Blend hard min and soft min for stability
            eff_min_dur = 0.6 * min_dur + 0.4 * soft_min
        else:
            min_dur = 0
            mean_dur = 0
            eff_min_dur = 0

        # 4. Number of machine options (more flexibility = less urgent)
        n_opts = len(machine_options) if machine_options else 1

        # Ratio-based urgency: remaining_work / effective_min_duration
        # Using the blended duration for a smoother gradient
        ratio_term = remaining_work / max(eff_min_dur, 1e-9)

        # Critical path proxy: jobs with fewer remaining ops are closer to completion
        # and their completion time directly impacts makespan
        completion_urgency = 1.0 / max(remaining_ops, 1)

        # Bottleneck awareness: ops on machines with fewer options are more constrained
        constraint_urgency = 1.0 / max(n_opts, 1)

        # Composite score with adjusted weights:
        # - Ratio term: high remaining work with short current op = very urgent
        # - Completion urgency: prioritize jobs close to finishing
        # - Constraint: fewer machine options = more constrained = more urgent
        # - Small penalty for very long current ops to avoid blocking
        
        score = 0.0
        score += 1.0 * ratio_term                    # primary: work-to-duration ratio
        score += 0.5 * completion_urgency            # prioritize near-completion jobs (increased weight)
        score += 0.2 * constraint_urgency            # urgency for inflexible ops (increased weight)
        score -= 0.03 * min_dur                      # reduced penalty for long current ops

        scores[i] = score

    return int(np.argmax(scores))


def dispatch(ready_operations, time):
    """Dispatch rule for job-shop scheduling.

    Returns the index of the ready operation to start next.
    """
    if not ready_operations:
        return 0

    # Deadline guard
    if _time.time() > _DEADLINE - 5.0:
        # Fallback: pick the one with most remaining work
        return int(np.argmax([op.get("remaining_work", 0) for op in ready_operations]))

    return _compute_score(ready_operations, time)
# EVOLVE-BLOCK-END
