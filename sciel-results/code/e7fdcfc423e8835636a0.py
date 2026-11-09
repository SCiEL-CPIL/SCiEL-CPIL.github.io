"""Job-shop dispatch rule: SPT + remaining work + urgency composite."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START = _time.monotonic()
_DEADLINE = 140.0  # safety margin before 150s kill


def dispatch(ready_operations, time):
    """Dispatch rule for job-shop scheduling.

    Returns the index of the ready operation to start next.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard
    if _time.monotonic() - _START > _DEADLINE - 5.0:
        # Fallback: pick min duration (SPT)
        durs = [min(op.get("machine_options", {0: 1}).values(), default=1) for op in ready_operations]
        return int(np.argmin(durs))

    # Extract features
    min_durs = np.zeros(n, dtype=np.float64)
    rem_work = np.zeros(n, dtype=np.float64)
    rem_ops = np.zeros(n, dtype=np.float64)
    n_opts = np.zeros(n, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if mo:
            vals = list(mo.values())
            min_durs[i] = min(vals)
            n_opts[i] = len(vals)
        else:
            min_durs[i] = 1.0
            n_opts[i] = 1.0
        rem_work[i] = op.get("remaining_work", 0)
        rem_ops[i] = op.get("remaining_ops", 0)

    # SPT term: prefer shorter operations (1/duration, clipped to avoid explosion)
    spt_term = 1.0 / np.maximum(min_durs, 0.1)

    # Remaining work: prefer jobs with more total work left
    # Normalize to [0,1] range for stable weighting
    rw_max = np.max(rem_work) if np.max(rem_work) > 0 else 1.0
    rw_term = rem_work / rw_max

    # Remaining ops: prefer jobs with more operations left
    ro_max = np.max(rem_ops) if np.max(rem_ops) > 0 else 1.0
    ro_term = rem_ops / ro_max

    # Flexibility: prefer operations with fewer machine options (more constrained)
    no_max = np.max(n_opts) if np.max(n_opts) > 0 else 1.0
    flex_term = 1.0 / np.maximum(n_opts, 1.0)

    # Composite score: higher = pick first
    # SPT is primary (reduces blocking), remaining work secondary, urgency tertiary
    scores = 2.0 * spt_term + 1.0 * rw_term + 0.5 * ro_term + 0.3 * flex_term

    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
