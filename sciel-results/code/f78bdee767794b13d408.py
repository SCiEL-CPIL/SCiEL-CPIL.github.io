"""Job-Shop Scheduling: Critical-path-aware dispatch with machine-load tiebreaking."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global deadline guard: evaluator kills at 150s, use 140s as hard limit
_DEADLINE = 140.0
_START_TIME = _time.monotonic()

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.

    Args:
        ready_operations: list of dicts, each with:
            - job: job id
            - op: operation index
            - machine_options: {machine: duration}
            - remaining_ops: number of ops remaining in this job
            - remaining_work: total remaining processing time for this job
        time: current simulation time

    Returns:
        index of the ready operation to start
    """
    # Deadline guard: if we're close to the limit, just return first op
    elapsed = _time.monotonic() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        if not ready_operations:
            return 0
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)

    # Pre-allocate arrays for performance
    min_durs = np.empty(n)
    rem_works = np.empty(n)
    rem_ops = np.empty(n)
    option_counts = np.empty(n)
    max_durs = np.empty(n)

    for i, op in enumerate(ready_operations):
        machine_opts = op["machine_options"]
        if not machine_opts:
            min_durs[i] = np.inf
            rem_works[i] = 0.0
            rem_ops[i] = 0
            option_counts[i] = 0
            max_durs[i] = 0.0
            continue

        vals = list(machine_opts.values())
        min_d = min(vals)
        max_d = max(vals)

        min_durs[i] = min_d
        max_durs[i] = max_d
        rem_works[i] = op.get("remaining_work", 0.0)
        rem_ops[i] = op.get("remaining_ops", 0)
        option_counts[i] = len(machine_opts)

    # Handle cases where no machine options were valid
    valid_mask = min_durs < np.inf
    if not np.any(valid_mask):
        return 0

    valid_indices = np.where(valid_mask)[0]
    nv = len(valid_indices)

    rw = rem_works[valid_mask]
    ro = rem_ops[valid_mask]
    md = min_durs[valid_mask]
    xd = max_durs[valid_mask]
    oc = option_counts[valid_mask]

    # Avoid division by zero
    safe_md = md + 1e-9
    safe_xd = xd + 1e-9

    # Composite score:
    # - Primary: remaining_work (MWR) - critical path, proven to help (measured improvement)
    # - Secondary: option_counts (constraint) - more constrained ops (fewer options) are more urgent
    #   Note: we want to prioritize ops with FEWER options, so we use -option_counts
    #   But the measured finding says increasing option_counts weight from 0.05 to 0.1 helped,
    #   which suggests prioritizing ops with MORE options (more flexible). Let me re-read:
    #   "option_counts from 0.05 to 0.1" - this likely means the weight on the option_counts term.
    #   If the term is +option_counts, more weight prioritizes flexible ops.
    #   If the term is -option_counts, more weight prioritizes constrained ops.
    #   Given it helped, and constrained ops are usually more urgent, it's likely -option_counts
    #   OR the interpretation is different. Let me use -option_counts (prioritize constrained).
    #   Actually, re-reading: "option_counts from 0.05 to 0.1" - the weight increased.
    #   If the score term is -w_oc * option_counts, increasing w_oc makes constrained ops (low count)
    #   score higher. That makes sense: constrained ops should be scheduled first.
    # - Tertiary: 1/min_duration (SPT-like) - reduces machine occupancy
    # - Quaternary: -remaining_ops (prefer finishing jobs)
    # - Quinary: -max_duration (avoid long ops that block machines, tiebreaker)

    w_mwr = 0.50      # Weight for remaining work (increased from measured finding)
    w_oc = 0.10       # Weight for -option_counts (increased from measured finding)
    w_spt = 0.20      # Weight for 1/min_duration
    w_ops = 0.05      # Weight for -remaining_ops
    w_maxd = 0.05     # Weight for -max_duration (avoid long blockers)

    # Normalize components to comparable scales to avoid one dominating
    # Normalize remaining_work by max
    rw_max = rw.max() if rw.max() > 0 else 1.0
    rw_norm = rw / rw_max

    # Normalize option_counts (we want to penalize high counts, so invert)
    # option_counts ranges from 1 to n_machines typically
    # Use -oc / max_oc to get negative scores for high counts
    oc_max = oc.max() if oc.max() > 0 else 1.0
    oc_norm = oc / oc_max

    # 1/min_duration: normalize by max of 1/min_dur
    inv_md = 1.0 / safe_md
    inv_md_max = inv_md.max() if inv_md.max() > 0 else 1.0
    inv_md_norm = inv_md / inv_md_max

    # -remaining_ops: normalize
    ro_max = ro.max() if ro.max() > 0 else 1.0
    ro_norm = ro / ro_max

    # -max_duration: normalize
    xd_max = xd.max() if xd_max > 0 else 1.0
    xd_norm = xd / xd_max

    # Composite score
    s = (w_mwr * rw_norm
         - w_oc * oc_norm
         + w_spt * inv_md_norm
         - w_ops * ro_norm
         - w_maxd * xd_norm)

    # Find the best index
    best_local_idx = int(np.argmax(s))
    best_idx = int(valid_indices[best_local_idx])

    return best_idx
# EVOLVE-BLOCK-END
