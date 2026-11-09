"""Job-Shop dispatch: pick the ready operation to start next to minimize makespan."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Wall-clock deadline guard: evaluator kills after 150s. Stay well under.
_START_TS = _time.time()
DEADLINE_S = 115.0  # safety margin

def _machine_load_map(ready_operations):
    """Approximate current machine load: sum of min durations assigned to each machine
    across all ready operations. This is a cheap proxy for which machines are busier."""
    load = {}
    for op in ready_operations:
        opts = op.get("machine_options", {})
        if not opts:
            continue
        # Attribute the operation's shortest duration to its least-loaded (preferred) machine
        best_m = min(opts, key=lambda m: (opts[m], m))
        load[best_m] = load.get(best_m, 0.0) + opts[best_m]
    return load

def dispatch(ready_operations, time):
    # Deadline guard: if we're near the kill limit, just return a quick choice.
    if _time.time() - _START_TS > DEADLINE_S:
        return 0

    if not ready_operations:
        return 0
    n = len(ready_operations)
    if n == 1:
        return 0

    # Approximate machine loads for load-balancing tie-break.
    mload = _machine_load_map(ready_operations)

    # Gather raw features
    rw = np.empty(n)      # remaining_work
    ro = np.empty(n)      # remaining_ops
    min_dur = np.empty(n) # min duration over machine options
    chosen_dur = np.empty(n)  # duration on the chosen (preferred) machine
    load_penalty = np.empty(n)  # load on the preferred machine (lower is better)
    avg_op_time = np.empty(n)   # average processing time of remaining ops (critical path proxy)
    critical_path_inc = np.empty(n)  # estimated critical path increase (duration + max remaining work of unblocked job)

    for i, op in enumerate(ready_operations):
        rw[i] = float(op.get("remaining_work", 0.0))
        ro[i] = max(1, int(op.get("remaining_ops", 0)))
        opts = op.get("machine_options", {})
        if not opts:
            min_dur[i] = 1.0
            chosen_dur[i] = 1.0
            load_penalty[i] = 0.0
            avg_op_time[i] = 0.0
            critical_path_inc[i] = 1.0 + rw[i]
            continue
        # Preferred machine: shortest duration; tie-break by lower current load, then id.
        best_m = min(opts, key=lambda m: (opts[m], mload.get(m, 0.0), m))
        chosen_dur[i] = opts[best_m]
        min_dur[i] = min(opts.values())
        load_penalty[i] = mload.get(best_m, 0.0)
        # Critical path proxy: average remaining work per remaining op
        avg_op_time[i] = rw[i] / ro[i]
        # Graph-theoretic lookahead: estimated increase in global critical path.
        # This is the duration of the chosen operation plus the maximum remaining work
        # of the job it unblocks (approximated by the job's own remaining_work, since
        # this is the next op in the job's sequence).
        critical_path_inc[i] = chosen_dur[i] + rw[i]

    # Robust normalization via ranks to avoid scale issues across instances.
    def rank(arr):
        # average rank (1-based) for ties
        order = np.argsort(arr)
        ranks = np.empty_like(arr, dtype=np.float64)
        sorted_vals = arr[order]
        i = 0
        while i < len(arr):
            j = i
            while j + 1 < len(arr) and sorted_vals[j + 1] == sorted_vals[i]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            ranks[order[i:j + 1]] = avg_rank
            i = j + 1
        return ranks

    # We want to FAVOR:
    #  - high remaining_work  -> higher rank is better (critical jobs first)
    #  - high avg_op_time     -> higher rank is better (critical path jobs)
    #  - short min_duration   -> lower is better (SPT)
    #  - high remaining_ops   -> higher is better (jobs with more work left)
    #  - short chosen_dur     -> lower is better
    #  - low load_penalty     -> lower is better (avoid bottleneck machines)
    #  - low critical_path_inc -> lower is better (minimize estimated makespan increase)

    rw_rank = rank(rw)          # high = good
    avg_rank = rank(avg_op_time) # high = good (critical path)
    dur_rank = rank(min_dur)    # high = long (bad), so invert
    ro_rank = rank(ro)          # high = many (good, more critical)
    cdur_rank = rank(chosen_dur)  # high = long (bad), so invert
    lp_rank = rank(load_penalty)  # high = loaded (bad), so invert
    cpi_rank = rank(critical_path_inc)  # high = large increase (bad), so invert

    inv_dur = (len(rw) + 1.0) - dur_rank
    inv_cdur = (len(rw) + 1.0) - cdur_rank
    inv_lp = (len(rw) + 1.0) - lp_rank
    inv_cpi = (len(rw) + 1.0) - cpi_rank

    # Composite score with refined weights:
    # - Strong weight on critical_path_inc (graph-theoretic lookahead)
    # - Strong weight on remaining_work and avg_op_time (critical path identification)
    # - Moderate weight on SPT (min_dur) for short operations
    # - Small weights on other features for tie-breaking
    w_rw = 1.2
    w_avg = 1.0
    w_dur = 0.6
    w_ro = 0.2
    w_cdur = 0.2
    w_lp = 0.3
    w_cpi = 1.5

    scores = (w_rw * rw_rank
              + w_avg * avg_rank
              + w_dur * inv_dur
              + w_ro * ro_rank
              + w_cdur * inv_cdur
              + w_lp * inv_lp
              + w_cpi * inv_cpi)

    best_idx = int(np.argmax(scores))

    return best_idx
# EVOLVE-BLOCK-END
