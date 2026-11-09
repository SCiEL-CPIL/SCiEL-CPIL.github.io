import numpy as np
import time as _time

_START_TS = _time.time()
DEADLINE_S = 115.0  # safety margin before 150s kill

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.time() - _START_TS > DEADLINE_S:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0

    # Approximate machine loads for load-balancing tie-break.
    mload = {}
    for op in ready_operations:
        opts = op.get("machine_options", {})
        if not opts:
            continue
        # Attribute the operation's shortest duration to its preferred machine
        best_m = min(opts, key=lambda m: (opts[m], m))
        mload[best_m] = mload.get(best_m, 0.0) + opts[best_m]

    # Precompute per-operation features
    rw = np.empty(n)      # remaining_work
    ro = np.empty(n)      # remaining_ops
    min_dur = np.empty(n) # min duration over machine options
    chosen_dur = np.empty(n)  # duration on the chosen (preferred) machine
    load_penalty = np.empty(n)  # load on the preferred machine (lower is better)
    avg_op_time = np.empty(n)   # average processing time of remaining ops (critical path proxy)
    cp_increase = np.empty(n)   # estimated critical path increase (graph-theoretic lookahead)

    for i, op in enumerate(ready_operations):
        rw[i] = float(op.get("remaining_work", 0.0))
        ro[i] = max(1, int(op.get("remaining_ops", 0)))
        opts = op.get("machine_options", {})
        if not opts:
            min_dur[i] = 1.0
            chosen_dur[i] = 1.0
            load_penalty[i] = 0.0
            avg_op_time[i] = 0.0
            cp_increase[i] = 1.0
            continue
        # Preferred machine: shortest duration; tie-break by lower current load, then id.
        best_m = min(opts, key=lambda m: (opts[m], mload.get(m, 0.0), m))
        chosen_dur[i] = opts[best_m]
        min_dur[i] = min(opts.values())
        load_penalty[i] = mload.get(best_m, 0.0)
        # Critical path proxy: average remaining work per remaining op
        avg_op_time[i] = rw[i] / ro[i]
        # Graph-theoretic lookahead: estimated increase in makespan if this op is scheduled
        # The critical path through this job is at least: current time + this op's duration + remaining work after this op
        # We approximate the "remaining work after this op" as rw[i] - chosen_dur[i]
        # So the critical path contribution is roughly: chosen_dur[i] + (rw[i] - chosen_dur[i]) = rw[i]
        # But we also consider that if this op is long, it delays the rest of the job.
        # A better proxy: the operation's duration plus the max remaining work of the job (which is rw[i] for the last op, but for non-last ops it's the sum of subsequent ops).
        # Since we don't have the exact sequence, we use rw[i] as the total remaining work including this op.
        # The critical path increase is approximately the duration of this op plus the remaining work of the job AFTER this op.
        # remaining_work_after = rw[i] - chosen_dur[i] (approximate, assuming chosen_dur is the actual processing time)
        cp_increase[i] = chosen_dur[i] + max(0, rw[i] - chosen_dur[i])
        # Simplifies to rw[i], but let's keep it explicit for clarity and potential future refinement.
        # Actually, a better heuristic: if this op is on a bottleneck machine, its delay affects the machine's availability for other ops.
        # But for now, the critical path through the job is the key factor.
        # Let's refine: the makespan is determined by the longest path. Scheduling an op with high rw[i] early helps if it's on the critical path.
        # The "increase" in makespan if we delay this op is roughly the critical path length through this job.
        # So we want to MINIMIZE the makespan, which means we should prioritize ops that are on the critical path.
        # An op is more critical if its job has high remaining work.
        # So cp_increase[i] = rw[i] is a reasonable proxy for the critical path length through this job.
        # But we also need to consider the duration of the op itself: if the op is very long, it takes a long time to complete, so the job's remaining work decreases slowly.
        # A better proxy: the ratio of remaining work to remaining ops (avg_op_time) is already a proxy for the critical path per op.
        # Let's use a combination: cp_increase[i] = rw[i] * (1.0 + chosen_dur[i] / max(1.0, rw[i]))
        # This increases the weight of ops with high remaining work AND long duration (which are more likely to be bottlenecks).
        cp_increase[i] = rw[i] * (1.0 + chosen_dur[i] / max(1.0, rw[i]))

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
    #  - high remaining_ops   -> higher is better (jobs with more work left are more critical)
    #  - short chosen_dur     -> lower is better
    #  - low load_penalty     -> lower is better (avoid bottleneck machines)
    #  - high cp_increase     -> higher rank is better (critical path ops first)

    rw_rank = rank(rw)          # high = good
    dur_rank = rank(min_dur)    # high = long (bad), so invert
    ro_rank = rank(ro)          # high = many (good, more critical)
    cdur_rank = rank(chosen_dur)  # high = long (bad), so invert
    lp_rank = rank(load_penalty)  # high = loaded (bad), so invert
    aot_rank = rank(avg_op_time)  # high = good (critical path)
    cp_rank = rank(cp_increase)   # high = good (critical path)

    inv_dur = (n + 1.0) - dur_rank
    inv_cdur = (n + 1.0) - cdur_rank
    inv_lp = (n + 1.0) - lp_rank

    # Composite score.
    # Refined weights: Added weight on cp_increase (graph-theoretic critical path lookahead)
    # to prioritize operations that are on the critical path of their job.
    w_rw = 1.2
    w_dur = 0.8
    w_ro = 0.2
    w_cdur = 0.2
    w_lp = 0.3
    w_aot = 0.5
    w_cp = 0.5

    scores = (w_rw * rw_rank
              + w_dur * inv_dur
              + w_ro * ro_rank
              + w_cdur * inv_cdur
              + w_lp * inv_lp
              + w_aot * aot_rank
              + w_cp * cp_rank)

    best_idx = int(np.argmax(scores))

    return best_idx
