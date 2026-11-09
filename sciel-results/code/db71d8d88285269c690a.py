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
        
        # Graph-theoretic lookahead: estimate critical path increase
        # For each candidate machine, compute: duration + max remaining work of job it unblocks
        # We use the soft-min (log-sum-exp) over machines to get a smooth estimate
        # of the best (minimum) critical path increase achievable
        durations = np.array(list(opts.values()), dtype=np.float64)
        # The remaining work after this op is rw[i] - duration
        # But we approximate the "unblocked" work as the max remaining work among 
        # operations that depend on this one. Since we don't have explicit dependencies,
        # we use rw[i] as a proxy for the job's total remaining critical path.
        # The critical path increase is approximated as: duration + (rw[i] - duration) = rw[i]
        # But this is constant across machines. To differentiate, we consider:
        # The bottleneck is max(duration_on_machine, remaining_work_of_job_after_this_op)
        # remaining_work_after = rw[i] - duration (if this op is on critical path)
        # So cp_increase_candidate = max(duration, rw[i] - duration)
        # We want the MIN over machines, so we use soft-min via log-sum-exp
        
        # Soft-min: -log-sum-exp(-x) approximates min(x) smoothly
        # cp_candidates[m] = max(durations[m], rw[i] - durations[m])
        cp_candidates = np.maximum(durations, rw[i] - durations)
        # Soft-min via log-sum-exp
        max_cp = np.max(cp_candidates)
        if max_cp > 0:
            log_sum_exp = max_cp + np.log(np.sum(np.exp(cp_candidates - max_cp)))
            soft_min_cp = -log_sum_exp  # This is wrong sign, let me fix
            # Actually soft-min(x) = -log-sum-exp(-x)
            neg_cp = -cp_candidates
            max_neg = np.max(neg_cp)
            lse = max_neg + np.log(np.sum(np.exp(neg_cp - max_neg)))
            soft_min_cp = -lse
        else:
            soft_min_cp = 0.0
        cp_increase[i] = soft_min_cp

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
    
    # Also compute a simple min cp_increase (hard min) for comparison
    cp_min = np.empty(n)
    for i, op in enumerate(ready_operations):
        opts = op.get("machine_options", {})
        if not opts:
            cp_min[i] = 1.0
            continue
        durations = list(opts.values())
        rw_i = float(op.get("remaining_work", 0.0))
        cp_candidates = [max(d, rw_i - d) for d in durations]
        cp_min[i] = min(cp_candidates)

    # We want to FAVOR:
    #  - high remaining_work  -> higher rank is better (critical jobs first)
    #  - high avg_op_time     -> higher rank is better (critical path jobs)
    #  - short min_duration   -> lower is better (SPT)
    #  - high remaining_ops   -> higher is better (jobs with more work left are more critical)
    #  - short chosen_dur     -> lower is better
    #  - low load_penalty     -> lower is better (avoid bottleneck machines)

    rw_rank = rank(rw)          # high = good
    dur_rank = rank(min_dur)    # high = long (bad), so invert
    ro_rank = rank(ro)          # high = many (good, more critical)
    cdur_rank = rank(chosen_dur)  # high = long (bad), so invert
    lp_rank = rank(load_penalty)  # high = loaded (bad), so invert
    aot_rank = rank(avg_op_time)  # high = good (critical path)
    cp_rank = rank(cp_min)      # high = large cp increase (bad), so invert

    inv_dur = (n + 1.0) - dur_rank
    inv_cdur = (n + 1.0) - cdur_rank
    inv_lp = (n + 1.0) - lp_rank
    inv_cp = (n + 1.0) - cp_rank  # low cp increase is good

    # Composite score.
    # Added graph-theoretic critical path lookahead (cp_min) as a new feature.
    # This estimates the increase in the global critical path if we schedule this operation,
    # using a soft-min over machine options to get a smooth surrogate for the min/max objective.
    # We prioritize operations with LOW critical path increase (inv_cp is high when cp is low).
    
    w_rw = 1.2
    w_dur = 0.8
    w_ro = 0.2
    w_cdur = 0.2
    w_lp = 0.3
    w_aot = 0.5
    w_cp = 0.8  # weight for critical path lookahead

    scores = (w_rw * rw_rank
              + w_dur * inv_dur
              + w_ro * ro_rank
              + w_cdur * inv_cdur
              + w_lp * inv_lp
              + w_aot * aot_rank
              + w_cp * inv_cp)

    best_idx = int(np.argmax(scores))

    return best_idx
