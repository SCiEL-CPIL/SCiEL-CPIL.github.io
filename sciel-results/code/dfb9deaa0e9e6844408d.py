"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math as _math

_START_TIME = None

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()

    # Deadline guard: evaluator kills at 150s; use 145s as safety margin.
    if _time.time() - _START_TIME > 145:
        # Fast fallback: shortest processing time
        best_idx = 0
        best_dur = float('inf')
        for i, op in enumerate(ready_operations):
            d = min(op["machine_options"].values())
            if d < best_dur:
                best_dur = d
                best_idx = i
        return int(best_idx)

    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0
    if n_ops == 1:
        return 0

    # Precompute statistics for normalization
    min_durs = []
    max_durs = []
    rem_works = []
    rem_ops = []
    
    # Machine load estimation: sum of min durations for each machine
    machine_load = {}
    for op in ready_operations:
        opts = op["machine_options"]
        min_d = min(opts.values())
        max_d = max(opts.values())
        min_durs.append(min_d)
        max_durs.append(max_d)
        rem_works.append(op["remaining_work"])
        rem_ops.append(op["remaining_ops"])
        
        for m, d in opts.items():
            machine_load[m] = machine_load.get(m, 0) + d

    total_machine_load = sum(machine_load.values())
    max_rem_work = max(rem_works) if rem_works else 1
    max_rem_ops = max(rem_ops) if rem_ops else 1
    min_min_dur = min(min_durs) if min_durs else 1
    max_min_dur = max(min_durs) if min_durs else 1
    
    # Weights for the composite score
    w_spt = 0.4
    w_cr = 0.3
    w_load = 0.3

    best_score = -float('inf')
    best_idx = 0

    for i in range(n_ops):
        op = ready_operations[i]
        opts = op["machine_options"]
        
        # Find the best machine (shortest duration)
        min_dur = min_durs[i]
        best_machine = None
        for m, d in opts.items():
            if d == min_dur:
                best_machine = m
                break
        
        # 1. Shortest Processing Time (SPT) component
        # Lower duration is better. Normalize to [0, 1] where 1 is best (shortest).
        if max_min_dur > min_min_dur:
            spt_score = (max_min_dur - min_dur) / (max_min_dur - min_min_dur)
        else:
            spt_score = 1.0
        
        # 2. Critical Ratio (CR) component
        # Higher remaining work is more urgent if we assume similar time horizons.
        # However, a common proxy is remaining_work / remaining_ops (average work per op).
        # Or simply remaining_work. Let's use normalized remaining_work.
        # Also consider remaining_ops: fewer ops left might mean it's closer to finishing.
        # Let's use a combined urgency: remaining_work / (remaining_ops + 1)
        avg_work_per_op = rem_works[i] / (rem_ops[i] + 1)
        max_avg_work = max(rem_works[j] / (rem_ops[j] + 1) for j in range(n_ops))
        
        if max_avg_work > 0:
            cr_score = avg_work_per_op / max_avg_work
        else:
            cr_score = 0.0
        
        # 3. Machine Load component
        # Lower load on the best machine is better.
        load_penalty = machine_load[best_machine] / (total_machine_load + 1e-9)
        load_score = 1.0 - load_penalty

        # Composite Score
        score = w_spt * spt_score + w_cr * cr_score + w_load * load_score

        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
