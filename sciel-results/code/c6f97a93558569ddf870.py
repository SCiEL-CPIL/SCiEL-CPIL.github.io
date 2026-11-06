"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import heapq

_START = _time.time()
_DEADLINE = 150.0 - 5.0  # safety margin


def _elapsed_ok():
    return (_time.time() - _START) < _DEADLINE


def _greedy_completion_makespan(remaining_ops, machine_available, time):
    """
    Simulate greedy completion of all remaining operations.
    remaining_ops: list of (job, op_id, machine_options, remaining_work, remaining_ops_count)
    machine_available: dict machine -> time when machine becomes free
    Returns estimated makespan.
    """
    if not remaining_ops:
        return time
    
    # Build a list of all (job, op_id, machine_options, remaining_work)
    # We'll simulate using a priority queue of ready operations
    # In open-shop, all ops of a job must be done, but they can be in any order.
    # Here we just have the remaining ops for the jobs that have ready ops.
    
    # For simplicity, simulate by always picking the op with the earliest
    # completion time (SPT on the best machine)
    
    ops_remaining = list(remaining_ops)  # list of dicts
    mach_avail = dict(machine_available)
    
    # Simulate until all ops are done
    sim_time = time
    ops_done = 0
    total_ops = len(ops_remaining)
    
    # Use a simple greedy: at each step, pick the op that can finish earliest
    # considering current machine availability
    for _ in range(total_ops):
        if not ops_done < total_ops:
            break
        
        best_completion = float('inf')
        best_idx = -1
        best_machine = -1
        
        for i, op in enumerate(ops_remaining):
            if op is None:
                continue
            opts = op["machine_options"]
            for m, dur in opts.items():
                start_time = max(sim_time, mach_avail.get(m, 0.0))
                end_time = start_time + dur
                if end_time < best_completion:
                    best_completion = end_time
                    best_idx = i
                    best_machine = m
        
        if best_idx == -1:
            break
        
        # Assign this op
        m = best_machine
        dur = ops_remaining[best_idx]["machine_options"][m]
        start_time = max(sim_time, mach_avail.get(m, 0.0))
        end_time = start_time + dur
        
        mach_avail[m] = end_time
        sim_time = end_time
        ops_remaining[best_idx] = None
        ops_done += 1
    
    # Makespan is the max machine availability
    if mach_avail:
        return max(mach_avail.values())
    return sim_time


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        return (0, best_m)

    # Pre-calculate basic metrics
    min_durs = []
    best_machines = []
    is_last_ops = []
    remaining_works = []

    for op in ready_operations:
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])
        is_last_ops.append(op["remaining_ops"] == 1)
        remaining_works.append(op["remaining_work"])

    # Deadline guard
    if not _elapsed_ok():
        idx = int(max(range(n), key=lambda i: remaining_works[i]))
        return (idx, best_machines[idx])

    # Collect all machines
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op["machine_options"].keys())
    
    if not all_machines:
        return (0, best_machines[0])

    # Estimate current machine loads from ready ops using greedy water-filling
    # This gives us an estimate of when machines will be free
    machine_loads = {m: 0.0 for m in all_machines}
    op_order = sorted(range(n), key=lambda i: (min_durs[i], -remaining_works[i]))
    for i in op_order:
        opts = ready_operations[i]["machine_options"]
        best_m_for_load = min(opts.keys(), key=lambda m: (machine_loads.get(m, 0.0), m))
        machine_loads[best_m_for_load] += opts[best_m_for_load]

    # Now, for each candidate operation, simulate what happens if we pick it
    # and then greedily complete all remaining ops
    scores = []
    for i in range(n):
        op = ready_operations[i]
        opts = op["machine_options"]
        
        # Try each compatible machine for this op
        best_score_for_i = float('inf')
        best_m_for_i = best_machines[i]
        
        for m, dur in opts.items():
            # Machine becomes available at time + dur (simplified: assume it's free now)
            # Actually, in the simulation, the machine might be busy from other ops
            # Let's estimate: this op starts at max(time, machine_loads[m]) and finishes at +dur
            # But we're just estimating, so use: finish_time = machine_loads[m] + dur
            
            new_machine_loads = dict(machine_loads)
            new_machine_loads[m] = new_machine_loads.get(m, 0.0) + dur
            
            # Build remaining ops list (all ops except the one we picked)
            remaining_ops = []
            for j, op_j in enumerate(ready_operations):
                if j == i:
                    continue
                remaining_ops.append({
                    "machine_options": op_j["machine_options"],
                    "remaining_work": op_j["remaining_work"],
                    "remaining_ops": op_j["remaining_ops"] - 1 if op_j["remaining_ops"] > 1 else 0
                })
            
            # Simulate greedy completion
            makespan = _greedy_completion_makespan(remaining_ops, new_machine_loads, time)
            
            if makespan < best_score_for_i:
                best_score_for_i = makespan
                best_m_for_i = m
        
        scores.append((best_score_for_i, best_m_for_i))
    
    # Pick the operation with the lowest estimated makespan
    # Tie-break: prefer last ops, then SPT, then LPT
    def score_key(i):
        ms, bm = scores[i]
        return (ms, 0 if is_last_ops[i] else 1, min_durs[i], -remaining_works[i])
    
    idx = min(range(n), key=score_key)
    best_m = scores[idx][1]
    return (idx, best_m)
# EVOLVE-BLOCK-END
