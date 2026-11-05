"""Open-Shop Scheduling dispatch rule using a bipartite-graph matching heuristic."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START_TIME = _time.time()
_DEADLINE = 150.0
_MARGIN = 10.0  # seconds of safety margin


def _elapsed() -> float:
    return _time.time() - _START_TIME


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next using a bipartite-graph matching heuristic.
    
    Model: jobs and machines form two partitions of a bipartite graph. Each ready
    operation is an edge between its job and one of its candidate machines, weighted
    by the processing time on that machine. We greedily select the edge (operation,
    machine) that maximizes a composite weight combining:
      - the processing time on the chosen machine (longer blocks reduce fragmentation),
      - the remaining workload of the job (criticality),
      - the remaining workload of the machine (prefer least-loaded machines),
      - a penalty for operations with many remaining ops (prefer near-completion).
    
    Returns (index, machine) tuple.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        durations = op.get("machine_options", {})
        if durations:
            best_m = min(durations.keys(), key=lambda m: durations[m])
        else:
            best_m = None
        return (0, best_m)

    # Deadline guard
    if _elapsed() > _DEADLINE - _MARGIN:
        # Fallback: pick operation with most remaining work, assign to shortest-duration machine
        rw_list = [op.get("remaining_work", 0) for op in ready_operations]
        best_idx = int(np.argmax(rw_list))
        op = ready_operations[best_idx]
        durations = op.get("machine_options", {})
        if durations:
            best_m = min(durations.keys(), key=lambda m: durations[m])
        else:
            best_m = None
        return (best_idx, best_m)

    # Pre-compute machine remaining workloads from ready operations
    # For each machine, sum of min durations across all ready ops that can use it
    # Actually, we want to estimate how much work is "pending" on each machine.
    # Use: for each machine, sum of the duration that each ready op would take on it.
    machine_workload = {}
    for op in ready_operations:
        durations = op.get("machine_options", {})
        for m, d in durations.items():
            machine_workload[m] = machine_workload.get(m, 0.0) + d

    # Pre-compute job remaining workloads
    job_workload = {}
    for op in ready_operations:
        job_id = op.get("job", None)
        rw = op.get("remaining_work", 0)
        job_workload[job_id] = job_workload.get(job_id, 0.0) + rw

    # For each ready operation, evaluate each candidate machine and compute a composite score
    best_score = -np.inf
    best_idx = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        job_id = op.get("job", None)
        rw = op.get("remaining_work", 0)
        rem_ops = op.get("remaining_ops", 0)
        durations = op.get("machine_options", {})

        if not durations:
            continue

        job_load = job_workload.get(job_id, 0.0)

        for m, d in durations.items():
            m_load = machine_workload.get(m, 0.0)
            
            # Composite score:
            # 1. Processing time on this machine (prefer longer to reduce fragmentation)
            # 2. Job criticality (prefer jobs with more remaining work)
            # 3. Machine load (prefer machines with less pending work)
            # 4. Prefer operations with fewer remaining ops (closer to completion)
            
            # Normalize components to comparable scales
            # d is the processing time on machine m
            # rw is remaining work for this op's job
            # m_load is total pending work on machine m across all ready ops
            # rem_ops is number of remaining operations for this job
            
            # We want to maximize:
            #   d * (rw / (m_load + 1)) - rem_ops * small_penalty
            # This favors long operations on critical jobs assigned to less-loaded machines.
            
            if m_load > 0:
                machine_factor = rw / (m_load + 1.0)
            else:
                machine_factor = rw
            
            score = d * machine_factor - rem_ops * 0.1
            
            if score > best_score:
                best_score = score
                best_idx = i
                best_machine = m

    # If no machine was found (shouldn't happen), fallback
    if best_machine is None:
        best_idx = int(np.argmax([op.get("remaining_work", 0) for op in ready_operations]))
        op = ready_operations[best_idx]
        durations = op.get("machine_options", {})
        if durations:
            best_machine = min(durations.keys(), key=lambda m: durations[m])
        else:
            best_machine = None

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
