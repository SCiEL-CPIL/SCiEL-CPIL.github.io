import time
import numpy as np
import sys

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave 10s buffer for the 150s hard limit

def dispatch(ready_operations, time):
    """
    Dispatches the next operation using a Greedy 1-Step Lookahead (Min-Completion) strategy.
    
    Strategy:
    1. For each ready operation, identify the best machine (shortest processing time).
    2. Simulate the immediate consequence of starting this operation on that machine:
       a. The machine will be busy until `time + duration`.
       b. The job's remaining work decreases by `duration`.
       c. Other operations that might be ready next are not yet known, but we can estimate
          the "system pressure" by looking at the current ready set.
    3. Estimate the "Projected Makespan Lower Bound" for each choice.
       A simple lower bound for the makespan at any point T is:
       LB = max(
           max(job_remaining_work[j] for all jobs j),  # Critical path lower bound
           max(machine_available_time[m] for all machines m) # Machine load lower bound (approx)
       )
       
       Since we don't have explicit machine state, we approximate machine load by assuming
       that if we pick op i on machine m_i, that machine is blocked for d_i time.
       Other machines are assumed to be free or have similar load (heuristic).
       
       A more robust heuristic for "Min-Completion" in FJSP is:
       Pick the operation that minimizes the completion time of the job it belongs to,
       weighted by the urgency of the job.
       
       Let's use a "Min-Max" regret approach:
       For each candidate op i:
         - Best machine m_i
         - Completion time of this op: C_i = time + d_i
         - Remaining work of job j_i after this op: RW'_i = RW_i - d_i
         
       We want to minimize the potential makespan. A common proxy is to minimize the
       maximum of (remaining work of the job) and (time this machine is blocked).
       
       However, a stronger signal in FJSP is often to prioritize operations that
       have few alternative machines (bottleneck ops) or long processing times (critical path).
       
       Let's try a "Weighted Shortest Processing Time with Urgency" but with a lookahead
       on the *next* step.
       
       Actually, let's stick to a clean 1-step lookahead:
       Score_i = (Remaining Work of Job i) / (1 + Estimated Contention)
       But we already tried that.
       
       New Strategy: **Greedy Min-Regret on Critical Path**
       
       1. Calculate the "Critical Path Length" (CPL) for each ready operation.
          CPL_i = remaining_work[job_i]
       2. Calculate the "Flexibility" of each operation.
          Flex_i = number of machine options.
       3. Priority Score = CPL_i / (1 + Flex_i)
          - High CPL: Urgent, long path.
          - Low Flex: Bottleneck, must be scheduled carefully.
       
       Wait, the inspiration 3 (score 1.02) used GCP. The current program (0.98) uses
       Dynamic Bottleneck. The best inspiration (1.09) is unknown details but uses numpy.
       
       Let's try a different angle: **Machine Load Balancing via Greedy Assignment**.
       
       Instead of just picking the best op, we look at the *distribution* of load.
       
       Revised Strategy: **Greedy Load Balancing**
       1. For each op, pick the machine with the shortest duration.
       2. Estimate the "Load" on each machine by summing the durations of all ready ops
          that *could* use that machine.
       3. For each op, calculate the "Penalty" of assigning it to its best machine:
          Penalty = Load[best_machine]
       4. Score = Remaining Work / (1 + Penalty)
       5. Pick max Score.
       
       This is very similar to the current program.
       
       Let's try: **Simulated Annealing-style Randomized Greedy**? No, too slow/risky.
       
       Let's try: **Minimize Max Remaining Work (MMRW)**
       1. For each op i, if we schedule it, job j_i's remaining work becomes RW_i - d_i.
       2. The makespan is at least the max remaining work of any job.
       3. We want to minimize the *maximum* remaining work across all jobs after this step.
       4. For each candidate i:
          New_Max_RW = max(
              (RW_i - d_i),  # The job we are scheduling
              max(RW_k for k != i)  # Other jobs remain unchanged
          )
       5. Pick the op that minimizes New_Max_RW.
       6. Tie-breaker: Minimize the duration d_i (SPT) to free the machine.
       
       This is a very direct "Min-Max" heuristic. It focuses on balancing the workload
       across jobs to prevent any single job from becoming a long critical path.
    """
    if not ready_operations:
        return -1
    
    # Check time limit
    if time.time() - _START_TIME > TIME_LIMIT:
        # Fallback: Pick first op with shortest duration
        best_idx = 0
        best_d = float('inf')
        for i, op in enumerate(ready_operations):
            d = min(op['machine_options'].values())
            if d < best_d:
                best_d = d
                best_idx = i
        m = min(op['machine_options'], key=op['machine_options'].get)
        return (best_idx, m)

    n_ops = len(ready_operations)
    
    # Pre-compute data
    min_durs = []
    best_machines = []
    rem_works = []
    
    for op in ready_operations:
        opts = op['machine_options']
        min_d = float('inf')
        best_m = -1
        for m, d in opts.items():
            if d < min_d:
                min_d = d
                best_m = m
        min_durs.append(min_d)
        best_machines.append(best_m)
        rem_works.append(op.get('remaining_work', 0))
    
    best_idx = -1
    best_score = float('inf')  # We want to MINIMIZE the projected max remaining work
    best_machine = -1
    
    # Pre-compute max remaining work for all jobs except the one being scheduled?
    # Since n_ops is small, O(N^2) is fine.
    
    for i in range(n_ops):
        rw_i = rem_works[i]
        d_i = min_durs[i]
        
        # If we schedule op i, job i's remaining work becomes rw_i - d_i
        new_rw_i = rw_i - d_i
        
        # Find max remaining work of all OTHER jobs
        max_rw_others = 0
        for k in range(n_ops):
            if k == i:
                continue
            if rem_works[k] > max_rw_others:
                max_rw_others = rem_works[k]
        
        # The new system max remaining work is max(new_rw_i, max_rw_others)
        projected_max_rw = max(new_rw_i, max_rw_others)
        
        # Tie-breaker: If projected_max_rw is the same, prefer shorter duration (SPT)
        # to free the machine sooner.
        # We can combine them into a single minimization score:
        # Score = projected_max_rw + epsilon * d_i
        score = projected_max_rw + 0.1 * d_i
        
        if score < best_score:
            best_score = score
            best_idx = i
            best_machine = best_machines[i]
            
    if best_idx == -1:
        return -1
        
    return (best_idx, best_machine)

