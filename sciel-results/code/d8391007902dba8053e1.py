"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
import math
import sys

_DEADLINE_SECONDS = 140.0
_START_TIME = time.time()

def _remaining_budget():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)

def _run_serial_sgs_min_regret(instance):
    """
    Serial Schedule Generation Scheme using a dynamic Min-Regret priority.
    At each step, for every eligible activity, we calculate a 'regret score'
    which estimates how much the makespan would increase if we delayed this activity.
    The activity with the highest regret is scheduled first.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    n = len(durations)
    if n == 0:
        return 0.0
        
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
        
    # Build successors and resource users
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            
    num_resources = 0
    d_mat = None
    c_vec = None
    if demands is not None:
        d_mat = np.asarray(demands, dtype=float)
        if len(d_mat.shape) == 2:
            num_resources = d_mat.shape[1]
        else:
            num_resources = 1
            d_mat = d_mat.reshape(-1, 1)
    if capacities is not None:
        c_vec = np.asarray(capacities, dtype=float)
        if len(c_vec.shape) == 1:
            c_vec = c_vec.reshape(-1)
            
    # State
    scheduled = [False] * n
    start_times = np.zeros(n, dtype=float)
    finish_times = np.zeros(n, dtype=float)
    
    # Pre-compute resource users for quick access
    resource_users = [[] for _ in range(num_resources)]
    if d_mat is not None:
        for i in range(n):
            for r in range(num_resources):
                if d_mat[i][r] > 0:
                    resource_users[r].append(i)

    # To speed up "longest path from node to end" calculations,
    # we will compute the "tail" (longest path from node to sink) dynamically or pre-compute static tails.
    # Since resource constraints change the effective critical path, static tails are a good baseline.
    
    # Static Tail Calculation (Longest path from i to any end node)
    # Topological sort
    in_degree = [len(p) for p in predecessors]
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    tail = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        max_succ_tail = 0.0
        for v in successors[u]:
            if tail[v] > max_succ_tail:
                max_succ_tail = tail[v]
        tail[u] = durations[u] + max_succ_tail
        
    # Initial lower bound for makespan (no resources)
    # Critical path length
    cp_len = 0.0
    # We can just use the max of (start_time + tail[i]) for any active/scheduled job, 
    # but since we are scheduling serially, the "current" lower bound is the max finish time of scheduled jobs 
    # plus the tail of the job we are considering?
    # Actually, a better lower bound at step k is:
    # max( max(finish_times[j] for scheduled j), max(start_time_candidate + tail[i] for eligible i) )
    # But we want to know the regret of *not* picking i.
    # Regret(i) = (Current LB without i) - (Current LB with i)? No.
    # Regret(i) = How much does the makespan increase if we don't start i now?
    # Approximation: If we don't start i now, it might be delayed due to resource contention.
    # A common heuristic is "Total Float" or "Critical Path Slack".
    # Let's use a dynamic "Urgency" score:
    # Score(i) = tail[i] + resource_pressure(i)
    # Where resource_pressure is high if i uses scarce resources that are currently busy.
    
    scheduled_count = 0
    current_time = 0.0
    
    while scheduled_count < n:
        eligible = []
        for i in range(n):
            if not scheduled[i]:
                # Check if all predecessors are scheduled
                is_eligible = True
                for p in predecessors[i]:
                    if not scheduled[p]:
                        is_eligible = False
                        break
                if is_eligible:
                    eligible.append(i)
                    
        if not eligible:
            break
            
        best_job = -1
        best_score = -np.inf
        
        # Calculate scores for eligible jobs
        # We want to maximize the score (urgency)
        for i in eligible:
            # Base urgency: Tail length (critical path distance to end)
            score = tail[i]
            
            # Resource Pressure Penalty/Bonus
            # If a job uses resources that are heavily loaded in the near future, it's more urgent.
            # Alternatively, if a job uses resources that are currently free, it's easy to schedule,
            # so maybe we should schedule "hard" jobs first?
            # Actually, "Min-Regret" usually implies: if I don't do this, I lose more.
            # Jobs with high resource demand on tight resources are risky to delay.
            
            if d_mat is not None and c_vec is not None:
                for r in range(num_resources):
                    d = d_mat[i][r]
                    if d > 0:
                        # Check current load at time current_time
                        load = 0.0
                        for j in resource_users[r]:
                            if scheduled[j]:
                                if start_times[j] <= current_time < finish_times[j]:
                                    load += d_mat[j][r]
                        
                        # If the job is close to filling the capacity, it's more critical
                        # because it leaves less room for others.
                        # Also, if the resource is currently very busy, starting this job might delay others,
                        # so maybe we should prioritize jobs that use *less* busy resources?
                        # No, standard heuristics often prioritize "critical" jobs.
                        # Let's add a term for resource intensity relative to capacity.
                        intensity = d / c_vec[r] if c_vec[r] > 0 else 0
                        score += intensity * 0.1 * tail[i] # Scale by tail to be consistent
                        
                        # Also, if the resource is currently free, we can start this job without delay.
                        # If it's busy, we have to wait. Waiting increases makespan risk.
                        # So, prioritize jobs that can start sooner?
                        # The SGS will calculate the earliest start time anyway.
                        # The priority just determines the *order* of attempts.
                        
                        # Let's refine: 
                        # If a job is on the critical path (high tail), it's urgent.
                        # If it uses scarce resources, it's urgent.
                        
            # Additional tie-breaker: Duration
            score += durations[i] * 0.01
            
            if score > best_score:
                best_score = score
                best_job = i
                
        # Schedule best_job
        # Find earliest feasible start time
        es_pred = 0.0
        for p in predecessors[best_job]:
            if finish_times[p] > es_pred:
                es_pred = finish_times[p]
                
        candidate_times = {es_pred}
        if d_mat is not None and c_vec is not None:
            for r in range(num_resources):
                if d_mat[best_job][r] > 0:
                    for j in resource_users[r]:
                        if scheduled[j]:
                            candidate_times.add(finish_times[j])
                            
        earliest_start = float('inf')
        for t in sorted(candidate_times):
            if t < es_pred:
                continue
            feasible = True
            if d_mat is not None and c_vec is not None:
                for r in range(num_resources):
                    if d_mat[best_job][r] == 0:
                        continue
                    load = 0.0
                    for j in resource_users[r]:
                        if scheduled[j]:
                            if start_times[j] <= t < finish_times[j]:
                                load += d_mat[j][r]
                    if load + d_mat[best_job][r] > c_vec[r]:
                        feasible = False
                        break
            if feasible:
                earliest_start = t
                break
                
        if earliest_start == float('inf'):
            earliest_start = es_pred
            
        start_times[best_job] = earliest_start
        finish_times[best_job] = earliest_start + durations[best_job]
        scheduled[best_job] = True
        scheduled_count += 1
        
        # Update current_time to max finish time for next iteration's "current load" check?
        # Actually, in SGS, time advances to the start time of the next job.
        # But for resource load checks, we check specific times.
        # We don't strictly need current_time to advance if we check specific t's.
        
    makespan = np.max(finish_times) if n > 0 else 0.0
    return makespan

def priority(instance):
    """
    Priority function that returns a static priority vector.
    However, since the task asks for a priority function that is used by a SGS,
    and the SGS is fixed (or we can assume the evaluator uses a standard SGS),
    we must return a vector.
    
    BUT, the prompt says: "Write a Python priority(instance) function that scores each activity; a serial schedule-generation scheme starts the highest-priority eligible activity".
    This implies the evaluator has a fixed SGS that uses my returned vector.
    My previous program used SA to find a good static vector.
    
    If I want to use the "Min-Regret" strategy, I cannot just return a static vector because the "regret" changes as the schedule is built.
    The evaluator's SGS is likely:
    1. Find eligible activities.
    2. Sort by priority (descending).
    3. Schedule the first one at the earliest feasible time.
    4. Repeat.
    
    If I return a static vector, I am stuck with static heuristics.
    To use dynamic heuristics, I would need to control the SGS, which I don't.
    
    Wait, let me re-read the task contract carefully.
    "Write a Python priority(instance) function that scores each activity... Scored by mean makespan on five generated single-mode instances... The evaluator measures the raw value... Keep the function names and signatures of the seed program."
    
    The seed program returns `best_vec` (a static vector).
    The evaluator uses this vector in a SGS.
    
    If I cannot change the SGS, I cannot implement dynamic Min-Regret directly.
    I must find a static priority vector that approximates the behavior of a good dynamic scheduler.
    
    However, the inspiration says: "simulate the scheduling process inside the priority function".
    This suggests that the `priority` function can be complex and run its own simulation to determine the best static vector to return.
    
    So, my strategy should be:
    1. Run the "Min-Regret" SGS internally to see which activities are critical.
    2. Or, use a more sophisticated static heuristic that captures the "regret" concept.
    
    Let's try a different structural change: **Lookahead Heuristic**.
    Instead of just Tail + Resource Intensity, let's calculate a "Lookahead Score" for each activity.
    For each activity i, simulate scheduling it next (in a simplified way) and see how it affects the lower bound of the makespan.
    
    Actually, the most robust static heuristic for RCPSP is often a combination of:
    1. Critical Path Length (Tail)
    2. Total Float (Slack)
    3. Resource Intensity
    
    Let's refine the static vector generation using a **Simulated Annealing** approach but with a much better initial guess and a more targeted perturbation, or better yet, a **Greedy Hill Climbing** on the priority vector, but starting from a very strong heuristic.
    
    Wait, the current program *is* Simulated Annealing.
    The directive says: "Make a STRUCTURAL change: a different design or modelling family, a different optimization or search method, or a different representation of the problem."
    
    Let's switch to **Genetic Algorithm (GA)** or **Particle Swarm Optimization (PSO)**?
    Or, let's try **Learning to Rank**? No, no ML libraries.
    
    How about **Constructive Heuristic with Local Search on the Schedule itself**?
    No, the interface is `priority(instance) -> vector`.
    
    Let's look at Inspiration 1: "simulate the scheduling process inside the priority function to identify the actual bottleneck".
    
    Let's try a **Bottleneck-Aware Priority Vector**.
    1. Run a fast SGS with a simple heuristic (e.g., Tail).
    2. Identify the critical path in the resulting schedule.
    3. Identify resources that are most constrained (highest utilization).
    4. Boost the priority of activities on the critical path AND activities using constrained resources.
    5. Repeat this process a few times (Iterative Refinement) to converge on a good vector.
    
    This is a structural change from "SA on random noise" to "Iterative Bottleneck Identification".
    """
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    if n == 0:
        return np.array([])
        
    # 1. Initial Vector: Tail + Duration
    predecessors = instance.get("predecessors", None)
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
        
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            
    in_degree = [len(p) for p in predecessors]
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    tail = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        max_succ_tail = 0.0
        for v in successors[u]:
            if tail[v] > max_succ_tail:
                max_succ_tail = tail[v]
        tail[u] = durations[u] + max_succ_tail
        
    initial_vec = tail.copy()
    
    # 2. Iterative Refinement
    # We will run a SGS, identify bottlenecks, and adjust the vector.
    # To save time, we do this for a fixed number of iterations or until time runs out.
    
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    num_resources = 0
    d_mat = None
    c_vec = None
    if demands is not None:
        d_mat = np.asarray(demands, dtype=float)
        if len(d_mat.shape) == 2:
            num_resources = d_mat.shape[1]
        else:
            num_resources = 1
            d_mat = d_mat.reshape(-1, 1)
    if capacities is not None:
        c_vec = np.asarray(capacities, dtype=float)
        if len(c_vec.shape) == 1:
            c_vec = c_vec.reshape(-1)
            
    resource_users = [[] for _ in range(num_resources)]
    if d_mat is not None:
        for i in range(n):
            for r in range(num_resources):
                if d_mat[i][r] > 0:
                    resource_users[r].append(i)

    def _run_sgs_with_vec(vec):
        """Fast SGS using the given static vector."""
        scheduled = [False] * n
        start_times = np.zeros(n, dtype=float)
        finish_times = np.zeros(n, dtype=float)
        scheduled_count = 0
        
        while scheduled_count < n:
            eligible = []
            for i in range(n):
                if not scheduled[i]:
                    is_eligible = True
                    for p in predecessors[i]:
                        if not scheduled[p]:
                            is_eligible = False
                            break
                    if is_eligible:
                        eligible.append(i)
                        
            if not eligible:
                break
                
            # Pick highest priority
            best_job = -1
            best_prio = -np.inf
            for i in eligible:
                if vec[i] > best_prio:
                    best_prio = vec[i]
                    best_job = i
                    
            es_pred = 0.0
            for p in predecessors[best_job]:
                if finish_times[p] > es_pred:
                    es_pred = finish_times[p]
                    
            candidate_times = {es_pred}
            if d_mat is not None and c_vec is not None:
                for r in range(num_resources):
                    if d_mat[best_job][r] > 0:
                        for j in resource_users[r]:
                            if scheduled[j]:
                                candidate_times.add(finish_times[j])
                                
            earliest_start = float('inf')
            for t in sorted(candidate_times):
                if t < es_pred:
                    continue
                feasible = True
                if d_mat is not None and c_vec is not None:
                    for r in range(num_resources):
                        if d_mat[best_job][r] == 0:
                            continue
                        load = 0.0
                        for j in resource_users[r]:
                            if scheduled[j]:
                                if start_times[j] <= t < finish_times[j]:
                                    load += d_mat[j][r]
                        if load + d_mat[best_job][r] > c_vec[r]:
                            feasible = False
                            break
                if feasible:
                    earliest_start = t
                    break
                    
            if earliest_start == float('inf'):
                earliest_start = es_pred
                
            start_times[best_job] = earliest_start
            finish_times[best_job] = earliest_start + durations[best_job]
            scheduled[best_job] = True
            scheduled_count += 1
            
        makespan = np.max(finish_times) if n > 0 else 0.0
        return makespan, start_times, finish_times

    current_vec = initial_vec.copy()
    best_vec = current_vec.copy()
    best_score = _run_sgs_with_vec(current_vec)[0]
    
    # Iterative Improvement
    # For each iteration, run SGS, find critical path, boost critical path nodes.
    # Also, find most utilized resources, boost nodes using them.
    
    max_iters = 10
    for _ in range(max_iters):
        if _remaining_budget() < 5.0:
            break
            
        score, st, ft = _run_sgs_with_vec(current_vec)
        
        if score < best_score:
            best_score = score
            best_vec = current_vec.copy()
            
        # Identify Critical Path in the current schedule
        # A node is on the critical path if its start/finish times align with the makespan
        # Specifically, if start_time[i] + tail[i] == makespan (approximately)
        # And it is part of the chain.
        
        makespan = score
        critical_mask = np.zeros(n, dtype=bool)
        for i in range(n):
            # If the job finishes at the makespan, it's critical
            # Or if start_time[i] + tail[i] is close to makespan
            if abs(st[i] + tail[i] - makespan) < 1e-6:
                critical_mask[i] = True
                
        # Identify Constrained Resources
        # Calculate resource utilization
        util = np.zeros(num_resources)
        for r in range(num_resources):
            if c_vec[r] == 0:
                continue
            # Total demand time
            total_demand = 0.0
            for i in range(n):
                if d_mat[i][r] > 0:
                    total_demand += d_mat[i][r] * durations[i]
            util[r] = total_demand / (makespan * c_vec[r]) if makespan > 0 else 0
            
        # Boost priorities
        new_vec = current_vec.copy()
        for i in range(n):
            boost = 0.0
            if critical_mask[i]:
                boost += 0.1 * makespan
            if d_mat is not None:
                for r in range(num_resources):
                    if d_mat[i][r] > 0 and util[r] > 0.8:
                        boost += 0.05 * makespan * (d_mat[i][r] / c_vec[r])
            new_vec[i] += boost
            
        # Normalize to prevent overflow
        if np.max(new_vec) > 0:
            new_vec = new_vec / np.max(new_vec) * np.max(current_vec)
            
        current_vec = new_vec
        
    return best_vec

# EVOLVE-BLOCK-END
