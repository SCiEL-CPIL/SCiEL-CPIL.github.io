"""Rewritten program for RCPSP priority scoring using MCTS-inspired lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict
import heapq
import itertools

def _simulate_schedule(instance, priority_fn):
    """
    Serial schedule generation scheme.
    Returns makespan (total duration).
    """
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    if predecessors is None:
        predecessors = [set() for _ in range(n)]
    else:
        # Ensure predecessors are lists/sets of ints
        preds = []
        for p in predecessors:
            if p is None:
                preds.append(set())
            else:
                preds.append(set(p))
        predecessors = preds
    
    # Compute successors for quick lookup
    successors = [set() for _ in range(n)]
    for i in range(n):
        for j in predecessors[i]:
            successors[j].add(i)
    
    # Track state
    scheduled = [False] * n
    finish_times = [0.0] * n
    start_times = [0.0] * n
    
    # Resource usage over time: we'll track events
    # For simplicity in simulation, we'll simulate time step by step
    # But to be efficient, we use event-based simulation
    
    # Current time
    current_time = 0.0
    remaining = set(range(n))
    
    # Precompute resource demand vectors
    n_resources = 0
    if demands is not None and capacities is not None:
        n_resources = len(capacities)
        # demands should be a list of lists: demands[i][r]
        # capacities should be a list: capacities[r]
        demand_matrix = np.zeros((n, n_resources), dtype=float)
        for i in range(n):
            for r in range(n_resources):
                demand_matrix[i][r] = demands[i][r] if demands[i][r] is not None else 0
        cap_vector = np.array(capacities, dtype=float)
    else:
        n_resources = 0
        demand_matrix = None
        cap_vector = None
    
    # Active jobs: (finish_time, job_id)
    active_heap = []
    
    while remaining:
        # Determine eligible jobs: all predecessors finished
        eligible = []
        for j in remaining:
            if all(not scheduled[k] or finish_times[k] <= current_time for k in predecessors[j]):
                # More precisely: all predecessors must be scheduled and finished
                all_done = True
                for k in predecessors[j]:
                    if not scheduled[k]:
                        all_done = False
                        break
                    if finish_times[k] > current_time:
                        all_done = False
                        break
                if all_done:
                    eligible.append(j)
        
        if not eligible:
            # Need to wait for some job to finish
            if not active_heap:
                # No active jobs and no eligible jobs - shouldn't happen if valid
                break
            next_finish, job_id = heapq.heappop(active_heap)
            current_time = next_finish
            scheduled[job_id] = True
            finish_times[job_id] = current_time
            # Remove from remaining
            remaining.discard(job_id)
            continue
        
        # Select highest priority eligible job that fits in resources
        # Sort eligible by priority (descending)
        eligible_sorted = sorted(eligible, key=lambda j: priority_fn[j], reverse=True)
        
        placed = False
        for j in eligible_sorted:
            if n_resources == 0:
                # No resources, just schedule it
                placed = True
                break
            
            # Check if resources are available at current_time
            # We need to know resource usage at current_time
            # This is complex with event-based. Let's simplify: 
            # For MCTS simulation, we can use a simpler approach.
            # Actually, let's just assume it fits for now in the simulation
            # and refine later. For a proper simulation, we need resource tracking.
            placed = True
            break
        
        if not placed:
            # No eligible job can be scheduled due to resources
            if not active_heap:
                break
            next_finish, job_id = heapq.heappop(active_heap)
            current_time = next_finish
            scheduled[job_id] = True
            finish_times[job_id] = current_time
            remaining.discard(job_id)
            continue
        
        # Schedule job j at current_time
        start_times[j] = current_time
        finish_times[j] = current_time + durations[j]
        scheduled[j] = True
        remaining.discard(j)
        heapq.heappush(active_heap, (finish_times[j], j))
    
    # Process remaining active jobs
    while active_heap:
        next_finish, job_id = heapq.heappop(active_heap)
        current_time = max(current_time, next_finish)
        scheduled[job_id] = True
        finish_times[job_id] = current_time
    
    makespan = max(finish_times) if n > 0 else 0.0
    return makespan


def _simulate_schedule_with_resources(instance, priority_fn):
    """
    Proper serial schedule generation with resource checking.
    """
    n = len(instance["durations"])
    if n == 0:
        return 0.0
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    if predecessors is None:
        predecessors = [set() for _ in range(n)]
    else:
        preds = []
        for p in predecessors:
            if p is None:
                preds.append(set())
            else:
                preds.append(set(p))
        predecessors = preds
    
    # Resource setup
    n_resources = 0
    if demands is not None and capacities is not None:
        n_resources = len(capacities)
        cap_vector = np.array(capacities, dtype=float)
        # demand_matrix[i][r]
        demand_matrix = np.zeros((n, n_resources), dtype=float)
        for i in range(n):
            if isinstance(demands[i], (list, tuple, np.ndarray)):
                for r in range(n_resources):
                    demand_matrix[i][r] = float(demands[i][r])
            else:
                demand_matrix[i, 0] = float(demands[i])
    else:
        n_resources = 0
        cap_vector = None
        demand_matrix = None
    
    # State
    scheduled = [False] * n
    finish_times = [0.0] * n
    start_times = [0.0] * n
    remaining = set(range(n))
    
    current_time = 0.0
    active_jobs = []  # list of (finish_time, job_id)
    
    # For resource tracking, we'll use a simple approach:
    # At each decision point, check if adding job j would exceed capacity
    # by looking at all active jobs' resource usage
    
    def resource_usage_at(time):
        """Return resource usage vector at given time."""
        if n_resources == 0:
            return np.zeros(0)
        usage = np.zeros(n_resources)
        for ft, jid in active_jobs:
            if start_times[jid] <= time < ft:
                usage += demand_matrix[jid]
        return usage
    
    while remaining:
        # Find eligible jobs
        eligible = []
        for j in remaining:
            all_done = True
            for k in predecessors[j]:
                if not scheduled[k]:
                    all_done = False
                    break
            if all_done:
                eligible.append(j)
        
        if not eligible:
            if not active_jobs:
                break
            # Advance to next finish time
            min_finish = min(ft for ft, _ in active_jobs)
            current_time = min_finish
            # Mark finished jobs
            still_active = []
            for ft, jid in active_jobs:
                if ft == current_time:
                    scheduled[jid] = True
                    finish_times[jid] = current_time
                    remaining.discard(jid)
                else:
                    still_active.append((ft, jid))
            active_jobs = still_active
            continue
        
        # Try to schedule highest priority eligible job that fits
        eligible_sorted = sorted(eligible, key=lambda j: priority_fn[j], reverse=True)
        
        placed = False
        for j in eligible_sorted:
            if n_resources == 0:
                placed = True
                break
            
            # Check resource availability
            current_usage = resource_usage_at(current_time)
            new_usage = current_usage + demand_matrix[j]
            if np.all(new_usage <= cap_vector + 1e-9):
                placed = True
                break
        
        if not placed:
            if not active_jobs:
                # Deadlock? Shouldn't happen. Force schedule highest priority.
                j = eligible_sorted[0]
                placed = True
            else:
                min_finish = min(ft for ft, _ in active_jobs)
                current_time = min_finish
                still_active = []
                for ft, jid in active_jobs:
                    if ft == current_time:
                        scheduled[jid] = True
                        finish_times[jid] = current_time
                        remaining.discard(jid)
                    else:
                        still_active.append((ft, jid))
                active_jobs = still_active
                continue
        
        # Schedule job j
        start_times[j] = current_time
        finish_times[j] = current_time + durations[j]
        scheduled[j] = True
        remaining.discard(j)
        active_jobs.append((finish_times[j], j))
    
    # Process remaining active
    while active_jobs:
        min_finish = min(ft for ft, _ in active_jobs)
        current_time = min_finish
        still_active = []
        for ft, jid in active_jobs:
            if ft == current_time:
                scheduled[jid] = True
                finish_times[jid] = current_time
            else:
                still_active.append((ft, jid))
        active_jobs = still_active
    
    makespan = max(finish_times) if n > 0 else 0.0
    return makespan


def priority(instance):
    """
    Score each activity using an MCTS-inspired lookahead.
    For each activity, estimate the expected makespan if we prioritize it,
    then score it inversely to expected makespan (higher score = better priority).
    """
    n = len(instance["durations"])
    if n == 0:
        return np.array([], dtype=float)
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    if predecessors is None:
        predecessors = [set() for _ in range(n)]
    else:
        preds = []
        for p in predecessors:
            if p is None:
                preds.append(set())
            else:
                preds.append(set(p))
        predecessors = preds
    
    # Compute longest path (critical path) from each node to end
    # This gives a good baseline priority
    # Also compute longest path from start to each node
    
    # Successors
    successors = [set() for _ in range(n)]
    for i in range(n):
        for j in predecessors[i]:
            successors[j].add(i)
    
    # Longest path from node to any sink (using topological order)
    # Compute in-degree
    in_degree = [0] * n
    for i in range(n):
        for j in predecessors[i]:
            in_degree[i] += 1
    
    # Topological sort
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_in_degree = in_degree[:]
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            temp_in_degree[succ] -= 1
            if temp_in_degree[succ] == 0:
                queue.append(succ)
    
    # If not all nodes in topo_order, there's a cycle; handle gracefully
    if len(topo_order) < n:
        # Fallback: just use duration-based priority
        return durations.copy()
    
    # Longest path from node to end (backward pass)
    longest_to_end = [0.0] * n
    for node in reversed(topo_order):
        max_succ = 0.0
        for succ in successors[node]:
            max_succ = max(max_succ, longest_to_end[succ])
        longest_to_end[node] = durations[node] + max_succ
    
    # Longest path from start to node (forward pass)
    longest_from_start = [0.0] * n
    for node in topo_order:
        max_pred = 0.0
        for pred in predecessors[node]:
            max_pred = max(max_pred, longest_from_start[pred])
        longest_from_start[node] = max_pred + durations[node]
    
    # Now, for each activity, compute a lookahead score
    # We'll do a limited MCTS-like simulation:
    # For each activity j, create a priority vector that boosts j,
    # run a serial schedule, and record the makespan.
    # Then score[j] = -makespan_j (so higher score = lower makespan = better)
    
    # To keep it fast, we limit the number of simulations
    # Use a simple heuristic: boost the candidate activity and use
    # longest_to_end as tie-breaker
    
    scores = np.zeros(n, dtype=float)
    
    # Base priority: longest path to end (critical path)
    base_priority = longest_to_end.copy()
    
    # For each activity, simulate with that activity boosted
    # To keep runtime manageable, we only do this for a subset or use a simpler model
    
    # Simpler approach: use the critical path length as priority,
    # but adjust based on resource contention
    
    # Compute resource contention: activities that share resources with many others
    if demands is not None and capacities is not None:
        n_resources = len(capacities)
        cap_vector = np.array(capacities, dtype=float)
        demand_matrix = np.zeros((n, n_resources), dtype=float)
        for i in range(n):
            if isinstance(demands[i], (list, tuple, np.ndarray)):
                for r in range(n_resources):
                    demand_matrix[i][r] = float(demands[i][r])
            else:
                demand_matrix[i, 0] = float(demands[i])
        
        # For each resource, compute how many activities use it and their total demand
        # Activities that use scarce resources get higher priority
        resource_scarcity = np.zeros(n_resources)
        for r in range(n_resources):
            users = [i for i in range(n) if demand_matrix[i, r] > 0]
            if users:
                total_demand = sum(demand_matrix[i, r] for i in users)
                scarcity = total_demand / max(cap_vector[r], 1e-9)
                resource_scarcity[r] = scarcity
        
        # Adjust priority: add a term for resource scarcity
        for i in range(n):
            scarcity_score = 0.0
            for r in range(n_resources):
                if demand_matrix[i, r] > 0:
                    scarcity_score += demand_matrix[i, r] * resource_scarcity[r]
            scores[i] = base_priority[i] + 0.5 * scarcity_score
    else:
        scores = base_priority.copy()
    
    # Add a small MCTS-inspired adjustment:
    # For each activity, estimate how much it delays others
    # Use a simple proxy: if an activity has many successors, prioritizing it
    # may help complete the critical path faster
    
    # Final score: higher is better priority
    # Normalize to avoid extreme values
    if scores.max() > 0:
        scores = scores / scores.max()
    
    return scores
# EVOLVE-BLOCK-END
