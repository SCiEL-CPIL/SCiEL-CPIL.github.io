"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def _simulate_sgs(instance, priority_scores):
    """
    Serial Schedule Generation Scheme.
    Returns makespan.
    instance: dict with 'durations', 'demands', 'capacities', 'predecessors'
    priority_scores: array of length N, higher is better.
    """
    N = len(instance["durations"])
    durations = np.array(instance["durations"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)  # (N, R)
    capacities = np.array(instance["capacities"], dtype=float)  # (R,)
    R = len(capacities)
    
    # Build predecessor sets
    preds = instance.get("predecessors", [])
    if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], (int, float, np.integer, np.floating)):
        # Maybe it's a flat list? Unlikely for this format, but handle just in case.
        # Standard format is list of lists.
        pass
    
    pred_sets = [set() for _ in range(N)]
    successors = [set() for _ in range(N)]
    
    if "predecessors" in instance:
        preds_list = instance["predecessors"]
        for i, p_list in enumerate(preds_list):
            if p_list is None:
                continue
            for p in p_list:
                pred_sets[i].add(p)
                successors[p].add(i)
    
    # Initial eligible set: nodes with no predecessors
    eligible = set(i for i in range(N) if len(pred_sets[i]) == 0)
    
    # Resource usage over time. We track events or just simulate time steps?
    # Since durations are likely floats or ints, and N=30, we can track resource availability by time points.
    # Simpler: keep a list of (finish_time, resource_id, amount) to release resources.
    # And current resource usage.
    
    current_usage = np.zeros(R)
    # List of (time, resource_idx, amount) to process
    release_events = []
    
    scheduled = set()
    start_times = np.zeros(N)
    
    while len(scheduled) < N:
        # Find the best eligible activity
        best_act = -1
        best_score = -np.inf
        
        for act in eligible:
            # Check if resources are available at time 0? No, time advances.
            # The SGS works by:
            # 1. Identify eligible activities (all preds scheduled).
            # 2. Among eligible, pick the one with highest priority.
            # 3. Check if resources are available. If not, wait?
            # Standard Serial SGS:
            # - Eligible set E(t) = {j | preds(j) subset of Scheduled(t)}
            # - Select j in E(t) with max priority.
            # - If resources available at t, start j at t.
            # - Else, find next time t' > t when resources are sufficient, and start j at t'.
            # - Update scheduled, E, resources.
            
            # Wait, the standard definition of Serial SGS usually implies that we select the highest priority eligible activity and schedule it as early as possible.
            # "As early as possible" means checking resource constraints.
            
            # Let's implement the "as early as possible" logic.
            # Current time is effectively determined by the max start time of predecessors?
            # No, in Serial SGS, time is continuous.
            # Actually, the standard algorithm is:
            # 1. E = {j | Pred(j) subset of S}
            # 2. j = argmax priority in E
            # 3. t_start = max(finish_time(pred) for pred in Pred(j))
            # 4. Check if resources are available for j in [t_start, t_start + d_j].
            #    If not, increase t_start to the next time when resources are free.
            # 5. Schedule j at t_start.
            
            # This requires checking resource availability over intervals.
            # Given N=30, this is feasible.
            
            # Let's refine the loop structure.
            # We maintain a set of scheduled activities and their start/finish times.
            # We maintain resource profiles.
            
            pass # Placeholder, will implement below
            
        # Implementation detail:
        # To check resource availability, we need to know the resource usage profile.
        # Since we are building the schedule incrementally, we can maintain a list of active intervals or a timeline.
        # With N=30, a simple timeline of events (start/end) is efficient.
        
        # Let's restart the logic inside the function properly.
        pass

    return 0 # Placeholder


def _simulate_sgs_fast(instance, priority_scores):
    """
    Optimized Serial SGS.
    """
    N = len(instance["durations"])
    durations = np.array(instance["durations"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    capacities = np.array(instance["capacities"], dtype=float)
    R = len(capacities)
    
    preds_list = instance.get("predecessors", [])
    pred_sets = [set() for _ in range(N)]
    succ_sets = [set() for _ in range(N)]
    
    for i, p_list in enumerate(preds_list):
        if p_list is None:
            continue
        for p in p_list:
            pred_sets[i].add(p)
            succ_sets[p].add(i)
            
    # State
    start_times = np.full(N, -1.0)
    finish_times = np.full(N, -1.0)
    scheduled = np.zeros(N, dtype=bool)
    
    # Resource usage tracking:
    # We need to check if a job can start at time t.
    # We can maintain a list of "resource events" (time, delta_vector).
    # Or, simpler for small N: maintain a list of active jobs and check overlap.
    # Even simpler: for a candidate job j at time t, check all scheduled jobs k.
    # If k overlaps with j in time AND they share resources, check if sum of demands <= capacity.
    # Overlap: start_k < t + d_j AND finish_k > t.
    
    # This is O(N^2 * R) per insertion, so O(N^3 * R) total.
    # N=30 -> 27000 * 4 = 108,000 ops. Very fast.
    
    while np.sum(scheduled) < N:
        # Find eligible
        eligible = [j for j in range(N) if not scheduled[j] and all(scheduled[p] for p in pred_sets[j])]
        
        if not eligible:
            # Should not happen if instance is valid
            break
            
        # Pick highest priority
        best_j = max(eligible, key=lambda j: priority_scores[j])
        
        # Determine earliest start time
        # Earliest start is max finish time of predecessors
        t_start = 0.0
        for p in pred_sets[best_j]:
            if finish_times[p] > t_start:
                t_start = finish_times[p]
                
        # Check resource feasibility at t_start
        # We need to find the earliest t >= t_start such that resources are available for [t, t + d_j]
        
        # Collect all relevant time points:
        # 1. t_start
        # 2. Finish times of all currently scheduled jobs that overlap with [t_start, t_start + d_j]
        #    Actually, we just need to check intervals.
        #    The resource usage is piecewise constant, changing only at start/finish times of scheduled jobs.
        
        # Let's collect candidate start times.
        candidates = set()
        candidates.add(t_start)
        
        d_j = durations[best_j]
        
        for k in range(N):
            if scheduled[k]:
                # If k finishes after t_start, it might affect resource availability
                if finish_times[k] > t_start:
                    # The resource profile changes at finish_times[k]
                    candidates.add(finish_times[k])
                    # Also, if k starts after t_start but before t_start + d_j,
                    # the resource usage increases at start_times[k].
                    # But we are looking for a start time t for j.
                    # If we delay j to t, we must ensure [t, t+d_j] is feasible.
                    # The constraints are active at the start times of other jobs too?
                    # Yes, if another job starts at t_k, resource usage goes up.
                    # So we should also consider start_times[k] if start_times[k] >= t_start.
                    if start_times[k] >= t_start:
                        candidates.add(start_times[k])
                        
        # Sort candidates
        candidates = sorted([c for c in candidates if c >= t_start])
        
        # Check each candidate time
        found = False
        for t in candidates:
            # Check if resources are sufficient for j in [t, t + d_j]
            feasible = True
            for r in range(R):
                # Base demand of j
                req = demands[best_j, r]
                if req == 0:
                    continue
                    
                # Sum demands of other jobs overlapping with [t, t + d_j]
                total = req
                for k in range(N):
                    if k == best_j or not scheduled[k]:
                        continue
                    # Check overlap
                    if start_times[k] < t + d_j and finish_times[k] > t:
                        total += demands[k, r]
                        
                if total > capacities[r]:
                    feasible = False
                    break
                    
            if feasible:
                # Schedule j at t
                start_times[best_j] = t
                finish_times[best_j] = t + d_j
                scheduled[best_j] = True
                found = True
                break
                
        if not found:
            # Fallback: this shouldn't happen if capacities are sufficient for individual jobs
            # But if a job alone exceeds capacity, it's infeasible.
            # Assume feasible instance.
            # If not found, try t_start + epsilon? Or just break.
            # For safety, if no candidate works, try a large time?
            # Actually, if no candidate in the set works, it might mean we missed a point.
            # But the set of start/finish times covers all changes.
            # If it's still infeasible, the instance might be infeasible.
            # Let's assume feasibility.
            pass
            
    makespan = np.max(finish_times)
    return makespan


def priority(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    N = len(instance["durations"])
    durations = np.array(instance["durations"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    capacities = np.array(instance["capacity"], dtype=float) if "capacity" in instance else np.array(instance["capacities"], dtype=float)
    
    # Precompute static features
    # 1. Longest Path (Critical Path) from start
    # 2. Longest Path to end (Successor path)
    # 3. Resource Intensity
    
    # Topological Sort
    preds_list = instance.get("predecessors", [])
    pred_sets = [set() for _ in range(N)]
    succ_sets = [set() for _ in range(N)]
    in_degree = np.zeros(N)
    
    for i, p_list in enumerate(preds_list):
        if p_list is None:
            continue
        for p in p_list:
            pred_sets[i].add(p)
            succ_sets[p].add(i)
            in_degree[i] += 1
            
    # Queue for topological sort
    from collections import deque
    q = deque([i for i in range(N) if in_degree[i] == 0])
    topo_order = []
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in succ_sets[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
                
    if len(topo_order) < N:
        # Cycle, return default
        return np.asarray(durations, dtype=float)
        
    # Earliest Start (ES) based on duration only
    ES = np.zeros(N)
    for u in topo_order:
        for v in succ_sets[u]:
            if ES[u] + durations[u] > ES[v]:
                ES[v] = ES[u] + durations[u]
                
    # Latest Finish (LF) based on duration only
    # Project duration without resources
    makespan_no_res = np.max(ES + durations)
    LF = np.full(N, makespan_no_res)
    for u in reversed(topo_order):
        for v in succ_sets[u]:
            if LF[v] - durations[u] < LF[u]:
                LF[u] = LF[v] - durations[u]
                
    # Slack
    slack = LF - (ES + durations)
    
    # Resource Intensity: sum of demands / capacity
    # Normalize by max possible intensity?
    res_intensity = np.sum(demands / capacities, axis=1)
    
    # Combine features
    # Weighting:
    # High ES (late) -> Low priority? No, usually we want to schedule critical path early.
    # Low Slack -> High priority.
    # High Duration -> High priority.
    # High Resource Intensity -> High priority (to avoid conflicts later? Or low to spread load?)
    # Usually, "Most Critical" is best. Critical = Low Slack.
    
    # Let's try a combination:
    # Score = w1 * (1 / (slack + 1)) + w2 * (duration / max_duration) + w3 * (res_intensity / max_res_intensity)
    
    max_duration = np.max(durations) if np.max(durations) > 0 else 1.0
    max_res_int = np.max(res_intensity) if np.max(res_intensity) > 0 else 1.0
    
    # Normalize
    norm_slack_inv = 1.0 / (slack + 1.0)
    norm_dur = durations / max_duration
    norm_res = res_intensity / max_res_int
    
    # Weights
    # Critical path (low slack) is most important.
    w1 = 1.0
    w2 = 0.5
    w3 = 0.5
    
    base_priority = w1 * norm_slack_inv + w2 * norm_dur + w3 * norm_res
    
    # Local Search / Hill Climbing
    # We will perturb the priority vector and check makespan.
    
    best_prio = base_priority.copy()
    best_makespan = _simulate_sgs_fast(instance, best_prio)
    
    # If time allows, do some random restarts or local search
    # Given N=30, we can do many evaluations.
    
    # Strategy: 
    # 1. Start with base_priority.
    # 2. Try swapping priorities of pairs of activities.
    # 3. If makespan improves, accept.
    
    # To speed up, we can use a limited number of iterations.
    
    iterations = 0
    while time.time() < deadline:
        # Pick two random indices
        i, j = random.sample(range(N), 2)
        
        # Swap priorities
        temp = best_prio[i]
        best_prio[i] = best_prio[j]
        best_prio[j] = temp
        
        new_makespan = _simulate_sgs_fast(instance, best_prio)
        
        if new_makespan < best_makespan:
            best_makespan = new_makespan
            iterations += 1
        else:
            # Revert
            best_prio[j] = best_prio[i]
            best_prio[i] = temp
            
        # Check deadline
        if time.time() > deadline - 5.0:
            break
            
    return best_prio

# EVOLVE-BLOCK-END
