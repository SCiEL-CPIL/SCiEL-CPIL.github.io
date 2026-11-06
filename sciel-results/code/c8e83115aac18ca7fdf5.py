"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_graph(instance):
    """Build predecessor/successor lists and compute static priorities."""
    N = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # (N, R)
    capacities = np.asarray(instance["capacities"], dtype=float)  # (R,)
    R = len(capacities)

    # Predecessors: list of lists, preds[i] = list of predecessors of i
    preds_raw = instance.get("predecessors", None)
    if preds_raw is None:
        preds = [[] for _ in range(N)]
    else:
        # Could be list of lists, list of arrays, or array of arrays
        preds = [list(p) for p in preds_raw]

    # Successors
    succs = [[] for _ in range(N)]
    for i in range(N):
        for p in preds[i]:
            succs[p].append(i)

    # Compute Latest Finish Time (LFT) via backward pass
    # First, compute topological order
    in_degree = [len(preds[i]) for i in range(N)]
    queue = [i for i in range(N) if in_degree[i] == 0]
    topo = []
    q = list(queue)
    while q:
        node = q.pop(0)
        topo.append(node)
        for s in succs[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                q.append(s)

    # Backward pass for LFT
    LFT = np.zeros(N, dtype=float)
    # Start with sink nodes (no successors)
    for i in reversed(topo):
        if not succs[i]:
            LFT[i] = durations[i]
        else:
            LFT[i] = durations[i] + min(LFT[s] for s in succs[i])

    # Resource intensity: sum of demands / capacities
    # For each activity, compute how much of each resource it uses relative to capacity
    resource_intensity = np.zeros(N, dtype=float)
    for r in range(R):
        if capacities[r] > 0:
            resource_intensity += demands[:, r] / capacities[r]
        else:
            resource_intensity += demands[:, r]

    # Combine LFT (higher = more urgent) and resource intensity
    # We want higher priority = should be scheduled earlier
    # LFT: lower LFT means the activity finishes earlier in the critical path, so it's more urgent
    # Actually, for makespan minimization, we want to schedule activities with smaller LFT first
    # So priority = -LFT + weight * resource_intensity
    # Let's normalize
    LFT_min = LFT.min()
    LFT_max = LFT.max()
    if LFT_max > LFT_min:
        LFT_norm = (LFT - LFT_min) / (LFT_max - LFT_min)
    else:
        LFT_norm = np.zeros(N, dtype=float)

    RI_min = resource_intensity.min()
    RI_max = resource_intensity.max()
    if RI_max > RI_min:
        RI_norm = (resource_intensity - RI_min) / (RI_max - RI_min)
    else:
        RI_norm = np.zeros(N, dtype=float)

    # Priority: higher is better. We want low LFT (urgent) and high resource intensity
    # priority = (1 - LFT_norm) * w1 + RI_norm * w2
    w1 = 0.7
    w2 = 0.3
    priority_scores = (1.0 - LFT_norm) * w1 + RI_norm * w2

    return priority_scores, durations, demands, capacities, preds, succs, N, R


def _simulate_sgs(priority_scores, durations, demands, capacities, preds, succs, N, R):
    """Run Serial SGS with given priority scores and return makespan."""
    # priority_scores: higher = schedule earlier
    # Build a priority queue: sort activities by priority descending
    # But we need to respect precedence constraints
    
    # Use a simple approach: repeatedly pick the eligible activity with highest priority
    # Eligible = all predecessors are completed
    
    # completed = set of completed activities
    # resource_usage = current resource usage per resource
    
    completed = [False] * N
    resource_usage = np.zeros(R, dtype=float)
    start_times = np.zeros(N, dtype=float)
    time = 0.0
    
    # Count of remaining predecessors for each activity
    remaining_preds = [len(preds[i]) for i in range(N)]
    
    # Activities that are eligible (all preds done)
    # We'll maintain a list of eligible activities
    
    # Priority order: sort by priority_scores descending
    # But we need to re-check eligibility as time progresses
    
    # Use a max-heap approach: at each step, find eligible activity with max priority
    # For N=30, this is fast enough
    
    n_scheduled = 0
    while n_scheduled < N:
        # Find eligible activities
        eligible = [i for i in range(N) if not completed[i] and remaining_preds[i] == 0]
        
        if not eligible:
            # Should not happen if graph is acyclic
            break
        
        # Pick the one with highest priority
        best = max(eligible, key=lambda i: priority_scores[i])
        
        # Check if resources are available
        can_start = True
        for r in range(R):
            if resource_usage[r] + demands[best, r] > capacities[r] + 1e-9:
                can_start = False
                break
        
        if can_start:
            # Schedule best at current time
            start_times[best] = time
            for r in range(R):
                resource_usage[r] += demands[best, r]
            completed[best] = True
            n_scheduled += 1
            
            # Update remaining_preds for successors
            for s in succs[best]:
                remaining_preds[s] -= 1
        else:
            # Can't start best due to resource constraints
            # Need to wait for some activity to finish
            # Find the earliest finishing activity among currently running ones
            # that uses at least one resource that best needs
            
            # Find activities that are running (started but not completed... wait, in serial SGS,
            # we schedule one at a time, so all previously scheduled are "done" in terms of being
            # placed, but their duration hasn't elapsed yet)
            
            # Actually in serial SGS, we place activities one by one. The time advances to the
            # earliest time when resources become available.
            
            # Find the minimum time when resources for 'best' become available
            # For each resource r, if best needs demand[best,r], we need to wait until
            # the sum of demands of running activities on r drops below capacities[r] - demand[best,r]
            
            # In serial SGS, "running" means scheduled but not yet finished.
            # Let me track finish times of scheduled activities.
            
            # This is getting complex. Let me use a simpler event-based approach.
            
            # Actually, in standard Serial SGS:
            # - We maintain a list of scheduled activities with their start times
            # - At each step, we find the earliest time t >= current_time when all resources
            #   for the selected activity are available
            # - The activity starts at that time
            
            # Let me redo this with proper time advancement
            
            pass  # Handled below
    
    # The above approach is flawed. Let me rewrite the simulation properly.
    return None


def _simulate_sgs_v2(priority_scores, durations, demands, capacities, preds, succs, N, R):
    """Proper Serial SGS simulation."""
    completed = [False] * N
    start_times = np.zeros(N, dtype=float)
    remaining_preds = [len(preds[i]) for i in range(N)]
    
    # List of (finish_time, activity_id) for scheduled activities
    scheduled = []  # (finish_time, activity_id)
    
    # Current resource usage at time 0
    resource_usage = np.zeros(R, dtype=float)
    
    n_scheduled = 0
    current_time = 0.0
    
    while n_scheduled < N:
        # Find eligible activities
        eligible = [i for i in range(N) if not completed[i] and remaining_preds[i] == 0]
        
        if not eligible:
            break
        
        # Pick the one with highest priority
        best = max(eligible, key=lambda i: priority_scores[i])
        
        # Determine the earliest start time for best
        earliest_start = current_time
        
        # Check resource availability at current_time
        # We need to know resource usage at each time point
        # In serial SGS, we advance time to when resources become free
        
        # For each resource, check if best can start at current_time
        # If not, find when it can
        
        # Build a timeline of resource releases
        # For each resource r, we have a list of (release_time, amount_released)
        
        # Simpler approach: iterate through scheduled activities to find when resources free up
        
        # Actually, let's just simulate time forward
        # At each candidate time, check if resources are available
        
        # Get all relevant time points: current_time and finish times of scheduled activities
        time_points = sorted(set([current_time] + [ft for ft, _ in scheduled]))
        
        found_start = False
        for t in time_points:
            if t < earliest_start:
                continue
            
            # Compute resource usage at time t
            usage_at_t = np.zeros(R, dtype=float)
            for ft, aid in scheduled:
                if start_times[aid] <= t < ft:
                    usage_at_t += demands[aid]
            
            # Check if best can start at t
            can_start = True
            for r in range(R):
                if usage_at_t[r] + demands[best, r] > capacities[r] + 1e-9:
                    can_start = False
                    break
            
            if can_start:
                earliest_start = t
                found_start = True
                break
        
        if not found_start:
            # Should not happen, but just in case
            earliest_start = current_time
        
        # Schedule best at earliest_start
        start_times[best] = earliest_start
        finish_time = earliest_start + durations[best]
        scheduled.append((finish_time, best))
        completed[best] = True
        n_scheduled += 1
        
        # Update remaining_preds for successors
        for s in succs[best]:
            remaining_preds[s] -= 1
        
        # Advance current_time to earliest_start (we may have jumped forward)
        current_time = earliest_start
    
    # Makespan is the maximum finish time
    if scheduled:
        makespan = max(ft for ft, _ in scheduled)
    else:
        makespan = 0.0
    
    return makespan


def priority(instance):
    """
    Returns a priority score for each activity to be used in a Serial SGS.
    Uses Latest Finish Time (LFT) combined with resource intensity.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    try:
        priority_scores, durations, demands, capacities, preds, succs, N, R = _build_graph(instance)
        
        # Verify with a quick simulation
        makespan = _simulate_sgs_v2(priority_scores, durations, demands, capacities, preds, succs, N, R)
        
        # Check deadline
        if time.time() > deadline:
            return priority_scores
        
        return priority_scores
        
    except Exception:
        # Fallback: just use durations
        try:
            return np.asarray(instance["durations"], dtype=float)
        except:
            return np.ones(len(instance["durations"]))
# EVOLVE-BLOCK-END
