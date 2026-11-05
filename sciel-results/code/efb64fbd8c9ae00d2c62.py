"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time


def priority(instance):
    """
    Empirical Resource-Bottleneck + Critical Path priority.
    
    Strategy:
    1. Compute static bottom-level (CPM longest path to end) for each activity.
    2. Run a fast greedy resource-feasible simulation using bottom-level as the
       priority rule. Track which activities are actually delayed by resource
       contention (empirical bottleneck detection).
    3. Combine: final_priority = w1 * bottom_level_norm + w2 * bottleneck_score
       where bottleneck_score reflects how much an activity is a resource
       bottleneck (high demand/capacity ratio AND causes delays to others).
    
    This is structurally different from the current program: instead of
    *predicting* contention from the non-resource schedule, we *simulate* a
    resource-feasible schedule and *observe* which activities actually create
    contention.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance.get("predecessors", [[] for _ in range(n)])

    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)

    if demands is None or capacities is None:
        return _bottom_level(n, durations, predecessors)

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    n_resources = len(capacities)

    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)

    # Compute topological order
    in_degree = [0] * n
    for i in range(n):
        for pred in predecessors[i]:
            in_degree[i] += 1

    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(topo_order) != n:
        return durations.copy()

    # Backward pass: bottom level (longest path to end)
    bottom_level = np.zeros(n)
    for node in reversed(topo_order):
        if not successors[node]:
            bottom_level[node] = durations[node]
        else:
            bottom_level[node] = durations[node] + max(
                bottom_level[s] for s in successors[node]
            )

    # Forward pass: earliest start (CPM, no resources)
    est = np.zeros(n)
    for node in topo_order:
        if predecessors[node]:
            est[node] = max(est[p] + durations[p] for p in predecessors[node])
        else:
            est[node] = 0.0

    # --- Greedy resource-feasible simulation to detect empirical bottlenecks ---
    # We simulate a serial SGS using bottom-level as the priority.
    # Track: for each activity, how much it was delayed due to resources,
    # and how much it "blocks" others (causes contention).
    
    # State:
    # - scheduled: set of completed activities
    # - start_time[i]: actual start in simulation
    # - resource_usage[r][t]: usage of resource r at time t (we use a list of events)
    
    # Use event-based resource tracking: for each resource, maintain a list of
    # (end_time, demand) tuples representing ongoing activities.
    
    scheduled = [False] * n
    sim_start = [0.0] * n
    sim_end = [0.0] * n
    # For each activity, track how much it was delayed from its EST due to resources
    delay_from_resources = np.zeros(n)
    # For each activity, track how much "contention" it creates:
    # sum over resources of demand[i][r] / capacity[r] weighted by how many
    # other activities are also using that resource at the same time
    contention_cause = np.zeros(n)

    # We'll use a simple approach: maintain for each resource a list of
    # (end_time, demand) for currently running activities.
    # At each step, find the earliest time when enough resources are free.
    
    # Activity indices sorted by bottom level (descending) as tie-breaker
    # But we need dynamic eligibility: an activity is eligible when all its
    # predecessors are scheduled.
    
    # For efficiency with n=30, we can use a simple loop.
    
    # resource_events[r] = list of (end_time, demand)
    resource_events = [[] for _ in range(n_resources)]
    
    # We need to process activities in priority order (bottom level descending)
    # but only among eligible ones.
    
    # Sort activities by bottom level descending
    order_by_bl = sorted(range(n), key=lambda i: -bottom_level[i])
    
    # Track which predecessors are done for each activity
    pred_count = [len(predecessors[i]) for i in range(n)]
    remaining_preds = pred_count[:]  # copy
    
    # Process
    for _ in range(n):
        # Find eligible activity with highest bottom level
        best_i = -1
        best_bl = -1.0
        for i in order_by_bl:
            if not scheduled[i] and remaining_preds[i] == 0:
                if bottom_level[i] > best_bl:
                    best_bl = bottom_level[i]
                    best_i = i
                break  # order_by_bl is sorted, first eligible is best
        
        if best_i == -1:
            # Should not happen if DAG is valid
            break
        
        # Determine earliest start for best_i considering resources
        # Precedence constraint:
        prec_start = 0.0
        for p in predecessors[best_i]:
            prec_start = max(prec_start, sim_end[p])
        
        # Resource constraint: find earliest time >= prec_start where
        # for all resources r, current usage + demand[best_i][r] <= capacity[r]
        earliest = prec_start
        
        # We need to find the earliest time when resources are available.
        # Collect all relevant end times from resource events.
        relevant_times = {earliest}
        for r in range(n_resources):
            if demands[best_i][r] > 0:
                for (et, d) in resource_events[r]:
                    if et > earliest:
                        relevant_times.add(et)
        
        # Sort candidate times and check feasibility
        candidate_times = sorted(relevant_times)
        found = False
        for t in candidate_times:
            feasible = True
            for r in range(n_resources):
                if demands[best_i][r] == 0:
                    continue
                # Sum demands of activities in resource_events[r] that are
                # still active at time t (i.e., end_time > t)
                current_usage = sum(d for (et, d) in resource_events[r] if et > t)
                if current_usage + demands[best_i][r] > capacities[r] + 1e-9:
                    feasible = False
                    break
            if feasible:
                earliest = t
                found = True
                break
        
        if not found:
            # This shouldn't happen if capacities are sufficient for single activity
            # but just in case, use the max end time
            earliest = max(
                (et for r in range(n_resources) for (et, d) in resource_events[r]),
                default=earliest
            )
        
        sim_start[best_i] = earliest
        sim_end[best_i] = earliest + durations[best_i]
        scheduled[best_i] = True
        
        # Track delay from resources
        delay_from_resources[best_i] = max(0.0, earliest - est[best_i])
        
        # Update resource events
        for r in range(n_resources):
            if demands[best_i][r] > 0:
                resource_events[r].append((sim_end[best_i], demands[best_i][r]))
        
        # Update remaining preds for successors
        for s in successors[best_i]:
            remaining_preds[s] -= 1
        
        # Check deadline
        if time.time() - start_time > DEADLINE:
            # Return what we have so far
            bl_norm = bottom_level / max(bottom_level.max(), 1e-9)
            return bl_norm

    # --- Compute bottleneck score ---
    # An activity is a "bottleneck" if:
    # 1. It has high demand-to-capacity ratio (structural)
    # 2. It was itself delayed by resources (empirical: it's in a contested region)
    # 3. It causes delays to successors (empirical: its presence blocks others)
    
    # Demand-to-capacity ratio (normalized)
    max_dcr = 0.0
    dcr = np.zeros(n)
    for i in range(n):
        for r in range(n_resources):
            if capacities[r] > 0:
                dcr[i] = max(dcr[i], demands[i][r] / capacities[r])
        max_dcr = max(max_dcr, dcr[i])
    if max_dcr > 0:
        dcr = dcr / max_dcr
    
    # Empirical: how much does this activity's scheduling cause contention?
    # Measure: for each activity i, look at the resource usage during [sim_start[i], sim_end[i])
    # and see how "tight" the resources are (usage close to capacity).
    contention_score = np.zeros(n)
    for i in range(n):
        s_i = sim_start[i]
        e_i = sim_end[i]
        dur_i = durations[i]
        if dur_i <= 0:
            continue
        for r in range(n_resources):
            if capacities[r] <= 0 or demands[i][r] <= 0:
                continue
            # Estimate average utilization of resource r during [s_i, e_i)
            # by checking all activities j that overlap with i
            total_demand = 0.0
            for j in range(n):
                if j == i:
                    continue
                if demands[j][r] <= 0:
                    continue
                overlap = min(e_i, sim_end[j]) - max(s_i, sim_start[j])
                if overlap > 0:
                    total_demand += demands[j][r] * overlap / dur_i
            utilization = total_demand / capacities[r]
            # Higher utilization means more contention
            contention_score[i] += min(2.0, utilization)
    
    # Normalize contention_score
    max_cs = contention_score.max()
    if max_cs > 0:
        contention_score = contention_score / max_cs
    
    # Also: delay propagation - how much does activity i delay its successors?
    # If i is on the critical path AND causes resource delays, it's a key bottleneck.
    # Measure: for each successor s of i, how much is s delayed beyond its EST?
    delay_propagation = np.zeros(n)
    for i in range(n):
        for s in successors[i]:
            delay_propagation[i] += delay_from_resources[s]
    
    max_dp = delay_propagation.max()
    if max_dp > 0:
        delay_propagation = delay_propagation / max_dp
    
    # Combine bottleneck indicators
    # bottleneck_score = weighted combination of dcr, contention, delay_propagation
    w_dcr = 0.3
    w_cont = 0.4
    w_delay = 0.3
    bottleneck_score = w_dcr * dcr + w_cont * contention_score + w_delay * delay_propagation
    
    # Normalize bottom level
    bl_max = bottom_level.max()
    if bl_max > 0:
        bl_norm = bottom_level / bl_max
    else:
        bl_norm = bottom_level
    
    # Final priority: combine critical path with bottleneck
    # Activities that are both critical AND bottlenecks get highest priority
    alpha = 0.6  # weight for critical path
    beta = 0.4   # weight for bottleneck
    
    priority_scores = alpha * bl_norm + beta * bottleneck_score
    
    # Ensure higher = more priority (schedule first)
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return priority_scores
    
    return priority_scores


def _bottom_level(n, durations, predecessors):
    """Fallback: compute bottom level (longest path to end)."""
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    in_degree = [0] * n
    for i in range(n):
        for pred in predecessors[i]:
            in_degree[i] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    if len(topo_order) != n:
        return durations.copy()
    
    bottom_level = np.zeros(n)
    for node in reversed(topo_order):
        if not successors[node]:
            bottom_level[node] = durations[node]
        else:
            bottom_level[node] = durations[node] + max(
                bottom_level[s] for s in successors[node]
            )
    
    return bottom_level
# EVOLVE-BLOCK-END
