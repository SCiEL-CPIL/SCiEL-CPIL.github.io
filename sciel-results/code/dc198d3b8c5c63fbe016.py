import numpy as np
import time

def priority(instance):
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    predecessors = instance["predecessors"]
    demands = np.asarray(instance["demands"], dtype=float)
    
    # Handle potential typo in instance key
    if "capacities" in instance:
        capacities = np.asarray(instance["capacities"], dtype=float)
    else:
        capacities = np.asarray(instance["capancies"], dtype=float)
        
    n_resources = len(capacities)

    # 1. Build successor lists and in-degrees
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1

    # 2. Topological Sort (Kahn's Algorithm)
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    current_queue = queue[:]
    in_degree_temp = in_degree[:]

    while current_queue:
        node = current_queue.pop(0)
        topo_order.append(node)
        for succ in successors[node]:
            in_degree_temp[succ] -= 1
            if in_degree_temp[succ] == 0:
                current_queue.append(succ)

    if len(topo_order) != n:
        # Cycle detected or invalid graph, fallback to simple duration-based
        return durations.copy()

    # 3. Forward Pass: Compute Earliest Start (ES) and Earliest Finish (EF)
    ES = np.zeros(n)
    EF = np.zeros(n)
    for i in topo_order:
        max_pred_ef = 0.0
        for p in predecessors[i]:
            if EF[p] > max_pred_ef:
                max_pred_ef = EF[p]
        ES[i] = max_pred_ef
        EF[i] = ES[i] + durations[i]

    project_duration = np.max(EF) if n > 0 else 0.0

    # 4. Backward Pass: Compute Latest Finish (LF) and Latest Start (LS)
    LF = np.full(n, project_duration)
    LS = np.full(n, project_duration)

    for i in reversed(topo_order):
        if successors[i]:
            min_succ_ls = float('inf')
            for s in successors[i]:
                if LS[s] < min_succ_ls:
                    min_succ_ls = LS[s]
            LF[i] = min_succ_ls
        LS[i] = LF[i] - durations[i]

    # 5. Compute Slack and Criticality
    slack = LS - ES
    max_slack = np.max(slack) if n > 0 else 1.0
    if max_slack < 1e-9:
        max_slack = 1.0
    # Criticality: 1 for critical (slack=0), 0 for max slack
    criticality = 1.0 - (slack / max_slack)

    # 6. Compute Resource Intensity (RI) and Resource Pressure
    # RI_i = sum(demand_ij / capacity_j)
    # This measures how much of the total resource pool an activity consumes.
    safe_caps = np.where(capacities == 0, 1.0, capacities)
    resource_usage = np.sum(demands / safe_caps[np.newaxis, :], axis=1)
    
    min_ru = np.min(resource_usage)
    max_ru = np.max(resource_usage)
    ru_range = max_ru - min_ru
    if ru_range > 1e-10:
        resource_usage_norm = (resource_usage - min_ru) / ru_range
    else:
        resource_usage_norm = np.zeros(n)

    # 7. Compute Resource Bottleneck Pressure
    # For each activity, look at the peak resource demand among its successors.
    # If an activity has successors with high resource demand, starting it early
    # might be good to clear the way, OR it might be bad if it blocks them.
    # Actually, a common heuristic is "Least Total Float" or "Minimum Slack".
    # Another is "Minimum Resource Intensity" (SPT equivalent for resources).
    
    # Let's try a combination:
    # Primary: Criticality (LS-based)
    # Secondary: Resource Intensity (higher is better for "big" tasks that block resources)
    # Tertiary: Duration (shorter is better for tie-breaking, SPT rule)
    
    # Normalize duration
    min_dur = np.min(durations)
    max_dur = np.max(durations)
    dur_range = max_dur - min_dur
    if dur_range > 1e-10:
        duration_norm = (durations - min_dur) / dur_range
    else:
        duration_norm = np.zeros(n)

    # We want HIGH score for:
    # - Low Slack (High Criticality)
    # - High Resource Intensity (to clear big blockers early? or avoid them? 
    #   Usually, in serial SGS, you pick the eligible activity with highest priority.
    #   If we pick high RI first, we use resources early. This might be good if resources are scarce.
    #   However, "Minimum Slack" is the most robust heuristic.
    #   Let's stick to Criticality as the main driver.
    
    # Let's refine the weights.
    # Weight 1: Criticality (LS - ES)
    # Weight 2: Resource Intensity
    # Weight 3: Duration (inverse, i.e., shorter is higher priority)
    
    # SPT (Shortest Processing Time) is often good for makespan.
    # So we want high score for LOW duration.
    # score_dur = 1 - duration_norm
    
    # Let's try:
    # score = 0.7 * criticality + 0.2 * resource_usage_norm + 0.1 * (1 - duration_norm)
    
    scores = 0.7 * criticality + 0.2 * resource_usage_norm + 0.1 * (1.0 - duration_norm)

    # Time check
    if time.time() - start_time > deadline:
         return scores

    return scores
