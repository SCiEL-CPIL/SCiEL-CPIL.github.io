import numpy as np
from collections import deque
import time

def priority(instance):
    """
    Dynamic Resource-Constrained Critical Path (RCCP) heuristic.
    
    Strategy:
    1. Compute static CPM slack for initial ordering.
    2. Simulate the serial schedule generation scheme.
    3. At each step, for eligible activities, compute a dynamic priority score:
       - Base: Negative static slack (critical path proximity).
       - Dynamic Penalty: Accumulated resource demand intensity relative to capacity 
         for the remaining duration, weighted by how "tight" the resource is.
       - Tie-break: Duration.
    4. Return the scores derived from this dynamic evaluation.
    
    Note: The 'priority' function is called once per instance in the context of the 
    serial SGS, but here we simulate the entire schedule to determine a static 
    priority vector that reflects the dynamic needs. However, since the evaluator 
    calls priority(instance) and expects a vector to be used by an external SGS, 
    we must return a static vector. 
    
    REVISION: The external SGS uses the returned vector. If we return a static 
    vector, we cannot adapt dynamically during the SGS. 
    However, the task says "a serial schedule-generation scheme starts the 
    highest-priority eligible activity". This implies the priority function 
    defines the ordering. 
    
    To make this "RCCP" in a static vector context, we can approximate the 
    dynamic pressure by calculating a "Resource Load Factor" for each activity 
    based on the total demand of all activities that share resources and have 
    overlapping potential time windows (approximated by CPM windows).
    
    Let's try a different structural change: **Min Late Finish (MLF)** with 
    **Resource Pressure Scaling**.
    
    Actually, a more distinct structural change from "Static CPM Slack" is 
    **Dynamic Slack** computed via a quick look-ahead simulation. But since we 
    must return a static array, we can compute a "Simulated Slack" by running 
    a greedy SGS using a simple heuristic (e.g., Min Slack) and then using the 
    *actual* slack from that simulation to refine the priorities for the real 
    SGS? No, the real SGS uses the returned array.
    
    Let's stick to a robust static heuristic that is structurally different from 
    simple CPM slack: **Weighted Sum of (1/Slack + Resource_Bottleneck_Score)**.
    
    Wait, the inspirations mention "Dynamic Critical Time Window" and "RCCP". 
    Since I can only return a static array, I will implement a **Look-Ahead 
    Resource Pressure** heuristic.
    
    1. Compute CPM Slack.
    2. For each resource, identify the "bottleneck" level (total demand / capacity).
    3. For each activity, calculate a "Resource Pressure Score" = Sum over resources 
       of (Demand_i,r / Capacity_r) * (1 / (Slack_i + 1)).
    4. Priority = -(Slack_i) - alpha * ResourcePressure_i.
    
    This is still somewhat close to CPM. Let's try **Schedule Simulation Based Priority**.
    I will run a fast, greedy SGS simulation inside the priority function to determine 
    which activities *actually* get delayed due to resources in a preliminary pass, 
    and then boost the priority of those that *would have been* critical but are 
    constrained.
    
    Actually, the most distinct structural change available in a static priority 
    function for Serial SGS is **Min Latest Start (LS)** vs **Min Slack**. 
    LS = LF - Duration. Min LS is equivalent to Min Slack + Duration. 
    The current code uses -Slack + 0.1*Duration.
    
    Let's try **Max Duration / (Slack + 1)**? Or **Critical Ratio**?
    
    Let's implement **RCCP approximation**:
    1. Compute CPM ES/LS.
    2. Compute "Resource Density" for each activity: 
       Density_i = Sum_r (Demand_ir / Capacity_r).
    3. Priority Score = -Slack_i - beta * Density_i.
       This penalizes activities that use scarce resources, pushing them earlier 
       to secure resources, which is the core of RCCP.
    
    Let's refine the beta. If a resource is heavily constrained (high total demand), 
    the penalty should be higher.
    GlobalResourcePressure_r = Sum_j Demand_jr / Capacity_r.
    ActivityPressure_i = Sum_r (Demand_ir * GlobalResourcePressure_r).
    
    Score = -Slack_i - 0.5 * ActivityPressure_i.
    """
    start_time = time.time()
    
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    # Assuming 'demands' is a list of lists or a 2D array: demands[i][r]
    # And 'capacities' is a list/array of resource capacities.
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    
    n = len(durations)
    num_resources = len(capacities)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            successors[i].append(j)
            
    # Topological Sort
    in_degree = [len(predecessors[j]) for j in range(n)]
    queue = deque([j for j in range(n) if in_degree[j] == 0])
    topo_order = []
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) != n:
        # Fallback
        return durations.copy()

    # Forward Pass: ES, EF
    ES = [0.0] * n
    EF = [0.0] * n
    for j in topo_order:
        if predecessors[j]:
            ES[j] = max(EF[i] for i in predecessors[j])
        else:
            ES[j] = 0.0
        EF[j] = ES[j] + durations[j]
        
    makespan_cpm = max(EF)
    
    # Backward Pass: LF, LS
    LF = [makespan_cpm] * n
    LS = [makespan_cpm] * n
    for j in reversed(topo_order):
        if successors[j]:
            LF[j] = min(LS[k] for k in successors[j])
        else:
            LF[j] = makespan_cpm
        LS[j] = LF[j] - durations[j]
        
    Slack = [LS[j] - ES[j] for j in range(n)]
    
    # Calculate Global Resource Pressure
    # Total demand per resource
    total_demands = np.sum(demands, axis=0) # Shape (num_resources,)
    # Pressure ratio: Total Demand / Capacity
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        resource_pressure = np.where(capacities > 0, total_demands / capacities, 0.0)
    
    # Calculate Activity Specific Pressure
    # For each activity i, pressure is sum over r of (Demand_ir * ResourcePressure_r)
    # This approximates how much this activity contributes to the bottleneck.
    # We weight it by the scarcity of the resource.
    activity_pressure = np.sum(demands * resource_pressure, axis=1)
    
    # Normalize pressure to be comparable to slack
    max_slack = max(Slack) if Slack else 1.0
    max_pressure = np.max(activity_pressure) if np.max(activity_pressure) > 0 else 1.0
    
    # Heuristic: 
    # Primary: Min Slack (Critical Path)
    # Secondary: Min Pressure (Resource Scarcity)
    # We want to prioritize activities that are on the critical path AND use scarce resources.
    # Score = -Slack - lambda * (Pressure / max_pressure)
    
    # Lambda tuning: 
    # If resources are very tight, pressure matters more.
    # Let's use a dynamic lambda based on average resource utilization.
    avg_utilization = np.mean(resource_pressure)
    # If avg_util > 1, resources are over-subscribed.
    # Scale lambda between 0.1 and 1.0
    lam = min(1.0, max(0.1, avg_utilization * 0.5))
    
    scores = np.array([-s - lam * (p / max_pressure) for s, p in zip(Slack, activity_pressure)])
    
    # Add small duration tie-breaker
    scores += 0.01 * durations
    
    # Safety check
    if time.time() - start_time > 100:
         return scores
         
    return scores
