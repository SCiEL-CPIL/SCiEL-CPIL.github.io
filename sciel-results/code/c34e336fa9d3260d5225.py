"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(instance):
    """
    Computes a priority score for each activity to be used by a serial schedule-generation scheme.
    Higher scores indicate higher priority.
    Uses a combination of Successor Critical Path (SCP) and Total Float (TF).
    Score = SCP / (TF + 1)
    """
    start_time = time.time()
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Get predecessors list
    pred_list = instance.get("predecessors", [])
    # Ensure pred_list is a list of lists of indices
    # Format assumption: pred_list[i] is a list of indices of predecessors of i
    if not pred_list:
        pred_list = [[] for _ in range(n)]
        
    # Build successor list
    succ_list = [[] for _ in range(n)]
    indegree = [0] * n
    for i in range(n):
        for p in pred_list[i]:
            succ_list[p].append(i)
            indegree[i] += 1

    # Topological Sort (Kahn's Algorithm)
    # We need this to compute forward and backward passes correctly
    from collections import deque
    q = deque()
    for i in range(n):
        if indegree[i] == 0:
            q.append(i)
            
    topo_order = []
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in succ_list[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)
                
    # If graph has cycles, topo_order will be shorter than n. 
    # Assuming valid DAG as per RCPSP standard.
    if len(topo_order) < n:
        # Fallback to simple duration if cycle detected (shouldn't happen)
        return durations.copy()

    # 1. Forward Pass: Earliest Start (ES) and Earliest Finish (EF)
    ES = np.zeros(n)
    EF = np.zeros(n)
    for u in topo_order:
        # ES[u] is already calculated by predecessors
        EF[u] = ES[u] + durations[u]
        for v in succ_list[u]:
            if EF[u] > ES[v]:
                ES[v] = EF[u]
                
    # Project Duration (Critical Path Length ignoring resources)
    C = np.max(EF) if n > 0 else 0

    # 2. Backward Pass: Latest Finish (LF) and Latest Start (LS)
    LF = np.full(n, C)
    LS = np.full(n, C)
    # Process in reverse topological order
    for u in reversed(topo_order):
        # LF[u] is min of LS of successors
        if not succ_list[u]:
            LF[u] = C
        else:
            min_succ_ls = C
            for v in succ_list[u]:
                # LS[v] is LF[v] - duration[v]
                if LS[v] < min_succ_ls:
                    min_succ_ls = LS[v]
            LF[u] = min_succ_ls
            
        LS[u] = LF[u] - durations[u]

    # 3. Total Float (TF)
    # TF = LS - ES
    TF = LS - ES
    # Ensure non-negative (numerical errors or cycles might cause issues, but DAG is safe)
    TF = np.maximum(TF, 0)

    # 4. Successor Critical Path (SCP)
    # SCP[u] = duration[u] + max(SCP[v] for v in succ_list[u])
    # If no successors, SCP[u] = duration[u]
    SCP = np.zeros(n)
    for u in reversed(topo_order):
        if not succ_list[u]:
            SCP[u] = durations[u]
        else:
            max_scp_succ = 0
            for v in succ_list[u]:
                if SCP[v] > max_scp_succ:
                    max_scp_succ = SCP[v]
            SCP[u] = durations[u] + max_scp_succ

    # 5. Calculate Priority
    # We want to prioritize activities with long SCP and low TF.
    # Score = SCP / (TF + epsilon)
    # Adding a small epsilon to avoid division by zero.
    epsilon = 1e-6
    scores = SCP / (TF + epsilon)
    
    # Optional: Incorporate Resource Demand
    # If an activity uses a lot of a scarce resource, it might be "heavy".
    # Let's check if demands and capacities are available.
    demands = instance.get("demands")
    capacities = instance.get("capacities")
    
    if demands is not None and capacities is not None:
        # Normalize demands by capacity to get "pressure"
        # demands is usually a list of lists or numpy array (n, R)
        # capacities is a list or numpy array (R,)
        try:
            D = np.asarray(demands, dtype=float)
            C_cap = np.asarray(capacities, dtype=float)
            # Avoid division by zero in capacities
            C_cap_safe = np.where(C_cap == 0, 1.0, C_cap)
            # Pressure per activity: sum of demand/capacity
            # D is (n, R), C_cap is (R,) -> result is (n,)
            pressure = np.sum(D / C_cap_safe, axis=1)
            
            # Normalize pressure to [0, 1] roughly to mix with SCP/TF
            # Or just add a weighted term.
            # High pressure means it consumes a lot of relative resource.
            # Scheduling high pressure early might be good to clear the bottleneck.
            # Let's add a small weight to pressure.
            # Score = SCP / (TF + eps) + alpha * pressure
            # We need to scale pressure to be comparable to SCP/TF.
            # SCP is in time units. Pressure is dimensionless (ratio).
            # Let's just use TF and SCP primarily, as they are more robust.
            # However, if we want to improve further:
            # Activities with high demand on tight resources should be prioritized if they are on the critical path.
            
            # Let's refine: If TF is low (critical), high pressure is very bad if delayed.
            # So, for low TF, we want high pressure to be high priority.
            # For high TF, pressure matters less.
            
            # New Score: (SCP / (TF + eps)) * (1 + pressure)
            # This boosts the priority of critical, high-pressure activities.
            scores = scores * (1.0 + pressure)
            
        except (ValueError, TypeError, IndexError):
            # Fallback if data format is unexpected
            pass

    # Deadline guard (though this is very fast)
    if time.time() - start_time > 110:
         return scores

    return scores

# EVOLVE-BLOCK-END
