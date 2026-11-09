import numpy as np
from collections import defaultdict, deque
import time as _time

def priority(instance):
    start_time = _time.time()
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor graph
    successors = defaultdict(list)
    in_degree = np.zeros(n, dtype=int)
    
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Topological sort (Kahn's algorithm)
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
            
    topo_order = []
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
                
    # Compute PW (Position Weight): longest path from task to end
    pw = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        total_succ_pw = 0.0
        for v in successors[u]:
            total_succ_pw += pw[v]
        pw[u] = times[u] + total_succ_pw
        
    # Compute number of successors (downstream task count)
    # Use reverse topological order to accumulate counts
    succ_count = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        total_succ_count = 0.0
        for v in successors[u]:
            total_succ_count += succ_count[v]
        succ_count[u] = 1.0 + total_succ_count
        
    # Compute "bottleneck density": how much time is concentrated downstream
    # For each task, compute the ratio of PW to (1 + succ_count)
    # This captures tasks that have high PW but few successors (bottlenecks)
    # vs tasks with many successors (branching points)
    bottleneck_density = pw / (1.0 + succ_count)
    
    # Combine: primary priority is PW, secondary is bottleneck density
    # Normalize both to [0, 1] range for combination
    pw_min, pw_max = pw.min(), pw.max()
    bd_min, bd_max = bottleneck_density.min(), bottleneck_density.max()
    
    if pw_max > pw_min:
        pw_norm = (pw - pw_min) / (pw_max - pw_min)
    else:
        pw_norm = np.zeros(n, dtype=float)
        
    if bd_max > bd_min:
        bd_norm = (bottleneck_density - bd_min) / (bd_max - bd_min)
    else:
        bd_norm = np.zeros(n, dtype=float)
        
    # Weighted combination: emphasize PW but add bottleneck density signal
    # Higher weight on PW since it's the dominant factor
    combined_priority = 0.7 * pw_norm + 0.3 * bd_norm
    
    # Safety check: if computation took too long, return simpler priority
    elapsed = _time.time() - start_time
    if elapsed > 100.0:
        return pw
        
    return combined_priority
