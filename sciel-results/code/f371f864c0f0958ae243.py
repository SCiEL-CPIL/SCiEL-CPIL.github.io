import numpy as np
import time

def priority(instance):
    start_time = time.time()
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    # Build successor list
    succ = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)
            in_degree[i] += 1
            
    # Calculate RPV (Rank Position Value)
    # rpv[i] = time[i] + sum(rpv[s] for s in succ[i])
    # Process nodes in reverse topological order (sinks first)
    out_degree = [len(succ[i]) for i in range(n)]
    od = out_degree[:]
    q = [i for i in range(n) if od[i] == 0]
    rpv = times.copy()
    
    while q:
        u = q.pop()
        for p in preds[u]:
            rpv[p] += rpv[u]
            od[p] -= 1
            if od[p] == 0:
                q.append(p)
                
    # Time guard
    if time.time() - start_time > 100:
        return rpv

    # Heuristic: RPV + weight * number of successors
    # Prioritizing tasks with more successors helps open up the graph
    # and can lead to fewer stations by reducing idle time due to blocked tasks.
    succ_counts = np.array([len(s) for s in succ], dtype=float)
    
    # Use a weight that is small enough not to override RPV differences
    # but large enough to break ties effectively.
    # If max time is ~10 and max succ is ~5, 0.5 * 5 = 2.5 is significant.
    # Let's try 0.5.
    scores = rpv + 0.5 * succ_counts
    
    return scores
