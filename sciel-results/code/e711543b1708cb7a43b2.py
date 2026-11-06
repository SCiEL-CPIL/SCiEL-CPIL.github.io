import numpy as np

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor list
    # predecessors[i] contains indices of tasks that must come before i
    # So for each pred p in predecessors[i], p is a predecessor of i, meaning i is a successor of p
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            
    # Calculate Position Weights (PW)
    # PW[i] = time[i] + sum(PW[j] for j in successors[i])
    # We use memoized DFS to calculate this efficiently.
    weights = [-1.0] * n
    visited = [False] * n
    
    def dfs(u):
        if visited[u]:
            return weights[u]
        visited[u] = True
        total = times[u]
        for v in successors[u]:
            total += dfs(v)
        weights[u] = total
        return total
        
    for i in range(n):
        dfs(i)
        
    return np.array(weights, dtype=float)
