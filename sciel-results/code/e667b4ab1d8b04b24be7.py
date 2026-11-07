import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    cycles = []
    start_time = time.time()
    deadline = start_time + 50.0 # Leave 10s margin
    
    for _ in range(3):
        if time.time() > deadline:
            break
            
        # Find a Hamiltonian cycle in the current graph
        cycle = find_cycle(adj, n)
        if not cycle:
            break
        
        cycles.append(cycle)
        
        # Remove edges of the cycle from the graph
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            adj[u].remove(v)
            adj[v].remove(u)
            
    return cycles

def find_cycle(adj, n):
    # Backtracking search
    # Start at 0 to break symmetry
    start = 0
    path = [start]
    visited = {start}
    
    def backtrack():
        if len(path) == n:
            # Check if we can close the cycle
            if path[0] in adj[path[-1]]:
                return True
            return False
        
        curr = path[-1]
        # Get neighbors, sorted by degree (Warnsdorff's rule)
        neighbors = list(adj[curr])
        # Filter to only unvisited vertices, except if it's the last step where we can go to start
        # Actually, we only allow going to start if len(path) == n-1
        candidates = []
        for v in neighbors:
            if v in visited:
                if v == start and len(path) == n - 1:
                    candidates.append(v)
            else:
                candidates.append(v)
                
        # Sort by degree in the remaining graph to prioritize constrained vertices
        # We need to consider the degree of v in the graph *excluding* the edges used so far?
        # The `adj` structure is static for this call, but it represents the available edges.
        # The "remaining graph" for pruning is tricky with simple backtracking.
        # Simple Warnsdorff: sort by len(adj[v])
        candidates.sort(key=lambda x: len(adj[x]))
        
        for v in candidates:
            path.append(v)
            visited.add(v)
            if backtrack():
                return True
            path.pop()
            visited.remove(v)
            
        return False

    if backtrack():
        return path
    return None
