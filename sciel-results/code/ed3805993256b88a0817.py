"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Candidate edges: only even-odd pairs not already in base
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u - v) % 2 != 0:  # one even, one odd
                if (u, v) not in base_edges and (v, u) not in base_edges:
                    candidates.append((u, v))
    
    def is_triangle_free(added):
        # Since we only add even-odd edges, graph stays bipartite, hence triangle-free
        # Just verify all added edges connect even to odd
        for u, v in added:
            if (u - v) % 2 == 0:
                return False
        return True
    
    def compute_diameter(added):
        # BFS from each node
        edge_set = set(base_edges)
        for u, v in added:
            edge_set.add((min(u,v), max(u,v)))
        
        # Build full adjacency
        full_adj = [set() for _ in range(n)]
        for u, v in edge_set:
            full_adj[u].add(v)
            full_adj[v].add(u)
        
        max_d = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = deque([src])
            while queue:
                cur = queue.popleft()
                for nb in full_adj[cur]:
                    if dist[nb] == -1:
                        dist[nb] = dist[cur] + 1
                        queue.append(nb)
            for d in dist:
                if d == -1:
                    return float('inf')  # disconnected
                if d > max_d:
                    max_d = d
            if max_d > 4:
                return max_d
        return max_d
    
    # Greedy: start with empty, try to add edges that reduce diameter
    # First, find a good initial solution using hub approach
    # Then try to remove edges
    
    # Initial: hub at 12 connected to all odd except 11,13
    initial_added = [(12, v) for v in [1, 3, 5, 7, 9, 15, 17, 19, 21, 23]]
    
    # Verify triangle-free (should be, since 12 is even and all are odd, bipartite)
    # Check diameter
    if compute_diameter(initial_added) <= 4 and is_triangle_free(initial_added):
        best = initial_added[:]
    else:
        best = []
    
    # Try to reduce: remove edges one by one
    improved = True
    while improved:
        improved = False
        for i in range(len(best)):
            trial = best[:i] + best[i+1:]
            if is_triangle_free(trial) and compute_diameter(trial) <= 4:
                best = trial
                improved = True
                break
    
    # Try adding different edges to potentially do better
    # Try all single-hub solutions
    for hub in range(0, n, 2):  # even hubs
        trial = [(hub, v) for v in range(1, n, 2) if v != hub-1 and v != hub+1]
        if len(trial) < len(best) and is_triangle_free(trial) and compute_diameter(trial) <= 4:
            best = trial
    
    # Try two-hub solutions
    for h1 in range(0, n, 2):
        for h2 in range(h1+2, n, 2):
            trial = []
            for v in range(1, n, 2):
                # Assign odd vertex to nearest hub
                d1 = abs(v - h1)
                d2 = abs(v - h2)
                if d1 <= d2:
                    if v not in (h1-1, h1+1):
                        trial.append((h1, v))
                else:
                    if v not in (h2-1, h2+1):
                        trial.append((h2, v))
            # Remove duplicates
            seen = set()
            unique_trial = []
            for e in trial:
                if e not in seen:
                    seen.add(e)
                    unique_trial.append(e)
            if len(unique_trial) < len(best) and is_triangle_free(unique_trial) and compute_diameter(unique_trial) <= 4:
                best = unique_trial
    
    # Try to reduce best further
    improved = True
    while improved:
        improved = False
        for i in range(len(best)):
            trial = best[:i] + best[i+1:]
            if is_triangle_free(trial) and compute_diameter(trial) <= 4:
                best = trial
                improved = True
                break
    
    # Return as list of [u,v]
    result = [[int(u), int(v)] for u, v in best]
    return result
# EVOLVE-BLOCK-END
