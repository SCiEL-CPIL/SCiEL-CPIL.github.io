import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set(tuple(sorted(e)) for e in P['edges'])
    
    # Precompute: only even-odd edges are valid (to maintain bipartiteness => triangle-free)
    # Build adjacency for quick triangle-free check
    def is_triangle_free(adj, edge_set):
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    # Check common neighbors
                    for w in adj[u]:
                        if w != v and w in adj[v]:
                            return False
        return True
    
    def get_diameter(adj):
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        q.append(nb)
                        if dist[nb] > max_d:
                            max_d = dist[nb]
                            if max_d > 4:
                                return max_d  # early exit
        return max_d
    
    def build_adj(added):
        adj = [[] for _ in range(n)]
        for u, v in P['edges']:
            adj[u].append(v)
            adj[v].append(u)
        for u, v in added:
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    def valid_config(added):
        if len(added) != len(set(tuple(sorted(e)) for e in added)):
            return False
        es = set(tuple(sorted(e)) for e in added)
        for e in es:
            if e in base_edges:
                return False
        adj = build_adj(added)
        if not is_triangle_free(adj, es):
            return False
        if get_diameter(adj) > 4:
            return False
        return True
    
    # Deadline guard
    deadline = time.time() + 80.0
    
    # Strategy: try hub-based configurations with different hubs,
    # then try to prune edges from the best found.
    
    best = None
    best_count = 100
    
    # Try each possible hub (must connect to opposite parity)
    for hub in range(n):
        if time.time() > deadline - 5.0:
            break
        # Hub connects to vertices of opposite parity
        # hub parity: even if hub%2==0, connect to odd; else connect to even
        if hub % 2 == 0:
            candidates = [v for v in range(n) if v % 2 == 1 and v != hub]
        else:
            candidates = [v for v in range(n) if v % 2 == 0 and v != hub]
        
        # Start with all candidate edges, try to prune
        # Greedy: start with full star, remove edges that don't break diameter
        added = [[hub, v] for v in candidates]
        # Remove edges that are already in base (shouldn't happen since path edges are consecutive)
        added = [[u, v] for u, v in added if tuple(sorted((u, v))) not in base_edges]
        
        # Try to prune: for each edge, see if removing it still gives valid config
        improved = True
        while improved:
            improved = False
            for i in range(len(added)):
                if time.time() > deadline - 5.0:
                    break
                trial = added[:i] + added[i+1:]
                if valid_config(trial):
                    added = trial
                    improved = True
                    break  # restart scan after modification
        
        if len(added) < best_count:
            best = added[:]
            best_count = len(added)
    
    if best is not None:
        return [[int(u), int(v)] for u, v in best]
    
    # Fallback: known 10-edge solution
    return [[11, 0], [11, 2], [11, 4], [11, 6], [11, 8], [11, 14], [11, 16], [11, 18], [11, 20], [11, 22]]
# EVOLVE-BLOCK-END
