import time
import sys
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = P['n']
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find maximum clique using Bron-Kerbosch with pivot
    best_clique = []
    
    def bronk(R, P_set, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        # Choose pivot u from P_set ∪ X
        candidates = list(P_set | X)
        if not candidates:
            return
        # Pick pivot with max degree in P_set
        pivot = max(candidates, key=lambda u: len(adj[u] & P_set))
        # For each v in P_set \ N(pivot)
        for v in list(P_set - adj[pivot]):
            bronk(R | {v}, P_set & adj[v], X & adj[v])
            P_set = P_set - {v}
            X = X | {v}
            if time.time() > deadline:
                return
    
    bronk(set(), set(range(n)), set())
    
    # Now try to extend: can we add more branches?
    # For a set of branches B, we need disjoint interior paths for all pairs.
    # Start with the max clique and try adding vertices one by one.
    
    def can_form_clique_subdivision(branches):
        """Check if branches can form a clique subdivision."""
        b_set = set(branches)
        # For each pair, find a path. Direct edge if possible.
        # We need to assign disjoint interior vertices.
        # Greedy: for each pair, if direct edge exists, use it.
        # Otherwise, find a path through unused non-branch vertices.
        
        # First, identify pairs that need interior vertices
        pairs = list(combinations(sorted(branches), 2))
        paths = {}
        used_interior = set()
        
        # Process pairs: direct edges first, then longer paths
        # Sort pairs: those with direct edge first
        def has_direct_edge(u, v):
            return v in adj[u]
        
        # Try to assign paths
        for u, v in pairs:
            if has_direct_edge(u, v):
                paths[(u, v)] = [u, v]
            else:
                # Find a path from u to v avoiding b_set (except endpoints)
                # and avoiding used_interior
                forbidden = b_set - {u, v}
                forbidden |= used_interior
                # BFS
                from collections import deque
                visited = {u}
                parent = {u: None}
                queue = deque([u])
                found = False
                while queue:
                    if time.time() > deadline:
                        return None
                    cur = queue.popleft()
                    if cur == v:
                        # Reconstruct path
                        path = []
                        node = v
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        path.reverse()
                        paths[(u, v)] = path
                        # Mark interior vertices
                        for mid in path[1:-1]:
                            used_interior.add(mid)
                        found = True
                        break
                    for nb in adj[cur]:
                        if nb not in visited and nb not in forbidden:
                            visited.add(nb)
                            parent[nb] = cur
                            queue.append(nb)
                if not found:
                    return None
        return paths
    
    # Start with max clique
    current_branches = list(best_clique)
    current_paths = can_form_clique_subdivision(current_branches)
    
    # Try to add more vertices
    remaining = [v for v in range(n) if v not in set(current_branches)]
    improved = True
    while improved:
        improved = False
        for v in remaining[:]:
            if time.time() > deadline:
                break
            test_branches = current_branches + [v]
            test_paths = can_form_clique_subdivision(test_branches)
            if test_paths is not None:
                current_branches = test_branches
                current_paths = test_paths
                remaining.remove(v)
                improved = True
    
    # Build result
    if current_paths is None:
        # Fallback: just use the clique with direct edges
        current_branches = list(best_clique)
        current_paths = {}
        for u, v in combinations(sorted(current_branches), 2):
            current_paths[(u, v)] = [u, v]
    
    branches = [int(b) for b in sorted(current_branches)]
    paths_list = []
    for u, v in combinations(sorted(current_branches), 2):
        p = current_paths.get((u, v), [u, v])
        paths_list.append([int(x) for x in p])
    
    return {'branches': branches, 'paths': paths_list}
