import time
import itertools
import random
from collections import deque

def construct():
    P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}
    
    n = P['n']
    adj = [set() for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    # Precompute all-pairs shortest paths for distance heuristics
    dist = [[n+1 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        queue = deque([i])
        while queue:
            u = queue.popleft()
            for v in adj_list[u]:
                if dist[i][v] > dist[i][u] + 1:
                    dist[i][v] = dist[i][u] + 1
                    queue.append(v)

    start_time = time.time()
    deadline = start_time + 55.0

    def time_left():
        return deadline - time.time()

    def find_paths_for_branches(branches):
        """
        Attempts to find internally disjoint paths for all pairs in 'branches'.
        Returns list of paths if successful, else None.
        """
        k = len(branches)
        if k < 2:
            return []
        
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        
        # Sort pairs by distance to handle "easy" ones first
        pairs.sort(key=lambda p: dist[p[0]][p[1]])
        
        used_internal = [0] * n  # bitmask or boolean list for used internal vertices
        # Actually, a set is faster for small n if we don't do bit ops, but n=24 fits in int.
        # Let's use a set for clarity, or a bitmask for speed.
        # Bitmask is better.
        used_mask = 0
        
        def get_mask(v):
            return 1 << v

        # We need to assign a path to each pair.
        # Backtracking search.
        # To speed up, we can pre-compute candidate paths for each pair.
        # But candidate paths depend on 'used_mask'.
        
        # Instead, let's try a greedy assignment with backtracking.
        # Order of pairs is fixed.
        
        assignment = [None] * len(pairs)
        
        def backtrack(idx, used_mask):
            if time.time() > deadline - 0.5:
                return False
            
            if idx == len(pairs):
                return True
            
            u, v = pairs[idx]
            
            # Find a path from u to v avoiding used_mask and branch_set (except u,v)
            # BFS to find shortest path
            # We can limit the search depth to avoid infinite loops in dense graphs, 
            # but BFS finds shortest, so it's fine.
            
            # Forbidden set: used_mask bits + all branches except u,v
            # For speed, we check on the fly in BFS.
            
            parent = [-1] * n
            visited = [False] * n
            visited[u] = True
            queue = deque([u])
            
            found_path = None
            while queue:
                curr = queue.popleft()
                if curr == v:
                    # Reconstruct path
                    path = []
                    c = v
                    while c != -1:
                        path.append(c)
                        c = parent[c]
                    path.reverse()
                    found_path = path
                    break
                for nb in adj_list[curr]:
                    if visited[nb]:
                        continue
                    # Check if nb is a branch (other than u,v)
                    if nb in branch_set and nb != u and nb != v:
                        continue
                    # Check if nb is used internal
                    if nb != u and nb != v:
                        if used_mask & get_mask(nb):
                            continue
                    visited[nb] = True
                    parent[nb] = curr
                    queue.append(nb)
            
            if not found_path:
                return False
            
            # Check if internal vertices are disjoint
            internal_ok = True
            new_mask = used_mask
            for node in found_path[1:-1]:
                if new_mask & get_mask(node):
                    internal_ok = False
                    break
                new_mask |= get_mask(node)
            
            if not internal_ok:
                return False
            
            assignment[idx] = found_path
            
            if backtrack(idx + 1, new_mask):
                return True
            
            assignment[idx] = None
            return False

        if backtrack(0, 0):
            return [assignment[i] for i in range(len(pairs))]
        
        return None

    def is_valid(branches):
        """Check if branches form a valid clique subdivision."""
        if len(branches) < 2:
            return False
        paths = find_paths_for_branches(branches)
        return paths is not None

    # Phase 1: Find Max Clique (lower bound)
    # Simple Bron-Kerbosch
    max_clique = []
    def bronk(R, P_set, X):
        if time.time() > deadline - 10:
            return
        if not P_set and not X:
            if len(R) > len(max_clique):
                max_clique[:] = list(R)
            return
        if P_set or X:
            pivot = max(P_set | X, key=lambda v: len(adj[v] & P_set))
        else:
            return
        candidates = [v for v in P_set if v not in adj[pivot]]
        for v in candidates:
            bronk(R | {v}, P_set & adj[v], X & adj[v])
            P_set = P_set - {v}
            X = X | {v}

    bronk(frozenset(), frozenset(range(n)), frozenset())
    
    best_branches = list(max_clique)
    best_paths = None
    if best_branches:
        best_paths = find_paths_for_branches(best_branches)
    
    best_size = len(best_branches) if best_paths else 0

    # Phase 2: GRASP / Local Search
    # We try to expand the set.
    # Candidates for addition: vertices not in best_branches.
    # We sort them by a heuristic: how well they connect to existing branches.
    
    for iteration in range(100):
        if time.time() > deadline - 5:
            break
            
        # Try to add a vertex
        added = False
        candidates = [v for v in range(n) if v not in set(best_branches)]
        
        # Heuristic score: sum of distances to existing branches (lower is better)
        # Or number of neighbors in existing branches (higher is better)
        def score_add(v):
            if not best_branches:
                return 0
            # Prefer vertices adjacent to many branches
            count = sum(1 for b in best_branches if v in adj[b])
            # Penalize if far away
            min_dist = min(dist[v][b] for b in best_branches)
            return (count, -min_dist)
        
        candidates.sort(key=score_add, reverse=True)
        
        for v in candidates:
            if time.time() > deadline - 5:
                break
            new_branches = best_branches + [v]
            paths = find_paths_for_branches(new_branches)
            if paths is not None:
                best_branches = new_branches
                best_paths = paths
                best_size = len(best_branches)
                added = True
                break
        
        if not added:
            # Try swapping
            swapped = False
            # Try removing one and adding another
            # Or just try adding a different one if we have time
            # Let's try a random restart of local search
            if iteration % 10 == 0 and best_size > 2:
                # Randomly drop a vertex and try to re-expand
                drop_idx = random.randint(0, len(best_branches) - 1)
                current = best_branches[:drop_idx] + best_branches[drop_idx+1:]
                # Try to add vertices to 'current'
                candidates = [v for v in range(n) if v not in set(current)]
                candidates.sort(key=lambda v: sum(1 for b in current if v in adj[b]), reverse=True)
                
                for v in candidates:
                    if time.time() > deadline - 5:
                        break
                    trial = current + [v]
                    paths = find_paths_for_branches(trial)
                    if paths is not None:
                        if len(trial) > best_size:
                            best_branches = trial
                            best_paths = paths
                            best_size = len(best_branches)
                            swapped = True
                            break
                        # Even if not better, we might use it as a new base? 
                        # No, we only want to improve.
            
            if not swapped and iteration > 20:
                # Stagnation, break or try harder?
                # Let's try one more random restart
                if iteration > 50:
                    break

    # Final validation
    if best_size >= 2 and best_paths is not None:
        branches = [int(b) for b in best_branches]
        formatted_paths = []
        for p in best_paths:
            formatted_paths.append([int(x) for x in p])
        
        return {'branches': branches, 'paths': formatted_paths}
    
    # Fallback
    if P['edges']:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    else:
        return {'branches': [0, 1], 'paths': [[0, 1]]}
