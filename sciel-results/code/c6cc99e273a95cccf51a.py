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
    
    start_time = time.time()
    deadline = start_time + 85.0

    def time_left():
        return deadline - time.time()

    def find_max_clique():
        """Bron-Kerbosch with pivot to find largest maximal clique."""
        max_clique = []
        
        def bronk(R, P_set, X):
            if time.time() > deadline:
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
        return max_clique

    def bfs_shortest(u, v, forbidden):
        """Find shortest path from u to v avoiding forbidden. Returns path or None."""
        if u == v:
            return [u]
        if v in forbidden:
            return None
        
        dist = {u: 0}
        parent = {u: None}
        queue = deque([u])
        while queue:
            curr = queue.popleft()
            if curr == v:
                break
            for nb in adj_list[curr]:
                if nb not in dist and nb not in forbidden:
                    dist[nb] = dist[curr] + 1
                    parent[nb] = curr
                    queue.append(nb)
        
        if v not in dist:
            return None
        
        path = []
        curr = v
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path.reverse()
        return path

    def find_k_shortest_paths(u, v, forbidden, k=5, max_len=4):
        """Find up to k shortest paths from u to v avoiding forbidden."""
        paths = []
        seen = set()
        
        # BFS for shortest
        p = bfs_shortest(u, v, forbidden)
        if p and len(p) - 1 <= max_len:
            paths.append(p)
            seen.add(tuple(p))
        
        if len(paths) >= k:
            return paths[:k]
        
        # DFS for more paths
        def dfs(path, visited):
            if len(paths) >= k:
                return
            if time.time() > deadline:
                return
            curr = path[-1]
            if curr == v:
                if len(path) - 1 <= max_len:
                    t = tuple(path)
                    if t not in seen:
                        paths.append(list(path))
                        seen.add(t)
                return
            
            if len(path) - 1 >= max_len:
                return
            
            for nb in adj_list[curr]:
                if nb == v:
                    if len(path) <= max_len:
                        p2 = path + [nb]
                        t = tuple(p2)
                        if t not in seen:
                            paths.append(p2)
                            seen.add(t)
                    continue
                if nb not in visited and nb not in forbidden:
                    visited.add(nb)
                    dfs(path + [nb], visited)
                    visited.remove(nb)
        
        dfs([u], {u})
        paths.sort(key=lambda x: (len(x), x))
        return paths[:k]

    def find_disjoint_paths(branches, max_path_len=4, max_candidates=8):
        """
        Check if 'branches' can be connected by internally disjoint paths.
        Returns list of paths if successful, else None.
        """
        if len(branches) < 2:
            return []
            
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        pairs.sort(key=lambda p: (p[0], p[1]))
        
        # Pre-compute candidate paths for each pair
        candidates = {}
        for u, v in pairs:
            forbidden = branch_set - {u, v}
            paths = find_k_shortest_paths(u, v, forbidden, k=max_candidates, max_len=max_path_len)
            candidates[(u, v)] = paths
        
        for u, v in pairs:
            if not candidates[(u, v)]:
                return None
        
        # Order pairs by scarcity of options
        pair_order = sorted(pairs, key=lambda p: len(candidates[p]))
        
        used_internal = set()
        assignment = {}
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(pair_order):
                return True
            
            u, v = pair_order[idx]
            for path in candidates[(u, v)]:
                internal = set(path[1:-1])
                if internal & used_internal:
                    continue
                if internal & branch_set:
                    continue
                
                assignment[(u, v)] = path
                used_internal.update(internal)
                
                if backtrack(idx + 1):
                    return True
                
                used_internal.difference_update(internal)
                del assignment[(u, v)]
            
            return False
        
        if backtrack(0):
            return [assignment[p] for p in pairs]
        
        return None

    # Phase 1: Find Max Clique as initial seed
    max_clique = find_max_clique()
    best_size = len(max_clique)
    best_branches = list(max_clique)
    best_paths = None
    
    if best_size >= 2:
        pairs = list(itertools.combinations(best_branches, 2))
        best_paths = [[u, v] for u, v in pairs]
    
    current_branches = list(best_branches)
    
    # Phase 2: Greedy expansion with swaps
    for iteration in range(100):
        if time.time() > deadline - 10:
            break
        
        added = False
        # Try adding a new vertex
        candidates = [v for v in range(n) if v not in set(current_branches)]
        candidates.sort(key=lambda v: -len(adj[v]))
        
        for v in candidates:
            if time.time() > deadline - 10:
                break
                
            new_branches = current_branches + [v]
            mpl = 4 if len(new_branches) <= 8 else 3
            mca = 8 if len(new_branches) <= 8 else 5
            
            paths = find_disjoint_paths(new_branches, max_path_len=mpl, max_candidates=mca)
            
            if paths is not None:
                current_branches = new_branches
                best_size = len(current_branches)
                best_branches = list(current_branches)
                best_paths = paths
                added = True
                break
        
        if added:
            continue
        
        # Try swapping: replace one branch with another
        if len(current_branches) >= 2:
            swap_found = False
            for i in range(len(current_branches)):
                if time.time() > deadline - 10:
                    break
                for v in candidates:
                    if time.time() > deadline - 10:
                        break
                    new_branches = current_branches[:i] + [v] + current_branches[i+1:]
                    new_branches.sort()
                    mpl = 4 if len(new_branches) <= 8 else 3
                    mca = 8 if len(new_branches) <= 8 else 5
                    
                    paths = find_disjoint_paths(new_branches, max_path_len=mpl, max_candidates=mca)
                    
                    if paths is not None:
                        # Check if this swap actually improves or maintains size
                        # We accept if it allows further expansion later
                        current_branches = new_branches
                        best_size = len(current_branches)
                        best_branches = list(current_branches)
                        best_paths = paths
                        swap_found = True
                        break
                if swap_found:
                    break
        
        if not added and not swap_found:
            # Try removing a branch to see if we can add two
            if len(current_branches) > 2:
                removed = False
                for i in range(len(current_branches)):
                    if time.time() > deadline - 10:
                        break
                    reduced = current_branches[:i] + current_branches[i+1:]
                    # Try to add back a different vertex
                    for v in candidates:
                        if time.time() > deadline - 10:
                            break
                        new_branches = reduced + [v]
                        new_branches.sort()
                        if len(new_branches) <= len(current_branches):
                            continue
                        mpl = 4 if len(new_branches) <= 8 else 3
                        mca = 8 if len(new_branches) <= 8 else 5
                        
                        paths = find_disjoint_paths(new_branches, max_path_len=mpl, max_candidates=mca)
                        
                        if paths is not None:
                            current_branches = new_branches
                            best_size = len(current_branches)
                            best_branches = list(current_branches)
                            best_paths = paths
                            removed = True
                            break
                    if removed:
                        break
                if removed:
                    continue
        
        if not added and not swap_found:
            break

    # Final validation and formatting
    if best_size >= 2 and best_paths is not None:
        branches = [int(b) for b in best_branches]
        formatted_paths = []
        for p in best_paths:
            formatted_paths.append([int(x) for x in p])
        
        return {'branches': branches, 'paths': formatted_paths}
    
    # Fallback: Any edge
    if P['edges']:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    else:
        return {'branches': [0, 1], 'paths': [[0, 1]]}

# EVOLVE-BLOCK-END
