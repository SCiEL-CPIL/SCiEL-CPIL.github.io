import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

DEADLINE = time.time() + 85.0

def construct():
    t0 = time.time()
    n = P['n']
    edges = P['edges']
    
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    
    def time_up():
        return time.time() - t0 > DEADLINE - 3.0
    
    def fallback():
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    # Max-flow based verification using vertex splitting
    # For a set of branches B, we need k(k-1)/2 pairwise internally disjoint paths
    # Model: split each non-branch vertex v into v_in -> v_out with capacity 1
    # For each pair (u,v) in B, we need a path. We can model this as:
    # Create a super-source connected to each branch, and check if we can route
    # all pairs simultaneously. Actually, this is the disjoint paths problem which
    # is NP-hard in general, but for small k we can use backtracking.
    
    # Better approach: Use the fact that for k branches, we need C(k,2) paths.
    # We'll use backtracking with BFS for path finding, but with better heuristics.
    
    def find_paths_backtrack(branches):
        """Find internally disjoint paths for all pairs in branches."""
        bs = set(branches)
        sorted_bs = sorted(branches)
        pairs = list(combinations(sorted_bs, 2))
        
        if not pairs:
            return []
        
        # Pre-compute: for each pair, find candidate paths
        # Use a recursive solver with good ordering
        
        def find_shortest_path(s, t, blocked):
            """BFS shortest path from s to t, internal vertices avoid blocked."""
            if s == t:
                return [s]
            parent = {s: None}
            q = deque([s])
            while q:
                curr = q.popleft()
                if curr == t:
                    break
                for nb in sorted(adj[curr]):
                    if nb in blocked and nb != s and nb != t:
                        continue
                    if nb in parent:
                        continue
                    parent[nb] = curr
                    if nb == t:
                        break
                    q.append(nb)
            if t not in parent:
                return None
            path = []
            node = t
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path
        
        def find_all_short_paths(s, t, blocked, max_paths=8, max_len=10):
            """Find up to max_paths simple paths, shortest first."""
            paths = []
            shortest = find_shortest_path(s, t, blocked)
            if shortest is not None:
                paths.append(shortest)
            
            if len(paths) >= max_paths:
                return paths
            
            # DFS for more paths
            def dfs(curr, path, visited):
                if len(paths) >= max_paths:
                    return
                if len(path) > max_len + 2:
                    return
                if curr == t:
                    if path not in paths:
                        paths.append(list(path))
                    return
                for nb in sorted(adj[curr]):
                    if nb in visited:
                        continue
                    if nb in blocked and nb != s and nb != t:
                        continue
                    visited.add(nb)
                    path.append(nb)
                    dfs(nb, path, visited)
                    path.pop()
                    visited.remove(nb)
                    if len(paths) >= max_paths:
                        return
            
            dfs(s, [s], {s})
            return paths
        
        # Try multiple orderings
        def pair_score(u, v):
            common = (adj[u] & adj[v]) - bs
            return len(common)
        
        orderings = [
            list(pairs),
            sorted(pairs, key=lambda p: pair_score(p[0], p[1])),
            sorted(pairs, key=lambda p: -pair_score(p[0], p[1])),
        ]
        
        rng = random.Random(123)
        for _ in range(5):
            sh = list(pairs)
            rng.shuffle(sh)
            orderings.append(sh)
        
        for ordering in orderings:
            if time_up():
                break
            
            chosen = [None] * len(ordering)
            
            def solve(idx, used_internal):
                if idx == len(ordering):
                    return True
                u, v = ordering[idx]
                blocked = used_internal | bs
                candidates = find_all_short_paths(u, v, blocked, max_paths=6, max_len=9)
                if not candidates:
                    return False
                candidates.sort(key=len)
                for path in candidates:
                    internal = set(path[1:-1])
                    if internal & used_internal or internal & bs:
                        continue
                    chosen[idx] = path
                    for vn in internal:
                        used_internal.add(vn)
                    if solve(idx + 1, used_internal):
                        return True
                    for vn in internal:
                        used_internal.discard(vn)
                    chosen[idx] = None
                return False
            
            if solve(0, set()):
                # Reorder to standard pair order
                std_pairs = list(combinations(sorted_bs, 2))
                path_map = {}
                for i, pair in enumerate(ordering):
                    path_map[pair] = chosen[i]
                result = []
                for p in std_pairs:
                    if p in path_map:
                        result.append(path_map[p])
                    else:
                        return None
                return result
        
        return None
    
    # Strategy: Greedily build up a valid branch set
    # Start with high-degree vertices, try to add more
    
    deg_sorted = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    best = None
    best_size = 2
    
    # Initial: any edge
    u, v = edges[0]
    best = {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    # Try to build larger sets greedily
    def try_add_vertex(current_branches, candidate):
        """Try to add candidate to current_branches."""
        new_branches = list(current_branches) + [candidate]
        paths = find_paths_backtrack(new_branches)
        if paths is not None:
            return sorted(new_branches), paths
        return None
    
    # Greedy construction
    current = [deg_sorted[0]]
    for v in deg_sorted[1:]:
        if time_up():
            break
        if v in current:
            continue
        result = try_add_vertex(current, v)
        if result is not None:
            current, paths = result
            if len(current) > best_size:
                best_size = len(current)
                sorted_cur = sorted(current)
                best = {'branches': [int(x) for x in sorted_cur], 'paths': [[int(x) for x in p] for p in paths]}
    
    # Local search: try to swap vertices
    if not time_up() and best_size >= 3:
        current_branches = best['branches']
        improved = True
        while improved and not time_up():
            improved = False
            current_set = set(current_branches)
            for v in deg_sorted:
                if v in current_set:
                    continue
                for old_v in list(current_branches):
                    new_branches = [x for x in current_branches if x != old_v] + [v]
                    paths = find_paths_backtrack(new_branches)
                    if paths is not None:
                        current_branches = sorted(new_branches)
                        best_size = len(current_branches)
                        best = {'branches': [int(x) for x in current_branches], 'paths': [[int(x) for x in p] for p in paths]}
                        improved = True
                        break
                if improved:
                    break
    
    # Try random subsets of increasing size
    for target_k in range(best_size + 1, min(n, 12) + 1):
        if time_up():
            break
        pool = deg_sorted[:min(target_k * 4, n)]
        rng = random.Random(42 + target_k * 7)
        max_tries = 40 if target_k <= 7 else 20
        found = False
        for _ in range(max_tries):
            if time_up():
                break
            subset = rng.sample(pool, target_k)
            paths = find_paths_backtrack(subset)
            if paths is not None:
                best = {'branches': [int(x) for x in sorted(subset)], 'paths': [[int(x) for x in p] for p in paths]}
                best_size = target_k
                found = True
                break
        if not found and target_k > best_size + 2:
            break
    
    if best is None:
        return fallback()
    
    return best

