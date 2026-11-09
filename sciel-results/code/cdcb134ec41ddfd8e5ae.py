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
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
        
    # Convert to list of lists for faster iteration if needed, but sets are good for intersection
    adj_list = [list(s) for s in adj]
    
    def time_up():
        return time.time() - t0 > DEADLINE - 2.0

    def fallback():
        if not edges:
            return {'branches': [0, 1], 'paths': [[0, 1]]}
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    def find_paths(s, t, blocked, max_paths=10, max_len=10):
        """
        Find up to max_paths simple paths from s to t avoiding blocked (except s,t).
        Returns list of paths.
        """
        if s == t:
            return [[s]]
        paths = []
        
        # BFS for shortest path first
        parent = {s: None}
        q = deque([s])
        found_shortest = False
        shortest_path = None
        
        while q and not found_shortest:
            curr = q.popleft()
            for nb in adj_list[curr]:
                if nb in blocked and nb != s and nb != t:
                    continue
                if nb in parent:
                    continue
                parent[nb] = curr
                if nb == t:
                    found_shortest = True
                    break
                q.append(nb)
        
        if found_shortest:
            path = []
            node = t
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            paths.append(path)
            
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
            for nb in adj_list[curr]:
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

    def solve_paths(branches):
        """
        Try to find disjoint internal paths for all pairs in branches.
        Returns list of paths or None.
        """
        bs = set(branches)
        sorted_bs = sorted(branches)
        pairs = list(combinations(sorted_bs, 2))
        
        if not pairs:
            return []
            
        # Heuristic: Sort pairs by "tightness"
        # Tightness: number of available internal vertices (common neighbors outside branches)
        def pair_tightness(u, v):
            common = (adj[u] & adj[v]) - bs
            return len(common)
        
        # Sort pairs: tightest first (fewest options)
        pairs_sorted = sorted(pairs, key=lambda p: pair_tightness(p[0], p[1]))
        
        chosen = [None] * len(pairs_sorted)
        
        def solve(idx, used_internal):
            if time_up():
                return False
            if idx == len(pairs_sorted):
                return True
            
            u, v = pairs_sorted[idx]
            blocked = used_internal | bs
            
            candidates = find_paths(u, v, blocked, max_paths=15, max_len=12)
            
            if not candidates:
                return False
            
            # Sort candidates by length (shorter is generally better for leaving space)
            candidates.sort(key=len)
            
            for path in candidates:
                internal = set(path[1:-1])
                if internal & used_internal:
                    continue
                if internal & bs:
                    continue
                
                chosen[idx] = path
                for node in internal:
                    used_internal.add(node)
                
                if solve(idx + 1, used_internal):
                    return True
                
                for node in internal:
                    used_internal.discard(node)
                chosen[idx] = None
            
            return False

        if solve(0, set()):
            # Map back to original pair order
            std_pairs = list(combinations(sorted_bs, 2))
            path_dict = {}
            for i, pair in enumerate(pairs_sorted):
                path_dict[pair] = chosen[i]
            
            result = []
            for p in std_pairs:
                if p in path_dict:
                    result.append(path_dict[p])
                else:
                    return None
            return result
        return None

    # Generate candidate branch sets
    # Strategy 1: Cliques
    def bron_kerbosch(R, P_set, X):
        if not P_set and not X:
            yield list(R)
            return
        pivot_candidates = P_set | X
        if pivot_candidates:
            pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set
            
        for v in list(candidates):
            yield from bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)

    all_cliques = []
    if n > 0:
        for clique in bron_kerbosch(set(), set(range(n)), set()):
            all_cliques.append(clique)
            
    all_cliques.sort(key=len, reverse=True)
    
    # Strategy 2: Greedy high-degree sets
    deg_sorted = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    candidate_sets = []
    seen_sets = set()
    
    # Add cliques
    for c in all_cliques:
        key = frozenset(c)
        if key not in seen_sets:
            seen_sets.add(key)
            candidate_sets.append(c)
            
    # Add greedy sets
    # Try starting from each high-degree node
    for start in deg_sorted[:10]:
        current_set = {start}
        for v in deg_sorted:
            if v in current_set:
                continue
            # Check if v is well-connected to current_set
            # Sum of edges to current_set
            connections = sum(1 for u in current_set if v in adj[u])
            if connections >= len(current_set) // 2: # Heuristic threshold
                new_set = current_set | {v}
                key = frozenset(new_set)
                if key not in seen_sets:
                    seen_sets.add(key)
                    candidate_sets.append(list(new_set))
                current_set = new_set
                if len(current_set) >= 8:
                    break
                    
    # Add random sets from top degree nodes
    rng = random.Random(42)
    top_nodes = deg_sorted[:15]
    for _ in range(50):
        if time_up():
            break
        k = rng.randint(3, 7)
        subset = rng.sample(top_nodes, min(k, len(top_nodes)))
        key = frozenset(subset)
        if key not in seen_sets:
            seen_sets.add(key)
            candidate_sets.append(subset)

    # Sort candidates by size descending
    candidate_sets.sort(key=len, reverse=True)
    
    best = None
    best_size = 0
    
    # Initial fallback
    if edges:
        u, v = edges[0]
        best = {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        best_size = 2

    # Try candidates
    for c in candidate_sets:
        if time_up():
            break
        if len(c) <= best_size:
            continue
            
        paths = solve_paths(c)
        if paths is not None:
            best = {'branches': [int(x) for x in sorted(c)], 'paths': [[int(x) for x in p] for p in paths]}
            best_size = len(c)
            
            # If we found a large set, we might stop early if it's very large?
            # But we want max. Keep looking for larger ones.
            # Since we sorted by size desc, if we find one of size K, 
            # we only need to check sets of size > K.
            # However, there might be other sets of size K that are also valid, 
            # but we only need one.
            # So we can break if we found a set of size K and no larger sets are left?
            # No, because there might be a set of size K+1 later in the list?
            # No, we sorted by size descending. So if we are at size K, 
            # all subsequent sets have size <= K.
            # So if we found a valid set of size K, we are done?
            # Wait, what if there is another set of size K that is valid, but we already have one?
            # We just need *a* valid set of max size.
            # So yes, if we find a valid set of size K, and all remaining candidates are <= K, 
            # we can stop?
            # Only if we are sure there are no larger sets.
            # Since we sorted by size desc, the first valid set we find for a given size K 
            # is the max size we will find in the remaining list.
            # So we can break.
            break

    if best is None:
        return fallback()
        
    return best
