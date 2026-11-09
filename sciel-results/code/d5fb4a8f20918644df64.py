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
    
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    edge_set = set()
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
        edge_set.add((a, b))
        edge_set.add((b, a))
    
    def time_up():
        return time.time() - t0 > DEADLINE - 3.0
    
    def fallback():
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    # Find all maximal cliques using Bron-Kerbosch with pivot
    def bron_kerbosch(R, P_set, X):
        if not P_set and not X:
            yield list(R)
            return
        # Pivot selection: choose vertex in P_set ∪ X with max degree in P_set
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
    
    # Generate all maximal cliques
    all_cliques = []
    if n > 0:
        for clique in bron_kerbosch(set(), set(range(n)), set()):
            all_cliques.append(clique)
    
    # Sort cliques by size descending
    all_cliques.sort(key=len, reverse=True)
    
    # Deduplicate (keep unique sets)
    seen = set()
    unique_cliques = []
    for c in all_cliques:
        key = frozenset(c)
        if key not in seen:
            seen.add(key)
            unique_cliques.append(c)
    
    best = None
    best_size = 0
    
    # Start with best 2-clique (any edge)
    if edges:
        u, v = edges[0]
        best = {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        best_size = 2
    
    # Core solver logic
    def try_branch_set(branches):
        """
        Try to find disjoint internal paths for all pairs in branches.
        Returns list of paths (in standard combination order) or None.
        """
        bs = set(branches)
        sorted_bs = sorted(branches)
        pairs = list(combinations(sorted_bs, 2))
        
        if not pairs:
            return []
        
        def pair_tightness(u, v):
            common = (adj[u] & adj[v]) - bs
            return len(common)
        
        # Try multiple orderings
        orderings = []
        orderings.append(list(pairs))
        orderings.append(sorted(pairs, key=lambda p: pair_tightness(p[0], p[1])))
        orderings.append(sorted(pairs, key=lambda p: -pair_tightness(p[0], p[1])))
        
        rng = random.Random(42)
        for _ in range(3):
            shuffled = list(pairs)
            rng.shuffle(shuffled)
            orderings.append(shuffled)
        
        solution_paths = None
        
        for ordering in orderings:
            if time_up():
                break
            
            sol = solve_paths(branches, bs, ordering)
            if sol is not None:
                solution_paths = sol
                break
        
        if solution_paths is None:
            return None
        
        # Reorder to standard combination order
        std_pairs = list(combinations(sorted_bs, 2))
        path_dict = {}
        for i, pair in enumerate(ordering):
            path_dict[pair] = solution_paths[i]
        
        result = []
        for p in std_pairs:
            if p in path_dict:
                result.append(path_dict[p])
            else:
                return None
        return result
    
    def solve_paths(branches, bs, ordered_pairs):
        """
        Backtracking solver for a specific pair ordering.
        Returns list of paths (one per pair in ordered_pairs order) or None.
        """
        num_pairs = len(ordered_pairs)
        chosen = [None] * num_pairs
        
        def find_shortest_path(s, t, blocked):
            """BFS to find shortest simple path from s to t avoiding blocked (except s,t)."""
            if s == t:
                return [s]
            
            parent = {s: None}
            q = deque([s])
            found = False
            
            while q and not found:
                curr = q.popleft()
                for nb in adj[curr]:
                    if nb in blocked and nb != s and nb != t:
                        continue
                    if nb in parent:
                        continue
                    parent[nb] = curr
                    if nb == t:
                        found = True
                        break
                    q.append(nb)
            
            if not found:
                return None
            
            path = []
            node = t
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path
        
        def find_paths(s, t, blocked, max_paths=5, max_len=8):
            """Find up to max_paths simple paths from s to t avoiding blocked (except s,t)."""
            paths = []
            
            shortest = find_shortest_path(s, t, blocked)
            if shortest is not None:
                paths.append(shortest)
            
            if len(paths) >= max_paths:
                return paths
            
            def dfs(curr, path, visited):
                if len(paths) >= max_paths:
                    return
                if len(path) > max_len + 2:
                    return
                if curr == t:
                    if path not in paths:
                        paths.append(list(path))
                    return
                for nb in adj[curr]:
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
        
        def solve(idx, used_internal):
            if idx == num_pairs:
                return True
            
            u, v = ordered_pairs[idx]
            blocked = used_internal | bs
            
            candidates = find_paths(u, v, blocked, max_paths=5, max_len=8)
            
            if not candidates:
                return False
            
            candidates.sort(key=len)
            
            for path in candidates:
                internal = set(path[1:-1])
                if internal & used_internal:
                    continue
                if internal & bs:
                    continue
                
                chosen[idx] = path
                for v_node in internal:
                    used_internal.add(v_node)
                
                if solve(idx + 1, used_internal):
                    return True
                
                for v_node in internal:
                    used_internal.discard(v_node)
                chosen[idx] = None
            
            return False
        
        if solve(0, set()):
            return chosen
        return None

    # 1. Try cliques in order of size (largest first)
    for clique in unique_cliques:
        if time_up():
            break
        
        k = len(clique)
        if k <= best_size:
            continue
        
        paths = try_branch_set(clique)
        if paths is not None:
            best = {'branches': [int(x) for x in sorted(clique)], 'paths': [[int(x) for x in p] for p in paths]}
            best_size = k
    
    # 2. Local Search with Swaps to escape local optima
    # We try to increase the size from best_size to best_size + 1, etc.
    
    # Candidate pool: top vertices by degree
    deg_sorted = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    # Initialize current best branches set
    current_branches = set(best['branches']) if best else set()
    current_size = best_size
    
    # If we didn't find anything better than 2, start with 2
    if current_size < 2:
         current_branches = set(edges[0])
         current_size = 2

    # Try to expand up to n
    for target_k in range(current_size + 1, min(n, 10) + 1):
        if time_up():
            break
            
        found = False
        # Limit iterations for local search
        max_iters = 100
        
        for _ in range(max_iters):
            if time_up():
                break
                
            # Strategy A: Direct extension
            # Try adding a vertex from the pool not in current_branches
            candidates_for_add = [v for v in deg_sorted if v not in current_branches]
            random.shuffle(candidates_for_add)
            
            extended = False
            for v in candidates_for_add:
                new_set = current_branches | {v}
                paths = try_branch_set(list(new_set))
                if paths is not None:
                    current_branches = new_set
                    current_size = target_k
                    best = {'branches': [int(x) for x in sorted(current_branches)], 'paths': [[int(x) for x in p] for p in paths]}
                    found = True
                    extended = True
                    break
            
            if extended:
                break # Move to next target_k
            
            # Strategy B: Swap to find a better base for extension
            # If direct extension failed, try swapping one vertex in current_branches with one outside
            # This might find a different valid set of size current_size that is "closer" to a valid set of size current_size + 1
            if current_size > 2: # Swapping is only useful if we have some size to work with
                # Pick a random vertex to remove
                remove_v = random.choice(list(current_branches))
                # Pick a random vertex to add
                candidates_for_swap = [v for v in deg_sorted if v not in current_branches]
                random.shuffle(candidates_for_swap)
                
                for add_v in candidates_for_swap:
                    if add_v == remove_v: continue
                    new_set = (current_branches - {remove_v}) | {add_v}
                    
                    # Check if this new set of size current_size is valid
                    paths = try_branch_set(list(new_set))
                    if paths is not None:
                        # We found a valid set of same size.
                        # Does it help us extend?
                        # We accept it as the new current_branches to try extending again in the next iteration of the while loop?
                        # Or try to extend it immediately?
                        # Let's try to extend it immediately.
                        candidates_for_add2 = [v for v in deg_sorted if v not in new_set]
                        random.shuffle(candidates_for_add2)
                        
                        for v2 in candidates_for_add2:
                            new_set_ext = new_set | {v2}
                            paths_ext = try_branch_set(list(new_set_ext))
                            if paths_ext is not None:
                                current_branches = new_set_ext
                                current_size = target_k
                                best = {'branches': [int(x) for x in sorted(current_branches)], 'paths': [[int(x) for x in p] for p in paths_ext]}
                                found = True
                                break
                        if found:
                            break
                        
                        # If we didn't extend, we can still update current_branches to this new valid set
                        # to explore its neighborhood in future iterations
                        current_branches = new_set
                        # And break the swap loop to try another swap? Or continue?
                        # Let's break and retry the main loop with the new base.
                        break
            
            if found:
                break
        
        if not found:
            # If we couldn't find target_k, maybe we can still find target_k + 1? Unlikely.
            # But we might have improved the base set.
            # If no progress after a few target_k's, break?
            # For now, just continue to next target_k, but it will likely fail fast.
            pass

    if best is None:
        return fallback()
    
    return best
