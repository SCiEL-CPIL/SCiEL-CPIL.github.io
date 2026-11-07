"""Solver for large clique subdivisions in a pinned graph."""

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    import time
    from collections import deque
    from itertools import combinations
    import random

    t_start = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_sorted = [sorted(adj[i]) for i in range(n)]
    deg = [len(adj[i]) for i in range(n)]

    def time_up():
        return time.time() - t_start > DEADLINE

    def find_shortest_paths_bfs(u, v, banned, limit=20):
        """
        Find up to 'limit' shortest simple paths from u to v avoiding 'banned' vertices in interior.
        Returns list of paths sorted by length.
        """
        if u == v:
            return [[u]]
        
        # BFS to find shortest distance
        dist = {u: 0}
        parent = {u: None}
        q = deque([u])
        found_v = False
        while q:
            cur = q.popleft()
            if cur == v:
                found_v = True
                break
            for nb in adj_sorted[cur]:
                if nb in banned and nb != v:
                    continue
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    parent[nb] = cur
                    q.append(nb)
        
        if not found_v or v not in dist:
            return []
        
        shortest_len = dist[v]
        
        # Now DFS to find all paths of length shortest_len
        paths = []
        
        def dfs(cur, path, visited):
            if len(paths) >= limit:
                return
            if cur == v:
                if len(path) - 1 == shortest_len:
                    paths.append(list(path))
                return
            
            # Pruning: if current path length + min possible remaining > shortest_len, prune
            if len(path) - 1 >= shortest_len:
                return
            
            for nb in adj_sorted[cur]:
                if nb in visited:
                    continue
                if nb in banned and nb != v:
                    continue
                visited.add(nb)
                path.append(nb)
                dfs(nb, path, visited)
                path.pop()
                visited.remove(nb)
                
        dfs(u, [u], {u})
        return paths

    def find_disjoint_paths(branches):
        """
        Given a list of branches, find disjoint paths for all pairs.
        Returns list of paths if successful, else None.
        """
        if len(branches) < 2:
            return None
            
        pairs = list(combinations(branches, 2))
        
        # Heuristic: Sort pairs by shortest path length in the original graph (ignoring bans for now)
        def pair_key(p):
            u, v = p
            # Quick BFS for length
            if u == v: return 0
            dist = {u: 0}
            q = deque([u])
            while q:
                cur = q.popleft()
                if cur == v:
                    return dist[cur]
                for nb in adj_sorted[cur]:
                    if nb not in dist:
                        dist[nb] = dist[cur] + 1
                        q.append(nb)
            return 1000
            
        pairs.sort(key=pair_key)
        
        paths = {}
        used_interiors = set()
        
        def backtrack(idx):
            if time_up():
                return False
            if idx == len(pairs):
                return True
                
            u, v = pairs[idx]
            # Find candidate paths avoiding used_interiors and other branches
            banned = set(branches) - {u, v}
            banned.update(used_interiors)
            
            # We need paths from u to v where interiors are not in banned
            candidates = find_shortest_paths_bfs(u, v, banned, limit=20)
            
            for path in candidates:
                # Update used_interiors
                new_interiors = set(path[1:-1])
                
                # Recursive step
                paths[(u, v)] = path
                old_used = set(used_interiors)
                used_interiors.update(new_interiors)
                
                if backtrack(idx + 1):
                    return True
                
                # Backtrack
                used_interiors.clear()
                used_interiors.update(old_used)
                del paths[(u, v)]
                
            return False
            
        if backtrack(0):
            # Return paths in a consistent order (list of lists)
            res_paths = []
            for p in pairs:
                res_paths.append(paths[p])
            return res_paths
        return None

    def verify_solution(branches, paths):
        """Verify a candidate solution. Returns True if valid."""
        if len(branches) < 2:
            return False
        
        branch_set = set(branches)
        if len(branch_set) != len(branches):
            return False
            
        pairs_found = set()
        used_interiors = set()
        
        for path in paths:
            if len(path) < 2:
                return False
            u, v = path[0], path[-1]
            if u not in branch_set or v not in branch_set:
                return False
            if u == v:
                return False
            
            pair = frozenset((u, v))
            if pair in pairs_found:
                return False
            pairs_found.add(pair)
            
            for node in path[1:-1]:
                if node in branch_set:
                    return False
                if node in used_interiors:
                    return False
                used_interiors.add(node)
                
            for i in range(len(path) - 1):
                if path[i+1] not in adj[path[i]]:
                    return False
                    
        return len(pairs_found) == len(branches) * (len(branches) - 1) // 2

    def find_cliques(max_size=8, limit=200):
        """Find maximal cliques or large cliques to use as seeds."""
        cliques = []
        candidates = sorted(range(n), key=lambda i: (-deg[i], i))
        for start in candidates:
            if time_up():
                break
            clique = [start]
            for v in candidates:
                if v == start:
                    continue
                if all(v in adj[w] for w in clique):
                    clique.append(v)
                if len(clique) >= max_size:
                    break
            if len(clique) >= 2:
                cliques.append(clique)
            if len(cliques) >= limit:
                break
        return cliques

    best = None
    best_size = 0
    rng_global = random.Random(42)

    # Phase 1: Greedy growth from clique seeds
    cliques = find_cliques(max_size=8, limit=100)
    cliques.sort(key=lambda c: (-len(c), -sum(deg[v] for v in c)))
    growth_pool = sorted(range(n), key=lambda i: (-deg[i], i))

    for seed in cliques:
        if time_up():
            break
        if len(seed) <= best_size:
            continue
        
        branches = list(seed)
        result_paths = find_disjoint_paths(branches)
        
        if result_paths is None:
            # Try smaller subsets of the seed
            found_sub = False
            for sz in range(len(seed) - 1, 1, -1):
                if sz <= best_size:
                    break
                for sub in combinations(seed, sz):
                    if time_up():
                        break
                    result_paths = find_disjoint_paths(list(sub))
                    if result_paths is not None:
                        branches = list(sub)
                        found_sub = True
                        break
                if found_sub:
                    break
            if result_paths is None:
                continue
            
        best = {'branches': branches, 'paths': result_paths}
        best_size = len(branches)

        # Greedy growth
        changed = True
        while changed and not time_up():
            changed = False
            for cand in growth_pool:
                if cand in branches:
                    continue
                if len(branches) + 1 <= best_size:
                    break
                new_branches = branches + [cand]
                result_paths = find_disjoint_paths(new_branches)
                if result_paths is not None:
                    branches = new_branches
                    best = {'branches': branches, 'paths': result_paths}
                    best_size = len(branches)
                    changed = True
                    break

    # Fallback
    if best is None:
        for a, b, c in combinations(range(n), 3):
            if b in adj[a] and c in adj[a] and c in adj[b]:
                best = {
                    'branches': [a, b, c],
                    'paths': [[a, b], [a, c], [b, c]]
                }
                break
        if best is None:
            u, v = edges[0]
            best = {'branches': [u, v], 'paths': [[u, v]]}

    branches = [int(x) for x in best['branches']]
    paths = [[int(x) for x in p] for p in best['paths']]
    
    # Final validation check before returning (just in case)
    if not verify_solution(branches, paths):
        # If validation fails, try to fix or fallback
        for a, b, c in combinations(range(n), 3):
            if b in adj[a] and c in adj[a] and c in adj[b]:
                branches = [a, b, c]
                paths = [[a, b], [a, c], [b, c]]
                break
        else:
            u, v = edges[0]
            branches = [u, v]
            paths = [[u, v]]

    return {'branches': branches, 'paths': paths}
