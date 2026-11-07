import time
import random
from itertools import combinations
from collections import deque
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t_start = time.time()
    DEADLINE = 85.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    degrees = [len(adj[v]) for v in range(n)]
    sorted_verts = sorted(range(n), key=lambda x: degrees[x], reverse=True)

    def time_up():
        return time.time() - t_start > DEADLINE

    def find_shortest_path(u, v, banned):
        if u == v:
            return None
        if v in adj[u]:
            return [u, v]
        parent = {u: None}
        q = deque([u])
        while q:
            cur = q.popleft()
            for nxt in adj[cur]:
                if nxt == v:
                    path = [v]
                    node = cur
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                if nxt in parent or nxt in banned:
                    continue
                parent[nxt] = cur
                q.append(nxt)
        return None

    def find_all_paths_bfs(u, v, banned, max_len=8):
        """Find all simple paths from u to v with length <= max_len, respecting banned."""
        if u == v:
            return []
        if v in adj[u] and v not in banned:
            return [[u, v]]
        if u in banned and u != v:
            return []
        
        all_paths = []
        # DFS with depth limit
        def dfs(cur, path, depth):
            if len(all_paths) >= 50:
                return
            if depth > max_len:
                return
            for nxt in adj[cur]:
                if nxt == v:
                    all_paths.append(path + [v])
                    return
                if nxt in banned or nxt in path:
                    continue
                dfs(nxt, path + [nxt], depth + 1)
        
        dfs(u, [u], 1)
        return all_paths

    def try_find_subdivision(branches, rng, max_attempts=30):
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        # Sort pairs by "difficulty": fewer available neighbors first
        def pair_difficulty(p):
            a, b = p
            avail_a = adj[a] - bset - {b}
            avail_b = adj[b] - bset - {a}
            return len(avail_a & avail_b)
        pairs.sort(key=pair_difficulty)
        
        for attempt in range(max_attempts):
            if time_up():
                break
            
            # Determine order
            if attempt == 0:
                order = list(pairs)
            elif attempt % 3 == 1:
                order = list(pairs)
                order.reverse()
            else:
                order = list(pairs)
                rng.shuffle(order)
            
            used_interior = set()
            paths = {}
            success = True
            
            for a, b in order:
                banned = used_interior | bset
                banned.discard(a)
                banned.discard(b)
                
                # Try shortest path first
                path = find_shortest_path(a, b, banned)
                
                if path is None:
                    # Try slightly longer paths
                    candidates = find_all_paths_bfs(a, b, banned, max_len=5)
                    found = False
                    for cand in candidates:
                        interior = set(cand[1:-1])
                        if not (interior & bset) and not (interior & used_interior):
                            path = cand
                            found = True
                            break
                    if not found:
                        success = False
                        break
                
                interior = set(path[1:-1])
                if interior & bset or interior & used_interior:
                    success = False
                    break
                
                used_interior |= interior
                paths[frozenset((a, b))] = path
            
            if success:
                return paths
        
        return None

    def try_find_subdivision_backtrack(branches, rng, max_nodes=30000):
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        # Sort by constraint: pairs with fewer joint neighbors first
        pairs.sort(key=lambda p: len(adj[p[0]] & adj[p[1]]))
        
        used_interior = set()
        paths = {}
        nodes_visited = 0
        
        def bt(idx):
            nonlocal nodes_visited
            if time_up():
                return False
            if nodes_visited > max_nodes:
                return False
            if idx == len(pairs):
                return True
            
            a, b = pairs[idx]
            banned = used_interior | bset
            banned.discard(a)
            banned.discard(b)
            
            # Find candidate paths
            candidates = []
            
            if b in adj[a]:
                candidates.append([a, b])
            
            # BFS for shortest
            path = find_shortest_path(a, b, banned)
            if path:
                if path not in candidates:
                    candidates.append(path)
            
            # If we have few candidates, try to find more via DFS limited depth
            if len(candidates) < 5:
                all_paths = find_all_paths_bfs(a, b, banned, max_len=6)
                for p in all_paths:
                    if p not in candidates:
                        candidates.append(p)
            
            # Sort candidates by length (shorter is usually better for packing)
            candidates.sort(key=len)
            
            for path in candidates:
                interior = set(path[1:-1])
                if interior & bset or interior & used_interior:
                    continue
                
                nodes_visited += 1
                used_interior_new = used_interior | interior
                paths[frozenset((a, b))] = path
                
                if bt(idx + 1):
                    return True
                
                del paths[frozenset((a, b))]
            
            return False
        
        if bt(0):
            return paths
        return None

    def try_find_subdivision_ilp(branches, rng, time_budget=5.0):
        """Solve the path-disjointness problem as an ILP for a fixed set of branches."""
        t0 = time.time()
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        npairs = len(pairs)
        if npairs == 0:
            return None
        
        # Index pairs
        pair_idx = {frozenset(p): i for i, p in enumerate(pairs)}
        
        # For each pair, find all simple paths up to some length limit
        # We'll use BFS/DFS to enumerate candidate paths
        max_path_len = 6  # vertices in path (including endpoints)
        
        candidate_paths = []  # list of (pair_idx, path)
        for pi, (a, b) in enumerate(pairs):
            # Find paths from a to b
            paths_found = []
            
            # Direct edge
            if b in adj[a]:
                paths_found.append([a, b])
            
            # BFS for short paths
            if len(paths_found) < 20:
                # DFS with depth limit
                def dfs_paths(start, end, banned, max_depth):
                    results = []
                    def _dfs(cur, path, depth):
                        if len(results) >= 20:
                            return
                        if depth > max_depth:
                            return
                        for nxt in adj[cur]:
                            if nxt == end:
                                results.append(path + [end])
                                return
                            if nxt in banned or nxt in path:
                                continue
                            _dfs(nxt, path + [nxt], depth + 1)
                    _dfs(start, [start], 1)
                    return results
                
                banned = bset - {a, b}
                extra = dfs_paths(a, b, banned, max_path_len - 1)
                for p in extra:
                    if p not in paths_found:
                        paths_found.append(p)
            
            for p in paths_found:
                candidate_paths.append((pi, p))
        
        if not candidate_paths:
            return None
        
        # ILP formulation:
        # x[pi][path_idx] = 1 if we choose this path for pair pi
        # Constraints:
        # 1. For each pair pi: sum of x[pi][*] = 1 (exactly one path)
        # 2. For each non-branch vertex v: sum over all (pi, path) containing v of x[pi][path] <= 1
        
        # Build the ILP
        n_vars = len(candidate_paths)
        var_to_path = candidate_paths  # (pair_idx, path)
        
        # Objective: maximize sum of x (all coefficients 1)
        c = np.ones(n_vars)
        
        # Constraints
        A_rows = []
        lb = []
        ub = []
        
        # Constraint 1: For each pair, exactly one path
        for pi in range(npairs):
            row = np.zeros(n_vars)
            for vi, (pidx, path) in enumerate(candidate_paths):
                if pidx == pi:
                    row[vi] = 1.0
            A_rows.append(row)
            lb.append(1.0)
            ub.append(1.0)
        
        # Constraint 2: For each non-branch vertex, at most one path uses it
        non_branch_verts = [v for v in range(n) if v not in bset]
        for v in non_branch_verts:
            row = np.zeros(n_vars)
            count = 0
            for vi, (pidx, path) in enumerate(candidate_paths):
                if len(path) > 2:  # has interior
                    if v in path[1:-1]:
                        row[vi] = 1.0
                        count += 1
            if count > 0:
                A_rows.append(row)
                lb.append(0.0)
                ub.append(1.0)
        
        if not A_rows:
            # No constraints beyond pair selection - just pick one path per pair
            paths_result = {}
            for pi, (a, b) in enumerate(pairs):
                # Find any valid path
                path = find_shortest_path(a, b, bset - {a, b})
                if path is None:
                    return None
                paths_result[frozenset((a, b))] = path
            return paths_result
        
        A = np.array(A_rows)
        lb_arr = np.array(lb)
        ub_arr = np.array(ub)
        
        # Binary variables
        integrality = np.ones(n_vars)
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        
        constraints = LinearConstraint(A, lb_arr, ub_arr)
        
        try:
            result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if time.time() - t0 > time_budget:
                return None
            
            if not result.success:
                return None
            
            # Extract solution
            x = result.x
            if x is None:
                return None
            
            paths_result = {}
            for vi, (pidx, path) in enumerate(candidate_paths):
                if x[vi] > 0.5:
                    a, b = pairs[pidx]
                    paths_result[frozenset((a, b))] = path
            
            if len(paths_result) != npairs:
                return None
            
            return paths_result
        
        except Exception:
            return None

    def validate(branches, paths):
        if not branches or len(branches) < 2:
            return 0
        bset = set(branches)
        if len(bset) != len(branches):
            return 0
        expected_pairs = set(frozenset((a, b)) for a, b in combinations(branches, 2))
        if len(paths) != len(expected_pairs):
            return 0
        used_interior = set()
        for fkey, path in paths.items():
            if len(path) < 2:
                return 0
            a, b = path[0], path[-1]
            if a not in bset or b not in bset:
                return 0
            if a == b:
                return 0
            if fkey not in expected_pairs:
                return 0
            interior = set(path[1:-1])
            if interior & bset:
                return 0
            if interior & used_interior:
                return 0
            for i in range(len(path) - 1):
                if path[i + 1] not in adj[path[i]]:
                    return 0
            used_interior |= interior
        return len(branches)

    best_count = 0
    best_result = None
    rng = random.Random(42)

    def find_max_clique():
        """Find a maximum clique using branch and bound with greedy coloring."""
        # Greedy coloring to get upper bound
        def greedy_color(vertices, adj):
            colors = {}
            color_map = {}
            for v in vertices:
                used_colors = set()
                for u in adj[v]:
                    if u in color_map:
                        used_colors.add(color_map[u])
                c = 0
                while c in used_colors:
                    c += 1
                color_map[v] = c
            return max(color_map.values()) + 1 if color_map else 0

        best_clique = []
        current = []

        def branch(vertices, clique):
            nonlocal best_clique
            if time_up():
                return
            if not vertices:
                if len(clique) > len(best_clique):
                    best_clique = list(clique)
                return

            # Upper bound via greedy coloring
            upper_bound = len(clique) + greedy_color(vertices, adj)
            if upper_bound <= len(best_clique):
                return

            # Pick vertex with highest degree in remaining
            v = max(vertices, key=lambda x: len(adj[x] & set(vertices)))
            new_vertices = [u for u in vertices if u != v and u in adj[v]]
            clique.append(v)
            branch(new_vertices, clique)
            clique.pop()

            vertices.remove(v)
            branch(vertices, clique)

        branch(list(range(n)), [])
        return best_clique

    # First, find a good clique as initial candidate
    clique = find_max_clique()
    if len(clique) >= 2:
        clique_set = set(clique)
        paths = {}
        for a, b in combinations(sorted(clique), 2):
            if b in adj[a]:
                paths[frozenset((a, b))] = [a, b]
        if len(paths) == len(list(combinations(clique, 2))):
            cnt = validate(clique, paths)
            if cnt > best_count:
                best_count = cnt
                best_result = (clique, paths)

    # Phase 1: Top-down k search
    for k in range(8, 1, -1):
        if time_up():
            break
        
        candidates = set()
        
        # Start with clique if large enough
        if len(clique) >= k:
            candidates.add(tuple(sorted(clique[:k])))
        elif len(clique) > 0:
            # Extend clique with high-degree vertices
            ext = sorted(clique) + [v for v in sorted_verts if v not in clique_set][:k - len(clique)]
            candidates.add(tuple(sorted(ext)))
        
        # Top k by degree
        candidates.add(tuple(sorted_verts[:k]))
        
        # Top k-1 + random
        for _ in range(8):
            rest = [v for v in range(n) if v not in sorted_verts[:k-1]]
            if rest:
                cand = sorted_verts[:k-1] + [rng.choice(rest)]
                candidates.add(tuple(cand))
        
        # Random samples
        for _ in range(40):
            candidates.add(tuple(rng.sample(range(n), k)))
        
        # Local search: start with top k, try swapping
        base = sorted_verts[:k]
        for _ in range(15):
            if time_up():
                break
            cand = list(base)
            idx = rng.randint(0, k-1)
            rest = [v for v in range(n) if v not in cand]
            if rest:
                cand[idx] = rng.choice(rest)
                candidates.add(tuple(cand))
        
        for cand_tuple in candidates:
            if time_up():
                break
            cand = list(cand_tuple)
            
            # Try greedy first
            paths = try_find_subdivision(cand, rng, max_attempts=40)
            if paths is None:
                # Try backtracking
                paths = try_find_subdivision_backtrack(cand, rng, max_nodes=30000)
            if paths is None and len(cand) <= 8:
                # Try ILP for small branch sets
                paths = try_find_subdivision_ilp(cand, rng, time_budget=3.0)
            
            if paths is not None:
                cnt = validate(cand, paths)
                if cnt > best_count:
                    best_count = cnt
                    best_result = (cand, paths)
                    break
        
        if best_result is not None and best_count >= k:
            break

    # Phase 2: Greedy incremental growth
    if not time_up():
        if best_result is not None:
            current = list(best_result[0])
        else:
            current = sorted_verts[:2]
        
        current_paths = try_find_subdivision(current, rng, max_attempts=50)
        if current_paths is None:
            current_paths = try_find_subdivision_backtrack(current, rng, max_nodes=30000)
        
        for _ in range(30):
            if time_up():
                break
            if current_paths is None:
                break
            
            added = False
            current_set = set(current)
            candidates_to_add = [v for v in range(n) if v not in current_set]
            # Prioritize vertices that form edges with many current branches
            candidates_to_add.sort(key=lambda v: sum(1 for u in current if u in adj[v]), reverse=True)
            
            for v in candidates_to_add:
                if time_up():
                    break
                new_branches = current + [v]
                new_paths = try_find_subdivision(new_branches, rng, max_attempts=25)
                if new_paths is None:
                    new_paths = try_find_subdivision_backtrack(new_branches, rng, max_nodes=20000)
                if new_paths is None and len(new_branches) <= 8:
                    new_paths = try_find_subdivision_ilp(new_branches, rng, time_budget=3.0)
                
                if new_paths is not None:
                    cnt = validate(new_branches, new_paths)
                    if cnt > best_count:
                        best_count = cnt
                        best_result = (new_branches, new_paths)
                        current = new_branches
                        current_paths = new_paths
                        added = True
                        break
            if not added:
                break

    # Fallback
    if best_result is None:
        u, v = edges[0]
        branches = [u, v]
        paths = {frozenset((u, v)): [u, v]}
        best_result = (branches, paths)

    branches, paths = best_result

    branch_list = [int(b) for b in sorted(branches)]
    path_list = []
    for a, b in combinations(sorted(branches), 2):
        fkey = frozenset((a, b))
        p = paths.get(fkey)
        if p is None:
            p = [a, b]
        path_list.append([int(x) for x in p])

    return {'branches': branch_list, 'paths': path_list}
