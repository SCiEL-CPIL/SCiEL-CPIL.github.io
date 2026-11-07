"""Solver for large clique subdivisions in a pinned graph."""

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    import time
    from collections import deque
    from itertools import combinations
    import random

    t_start = time.time()
    DEADLINE = 82.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_sorted = [sorted(adj[i]) for i in range(n)]

    def time_up():
        return time.time() - t_start > DEADLINE

    def validate(branches, paths):
        bset = set(branches)
        if len(branches) < 2:
            return False
        if len(branches) != len(bset):
            return False
        expected_paths = len(branches) * (len(branches) - 1) // 2
        if len(paths) != expected_paths:
            return False
        used_interiors = set()
        for p in paths:
            if not isinstance(p, (list, tuple)):
                return False
            if len(p) < 2:
                return False
            u, v = p[0], p[-1]
            if u not in bset or v not in bset:
                return False
            if u == v:
                return False
            seen = set()
            for node in p:
                if node in seen:
                    return False
                seen.add(node)
            for node in p[1:-1]:
                if node in bset:
                    return False
                if node in used_interiors:
                    return False
            used_interiors.update(p[1:-1])
            for i in range(len(p) - 1):
                if p[i+1] not in adj[p[i]]:
                    return False
        path_pairs = set()
        for p in paths:
            path_pairs.add(frozenset((p[0], p[-1])))
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                if frozenset((branches[i], branches[j])) not in path_pairs:
                    return False
        return True

    def _bfs_shortest(src, dst, forbidden, adj_sorted):
        """BFS shortest path avoiding forbidden in interior."""
        if src == dst:
            return [src]
        if dst in adj[src]:
            return [src, dst]
        parent = {src: None}
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            if cur == dst:
                path = []
                node = dst
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in adj_sorted[cur]:
                if nb not in parent:
                    if nb != dst and nb in forbidden:
                        continue
                    parent[nb] = cur
                    queue.append(nb)
        return None

    def _dfs_random(src, dst, forbidden, rng, adj_sorted, max_depth=15):
        """Randomized DFS path finding."""
        if src == dst:
            return [src]
        if dst in adj[src]:
            return [src, dst]

        def dfs(cur, path, visited):
            if len(path) > max_depth + 2:
                return None
            if cur == dst:
                return list(path)
            nbrs = list(adj_sorted[cur])
            rng.shuffle(nbrs)
            for nb in nbrs:
                if nb in visited:
                    continue
                if nb != dst and nb in forbidden:
                    continue
                visited.add(nb)
                path.append(nb)
                result = dfs(nb, path, visited)
                if result is not None:
                    return result
                path.pop()
                visited.discard(nb)
            return None

        visited = {src}
        return dfs(src, [src], visited)

    def _count_short_paths(src, dst, forbidden, adj_sorted, max_len=4):
        """Count number of short paths (length <= max_len) from src to dst."""
        # BFS up to max_len
        count = 0
        # Use iterative deepening or BFS with path tracking
        # For simplicity, BFS to find all paths up to max_len
        if dst in adj[src]:
            return 1
        # BFS layer by layer
        from collections import defaultdict
        paths_at_dist = {1: [[src]]}
        for d in range(1, max_len):
            if not paths_at_dist.get(d):
                break
            next_paths = []
            for p in paths_at_dist[d]:
                cur = p[-1]
                for nb in adj_sorted[cur]:
                    if nb in p:
                        continue
                    if nb != dst and nb in forbidden:
                        continue
                    new_p = p + [nb]
                    if nb == dst:
                        count += 1
                    else:
                        next_paths.append(new_p)
            paths_at_dist[d+1] = next_paths
        return count

    def find_paths_constrained_first(branches, rng, max_iters=200):
        """Find paths ordering by constraint: pairs with fewer options first."""
        k = len(branches)
        if k < 2:
            return None

        for attempt in range(max_iters):
            if time_up():
                return None

            paths_dict = {}
            used_interiors = set()
            success = True

            pair_list = []
            for i in range(k):
                for j in range(i + 1, k):
                    pair_list.append((branches[i], branches[j]))

            # Compute constraint score: fewer available paths = more constrained
            def constraint_score(pair):
                u, v = pair
                if v in adj[u]:
                    return 0  # Direct edge, always available
                # Estimate: count common neighbors (proxy for path options)
                common = len(adj[u] & adj[v])
                # Fewer common neighbors = more constrained
                return common

            # Sort by constraint: most constrained (fewest options) first
            pair_list.sort(key=lambda p: constraint_score(p))

            # Shuffle within same constraint level for diversity
            groups = {}
            for p in pair_list:
                key = constraint_score(p)
                if key not in groups:
                    groups[key] = []
                groups[key].append(p)
            ordered = []
            for key in sorted(groups.keys()):
                group = groups[key]
                if key > 0:  # Don't shuffle direct edges
                    rng.shuffle(group)
                ordered.extend(group)
            pair_list = ordered

            for (u, v) in pair_list:
                if time_up():
                    return None
                forbidden = set(used_interiors)
                forbidden.update(branches)
                forbidden.discard(u)
                forbidden.discard(v)

                if v in adj[u]:
                    path = [u, v]
                else:
                    # Try multiple strategies
                    path = _bfs_shortest(u, v, forbidden, adj_sorted)
                    if path is None:
                        path = _dfs_random(u, v, forbidden, rng, adj_sorted)

                if path is None:
                    success = False
                    break

                interior = set(path[1:-1])
                if interior & set(branches):
                    success = False
                    break
                if interior & used_interiors:
                    success = False
                    break

                used_interiors |= interior
                paths_dict[(u, v)] = path
                paths_dict[(v, u)] = path

            if success:
                path_list = []
                for (u, v) in pair_list:
                    path_list.append(paths_dict[(u, v)])
                return path_list

        return None

    def find_paths_with_backtrack(branches, rng, max_iters=100):
        """Find paths with limited backtracking when a path assignment fails."""
        k = len(branches)
        if k < 2:
            return None

        for attempt in range(max_iters):
            if time_up():
                return None

            pair_list = []
            for i in range(k):
                for j in range(i + 1, k):
                    pair_list.append((branches[i], branches[j]))

            # Order: direct edges first, then by constraint
            def pair_key(pair):
                u, v = pair
                if v in adj[u]:
                    return (0, 0)
                common = len(adj[u] & adj[v])
                return (1, common)

            pair_list.sort(key=pair_key)
            edge_pairs = [(u,v) for u,v in pair_list if v in adj[u]]
            non_edge_pairs = [(u,v) for u,v in pair_list if v not in adj[u]]
            rng.shuffle(non_edge_pairs)
            pair_list = edge_pairs + non_edge_pairs

            # Try with backtracking: if we fail at pair i, try alternative paths
            # for earlier pairs
            paths_dict = {}
            used_interiors = set()
            success = True

            for idx, (u, v) in enumerate(pair_list):
                if time_up():
                    return None
                forbidden = set(used_interiors)
                forbidden.update(branches)
                forbidden.discard(u)
                forbidden.discard(v)

                if v in adj[u]:
                    path = [u, v]
                else:
                    # Try BFS first, then DFS
                    path = _bfs_shortest(u, v, forbidden, adj_sorted)
                    if path is None:
                        path = _dfs_random(u, v, forbidden, rng, adj_sorted)

                if path is None:
                    # Backtrack: try to find alternative for an earlier non-edge pair
                    backtracked = False
                    for prev_idx in range(idx - 1, -1, -1):
                        pu, pv = pair_list[prev_idx]
                        if pv in adj[pu]:
                            continue  # Direct edge, no alternative
                        # Remove this path and try to find alternative
                        old_path = paths_dict.get((pu, pv))
                        if old_path is None:
                            continue
                        old_interior = set(old_path[1:-1])
                        used_interiors -= old_interior
                        del paths_dict[(pu, pv)]
                        if (pv, pu) in paths_dict:
                            del paths_dict[(pv, pu)]

                        # Find alternative path for (pu, pv)
                        new_forbidden = set(used_interiors)
                        new_forbidden.update(branches)
                        new_forbidden.discard(pu)
                        new_forbidden.discard(pv)
                        # Also exclude the old interior to force different path
                        new_forbidden.update(old_interior)

                        new_path = _bfs_shortest(pu, pv, new_forbidden, adj_sorted)
                        if new_path is None:
                            new_path = _dfs_random(pu, pv, new_forbidden, rng, adj_sorted)

                        if new_path is not None:
                            new_interior = set(new_path[1:-1])
                            if not (new_interior & set(branches)) and not (new_interior & used_interiors):
                                used_interiors |= new_interior
                                paths_dict[(pu, pv)] = new_path
                                paths_dict[(pv, pu)] = new_path
                                backtracked = True
                                break
                        else:
                            # Restore old path
                            used_interiors |= old_interior
                            paths_dict[(pu, pv)] = old_path
                            paths_dict[(pv, pu)] = old_path

                    if not backtracked:
                        success = False
                        break
                    # Continue with current pair after backtrack
                    continue

                interior = set(path[1:-1])
                if interior & set(branches) or interior & used_interiors:
                    success = False
                    break

                used_interiors |= interior
                paths_dict[(u, v)] = path
                paths_dict[(v, u)] = path

            if success:
                path_list = []
                for (u, v) in pair_list:
                    path_list.append(paths_dict[(u, v)])
                return path_list

        return None

    def find_paths_multi_strategy(branches, rng, max_iters=300):
        """Try multiple strategies and return the first success."""
        k = len(branches)
        if k < 2:
            return None

        # Strategy 1: Constrained first
        result = find_paths_constrained_first(branches, rng, max_iters=max_iters//3)
        if result is not None:
            return result

        # Strategy 2: With backtracking
        result = find_paths_with_backtrack(branches, rng, max_iters=max_iters//3)
        if result is not None:
            return result

        # Strategy 3: Randomized multiple attempts
        for _ in range(max_iters//3):
            if time_up():
                return None
            pair_list = []
            for i in range(k):
                for j in range(i + 1, k):
                    pair_list.append((branches[i], branches[j]))
            rng.shuffle(pair_list)

            paths_dict = {}
            used_interiors = set()
            success = True

            for (u, v) in pair_list:
                if time_up():
                    return None
                forbidden = set(used_interiors)
                forbidden.update(branches)
                forbidden.discard(u)
                forbidden.discard(v)

                if v in adj[u]:
                    path = [u, v]
                else:
                    path = _bfs_shortest(u, v, forbidden, adj_sorted)
                    if path is None:
                        path = _dfs_random(u, v, forbidden, rng, adj_sorted)

                if path is None:
                    success = False
                    break

                interior = set(path[1:-1])
                if interior & set(branches) or interior & used_interiors:
                    success = False
                    break

                used_interiors |= interior
                paths_dict[(u, v)] = path
                paths_dict[(v, u)] = path

            if success:
                path_list = []
                for (u, v) in pair_list:
                    path_list.append(paths_dict[(u, v)])
                return path_list

        return None

    # Degree ranking
    deg = [len(adj[i]) for i in range(n)]
    candidates_sorted = sorted(range(n), key=lambda i: (-deg[i], i))

    best = None
    best_size = 0

    def try_branch_set(branches, rng, max_iters=200):
        nonlocal best, best_size
        if len(branches) < 2:
            return False
        path_list = find_paths_multi_strategy(branches, rng, max_iters=max_iters)
        if path_list is not None:
            if validate(branches, path_list):
                if len(branches) > best_size:
                    best = {'branches': list(branches), 'paths': path_list}
                    best_size = len(branches)
                    return True
        return False

    rng = random.Random(42)

    # Strategy 1: Greedy growth from promising seeds
    top_candidates = candidates_sorted[:15]

    # Try all pairs of top candidates as seeds
    for i in range(len(top_candidates)):
        if time_up():
            break
        for j in range(i + 1, len(top_candidates)):
            if time_up():
                break
            seed = [top_candidates[i], top_candidates[j]]
            current = list(seed)
            try_branch_set(current, rng, max_iters=50)

            # Greedy growth
            for _ in range(20):
                if time_up():
                    break
                added = False
                candidates = [v for v in range(n) if v not in current]
                # Prioritize high-degree and well-connected vertices
                def cand_score(v):
                    if v in current:
                        return -1
                    # Score based on connectivity to current branches
                    conn = sum(1 for b in current if v in adj[b])
                    return (-deg[v], -conn, v)
                candidates.sort(key=cand_score)
                rng.shuffle(candidates[:min(10, len(candidates))])

                for cand in candidates[:15]:
                    if time_up():
                        break
                    new_set = current + [cand]
                    if try_branch_set(new_set, rng, max_iters=100):
                        current = new_set
                        added = True
                        break
                if not added:
                    break

    # Strategy 2: Random subset search
    for k in range(3, 12):
        if time_up():
            break
        if best_size >= k:
            continue
        for _ in range(80):
            if time_up():
                break
            # Sample from top candidates with some randomness
            sample_size = min(k, len(top_candidates))
            subset = rng.sample(top_candidates, sample_size)
            if k > sample_size:
                remaining = [v for v in range(n) if v not in subset]
                subset += rng.sample(remaining, k - sample_size)
            try_branch_set(subset, rng, max_iters=150)

    # Strategy 3: Hill climbing from best
    if best is not None and not time_up():
        current_branches = list(best['branches'])
        current_paths = best['paths']

        for _ in range(300):
            if time_up():
                break
            improved = False
            candidates = [v for v in range(n) if v not in current_branches]
            def cand_score(v):
                conn = sum(1 for b in current_branches if v in adj[b])
                return (-deg[v], -conn, v)
            candidates.sort(key=cand_score)
            rng.shuffle(candidates[:min(15, len(candidates))])

            for cand in candidates[:20]:
                if time_up():
                    break
                new_set = current_branches + [cand]
                new_paths = find_paths_multi_strategy(new_set, rng, max_iters=150)
                if new_paths is not None and validate(new_set, new_paths):
                    current_branches = new_set
                    current_paths = new_paths
                    best = {'branches': list(current_branches), 'paths': current_paths}
                    best_size = len(current_branches)
                    improved = True
                    break
            if not improved:
                # Try swaps
                if len(current_branches) > 2:
                    swap_found = False
                    for remove_idx in range(len(current_branches)):
                        if time_up():
                            break
                        for cand in candidates[:10]:
                            if time_up():
                                break
                            new_set = current_branches[:remove_idx] + current_branches[remove_idx+1:] + [cand]
                            new_paths = find_paths_multi_strategy(new_set, rng, max_iters=150)
                            if new_paths is not None and validate(new_set, new_paths):
                                current_branches = new_set
                                current_paths = new_paths
                                best = {'branches': list(current_branches), 'paths': current_paths}
                                best_size = len(current_branches)
                                improved = True
                                swap_found = True
                                break
                        if swap_found:
                            break
            if not improved:
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
            for u, v in edges:
                best = {'branches': [u, v], 'paths': [[u, v]]}
                break
        if best is None:
            best = {'branches': [0, 1], 'paths': [[0, 1]]}

    if not validate(best['branches'], best['paths']):
        for u, v in edges:
            if validate([u, v], [[u, v]]):
                best = {'branches': [u, v], 'paths': [[u, v]]}
                break

    branches = [int(x) for x in best['branches']]
    paths = [[int(x) for x in p] for p in best['paths']]
    return {'branches': branches, 'paths': paths}
