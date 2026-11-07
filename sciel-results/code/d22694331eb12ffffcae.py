"""Solver for large clique subdivisions in a pinned graph."""

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    import time
    from collections import deque
    from itertools import combinations
    import random

    t_start = time.time()
    DEADLINE = 85.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    adj_list = [list() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        adj_list[i] = sorted(adj[i])
    deg = [len(adj[i]) for i in range(n)]

    def time_up():
        return time.time() - t_start > DEADLINE

    # Precompute all-pairs shortest path distances
    sp_dist = [[10**9] * n for _ in range(n)]
    for s in range(n):
        dist = [10**9] * n
        dist[s] = 0
        q = deque([s])
        while q:
            cur = q.popleft()
            for nb in adj_list[cur]:
                if dist[nb] > dist[cur] + 1:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        sp_dist[s] = dist

    def bfs_path(src, dst, forbidden):
        """BFS simple path src->dst; interior vertices avoid forbidden."""
        if src == dst:
            return [src]
        parent = {src: -1}
        q = deque([src])
        while q:
            cur = q.popleft()
            if cur == dst:
                path = []
                node = dst
                while node != -1:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in adj_list[cur]:
                if nb not in parent:
                    if nb != dst and nb in forbidden:
                        continue
                    parent[nb] = cur
                    q.append(nb)
        return None

    def find_alternatives(u, v, used, max_alts=5):
        """Find up to max_alts distinct simple paths u->v avoiding used interiors."""
        alts = []
        p0 = bfs_path(u, v, used)
        if p0 is None:
            return alts
        alts.append(tuple(p0))
        seen = {tuple(p0)}
        # Try excluding each interior vertex of p0 to force different routes
        for mid in p0[1:-1]:
            if len(alts) >= max_alts:
                break
            alt = bfs_path(u, v, used | {mid})
            if alt is not None:
                t = tuple(alt)
                if t not in seen:
                    seen.add(t)
                    alts.append(t)
        # Try excluding pairs of interior vertices
        if len(alts) < max_alts:
            interior = p0[1:-1]
            for a_idx in range(len(interior)):
                for b_idx in range(a_idx + 1, len(interior)):
                    if len(alts) >= max_alts:
                        break
                    a, b = interior[a_idx], interior[b_idx]
                    alt = bfs_path(u, v, used | {a, b})
                    if alt is not None:
                        t = tuple(alt)
                        if t not in seen:
                            seen.add(t)
                            alts.append(t)
        return alts

    def solve_paths_backtrack(branches, order, max_alts=5):
        """Backtracking: assign internally-disjoint paths for all pairs in order.
        Returns dict {(u,v): path} or None."""
        bset = set(branches)
        k = len(order)

        def bt(idx, used, paths):
            if time_up():
                return None
            if idx == k:
                return dict(paths)
            u, v = order[idx]
            alts = find_alternatives(u, v, used, max_alts)
            if not alts:
                return None
            for path_t in alts:
                interior = set(path_t[1:-1])
                if interior & bset or interior & used:
                    continue
                new_used = used | interior
                paths[(min(u, v), max(u, v))] = list(path_t)
                result = bt(idx + 1, new_used, paths)
                if result is not None:
                    return result
                del paths[(min(u, v), max(u, v))]
            return None

        return bt(0, set(), {})

    def make_order(branches, mode):
        """Generate a pair ordering for the given branch set."""
        pairs = list(combinations(branches, 2))
        if mode == 0:
            # direct edges first, then by distance
            pairs.sort(key=lambda p: (0 if p[1] in adj[p[0]] else 1,
                                       sp_dist[p[0]][p[1]], p[0], p[1]))
        elif mode == 1:
            # high degree sum first
            pairs.sort(key=lambda p: -(deg[p[0]] + deg[p[1]]))
        elif mode == 2:
            # low degree first (constrained endpoints)
            pairs.sort(key=lambda p: (deg[p[0]], deg[p[1]]))
        elif mode == 3:
            # reverse of mode 0
            pairs.sort(key=lambda p: (0 if p[1] in adj[p[0]] else 1,
                                       sp_dist[p[0]][p[1]], p[0], p[1]))
            pairs.reverse()
        elif mode == 4:
            # by distance only
            pairs.sort(key=lambda p: (sp_dist[p[0]][p[1]], p[0], p[1]))
        elif mode == 5:
            # random shuffle
            random.shuffle(pairs)
        else:
            pairs.sort(key=lambda p: (p[0], p[1]))
        return pairs

    def make_result(branches, paths_dict):
        bset = set(branches)
        path_list = []
        for u, v in combinations(branches, 2):
            key = (min(u, v), max(u, v))
            path_list.append(list(paths_dict[key]))
        return {'branches': list(branches), 'paths': path_list}

    def validate(branches, paths):
        """Validate a clique subdivision. Returns True if valid."""
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
                if p[i + 1] not in adj[p[i]]:
                    return False
        path_pairs = set()
        for p in paths:
            path_pairs.add(frozenset((p[0], p[-1])))
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                if frozenset((branches[i], branches[j])) not in path_pairs:
                    return False
        return True

    def try_branch_set(branches, max_alts=5, num_orderings=5):
        """Try multiple orderings with backtracking; return result dict or None."""
        if len(branches) < 2:
            return None
        for mode in range(num_orderings):
            if time_up():
                return None
            order = make_order(branches, mode)
            paths = solve_paths_backtrack(branches, order, max_alts)
            if paths is not None:
                result = make_result(branches, paths)
                if validate(result['branches'], result['paths']):
                    return result
        return None

    # Degree ranking
    candidates_sorted = sorted(range(n), key=lambda i: (-deg[i], i))
    top_candidates = candidates_sorted[:12]

    best = None
    best_size = 0

    def try_and_update(branches, max_alts=5, num_orderings=5):
        """Try to find valid paths for a given branch set. Update best if successful."""
        nonlocal best, best_size
        if len(branches) < 2:
            return False
        result = try_branch_set(branches, max_alts, num_orderings)
        if result is not None:
            if len(branches) > best_size:
                best = result
                best_size = len(branches)
                return True
        return False

    # Strategy 1: Greedy growth from pairs of top-degree vertices
    for i in range(len(top_candidates)):
        if time_up():
            break
        for j in range(i + 1, len(top_candidates)):
            if time_up():
                break
            current = [top_candidates[i], top_candidates[j]]
            try_and_update(current, max_alts=4, num_orderings=3)

            # Greedy growth
            for _ in range(15):
                if time_up():
                    break
                added = False
                candidates = [v for v in top_candidates if v not in current]
                candidates += [v for v in range(n) if v not in current and v not in top_candidates]
                random.shuffle(candidates)

                for cand in candidates:
                    if time_up():
                        break
                    new_set = current + [cand]
                    if try_and_update(new_set, max_alts=4, num_orderings=3):
                        current = new_set
                        added = True
                        break
                if not added:
                    break

    # Strategy 2: Systematic search over random subsets
    for k in range(3, 12):
        if time_up():
            break
        if best_size >= k:
            continue
        for _ in range(60):
            if time_up():
                break
            subset = random.sample(top_candidates, min(k, len(top_candidates)))
            try_and_update(subset, max_alts=4, num_orderings=4)

    # Strategy 3: Local search / hill climbing
    if best is not None and not time_up():
        current_branches = list(best['branches'])
        current_paths = best['paths']

        for _ in range(300):
            if time_up():
                break
            improved = False
            # Try adding a vertex
            candidates = [v for v in range(n) if v not in current_branches]
            random.shuffle(candidates)
            for cand in candidates[:25]:
                if time_up():
                    break
                new_set = current_branches + [cand]
                new_result = try_branch_set(new_set, max_alts=5, num_orderings=5)
                if new_result is not None:
                    current_branches = new_set
                    current_paths = new_result['paths']
                    best = new_result
                    best_size = len(current_branches)
                    improved = True
                    break
            if not improved:
                # Try swapping: remove one, add another
                if len(current_branches) > 2:
                    for remove_idx in range(len(current_branches)):
                        if time_up():
                            break
                        for cand in candidates[:15]:
                            if time_up():
                                break
                            new_set = current_branches[:remove_idx] + current_branches[remove_idx + 1:] + [cand]
                            new_result = try_branch_set(new_set, max_alts=5, num_orderings=5)
                            if new_result is not None:
                                current_branches = new_set
                                current_paths = new_result['paths']
                                best = new_result
                                best_size = len(current_branches)
                                improved = True
                                break
                        if improved:
                            break
            if not improved:
                break

    # Fallback: find a triangle or edge
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

    # Final validation
    if not validate(best['branches'], best['paths']):
        # Emergency fallback
        for u, v in edges:
            if validate([u, v], [[u, v]]):
                best = {'branches': [u, v], 'paths': [[u, v]]}
                break

    # Convert to proper types
    branches = [int(x) for x in best['branches']]
    paths = [[int(x) for x in p] for p in best['paths']]
    return {'branches': branches, 'paths': paths}
