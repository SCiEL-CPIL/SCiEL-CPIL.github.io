import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    deadline = time.time() + 55.0
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # All-pairs shortest paths
    D = [[n]*n for _ in range(n)]
    for s in range(n):
        D[s][s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if D[s][w] > D[s][u] + 1:
                    D[s][w] = D[s][u] + 1
                    q.append(w)

    def find_paths_bounded(src, dst, forbidden, max_len, limit=20):
        """Find simple paths src->dst avoiding forbidden, length <= max_len."""
        if src == dst or dst in forbidden:
            return []
        # BFS distance from src for pruning
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                break
            for w in adj[u]:
                if w not in dist and w not in forbidden:
                    dist[w] = dist[u] + 1
                    q.append(w)
        if dst not in dist or dist[dst] > max_len:
            return []
        paths = []
        def dfs(u, path, depth):
            if len(paths) >= limit:
                return
            if depth > max_len:
                return
            if u == dst:
                paths.append(list(path))
                return
            for w in sorted(adj[u]):
                if w in forbidden:
                    continue
                if len(path) > 1 and w in path[:-1]:
                    continue
                if dist.get(w, n) <= depth + 1 and depth + 1 <= max_len:
                    path.append(w)
                    dfs(w, path, depth + 1)
                    path.pop()
        dfs(src, [src], 0)
        paths.sort(key=len)
        return paths

    def solve_branches(branches, time_limit):
        """Try to find internally-disjoint paths for all pairs of branches."""
        branch_set = frozenset(branches)
        pairs = list(combinations(branches, 2))

        # Sort pairs by difficulty (fewest options first)
        pair_opts = []
        for u, v in pairs:
            forb = branch_set - {u, v}
            cnt = len(find_paths_bounded(u, v, forb, D[u][v] + 1, limit=15))
            pair_opts.append((cnt, u, v))
        pair_opts.sort(key=lambda x: x[0])
        ordered_pairs = [(u, v) for _, u, v in pair_opts]

        used = set()
        found_paths = {}
        t0 = time.time()

        def bt(idx):
            if time.time() - t0 > time_limit:
                return False
            if idx == len(ordered_pairs):
                return True
            u, v = ordered_pairs[idx]
            forb = branch_set - {u, v} | used
            cands = find_paths_bounded(u, v, forb, D[u][v] + 2, limit=25)
            for p in cands:
                interior = set(p[1:-1])
                if interior & (used | branch_set):
                    continue
                found_paths[(u, v)] = p
                if bt(idx + 1):
                    return True
                del found_paths[(u, v)]
            return False

        if bt(0):
            return found_paths
        return None

    def make_result(branches, paths_dict):
        branches = [int(x) for x in branches]
        res = []
        for u, v in combinations(branches, 2):
            res.append([int(x) for x in paths_dict[(u, v)]])
        return {'branches': branches, 'paths': res}

    # ---- Greedy construction ----
    def greedy_build(seed, max_iter=3000):
        """Greedy: start from seed, add vertices one at a time if feasible."""
        branches = list(seed)
        paths_dict = solve_branches(branches, time_limit=2.0)
        if not paths_dict:
            return None
        best = (len(branches), make_result(branches, paths_dict))
        remaining = [v for v in range(n) if v not in branches]
        # Sort remaining by degree descending
        remaining.sort(key=lambda v: -len(adj[v]))

        for _ in range(max_iter):
            if time.time() > deadline - 3:
                break
            added = False
            for v in remaining:
                if time.time() > deadline - 3:
                    break
                trial = branches + [v]
                trial.sort()
                # Quick check: all pairs connected
                skip = False
                for a, b in combinations(trial, 2):
                    if D[a][b] == n:
                        skip = True
                        break
                if skip:
                    continue
                pd = solve_branches(trial, time_limit=1.0)
                if pd:
                    branches = trial
                    paths_dict = pd
                    remaining.remove(v)
                    added = True
                    if len(branches) > best[0]:
                        best = (len(branches), make_result(branches, paths_dict))
                    break
            if not added:
                break
        return best[1] if best[1] else None

    # ---- Local search (swap) ----
    def local_search(initial_result, iterations=2000):
        """Try to improve by swapping one branch for another."""
        if not initial_result:
            return None
        branches = list(initial_result['branches'])
        paths_dict = solve_branches(branches, time_limit=2.0)
        if not paths_dict:
            return None
        best_size = len(branches)
        best_result = make_result(branches, paths_dict)
        remaining = [v for v in range(n) if v not in branches]
        remaining.sort(key=lambda v: -len(adj[v]))
        rng = random.Random(42)

        for _ in range(iterations):
            if time.time() > deadline - 2:
                break
            # Try adding a new vertex
            added = False
            for v in remaining[:10]:
                trial = branches + [v]
                trial.sort()
                skip = any(D[a][b] == n for a, b in combinations(trial, 2))
                if skip:
                    continue
                pd = solve_branches(trial, time_limit=0.8)
                if pd:
                    branches = trial
                    paths_dict = pd
                    remaining.remove(v)
                    if len(branches) > best_size:
                        best_size = len(branches)
                        best_result = make_result(branches, paths_dict)
                    added = True
                    break
            if not added:
                # Try swap: remove one, add another
                for i, old in enumerate(branches):
                    for v in remaining[:8]:
                        trial = [x for j, x in enumerate(branches) if j != i] + [v]
                        trial.sort()
                        skip = any(D[a][b] == n for a, b in combinations(trial, 2))
                        if skip:
                            continue
                        pd = solve_branches(trial, time_limit=0.6)
                        if pd:
                            branches = trial
                            paths_dict = pd
                            if old not in remaining:
                                remaining.append(old)
                            if v in remaining:
                                remaining.remove(v)
                            if len(branches) > best_size:
                                best_size = len(branches)
                                best_result = make_result(branches, paths_dict)
                            break
                    if len(branches) > best_size:
                        break
        return best_result

    # ---- Search ----
    best_result = None
    best_size = 0

    # Strategy 1: Greedy from high-degree seeds
    high_deg = sorted(range(n), key=lambda v: -len(adj[v]))
    for seed_size in [4, 5, 6]:
        if time.time() > deadline - 10:
            break
        for seed in combinations(high_deg[:15], seed_size):
            if time.time() > deadline - 5:
                break
            seed_list = list(seed)
            # Quick connectivity
            if any(D[a][b] == n for a, b in combinations(seed_list, 2)):
                continue
            res = greedy_build(seed_list)
            if res and len(res['branches']) > best_size:
                best_size = len(res['branches'])
                best_result = res
                # Continue improving with local search
                improved = local_search(res, iterations=500)
                if improved and len(improved['branches']) > best_size:
                    best_size = len(improved['branches'])
                    best_result = improved

    # Strategy 2: Try large random subsets
    if time.time() < deadline - 5:
        rng2 = random.Random(7)
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            size = rng2.randint(7, 12)
            if size <= best_size:
                continue
            subset = rng2.sample(range(n), size)
            if any(D[a][b] == n for a, b in combinations(subset, 2)):
                continue
            pd = solve_branches(subset, time_limit=1.5)
            if pd:
                res = make_result(subset, pd)
                if len(subset) > best_size:
                    best_size = len(subset)
                    best_result = res
                # Try to extend
                extended = local_search(res, iterations=300)
                if extended and len(extended['branches']) > best_size:
                    best_size = len(extended['branches'])
                    best_result = extended

    # Strategy 3: Exhaustive over top candidates for smaller sizes
    if time.time() < deadline - 2 and best_size < 9:
        top = high_deg[:12]
        for size in range(min(best_size + 1, 9), 6, -1):
            if time.time() > deadline - 2:
                break
            for combo in combinations(top, size):
                if time.time() > deadline - 1:
                    break
                if any(D[a][b] == n for a, b in combinations(combo, 2)):
                    continue
                pd = solve_branches(combo, time_limit=0.5)
                if pd:
                    res = make_result(combo, pd)
                    if size > best_size:
                        best_size = size
                        best_result = res
                        break

    if best_result is None:
        # Fallback
        for u, v in P['edges']:
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        return {'branches': [0, 1], 'paths': [[0, 1]]}

    return best_result
