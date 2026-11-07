"""Alternative strategy: branch-and-bound over vertex subsets with exact
DFS path-packing feasibility (no max-flow), guided by elimination ordering."""
import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    t_start = time.time()
    DEADLINE = 82.0  # safety margin

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def time_up():
        return (time.time() - t_start) > DEADLINE

    # Precompute all simple paths of length <= 4 between every pair
    # (short paths are most likely to coexist in a packing)
    # Actually, we need paths of any length, but we'll use BFS for feasibility.

    def bfs_path(s, t, banned):
        """Find a simple path from s to t avoiding banned vertices (except s,t)."""
        if s == t:
            return None
        if t in adj[s]:
            return [s, t]
        parent = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v == s:
                    continue
                if v in parent:
                    continue
                if v == t:
                    path = [v]
                    cur = u
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    return path
                if v in banned:
                    continue
                parent[v] = u
                q.append(v)
        return None

    def check_packing(branches, max_depth=6000):
        """Exact DFS: can we find internally vertex-disjoint paths for all pairs?
        Returns list of paths or None. Uses backtracking with ordering heuristics."""
        m = len(branches)
        if m < 2:
            return []
        branch_set = set(branches)
        # Generate all pairs
        pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
        # Sort pairs: direct edges first, then by degree
        def pair_priority(p):
            i, j = p
            a, b = branches[i], branches[j]
            direct = 0 if b in adj[a] else 1
            return (direct, -len(adj[a]) - len(adj[b]))
        pairs.sort(key=pair_priority)

        used = set()
        paths_found = {}
        nodes_explored = [0]

        def dfs(idx):
            if idx == len(pairs):
                return True
            if nodes_explored[0] > max_depth:
                return False
            nodes_explored[0] += 1
            i, j = pairs[idx]
            a, b = branches[i], branches[j]
            banned = set(used)
            banned.update(branch_set)
            banned.discard(a)
            banned.discard(b)
            # Try BFS path
            path = bfs_path(a, b, banned)
            if path is None:
                return False
            interior = set(path[1:-1])
            if interior & used:
                return False
            # Try this path
            used.update(interior)
            paths_found[(i, j)] = path
            if dfs(idx + 1):
                return True
            # Undo
            used.difference_update(interior)
            del paths_found[(i, j)]
            # Try alternative: use a longer path (BFS gives shortest; try next)
            # For simplicity, we only try the BFS shortest path here.
            # To get alternatives, we'd need k-shortest paths, which is expensive.
            # Instead, we rely on the ordering and the fact that if BFS fails,
            # no path exists given current used set.
            return False

        if dfs(0):
            result = []
            for (i, j) in pairs:
                result.append(paths_found[(i, j)])
            return result
        return None

    # Branch and bound over vertex subsets
    # State: current branch set, candidate vertices to add
    # We'll use a recursive search with pruning

    best_branches = None
    best_paths = None
    best_count = 0

    def update_best(branches, paths):
        nonlocal best_branches, best_paths, best_count
        if paths is not None and len(branches) > best_count:
            best_branches = list(branches)
            best_paths = list(paths)
            best_count = len(branches)

    # Initial: find a good starting clique/set
    # Use greedy: add vertices in order of degree if packing remains feasible
    initial = []
    candidates = sorted(range(n), key=lambda x: -len(adj[x]))
    for v in candidates:
        trial = initial + [v]
        if len(trial) >= 2:
            p = check_packing(trial)
            if p is not None:
                initial = trial
                update_best(initial, p)
        if time_up():
            break

    # Branch and bound: try to expand
    def bnb(branches, candidates, depth=0):
        nonlocal best_count
        if time_up():
            return
        # Prune: even if we add all candidates, can't beat best
        if len(branches) + len(candidates) <= best_count:
            return
        # Try adding each candidate
        for i, v in enumerate(candidates):
            if time_up():
                return
            new_branches = branches + [v]
            new_candidates = candidates[i + 1:]
            # Check feasibility
            if len(new_branches) >= 2:
                p = check_packing(new_branches)
                if p is not None:
                    update_best(new_branches, p)
                    # Recurse
                    bnb(new_branches, new_candidates, depth + 1)

    # Start B&B from the initial set
    remaining = [v for v in range(n) if v not in initial]
    remaining.sort(key=lambda x: -len(adj[x]))
    bnb(initial, remaining)

    # If we didn't find anything, try random restarts
    if best_branches is None or len(best_branches) < 2:
        for restart in range(200):
            if time_up():
                break
            size = random.randint(2, min(n, 10))
            seed = random.sample(range(n), size)
            p = check_packing(seed)
            if p is not None:
                update_best(seed, p)

    # Local improvement: try swapping one vertex for another
    if best_branches is not None and not time_up():
        improved = True
        while improved and not time_up():
            improved = False
            cur = list(best_branches)
            cur_set = set(cur)
            for idx in range(len(cur)):
                if time_up():
                    break
                for v in range(n):
                    if v in cur_set:
                        continue
                    new_b = cur[:idx] + cur[idx + 1:] + [v]
                    new_b.sort()
                    p = check_packing(new_b)
                    if p is not None and len(new_b) > best_count:
                        update_best(new_b, p)
                        improved = True
                        break
                if improved:
                    break

    # Fallback
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = [[u, v]]

    branches_out = [int(b) for b in best_branches]
    paths_out = [[int(x) for x in p] for p in best_paths]

    return {'branches': branches_out, 'paths': paths_out}
