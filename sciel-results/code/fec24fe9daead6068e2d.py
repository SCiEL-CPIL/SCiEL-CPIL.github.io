"""Find longest simple path in the pinned graph.

Combines forced-move / multi-heuristic randomized restarts (structural search)
with a refinement pipeline (extend both ends, insert unvisited vertices,
limited backtracking) to maximize the simple path length in the pinned graph.
"""
import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)


def find_components():
    """Find all connected components."""
    visited = [False] * N
    comps = []
    for i in range(N):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in ADJ[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            comps.append(comp)
    return comps


def greedy_path_from(start, adj, seen=None):
    """Greedy path starting from start (Warnsdorff: fewest remaining options)."""
    if seen is None:
        seen = set()
    path = [start]
    seen.add(start)
    while True:
        cur = path[-1]
        candidates = [nb for nb in adj[cur] if nb not in seen]
        if not candidates:
            break
        v = min(candidates, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
        path.append(v)
        seen.add(v)
    return path


def extend_both_ends(path, adj):
    """Try to extend path from both ends (Warnsdorff)."""
    path = list(path)
    seen = set(path)
    changed = True
    while changed:
        changed = False
        if path:
            cur = path[0]
            candidates = [nb for nb in adj[cur] if nb not in seen]
            if candidates:
                v = min(candidates, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
                path.insert(0, v)
                seen.add(v)
                changed = True
        if path:
            cur = path[-1]
            candidates = [nb for nb in adj[cur] if nb not in seen]
            if candidates:
                v = min(candidates, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
                path.append(v)
                seen.add(v)
                changed = True
    return path


def try_insert(path, adj):
    """Try to insert unvisited vertices between consecutive path nodes."""
    path = list(path)
    seen = set(path)
    unvisited = set(range(N)) - seen
    improved = True
    while improved and unvisited:
        improved = False
        for w in list(unvisited):
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                if w in adj[u] and w in adj[v]:
                    path.insert(i + 1, w)
                    seen.add(w)
                    unvisited.remove(w)
                    improved = True
                    break
            if improved:
                break
    return path


def backtracking_extend(path, adj, max_depth=40):
    """Try to extend path using limited backtracking from both ends."""
    path = list(path)
    seen = set(path)

    def dfs_end(cur, depth):
        if depth >= max_depth:
            return False
        candidates = [nb for nb in adj[cur] if nb not in seen]
        if not candidates:
            return False
        candidates.sort(key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
        for v in candidates:
            path.append(v)
            seen.add(v)
            if dfs_end(v, depth + 1):
                return True
            path.pop()
            seen.remove(v)
        return False

    def dfs_start(cur, depth):
        if depth >= max_depth:
            return False
        candidates = [nb for nb in adj[cur] if nb not in seen]
        if not candidates:
            return False
        candidates.sort(key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
        for nv in candidates:
            path.insert(0, nv)
            seen.add(nv)
            if dfs_start(nv, depth + 1):
                return True
            path.pop(0)
            seen.remove(nv)
        return False

    if path:
        dfs_end(path[-1], 0)
    if path:
        start = path[0]
        for v in list(adj[start]):
            if v in seen:
                continue
            path.insert(0, v)
            seen.add(v)
            dfs_start(v, 0)
            if path and path[0] == v:
                path.pop(0)
                seen.remove(v)
    return path


def forced_move_path(start, adj, rng, mode):
    """Build a simple path with forced-move priority and a randomized tie-break.

    At each step, if the current vertex has exactly one unvisited neighbor it
    MUST be taken (forced move). Otherwise pick by the given heuristic:
      'warns' : fewest unvisited neighbors (Warnsdorff)
      'max'   : most total neighbors (max degree)
      'rand'  : uniform random
    """
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        candidates = [nb for nb in adj[cur] if nb not in seen]
        if not candidates:
            break
        if len(candidates) == 1:
            v = candidates[0]
        elif mode == 'warns':
            v = min(candidates, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
        elif mode == 'max':
            v = max(candidates, key=lambda w: (len(adj[w]), -w))
        else:
            v = rng.choice(candidates)
        path.append(v)
        seen.add(v)
    return path


def refine(path, adj, deadline):
    """Polish a candidate path with the refinement pipeline."""
    path = extend_both_ends(path, adj)
    path = try_insert(path, adj)
    path = extend_both_ends(path, adj)
    return path


def construct():
    start_time = time.time()
    deadline = start_time + 85  # 90s wall-clock kill; keep a safety margin

    comps = find_components()
    comps.sort(key=len, reverse=True)
    if not comps:
        return [0]

    best_path = list(comps[0][:1])
    best_len = len(best_path)
    modes = ['warns', 'max', 'rand']

    # Phase 1: deterministic greedy from every vertex of the largest component.
    for start in comps[0]:
        if time.time() > deadline - 20:
            break
        p = greedy_path_from(start, ADJ)
        p = refine(p, ADJ, deadline)
        if len(p) > best_len:
            best_path, best_len = p, len(p)

    # Phase 2: forced-move / multi-heuristic randomized restarts (structural search).
    rng = random.Random(748)
    for _ in range(20000):
        if time.time() > deadline - 8:
            break
        start = rng.choice(comps[0])
        mode = modes[rng.randrange(len(modes))]
        p = forced_move_path(start, ADJ, rng, mode)
        p = refine(p, ADJ, deadline)
        if len(p) > best_len:
            best_path, best_len = p, len(p)

    # Phase 3: limited backtracking extension on the best path.
    if best_path and time.time() < deadline - 3:
        extended = backtracking_extend(best_path, ADJ, max_depth=40)
        extended = refine(extended, ADJ, deadline)
        if len(extended) > best_len:
            best_path, best_len = extended, len(extended)

    # Final polish.
    if best_path and time.time() < deadline - 1:
        extended = refine(best_path, ADJ, deadline)
        if len(extended) > best_len:
            best_path, best_len = extended, len(extended)

    # Validate the path; fall back to a valid greedy path if needed.
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in ADJ[best_path[i]]:
            for s in range(N):
                p = greedy_path_from(s, ADJ)
                if p:
                    best_path = p
                    break
            break

    if not best_path:
        best_path = [0]
    return best_path
