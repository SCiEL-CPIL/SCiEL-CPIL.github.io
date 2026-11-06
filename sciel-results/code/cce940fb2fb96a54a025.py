"""Rewritten program: multi-strategy longest simple path search in the pinned G(100,1.8/100) graph."""
import time

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

_N = P['n']
_EDGES = P['edges']
_DEADLINE = 85.0  # safety margin under 90s wall-clock kill
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _build_adj():
    adj = [set() for _ in range(_N)]
    for u, v in _EDGES:
        adj[u].add(v)
        adj[v].add(u)
    # sort for determinism
    for i in range(_N):
        adj[i] = sorted(adj[i])
    return adj

_ADJ = _build_adj()
_DEG = [len(_ADJ[i]) for i in range(_N)]

def _greedy_path(adj, start, mode='dead_end'):
    """Greedy path from start. mode: 'dead_end' (fewest remaining unvisited neighbors),
    'high_deg' (highest degree), 'low_deg' (lowest degree)."""
    path = [start]
    seen = [False] * _N
    seen[start] = True
    cur = start
    while True:
        cand = [w for w in adj[cur] if not seen[w]]
        if not cand:
            break
        if mode == 'dead_end':
            nxt = min(cand, key=lambda w: (sum(1 for x in adj[w] if not seen[x]), w))
        elif mode == 'high_deg':
            nxt = max(cand, key=lambda w: (_DEG[w], -w))
        elif mode == 'low_deg':
            nxt = min(cand, key=lambda w: (_DEG[w], w))
        else:
            nxt = cand[0]
        path.append(nxt)
        seen[nxt] = True
        cur = nxt
    return path

def _extend_both_ends(path, adj):
    """Try to extend a path at both ends repeatedly using dead-end-avoidance greedy."""
    path = list(path)
    seen = [False] * _N
    for v in path:
        seen[v] = True
    changed = True
    while changed and _remaining() > 1.0:
        changed = False
        # try left end
        cur = path[0]
        cand = [w for w in adj[cur] if not seen[w]]
        if cand:
            nxt = min(cand, key=lambda w: (sum(1 for x in adj[w] if not seen[x]), w))
            path.insert(0, nxt)
            seen[nxt] = True
            changed = True
        # try right end
        cur = path[-1]
        cand = [w for w in adj[cur] if not seen[w]]
        if cand:
            nxt = min(cand, key=lambda w: (sum(1 for x in adj[w] if not seen[x]), w))
            path.append(nxt)
            seen[nxt] = True
            changed = True
    return path

def _try_insert(path, adj, limit=200):
    """Try to insert unused vertices into the path to lengthen it."""
    path = list(path)
    seen = [False] * _N
    for v in path:
        seen[v] = True
    unused = [v for v in range(_N) if not seen[v]]
    improved = True
    tries = 0
    while improved and tries < limit and _remaining() > 1.0:
        improved = False
        tries += 1
        # For each unused vertex, check if it can be inserted between consecutive path vertices
        # or appended at ends
        best_gain = -1
        best_pos = -1
        best_v = -1
        for v in unused:
            # try inserting between path[i] and path[i+1]
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                if v in adj[a] and v in adj[b]:
                    # gain 1
                    if best_gain < 1:
                        best_gain = 1
                        best_pos = i
                        best_v = v
                        break
            else:
                continue
            break
        # also try appending to ends
        if best_gain < 1:
            if unused:
                # try append to right
                cur = path[-1]
                for v in unused:
                    if v in adj[cur]:
                        best_gain = 1
                        best_pos = len(path)  # append
                        best_v = v
                        break
            if best_gain < 1:
                cur = path[0]
                for v in unused:
                    if v in adj[cur]:
                        best_gain = 1
                        best_pos = -1  # prepend
                        best_v = v
                        break
        if best_gain > 0 and best_v >= 0:
            if best_pos == -1:
                path.insert(0, best_v)
            elif best_pos == len(path):
                path.append(best_v)
            else:
                path.insert(best_pos + 1, best_v)
            seen[best_v] = True
            unused.remove(best_v)
            improved = True
    return path

def _backtrack_search(adj, start, depth_limit, time_limit):
    """Bounded DFS backtracking with bipartite-matching-inspired heuristic.
    Prioritizes vertices with low 'matching slack' (few unvisited neighbors)
    to avoid trapping path ends."""
    best_path = [start]
    best_len = 1
    seen = [False] * _N
    seen[start] = True

    def dfs(path):
        nonlocal best_path, best_len
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        if _remaining() < time_limit or len(path) >= depth_limit:
            return
        cur = path[-1]
        cand = [w for w in adj[cur] if not seen[w]]
        if not cand:
            return
        # Bipartite-matching-inspired heuristic:
        # For each candidate, compute its "slack" = number of unvisited neighbors.
        # Vertices with low slack are more constrained and should be explored first
        # to avoid creating isolated vertices that can't be placed later.
        # Also consider: if choosing w would leave another unvisited vertex with
        # zero unvisited neighbors (except possibly path[-1]), that's a trap.
        scored = []
        for w in cand:
            slack = sum(1 for x in adj[w] if not seen[x])
            # Penalty if choosing w would orphan another vertex
            orphan_penalty = 0
            for x in adj[w]:
                if not seen[x] and x != cur:
                    # If x only connects to w among unvisited (besides cur), it becomes isolated
                    x_slack = sum(1 for y in adj[x] if not seen[y] and y != w)
                    if x_slack == 0:
                        orphan_penalty += 2
            scored.append((slack + orphan_penalty, w))
        scored.sort()
        for _, w in scored:
            if _remaining() < time_limit:
                return
            path.append(w)
            seen[w] = True
            dfs(path)
            path.pop()
            seen[w] = False

    dfs([start])
    return best_path

def _merge_paths(path1, path2, adj):
    """Try to merge two disjoint paths into one longer path.
    Check if endpoints of path1 can connect to endpoints of path2."""
    if not path1 or not path2:
        return None
    set1 = set(path1)
    set2 = set(path2)
    if set1 & set2:
        return None
    # Try all four endpoint combinations
    e1a, e1b = path1[0], path1[-1]
    e2a, e2b = path2[0], path2[-1]
    # path1 + path2: e1b-e2a or e1b-e2b
    if e1b in adj[e2a]:
        return path1 + path2
    if e1b in adj[e2b]:
        return path1 + list(reversed(path2))
    # reversed path1 + path2: e1a-e2a or e1a-e2b
    if e1a in adj[e2a]:
        return list(reversed(path1)) + path2
    if e1a in adj[e2b]:
        return list(reversed(path1)) + list(reversed(path2))
    return None

def _largest_component(adj):
    visited = [False] * _N
    comp = []
    for s in range(_N):
        if visited[s]:
            continue
        stack = [s]
        visited[s] = True
        c = []
        while stack:
            u = stack.pop()
            c.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        if len(c) > len(comp):
            comp = c
    return comp

def _try_2_insert(path, adj, limit=100):
    """Try to insert pairs of unused vertices into the path."""
    path = list(path)
    seen = [False] * _N
    for v in path:
        seen[v] = True
    unused = [v for v in range(_N) if not seen[v]]
    improved = True
    tries = 0
    while improved and tries < limit and _remaining() > 1.0:
        improved = False
        tries += 1
        best_gain = -1
        best_pos = -1
        best_v1 = -1
        best_v2 = -1
        # Try inserting two consecutive vertices between path[i] and path[i+1]
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            for v1 in unused:
                if v1 in adj[a]:
                    for v2 in unused:
                        if v1 != v2 and v2 in adj[v1] and v2 in adj[b]:
                            if best_gain < 2:
                                best_gain = 2
                                best_pos = i
                                best_v1 = v1
                                best_v2 = v2
                                break
                if best_gain >= 2:
                    break
            if best_gain >= 2:
                break
        # Also try single insertions
        if best_gain < 1:
            for v in unused:
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    if v in adj[a] and v in adj[b]:
                        best_gain = 1
                        best_pos = i
                        best_v1 = v
                        break
                if best_gain >= 1:
                    break
        if best_gain < 1 and unused:
            cur = path[-1]
            for v in unused:
                if v in adj[cur]:
                    best_gain = 1
                    best_pos = len(path)
                    best_v1 = v
                    break
        if best_gain < 1 and unused:
            cur = path[0]
            for v in unused:
                if v in adj[cur]:
                    best_gain = 1
                    best_pos = -1
                    best_v1 = v
                    break
        if best_gain > 0 and best_v1 >= 0:
            if best_gain == 2:
                if best_pos == -1:
                    path.insert(0, best_v2)
                    path.insert(0, best_v1)
                elif best_pos == len(path):
                    path.append(best_v1)
                    path.append(best_v2)
                else:
                    path.insert(best_pos + 1, best_v1)
                    path.insert(best_pos + 2, best_v2)
                seen[best_v1] = True
                seen[best_v2] = True
                unused.remove(best_v1)
                unused.remove(best_v2)
            else:
                if best_pos == -1:
                    path.insert(0, best_v1)
                elif best_pos == len(path):
                    path.append(best_v1)
                else:
                    path.insert(best_pos + 1, best_v1)
                seen[best_v1] = True
                unused.remove(best_v1)
            improved = True
    return path

def construct():
    global _START, _DEADLINE
    _START = time.time()
    adj = _ADJ

    best = []

    # Strategy 1: dead-end-avoidance greedy from each start
    for s in range(_N):
        if _remaining() < 5.0:
            break
        p = _greedy_path(adj, s, mode='dead_end')
        if len(p) > len(best):
            best = p

    # Strategy 2: high-degree greedy from each start
    for s in range(_N):
        if _remaining() < 5.0:
            break
        p = _greedy_path(adj, s, mode='high_deg')
        if len(p) > len(best):
            best = p

    # Strategy 3: low-degree greedy from each start
    for s in range(_N):
        if _remaining() < 5.0:
            break
        p = _greedy_path(adj, s, mode='low_deg')
        if len(p) > len(best):
            best = p

    # Strategy 4: extend both ends of best greedy paths
    if _remaining() > 10.0:
        for s in range(_N):
            if _remaining() < 5.0:
                break
            for mode in ('dead_end', 'high_deg', 'low_deg'):
                p = _greedy_path(adj, s, mode=mode)
                p = _extend_both_ends(p, adj)
                if len(p) > len(best):
                    best = p

    # Strategy 5: vertex insertion on best path
    if _remaining() > 10.0:
        p = _try_insert(list(best), adj, limit=500)
        if len(p) > len(best):
            best = p

    # Strategy 6: bounded DFS backtracking on largest component
    if _remaining() > 20.0:
        comp = _largest_component(adj)
        # pick promising starts in the largest component
        starts = sorted(comp, key=lambda v: -_DEG[v])[:min(30, len(comp))]
        # also include endpoints of best path
        if best:
            starts = list(set(starts + [best[0], best[-1]]))
        for s in starts:
            if _remaining() < 5.0:
                break
            depth = min(len(comp), 80)
            tl = max(1.0, min(5.0, _remaining() * 0.3))
            p = _backtrack_search(adj, s, depth_limit=depth, time_limit=tl)
            if len(p) > len(best):
                best = p

    # Strategy 6b: path merging - combine shorter greedy paths into longer ones
    if _remaining() > 8.0:
        # Generate several candidate paths from different starts
        candidates = []
        for s in range(0, _N, 3):  # sample every 3rd vertex as start
            if _remaining() < 4.0:
                break
            for mode in ('dead_end', 'high_deg'):
                p = _greedy_path(adj, s, mode=mode)
                if len(p) > 5:
                    candidates.append(p)
        # Try to merge pairs of disjoint paths
        candidates.sort(key=len, reverse=True)
        for i in range(len(candidates)):
            if _remaining() < 3.0:
                break
            for j in range(i + 1, len(candidates)):
                if _remaining() < 2.0:
                    break
                merged = _merge_paths(candidates[i], candidates[j], adj)
                if merged and len(merged) > len(best):
                    best = merged
                    break
            if _remaining() < 3.0:
                break

    # Strategy 7: 2-vertex insertion pass
    if _remaining() > 5.0:
        p = _try_2_insert(list(best), adj, limit=200)
        if len(p) > len(best):
            best = p

    # Strategy 8: final single-vertex insertion pass on the overall best
    if _remaining() > 2.0:
        p = _try_insert(list(best), adj, limit=1000)
        if len(p) > len(best):
            best = p

    # Ensure nonempty and valid
    if not best:
        # fallback: single vertex
        best = [0]

    # Validate: distinct vertices, consecutive edges
    valid = len(best) >= 1 and len(set(best)) == len(best)
    if valid:
        for i in range(len(best) - 1):
            if best[i + 1] not in adj[best[i]]:
                valid = False
                break
    if not valid:
        # reconstruct a valid path by trimming
        trimmed = [best[0]]
        for v in best[1:]:
            if v in adj[trimmed[-1]] and v not in trimmed:
                trimmed.append(v)
            else:
                break
        if not trimmed:
            trimmed = [best[0]]
        best = trimmed

    return best
