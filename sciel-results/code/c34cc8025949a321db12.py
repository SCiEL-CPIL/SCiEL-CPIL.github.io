"""Rewritten program: multi-strategy longest simple path search in the pinned G(100,1.8/100) graph."""
import time
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

_N = P['n']
_EDGES = P['edges']
_DEADLINE = 80.0  # safety margin under 90s wall-clock kill
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
    seen = {start}
    cur = start
    while True:
        cand = [w for w in adj[cur] if w not in seen]
        if not cand:
            break
        if mode == 'dead_end':
            nxt = min(cand, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
        elif mode == 'high_deg':
            nxt = max(cand, key=lambda w: (_DEG[w], -w))
        elif mode == 'low_deg':
            nxt = min(cand, key=lambda w: (_DEG[w], w))
        else:
            nxt = cand[0]
        path.append(nxt)
        seen.add(nxt)
        cur = nxt
    return path

def _extend_both_ends(path, adj):
    """Try to extend a path at both ends repeatedly using dead-end-avoidance greedy."""
    path = list(path)
    seen = set(path)
    changed = True
    while changed and _remaining() > 1.0:
        changed = False
        # try left end
        cur = path[0]
        cand = [w for w in adj[cur] if w not in seen]
        if cand:
            nxt = min(cand, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
            path.insert(0, nxt)
            seen.add(nxt)
            changed = True
        # try right end
        cur = path[-1]
        cand = [w for w in adj[cur] if w not in seen]
        if cand:
            nxt = min(cand, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
            path.append(nxt)
            seen.add(nxt)
            changed = True
    return path

def _try_insert(path, adj, limit=200):
    """Try to insert unused vertices into the path to lengthen it."""
    path = list(path)
    seen = set(path)
    unused = [v for v in range(_N) if v not in seen]
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
            seen.add(best_v)
            unused.remove(best_v)
            improved = True
    return path

def _backtrack_search(adj, start, depth_limit, time_limit):
    """Bounded DFS backtracking to find a long path starting at start."""
    best_path = [start]
    best_len = 1

    def dfs(path, seen):
        nonlocal best_path, best_len
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        if _remaining() < time_limit or len(path) >= depth_limit:
            return
        cur = path[-1]
        # order neighbors by fewest remaining options first (dead-end avoidance)
        cand = [w for w in adj[cur] if w not in seen]
        cand.sort(key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
        for w in cand:
            if _remaining() < time_limit:
                return
            path.append(w)
            seen.add(w)
            dfs(path, seen)
            path.pop()
            seen.remove(w)

    dfs([start], {start})
    return best_path

def _largest_component(adj):
    visited = set()
    comp = []
    for s in range(_N):
        if s in visited:
            continue
        stack = [s]
        visited.add(s)
        c = []
        while stack:
            u = stack.pop()
            c.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        if len(c) > len(comp):
            comp = c
    return comp

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
    if _remaining() > 15.0:
        comp = _largest_component(adj)
        # pick a few promising starts in the largest component
        starts = sorted(comp, key=lambda v: -_DEG[v])[:min(20, len(comp))]
        # also include endpoints of best path
        if best:
            starts = list(set(starts + [best[0], best[-1]]))
        for s in starts:
            if _remaining() < 5.0:
                break
            depth = min(len(comp), 60)
            tl = max(1.0, min(3.0, _remaining() * 0.25))
            p = _backtrack_search(adj, s, depth_limit=depth, time_limit=tl)
            if len(p) > len(best):
                best = p

    # Strategy 7: final insertion pass on the overall best
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
