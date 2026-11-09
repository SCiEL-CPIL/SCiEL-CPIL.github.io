"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(added_edges):
    n = P['n']
    base = set()
    for e in P['edges']:
        base.add((min(e[0], e[1]), max(e[0], e[1])))
    all_edges = set(base)
    for e in added_edges:
        u, v = int(e[0]), int(e[1])
        u, v = min(u, v), max(u, v)
        if (u, v) in all_edges:
            return None  # duplicate
        all_edges.add((u, v))
    adj = [[] for _ in range(n)]
    for u, v in all_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check(adj):
    n = P['n']
    if adj is None:
        return False
    # Triangle-free: for each vertex, no two neighbors are adjacent
    for u in range(n):
        nbrs = adj[u]
        nbr_set = set(nbrs)
        for v in nbrs:
            for w in adj[v]:
                if w in nbr_set and w != u:
                    return False
    # Connected + diameter via BFS from each vertex
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = [src]
        qi = 0
        while qi < len(q):
            node = q[qi]
            qi += 1
            if dist[node] >= 4:
                continue
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    q.append(nb)
        if any(d == -1 for d in dist):
            return False
    return True

def _valid(added_edges):
    return _check(_build_adj(added_edges))

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill

    n = 24
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))

    # All valid candidate edges (even-odd, not in base)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u % 2) != (v % 2) and (u, v) not in base_edges:
                candidates.append((u, v))

    # Initial solution: hub at 12
    hub = 12
    initial = []
    for v in range(n):
        if v != hub and v % 2 != hub % 2 and (min(hub, v), max(hub, v)) not in base_edges:
            initial.append([hub, v])

    best = [list(e) for e in initial]
    best_adj = _build_adj(best)
    if not _check(best_adj):
        # Should not happen, but fallback
        return best

    best_len = len(best)

    # Local search: try removing one edge and adding one different edge
    # Repeat until no improvement or time runs out
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for i in range(best_len):
            if (time.time() - start_time) > deadline:
                break
            removed = best[i]
            remaining = [e for j, e in enumerate(best) if j != i]
            rem_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in remaining)
            for cand in candidates:
                if cand in rem_set:
                    continue
                if cand == (min(removed[0], removed[1]), max(removed[0], removed[1])):
                    continue
                test = remaining + [[cand[0], cand[1]]]
                if _valid(test):
                    best = test
                    best_adj = _build_adj(best)
                    best_len = len(best)
                    improved = True
                    break
        # Also try pure removal (if valid without the edge)
        if (time.time() - start_time) > deadline:
            break
        for i in range(best_len):
            remaining = [e for j, e in enumerate(best) if j != i]
            if _valid(remaining):
                best = remaining
                best_adj = _build_adj(best)
                best_len = len(best)
                improved = True
                break

    # Ensure return type is list of [int, int]
    result = [[int(e[0]), int(e[1])] for e in best]
    return result
# EVOLVE-BLOCK-END
