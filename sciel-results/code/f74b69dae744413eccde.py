"""C6-free graph on 40 vertices: greedy seed + local-search edge swaps."""
import time
import random

_N = 40
_DEADLINE = 150.0
_MARGIN = 6.0


def _has_c6(mat, cap=2):
    """True iff some pair has more than `cap` common neighbours (=> a 6-cycle)."""
    n = len(mat)
    for i in range(n):
        row = mat[i]
        for j in range(i + 1, n):
            if row[j] > cap:
                return True
    return False


def _build_seed():
    """5-partite Turan (no 4-cycle) + a spanning path -> large C6-free graph."""
    n = _N
    groups = [list(range(8 * g, 8 * g + 8)) for g in range(5)]
    mat = [[0] * n for _ in range(n)]
    edges = []
    # Turan: all edges between different groups
    for a in range(5):
        for b in range(a + 1, 5):
            for u in groups[a]:
                for v in groups[b]:
                    mat[u][v] += 1
                    mat[v][u] += 1
                    edges.append([u, v])
    # spanning path
    for i in range(n - 1):
        u, v = i, i + 1
        if mat[u][v] == 0:
            mat[u][v] += 1
            mat[v][u] += 1
            edges.append([u, v])
    return mat, edges


def _local_search(mat, edges, rng, deadline):
    n = _N
    m = len(edges)
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v) if u < v else (v, u))
    best_m = m
    best_mat = [row[:] for row in mat]
    best_eset = set(edge_set)
    # missing edges (potential gains)
    missing = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edge_set]
    rng.shuffle(missing)
    idx = 0
    stall = 0
    while time.time() < deadline:
        if idx >= len(missing):
            if stall > 4000:
                break
            idx = 0
            rng.shuffle(missing)
            stall = 0
            continue
        u, v = missing[idx]
        idx += 1
        if (u, v) in edge_set:
            continue
        # try to add (u, v); if it creates a C6, try to fix by removing one edge
        mat[u][v] += 1
        mat[v][u] += 1
        if _has_c6(mat):
            mat[u][v] -= 1
            mat[v][u] -= 1
            # collect candidate removals (edges adjacent to u or v)
            rem = []
            for w in range(n):
                if w == u or w == v:
                    continue
                if (u, w) in edge_set or (v, w) in edge_set:
                    rem.append((u, w) if (u, w) in edge_set else (v, w))
            rng.shuffle(rem)
            added = False
            for a, b in rem:
                if (a, b) not in edge_set:
                    continue
                mat[a][b] -= 1
                mat[b][a] -= 1
                if not _has_c6(mat):
                    # accept swap: remove (a,b), keep (u,v)
                    edge_set.discard((a, b))
                    edge_set.add((u, v))
                    missing.remove((u, v))
                    added = True
                    break
                mat[a][b] += 1
                mat[b][a] += 1
            if added:
                if m > best_m:
                    best_m = m
                    best_mat = [row[:] for row in mat]
                    best_eset = set(edge_set)
                stall = 0
            else:
                stall += 1
        else:
            # plain add
            edge_set.add((u, v))
            missing.remove((u, v))
            m += 1
            if m > best_m:
                best_m = m
                best_mat = [row[:] for row in mat]
                best_eset = set(edge_set)
            stall = 0
    return best_mat, best_eset, best_m


def construct():
    start = time.time()
    deadline = start + _DEADLINE - _MARGIN
    rng = random.Random(12345)
    mat, edges = _build_seed()
    best_mat, best_eset, best_m = _local_search(mat, edges, rng, deadline)
    # restart from best to escape local plateaus
    while time.time() < deadline:
        mat2 = [row[:] for row in best_mat]
        eset2 = set(best_eset)
        mat2b, eset2b, mb = _local_search(mat2, list(eset2), rng, deadline)
        if mb > best_m:
            best_mat, best_eset, best_m = mat2b, eset2b, mb
    out = [[u, v] for (u, v) in best_eset if u < v]
    return out
