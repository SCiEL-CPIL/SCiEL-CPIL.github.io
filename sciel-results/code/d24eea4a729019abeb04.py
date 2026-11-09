"""C6-free graph on 40 vertices: finite-field C4-free bipartite base + greedy edge addition."""
import time
import random

N = 40
DEADLINE = 145.0  # safety margin under 150s wall-clock


def _has_c6(adj, n):
    """Return True if graph (adjacency bitmasks) contains a C6."""
    # L2[i][j] = number of common neighbors of i and j
    L2 = [[0] * n for _ in range(n)]
    for x in range(n):
        ax = adj[x]
        ys = []
        while ax:
            lsb = ax & (-ax)
            ys.append(lsb.bit_length() - 1)
            ax ^= lsb
        for i in range(len(ys)):
            yi = ys[i]
            row = L2[yi]
            for j in range(i + 1, len(ys)):
                yj = ys[j]
                row[yj] += 1
                L2[yj][yi] += 1
    # A C6 exists iff some non-adjacent pair (a,d) has >=2 length-3 paths.
    allmask = (1 << n) - 1
    for a in range(n):
        da = adj[a]
        nd = allmask & ~(da | (1 << a))
        La = L2[a]
        while nd:
            lsb = nd & (-nd)
            d = lsb.bit_length() - 1
            cnt = 0
            cd = adj[d]
            while cd:
                lsb2 = cd & (-cd)
                c = lsb2.bit_length() - 1
                cnt += La[c]
                cd ^= lsb2
                if cnt >= 2:
                    return True
            nd ^= lsb
    return False


def _edge_count(adj, n):
    c = 0
    for i in range(n):
        c += adj[i].bit_count()
    return c // 2


def _build_base():
    """C4-free bipartite graph (hence C6-free) via finite-field squares, then densify."""
    adj = [0] * N
    p = 19  # prime
    half = 20
    # vertices 0..19 = U, 20..39 = V
    for i in range(half):
        # two quadratic residues mod p
        for c in (0, 1):
            j = (i * i + c) % p
            if j < half:
                u = i
                v = half + j
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj


def _greedy_add(adj, n, deadline, seed):
    """Repeatedly add any edge that preserves C6-freeness, in random order."""
    random.seed(seed)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)
        for i, j in pairs:
            if adj[i] & (1 << j):
                continue
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            if not _has_c6(adj, n):
                improved = True
            else:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
    return adj


def construct():
    start = time.time()
    deadline = start + DEADLINE
    adj = _build_base()
    # Several randomized greedy passes with different seeds to escape local structure
    for seed in range(50):
        if time.time() > deadline:
            break
        _greedy_add(adj, N, deadline, seed)
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges
