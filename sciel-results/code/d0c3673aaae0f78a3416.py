"""Reconstruct unrooted phylogeny from Jukes-Cantor-simulated DNA."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 250.0  # wall-clock safety margin (evaluator kills at 270s)
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def _jc_matrix(seq_array):
    """Pairwise Jukes-Cantor-corrected distances (symmetric, zero diagonal)."""
    code = (seq_array == ord('A')).astype(np.int8) * 1 \
        + (seq_array == ord('C')).astype(np.int8) * 2 \
        + (seq_array == ord('G')).astype(np.int8) * 4 \
        + (seq_array == ord('T')).astype(np.int8) * 8
    diff = (code[:, None] != code[None, :]).astype(np.float64)
    p = diff / seq_array.shape[1]
    np.fill_diagonal(p, 0.0)
    p = np.clip(p, 0.0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(d, 0.0)
    return d


def _nj_build(d0, n):
    """Vectorised Neighbour-Joining. Returns (joins, final3)."""
    size = 2 * n
    dist = np.zeros((size, size), dtype=np.float64)
    dist[:n, :n] = d0
    active = list(range(n))
    joins = []
    node_count = n
    while len(active) > 3:
        m = len(active)
        sub = dist[active][:, active]
        r = sub.sum(axis=1) / (m - 2)
        q = (m - 2) * sub - r[:, None] - r[None, :]
        qmask = np.full((m, m), np.inf)
        np.fill_diagonal(qmask, np.inf)
        qm = np.where(q < qmask, q, qmask)
        idx = np.unravel_index(np.argmin(qm), qm.shape)
        if idx[0] > idx[1]:
            idx = (idx[1], idx[0])
        i, j = idx
        a = active[i]
        b = active[j]
        dij = dist[a, b]
        la = 0.5 * dij + (r[i] - r[j]) / (2.0 * (m - 2))
        lb = dij - la
        if la < 0:
            la = 0.0
        if lb < 0:
            lb = 0.0
        new = node_count
        node_count += 1
        for x in active:
            if x == a or x == b:
                continue
            val = 0.5 * (dist[a, x] + dist[b, x] - dij)
            if val < 0:
                val = 0.0
            dist[new, x] = val
            dist[x, new] = val
        active = [x for x in active if x != a and x != b]
        active.append(new)
        joins.append((a, b, la, lb))
    return joins, list(active)


def _reconstruct_tree(joins, final3, labels):
    n_taxa = len(labels)
    edges = []
    node_count = n_taxa
    for (a, b, la, lb) in joins:
        new = node_count
        node_count += 1
        edges.append((a, new, la))
        edges.append((b, new, lb))
    center = node_count
    node_count += 1
    for x in final3:
        edges.append((x, center, 0.0))
    adj = {i: [] for i in range(node_count)}
    for (u, v, w) in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    import sys
    sys.setrecursionlimit(10000)
    out = []

    def dfs(u, parent):
        children = [v for (v, w) in adj[u] if v != parent]
        if u < n_taxa:
            out.append(labels[u])
            return
        if not children:
            out.append("()")
            return
        parts = [dfs(v, u) for v in children]
        out.append("(" + ",".join(parts) + ")")

    dfs(center, -1)
    return out[0] + ";"


def _fallback_star(labels):
    return "(" + ",".join(labels) + ");"


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n < 2:
        return "(" + ",".join(labels) + ");"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    if n == 3:
        return "(" + ",".join(labels) + ");"
    L = len(sequences[labels[0]])
    seq_array = np.empty((n, L), dtype=np.uint8)
    for i, lab in enumerate(labels):
        seq_array[i] = np.frombuffer(sequences[lab].encode('ascii'), dtype=np.uint8)
    d = _jc_matrix(seq_array)
    if _elapsed() > _DEADLINE:
        return _fallback_star(labels)
    joins, final3 = _nj_build(d, n)
    if _elapsed() > _DEADLINE:
        return _fallback_star(labels)
    try:
        return _reconstruct_tree(joins, final3, labels)
    except Exception:
        return _fallback_star(labels)
# EVOLVE-BLOCK-END
