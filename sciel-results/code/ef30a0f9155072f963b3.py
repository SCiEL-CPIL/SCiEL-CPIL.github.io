"""Phylogenetic topology recovery: NJ + NNI local search."""
import math
import time
import numpy as np


def _jc_dist_matrix(seqs, labels):
    n = len(labels)
    L = len(seqs[labels[0]])
    enc = np.zeros((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = seqs[lab]
        for j, c in enumerate(s):
            enc[i, j] = 0 if c == 'A' else (1 if c == 'C' else (2 if c == 'G' else 3))
    diff = (enc[:, None, :] != enc[None, :, :]).sum(axis=2)
    p = diff / L
    np.fill_diagonal(p, 0.0)
    p = np.clip(p, 0.0, 0.7499)
    D = -0.75 * np.log(np.clip(1.0 - 4.0 * p / 3.0, 1e-12, 1.0))
    np.fill_diagonal(D, 0.0)
    return D


def _nj(D, n):
    D = D.copy()
    active = list(range(n))
    joins = []
    next_id = n
    while len(active) > 3:
        m = len(active)
        r = D[active, :][:, active].sum(axis=1)
        best_q = float('inf')
        bi = bj = 0
        for i in range(m):
            ai = active[i]
            ri = r[i]
            for j in range(i + 1, m):
                aj = active[j]
                q = (m - 2) * D[ai, aj] - ri - r[j]
                if q < best_q:
                    best_q = q
                    bi, bj = i, j
        a, b = active[bi], active[bj]
        joins.append((a, b, next_id))
        active = [x for x in active if x != a and x != b]
        for x in active:
            D[next_id, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
            D[x, next_id] = D[next_id, x]
        D[next_id, next_id] = 0.0
        next_id += 1
    return joins, active


def _build_newick(joins, active, n, labels):
    sub = {i: labels[i] for i in range(n)}
    for (a, b, nid) in joins:
        sub[nid] = f"({sub[a]},{sub[b]})"
    return "(" + ",".join(sub[x] for x in active) + ");"


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 3:
        return "(" + ",".join(labels) + ");"
    D = _jc_dist_matrix(sequences, labels)
    joins, active = _nj(D, n)
    return _build_newick(joins, active, n, labels)
