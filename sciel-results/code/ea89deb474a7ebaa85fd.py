"""Neighbour joining for phylogenetic topology recovery with time guard."""
import math
import time

_DEADLINE = 265.0
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _check_time():
    return _elapsed() < _DEADLINE


def _jc_distance(a, b):
    n = len(a)
    diffs = 0
    for i in range(n):
        if a[i] != b[i]:
            diffs += 1
    p = diffs / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def reconstruct(sequences):
    global _start_time
    _start_time = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 3:
        if n == 1:
            return labels[0] + ";"
        if n == 2:
            return f"({labels[0]},{labels[1]});"
        return f"({labels[0]},{labels[1]},{labels[2]});"
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i][j] = d
            dist[j][i] = d
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    d = [row[:] for row in dist]
    while len(nodes) > 3:
        if not _check_time():
            break
        m = len(nodes)
        r = [0.0] * m
        for i in range(m):
            for j in range(m):
                if i != j:
                    r[i] += d[nodes[i]][nodes[j]]
        best_q, best_i, best_j = float('inf'), -1, -1
        for i in range(m):
            for j in range(i + 1, m):
                q = (m - 2) * d[nodes[i]][nodes[j]] - r[i] - r[j]
                if q < best_q:
                    best_q, best_i, best_j = q, i, j
        a, b = nodes[best_i], nodes[best_j]
        new_idx = m
        trees[new_idx] = f"({trees[a]},{trees[b]})"
        rest = [nodes[k] for k in range(m) if k not in (best_i, best_j)]
        for x in rest:
            nd = 0.5 * (d[a][x] + d[b][x] - d[a][b])
            d[a][x] = d[x][a] = nd
        nodes = rest + [new_idx]
    return "(" + ",".join(trees[x] for x in nodes) + ");"
