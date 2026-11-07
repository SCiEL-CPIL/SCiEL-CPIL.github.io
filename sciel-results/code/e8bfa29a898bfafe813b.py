"""Spectral graph lifts: analytic sign pattern + fast polish."""
# EVOLVE-BLOCK-START
import time

import numpy as np

_DEADLINE = 80.0          # safety margin well under the 90 s wall-clock kill
_START = time.monotonic()


def _signed_adjacency(graph, signs):
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n), dtype=np.float64)
    for k, (u, v) in enumerate(edges):
        s = float(signs[k])
        A[u, v] = s
        A[v, u] = s
    return A


def _spectral_radius(A):
    return float(np.max(np.abs(np.linalg.eigvalsh(A))))


def _is_connected_lift(graph, signs):
    """Lift connectivity via BFS over the 2n lifted vertices."""
    n = graph['n']
    edges = graph['edges']
    N = 2 * n
    adj = [[] for _ in range(N)]
    for k, (u, v) in enumerate(edges):
        if signs[k] > 0:
            adj[u * 2].append(v * 2)
            adj[v * 2].append(u * 2)
            adj[u * 2 + 1].append(v * 2 + 1)
            adj[v * 2 + 1].append(u * 2 + 1)
        else:
            adj[u * 2].append(v * 2 + 1)
            adj[v * 2 + 1].append(u * 2)
            adj[u * 2 + 1].append(v * 2)
            adj[v * 2].append(u * 2 + 1)
    seen = [False] * N
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        cur = stack.pop()
        for nb in adj[cur]:
            if not seen[nb]:
                seen[nb] = True
                cnt += 1
                stack.append(nb)
    return cnt == N


def _score(graph, signs):
    if not _is_connected_lift(graph, signs):
        return None
    return _spectral_radius(_signed_adjacency(graph, signs))


def _analytic_signs(graph):
    """Uniform pattern: bottom +1, matching +1, top -1  (guarantees connectivity)."""
    n = graph['n']
    m = n // 2
    signs = []
    for k, (u, v) in enumerate(graph['edges']):
        if u < m and v < m:          # bottom cycle
            signs.append(1)
        elif u >= m and v >= m:      # top cycle
            signs.append(-1)
        else:                        # matching
            signs.append(1)
    return np.array(signs, dtype=np.int64)


def _hill_climb(graph, signs, max_iter=60000):
    signs = np.array(signs, dtype=np.int64)
    sc = _score(graph, signs)
    if sc is None:
        return None, None
    best_signs = signs.copy()
    best_sc = sc
    n = len(signs)
    for _ in range(max_iter):
        if time.monotonic() - _START > _DEADLINE - 1.0:
            break
        improved = False
        for i in range(n):
            if signs[i] == 1:
                ns = -1
            else:
                ns = 1
            signs[i] = ns
            s = _score(graph, signs)
            if s is not None and s < best_sc - 1e-12:
                best_sc = s
                best_signs = signs.copy()
                improved = True
            else:
                signs[i] = -ns  # revert
        if not improved:
            break
    return best_signs, best_sc


def sign_edges(graph, budget):
    # 1) Analytic pattern (valid, low spectral radius)
    signs = _analytic_signs(graph)
    sc = _score(graph, signs)

    # 2) Fast hill-climb polish from the analytic start
    best_signs, best_sc = _hill_climb(graph, signs)
    if best_sc is not None:
        signs = best_signs
        sc = best_sc
    elif sc is not None:
        pass  # keep analytic

    # 3) Fallback to the seed baseline if the analytic pattern is somehow invalid
    if sc is None:
        from autoresearch_suite.science_tcs_expansion import lift_baseline
        return lift_baseline(graph, budget)

    # 4) Final safety: verify validity; fall back if it fails
    if not _is_connected_lift(graph, signs):
        from autoresearch_suite.science_tcs_expansion import lift_baseline
        return lift_baseline(graph, budget)

    if time.monotonic() - _START > _DEADLINE:
        from autoresearch_suite.science_tcs_expansion import lift_baseline
        return lift_baseline(graph, budget)

    return signs
# EVOLVE-BLOCK-END
