"""Fresh design: exact spectral model + fast exhaustive search for prism two-lifts."""
import time
import numpy as np
from itertools import product

_TIME_LIMIT = 80.0  # wall-clock safety margin below the 90 s kill


def _prism_structure(edges):
    """Return (m, top_edges, bot_edges, rung_edges) with each a list of (u, v)."""
    n = len(edges)
    m = n // 3
    top, bot, rung = [], [], []
    for (u, v) in edges:
        if u < m and v < m:
            top.append((u, v))
        elif u >= m and v >= m:
            bot.append((u - m, v - m))
        else:
            i = u if u < m else v
            rung.append((i, i + m))
    return m, top, bot, rung


def _signed_radius(m, t, u, v):
    """Exact spectral radius of the base signed adjacency for given cycle products.

    t = product of top cycle signs, u = product of bottom cycle signs,
    v = product of rung signs.  The lift eigenvalues are those of A+D and A-D.
    """
    c_rung = 1.0 if v == 1 else -1.0
    if m % 2 == 0:
        k = m // 2
        base = np.cos(np.pi * k / m)  # = 0
        r_plus = np.hypot(2.0, 2.0 * base)
        r_minus = np.hypot(2.0 * c_rung, 2.0 * base)
    else:
        k = (m - 1) // 2
        base = np.cos(np.pi * k / m)
        s = np.sin(np.pi * k / m)
        r_plus = np.hypot(2.0, 2.0 * base) + 2.0 * s
        r_minus = np.hypot(2.0 * c_rung, 2.0 * base) + 2.0 * s
    return max(r_plus, r_minus)


def _true_radius(graph, signs):
    """Spectral radius of the base signed adjacency (max |eig| of A+D and A-D)."""
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n))
    D = np.zeros((n, n))
    for (a, b), s in zip(edges, signs):
        A[a, b] = 1.0
        A[b, a] = 1.0
        D[a, b] = float(s)
        D[b, a] = float(s)
    r = 0.0
    for M in (A + D, A - D):
        w = np.linalg.eigvalsh(M)
        r = max(r, float(np.max(np.abs(w))))
    return r


def _is_connected_lift(m, top, bot, rung, signs, top_idx, bot_idx, rung_idx):
    """Check the two-lift is connected (single component)."""
    # Build lift graph on 2m*2 vertices.
    N = 2 * m
    adj = [[] for _ in range(2 * N)]
    def add(a, b):
        adj[a].append(b); adj[b].append(a)
    for (a, b) in top:
        s = signs[top_idx[(a, b)]]
        if s == 1:
            add(a, b); add(N + a, N + b)
        else:
            add(a, N + b); add(N + a, b)
    for (a, b) in bot:
        s = signs[bot_idx[(a, b)]]
        if s == 1:
            add(N + a, N + b); add(a, b)
        else:
            add(N + a, b); add(a, N + b)
    for (i, j) in rung:
        s = signs[rung_idx[i]]
        if s == 1:
            add(i, N + i); add(N + i, i)
        else:
            add(i, i); add(N + i, N + i)
    # BFS
    seen = [False] * (2 * N)
    seen[0] = True
    stack = [0]
    cnt = 1
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if not seen[y]:
                seen[y] = True
                cnt += 1
                stack.append(y)
    return cnt == 2 * N


def sign_edges(graph, budget):
    edges = graph['edges']
    n = len(edges)
    m, top, bot, rung = _prism_structure(edges)
    start = time.time()

    # Edge index maps.
    top_idx, bot_idx, rung_idx = {}, {}, {}
    for (u, v) in top:
        top_idx[(u, v)] = edges.index((u, v))
    for (u, v) in bot:
        bot_idx[(u, v)] = edges.index((m + u, m + v))
    for (i, j) in rung:
        rung_idx[i] = edges.index((i, j))

    # Exhaustive search over the 8 distinct (t, u, v) cycle-product combos.
    best = None
    best_r = float('inf')
    for t, u, v in product([1, -1], repeat=3):
        r = _signed_radius(m, t, u, v)
        if r < best_r:
            best_r = r
            best = (t, u, v)

    def make_signs(t, u, v):
        s = np.ones(n, dtype=np.int8)
        for (a, b) in top[:-1]:
            s[top_idx[(a, b)]] = 1
        s[top_idx[top[-1]]] = t
        for (a, b) in bot[:-1]:
            s[bot_idx[(a, b)]] = 1
        s[bot_idx[bot[-1]]] = u
        for (i, j) in rung[:-1]:
            s[rung_idx[i]] = 1
        s[rung_idx[rung[-1][0]]] = v
        return s

    # Try each of the 8 combos, keep the valid one with smallest true radius.
    best_signs = None
    best_true = float('inf')
    for t, u, v in product([1, -1], repeat=3):
        s = make_signs(t, u, v)
        if not _is_connected_lift(m, top, bot, rung, s, top_idx, bot_idx, rung_idx):
            continue
        r = _true_radius(graph, s)
        if r < best_true:
            best_true = r
            best_signs = s

    if best_signs is None:
        # Fallback: brute force a connected valid lift.
        rng = np.random.default_rng(3)
        for _ in range(budget):
            if time.time() - start > _TIME_LIMIT:
                break
            s = rng.choice([-1, 1], size=n)
            if _is_connected_lift(m, top, bot, rung, s, top_idx, bot_idx, rung_idx):
                best_signs = s
                break

    if best_signs is None:
        # Last resort: seed baseline.
        from autoresearch_suite.science_tcs_expansion import lift_baseline
        best_signs = lift_baseline(graph, budget)

    return best_signs
