"""Rewritten: analytic spectral radius for prism lifts + deadline-guarded local search."""
import time
import numpy as np


def _edge_indices(m, edges):
    """Map each (u,v) edge to a structural index: 0..m-1 cycle0, m..2m-1 cycle1, 2m..3m-1 matching."""
    idx = []
    for (u, v) in edges:
        if u < m and v < m:
            idx.append((v - u) % m)
        elif u >= m and v >= m:
            idx.append(m + ((v - m) - (u - m)) % m)
        else:
            idx.append(2 * m + min(u, v) % m)
    return idx


def _cycle_prod(sgn, m):
    p = 1
    for a in sgn[:m]:
        p *= a
    return p


def _spectral_radius(sgn, m):
    """Max |eigenvalue| of the signed adjacency of the prism lift (block-circulant)."""
    c1 = _cycle_prod(sgn, m)
    c2 = _cycle_prod(sgn[m:], m)
    c = sgn[2 * m:]
    best = 0.0
    for k in range(m):
        th = 2.0 * np.pi * k / m
        ck = c1 * np.cos(th) + c2 * np.sin(th)
        for j in range(m):
            ph = 2.0 * np.pi * j / m
            cj = c[j] * np.cos(ph)
            val = ck + cj
            if val < 0.0:
                val = -val
            if val > best:
                best = val
    return best


def _is_connected(sgn, m):
    """Check the two-lift is connected (single connected component)."""
    m2 = 2 * m
    adj = [[] for _ in range(m2)]
    for k in range(m):
        a, b = k, (k + 1) % m
        s = sgn[k]
        adj[a].append(b if s > 0 else b ^ 1)
        adj[b].append(a if s > 0 else a ^ 1)
    for k in range(m):
        a, b = m + k, m + ((k + 1) % m)
        s = sgn[m + k]
        adj[a].append(b if s > 0 else b ^ 1)
        adj[b].append(a if s > 0 else a ^ 1)
    for k in range(m):
        a, b = k, m + k
        s = sgn[2 * m + k]
        adj[a].append(b if s > 0 else b ^ 1)
        adj[b].append(a if s > 0 else a ^ 1)
    seen = [False] * m2
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    return cnt == m2


def sign_edges(graph, budget):
    m = graph["n"] // 2
    edges = graph["edges"]
    idx = _edge_indices(m, edges)

    t0 = time.time()
    deadline = t0 + 55.0  # safety margin under the 90s wall-clock kill

    # ---- seed candidates: all-+1, all -1, random ----
    rng = np.random.default_rng(3)
    seeds = [np.ones(3 * m, dtype=np.int8), np.ones(3 * m, dtype=np.int8) * -1]
    for _ in range(64):
        seeds.append(rng.choice(np.array([-1, 1], dtype=np.int8), size=3 * m))

    best = None
    best_r = float("inf")
    for sgn in seeds:
        if not _is_connected(sgn, m):
            continue
        r = _spectral_radius(sgn, m)
        if r < best_r:
            best_r = r
            best = sgn.copy()

    # ---- local search: flip single edges ----
    if best is not None:
        cur = best.copy()
        cur_r = best_r
        for it in range(2000000):
            if time.time() > deadline:
                break
            e = rng.integers(0, 3 * m)
            cur[e] = -cur[e]
            if not _is_connected(cur, m):
                cur[e] = -cur[e]
                continue
            r = _spectral_radius(cur, m)
            if r < cur_r - 1e-12:
                cur_r = r
                best = cur.copy()
                best_r = r
            else:
                cur[e] = -cur[e]

    # ---- map structural signs back to the given edge order ----
    out = np.empty(len(edges), dtype=np.int8)
    for i, k in enumerate(idx):
        out[i] = best[k]
    return out
