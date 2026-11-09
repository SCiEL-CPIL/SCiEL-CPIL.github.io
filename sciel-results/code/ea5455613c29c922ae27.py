"""Fewest 4-cycles in graphs with 31 vertices and 91 edges.

Strategy:
  1. Build the polarity graph of PG(2,5) (31 vertices, 90 edges, 0 four-cycles).
  2. Add the single non-edge with the fewest common neighbors (>= 2 by the
     extremal bound ex(31;C4)=90).  A non-edge with exactly 2 common neighbors
     creates exactly C(2,2) = 1 new four-cycle, so the total is 1.
  3. Fallback: if the polarity construction is imperfect, run a bounded
     local search to minimise the four-cycle count.
"""
import time
import random
import itertools

import numpy as np

DEADLINE = 150.0
SAFETY_MARGIN = 8.0
N = 31
M_EDGES = 91


# --------------------------------------------------------------------------- #
#  Four-cycle counting (exact, fast via adjacency-bitset)                     #
# --------------------------------------------------------------------------- #
def _count_c4(adj):
    """adj: list of int bitsets.  Return the number of 4-cycles (subgraphs)."""
    total = 0
    n = len(adj)
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                c = common.bit_count()
                if c >= 2:
                    total += c * (c - 1) // 2
    return total // 2  # each C4 counted once per its two opposite-vertex pairs


# --------------------------------------------------------------------------- #
#  Polarity graph of PG(2,5)                                                 #
# --------------------------------------------------------------------------- #
def _pg2_points(q):
    """All 1-dim subspaces of GF(q)^3 as representative vectors (list of tuples)."""
    pts = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == b == c == 0:
                    continue
                # normalise: make first non-zero coordinate 1
                if a != 0:
                    rep = (1, b * pow(a, q - 2, q), c * pow(a, q - 2, q))
                elif b != 0:
                    rep = (0, 1, c * pow(b, q - 2, q))
                else:
                    rep = (0, 0, 1)
                if rep not in seen:
                    seen.add(rep)
                    pts.append(rep)
    return pts


def _polarity_graph(q):
    """Return (adj_bitsets, edge_list) for the polarity graph of PG(2,q)."""
    pts = _pg2_points(q)
    n = len(pts)
    # symmetric bilinear form matrix (a standard polarity for q odd)
    # M = [[0,0,1],[0,1,0],[1,0,0]]
    def B(p, r):
        return (p[0] * r[2] + p[1] * r[1] + p[2] * r[0]) % q

    adj = [0] * n
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if B(pts[i], pts[j]) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append([i, j])
    return adj, edges


# --------------------------------------------------------------------------- #
#  Local search (bounded, used as a safety net)                              #
# --------------------------------------------------------------------------- #
def _local_search(adj, edges, budget):
    """Try to reduce the C4 count via edge swaps.  Returns (best_adj, best_edges)."""
    n = len(adj)
    cur_c4 = _count_c4(adj)
    best_adj = list(adj)
    best_edges = [list(e) for e in edges]
    best_c4 = cur_c4
    edge_set = set((min(e), max(e)) for e in edges)
    rng = random.Random(12345)

    start = time.time()
    while time.time() < budget:
        # pick a random edge to remove
        e = rng.choice(edges)
        u, v = e
        # candidate non-edges to add
        # prefer pairs that are "far" (few common neighbours) to keep C4 low
        candidates = []
        for _ in range(200):
            a = rng.randrange(n)
            b = rng.randrange(n)
            if a == b:
                continue
            a, b = (a, b) if a < b else (b, a)
            if (a, b) in edge_set:
                continue
            if a == u and b == v:
                continue
            common = (adj[a] & adj[b]).bit_count()
            candidates.append((common, a, b))
        if not candidates:
            continue
        candidates.sort()
        # try the best few
        for common, a, b in candidates[:3]:
            # perform swap: remove (u,v), add (a,b)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            edge_set.discard((u, v))
            edge_set.add((a, b))
            new_c4 = _count_c4(adj)
            if new_c4 < best_c4:
                best_c4 = new_c4
                best_adj = list(adj)
                best_edges = [list(e) for e in edge_set]
            # revert
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            edge_set.add((u, v))
            edge_set.discard((a, b))
    return best_adj, best_edges


# --------------------------------------------------------------------------- #
#  Entry point                                                               #
# --------------------------------------------------------------------------- #
def construct():
    start = time.time()
    deadline = start + DEADLINE - SAFETY_MARGIN

    # 1. Polarity graph (90 edges, 0 C4).
    adj, edges = _polarity_graph(5)
    # sanity: should be 90 edges
    if len(edges) != 90:
        # fall back to a random-ish construction if polarity misbehaves
        # (should not happen for q=5)
        pass

    cur_c4 = _count_c4(adj)

    # 2. Find the non-edge with the minimum number of common neighbours.
    best_common = 10 ** 9
    best_pair = None
    edge_set = set((min(e), max(e)) for e in edges)
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in edge_set:
                continue
            common = (adj[i] & adj[j]).bit_count()
            if common < best_common:
                best_common = common
                best_pair = (i, j)

    if best_pair is not None:
        u, v = best_pair
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges.append([u, v])

    final_c4 = _count_c4(adj)

    # 3. Bounded local search as a safety net (only if we have time and
    #    the count is not already minimal).
    if final_c4 > 1 and time.time() < deadline:
        b_adj, b_edges = _local_search(adj, edges, deadline)
        b_c4 = _count_c4(b_adj)
        if b_c4 < final_c4:
            adj, edges = b_adj, b_edges
            final_c4 = b_c4

    # Return exactly 91 distinct edges as [u, v] lists.
    result = [list(e) for e in edges]
    # deduplicate / normalise just in case
    seen = set()
    out = []
    for e in result:
        a, b = e
        if a == b:
            continue
        a, b = (a, b) if a < b else (b, a)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append([a, b])
    # pad if (unexpectedly) short with random non-edges
    if len(out) < M_EDGES:
        rng = random.Random(7)
        while len(out) < M_EDGES:
            a = rng.randrange(N)
            b = rng.randrange(N)
            if a == b:
                continue
            a, b = (a, b) if a < b else (b, a)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            out.append([a, b])
    return out[:M_EDGES]
