"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph.

Strategy:
  - The largest triangle-free induced subgraph of G equals the largest independent
    set of the "triangle-conflict" graph H, where H has an edge (u,v) iff u and v
    are adjacent in G and share a common neighbor in G.
  - Max independent set is solved exactly with a fast bit-parallel branch-and-bound.
  - Local search (edge add/remove) minimizes this value while keeping G K4-free.
"""
import time
import random

N = 18
DEADLINE = 115.0  # safety margin under 150s wall-clock


def _time_left(start):
    return DEADLINE - (time.time() - start)


def _is_k4_free(adj):
    """True iff G is K4-free (no K4 subgraph)."""
    for i in range(N):
        for j in range(i + 1, N):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j]
            c = common
            while c:
                lsb = c & (-c)
                k = lsb.bit_length() - 1
                c ^= lsb
                if adj[k] & (common & ~(1 << k)):
                    return False
    return True


def _triangle_conflict_adj(adj):
    """H[u] = bitmask of vertices v that form a triangle with u (u~v and share neighbor)."""
    H = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if (adj[i] & (1 << j)) and (adj[i] & adj[j]):
                H[i] |= (1 << j)
                H[j] |= (1 << i)
    return H


def _max_independent_set(H):
    """Exact max independent set size of graph H (bitmask adjacency), via branch-and-bound."""
    full = (1 << N) - 1
    best = 0

    def search(P, R):
        nonlocal best
        p = P.bit_count()
        if p == 0:
            if R.bit_count() > best:
                best = R.bit_count()
            return
        if R.bit_count() + p <= best:
            return
        # Branch on a vertex of max degree within P for strong pruning
        m = P
        best_v = -1
        best_d = -1
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            d = (H[v] & P).bit_count()
            if d > best_d:
                best_d = d
                best_v = v
            m ^= lsb
        v = best_v
        bv = 1 << v
        # Include v: neighbors of v cannot be in the independent set
        search(P & ~(H[v] | bv), R | bv)
        # Exclude v
        search(P & ~bv, R)

    search(full, 0)
    return best


def _score_fast(adj):
    return _max_independent_set(_triangle_conflict_adj(adj))


def _complement_of_3_regular():
    """Complement of a 3-regular circulant graph on 18 vertices (dense, often K4-free)."""
    adj = [0] * N
    for v in range(N):
        for d in (1, 2, 9):
            w = (v + d) % N
            if w != v:
                adj[v] |= (1 << w)
                adj[w] |= (1 << v)
    comp = [0] * N
    for u in range(N):
        for v in range(u + 1, N):
            if not (adj[u] & (1 << v)):
                comp[u] |= (1 << v)
                comp[v] |= (1 << u)
    return comp


def _turan_graph():
    """Balanced complete 4-partite graph (K4-free). Parts sizes 5,4,4,5."""
    adj = [0] * N
    sizes = [5, 4, 4, 5]
    part = [0] * N
    idx = 0
    for p, s in enumerate(sizes):
        for _ in range(s):
            part[idx] = p
            idx += 1
    for i in range(N):
        for j in range(i + 1, N):
            if part[i] != part[j]:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _random_k4_free(rng, p):
    """Random graph, then strip edges to become K4-free."""
    adj = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    changed = True
    while changed:
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                if not (adj[i] & (1 << j)):
                    continue
                common = adj[i] & adj[j]
                c = common
                has_edge = False
                while c:
                    lsb = c & (-c)
                    k = lsb.bit_length() - 1
                    c ^= lsb
                    if adj[k] & (common & ~(1 << k)):
                        has_edge = True
                        break
                if has_edge:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                    changed = True
    return adj


def _local_search(adj, start, rng):
    """Minimize score via single-edge add/remove moves. Returns (best_adj, best_score)."""
    best_adj = adj[:]
    best_score = _score_fast(adj)

    improved = True
    while improved and _time_left(start) > 30:
        improved = False
        # Try removing edges
        edges = [(i, j) for i in range(N) for j in range(i + 1, N) if adj[i] & (1 << j)]
        rng.shuffle(edges)
        for (u, v) in edges:
            if _time_left(start) < 30:
                break
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            s = _score_fast(adj)
            if s < best_score:
                best_score = s
                best_adj = adj[:]
                improved = True
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        # Try adding edges (must stay K4-free)
        non_edges = [(i, j) for i in range(N) for j in range(i + 1, N) if not (adj[i] & (1 << j))]
        rng.shuffle(non_edges)
        for (u, v) in non_edges:
            if _time_left(start) < 30:
                break
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if _is_k4_free(adj):
                s = _score_fast(adj)
                if s < best_score:
                    best_score = s
                    best_adj = adj[:]
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
    return best_adj, best_score


def construct():
    start = time.time()
    rng = random.Random(42)

    candidates = [
        _complement_of_3_regular(),
        _turan_graph(),
    ]
    for p in (0.3, 0.4, 0.5, 0.6, 0.7):
        candidates.append(_random_k4_free(rng, p))

    best_adj = None
    best_score = N + 1

    for adj in candidates:
        if _time_left(start) < 60:
            break
        if not _is_k4_free(adj):
            continue
        s = _score_fast(adj)
        if s < best_score:
            best_score = s
            best_adj = adj[:]

    if best_adj is None:
        # Fallback: Turan graph is guaranteed K4-free
        best_adj = _turan_graph()
        best_score = _score_fast(best_adj)

    # Local search from the best candidate
    best_adj, best_score = _local_search(best_adj, start, rng)

    # If time remains, try local search from a couple of other good seeds
    for adj in candidates:
        if _time_left(start) < 20:
            break
        if not _is_k4_free(adj):
            continue
        s0 = _score_fast(adj)
        if s0 >= best_score:
            continue
        a2, s2 = _local_search(adj[:], start, rng)
        if s2 < best_score:
            best_score = s2
            best_adj = a2

    result = []
    for i in range(N):
        for j in range(i + 1, N):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    return result


if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
