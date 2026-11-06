"""Model-and-optimize: even-factor / minimum cycle decomposition via matching."""
import time
import random
from itertools import combinations
import networkx as nx

_DEADLINE = time.time() + 140.0
_DEADLINE_MARGIN = 3.0


def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_one_cycle(G, start=None):
    """Find any cycle in G (undirected). Returns list of vertices or None."""
    from collections import deque
    if G.number_of_nodes() == 0:
        return None
    start = start if start is not None else next(iter(G))
    parent = {start: None}
    depth = {start: 0}
    q = deque([start])
    while q:
        v = q.popleft()
        for w in G.neighbors(v):
            if w not in depth:
                depth[w] = depth[v] + 1
                parent[w] = v
                q.append(w)
    # find an edge (u,v) where depth[v] < depth[u]-1 (non-tree edge)
    for u in G:
        for v in G.neighbors(u):
            if u < v and v in depth and u in depth and depth[v] < depth[u] - 1:
                # reconstruct cycle: u up to v, then edge v-u
                path = []
                cur = u
                while cur != v:
                    path.append(cur)
                    cur = parent[cur]
                path.append(v)
                return path
    return None


def _decompose_even(G):
    """Decompose an even-degree subgraph G into cycles. Returns list of cycles."""
    cycles = []
    remaining = G.copy()
    while remaining.number_of_edges() > 0:
        c = _find_one_cycle(remaining)
        if c is None:
            # shouldn't happen for even subgraph, break to avoid infinite loop
            break
        for i in range(len(c)):
            u, v = c[i], c[(i + 1) % len(c)]
            remaining.remove_edge(u, v)
        cycles.append(c)
    return cycles


def _heuristic_evenfactor(G):
    """Find a large even-degree spanning subgraph via greedy cycle removal."""
    cur = G.copy()
    # Greedy: find cycles, remove them until no cycles remain
    while time.time() < _DEADLINE - 5:
        c = _find_one_cycle(cur)
        if c is None:
            break
        for i in range(len(c)):
            u, v = c[i], c[(i + 1) % len(c)]
            cur.remove_edge(u, v)
    return cur


def _refine_with_matching(G, initial_even):
    """Refine even-factor using min-weight perfect matching on Tutte graph.
    Returns (improved_even_subgraph, cost_reduction)."""
    n = G.number_of_nodes()
    # Odd-degree vertices in current even-subgraph
    odd = [v for v in initial_even if initial_even.degree(v) % 2 == 1]
    if not odd:
        return initial_even, 0
    # Cost matrix: cost(u,v) = 1 if (u,v) in G else INF
    INF = 10**9
    cost = {}
    for (u, v) in G.edges():
        cost[(u, v)] = cost[(v, u)] = 1
    # Tutte graph: n original + n dummy. Original i connects to dummy n+i with weight 0.
    # Original u-v edge with weight w(u,v).
    # Find min-weight perfect matching; edges matched to dummies = odd set.
    # Build full weight matrix
    N = 2 * n
    W = [[0] * N for _ in range(N)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            W[i][j] = cost.get((i, j), INF)
        W[i][n + i] = 0
        W[n + i][i] = 0
    # dummy-dummy edges: large (shouldn't be used)
    for i in range(n):
        for j in range(n):
            W[n + i][n + j] = INF if i != j else 0
    try:
        res = nx.algorithms.matching.minimum_weight_perfect_matching(W)
    except Exception:
        return initial_even, 0
    # Determine odd set: vertices i matched to a dummy (n + j)
    odd_set = set()
    for (a, b) in res:
        if a < n and b >= n:
            odd_set.add(a)
        elif b < n and a >= n:
            odd_set.add(b)
    # Build new even subgraph: start from initial_even, flip edges to match parity
    newG = nx.Graph()
    newG.add_nodes_from(G.nodes())
    for e in initial_even.edges():
        newG.add_edge(*e)
    # For each original-original matched edge, remove it (toggles both endpoints' parity)
    for (a, b) in res:
        if a < n and b < n and a != b:
            if newG.has_edge(a, b):
                newG.remove_edge(a, b)
    # Verify and fix: any vertex with wrong parity, find edge to flip.
    for v in newG:
        target = 1 if v in odd_set else 0
        cur = newG.degree(v) % 2
        if cur != target:
            nb = list(newG.neighbors(v))
            if nb:
                newG.remove_edge(v, nb[0])
    # Final check: ensure even-degree subgraph (all even). If not, drop problem edges.
    for v in list(newG.nodes()):
        while newG.degree(v) % 2 == 1:
            nb = list(newG.neighbors(v))
            if not nb:
                break
            newG.remove_edge(v, nb[0])
    return newG, 0


def _to_pieces(G, even_sub):
    """Produce pieces: cycles from even_sub, single edges from the rest."""
    pieces = []
    edges_in_even = set(tuple(sorted(e)) for e in even_sub.edges())
    for c in _decompose_even(even_sub):
        pieces.append(c)
    for (u, v) in G.edges():
        e = tuple(sorted((u, v)))
        if e not in edges_in_even:
            pieces.append([u, v])
    return pieces


def _validate(G, pieces):
    """Check pieces are edge-disjoint and cover all edges exactly."""
    seen = set()
    for p in pieces:
        k = len(p)
        if k == 2:
            e = tuple(sorted(p))
        elif k >= 3:
            for i in range(k):
                e = tuple(sorted((p[i], p[(i + 1) % k])))
                if e in seen:
                    return False
                seen.add(e)
            continue
        else:
            return False
        if e in seen:
            return False
        seen.add(e)
    all_edges = {tuple(sorted(e)) for e in G.edges()}
    return seen == all_edges


def _solve_one(G):
    m = G.number_of_edges()
    # Baseline: all single edges
    best_pieces = [[u, v] for (u, v) in G.edges()]
    best_count = m

    # Heuristic even-factor
    even_sub = None
    try:
        even_sub = _heuristic_evenfactor(G)
        pieces = _to_pieces(G, even_sub)
        if _validate(G, pieces):
            if len(pieces) < best_count:
                best_count = len(pieces)
                best_pieces = pieces
    except Exception:
        pass

    # Matching refinement
    if even_sub is not None and time.time() < _DEADLINE - 5:
        try:
            even_sub2 = _refine_with_matching(G, even_sub)
            pieces2 = _to_pieces(G, even_sub2)
            if _validate(G, pieces2):
                if len(pieces2) < best_count:
                    best_count = len(pieces2)
                    best_pieces = pieces2
        except Exception:
            pass

    return best_pieces


def construct():
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    for spec in specs:
        G = nx.Graph()
        G.add_edges_from(_instance(*spec))
        pieces = _solve_one(G)
        # Fallback if invalid
        if not _validate(G, pieces):
            pieces = [[u, v] for (u, v) in G.edges()]
        results.append(pieces)
        if time.time() > _DEADLINE - _DEADLINE_MARGIN:
            # fill remaining with all-single-edges fallback
            while len(results) < 3:
                spec2 = specs[len(results)]
                G2 = nx.Graph()
                G2.add_edges_from(_instance(*spec2))
                results.append([[u, v] for (u, v) in G2.edges()])
            break
    return results
