"""Decompose three fixed random graphs into edge-disjoint cycles + edges, minimizing pieces."""
import time
import random
from random import Random

_START = None

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = set()
    for i in range(n - 1):
        u, v = order[i], order[i + 1]
        edges.add((min(u, v), max(u, v)))
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
    return sorted(edges)


def _build_adj(edges):
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def _find_long_cycle(adj, edges_set, rng, max_iter=60):
    """Find a long simple cycle using randomized DFS with backtracking, then 2-opt improve."""
    n = max(adj.keys()) + 1 if adj else 0
    if n < 3:
        return None
    best_cycle = None
    best_len = 0

    # Try multiple random starts and DFS walks
    for attempt in range(max_iter):
        start = rng.choice(list(adj.keys()))
        # Randomized DFS to find a cycle back to start
        visited = {start}
        stack = [(start, iter(sorted(adj[start], key=lambda x: rng.random())))]
        path = [start]
        found = False
        # We want a cycle: find a path from start back to start
        # Use DFS that tries to close the cycle
        def dfs():
            nonlocal found
            if found:
                return
            while stack:
                node, it = stack[-1]
                advanced = False
                for nb in it:
                    if nb == start and len(path) >= 3:
                        # Found a cycle
                        found = True
                        return
                    if nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        stack.append((nb, iter(sorted(adj[nb], key=lambda x: rng.random()))))
                        advanced = True
                        break
                if not advanced:
                    stack.pop()
                    visited.discard(node)
                    path.pop()
            return
        dfs()
        if found:
            cycle = path[:]  # path[0]==start, and last node adjacent to start
            # The cycle is path[0], path[1], ..., path[-1], path[0]
            clen = len(cycle)
            if clen > best_len:
                best_len = clen
                best_cycle = cycle[:]
            # 2-opt improve: try to make the cycle longer by swapping
            # Actually for a cycle, we can try to extend it by finding a vertex
            # on the cycle with an edge to a vertex not on the cycle
            improved = True
            while improved:
                improved = False
                cycle_set = set(best_cycle)
                # For each consecutive pair in the cycle, check if we can insert a vertex
                for i in range(len(best_cycle)):
                    a = best_cycle[i]
                    b = best_cycle[(i + 1) % len(best_cycle)]
                    # Look for a vertex w not in cycle adjacent to both a and b
                    candidates = (adj.get(a, set()) & adj.get(b, set())) - cycle_set
                    if candidates:
                        w = rng.choice(list(candidates))
                        # Insert w between a and b
                        # Need to rebuild the cycle
                        new_cycle = []
                        for j, v in enumerate(best_cycle):
                            new_cycle.append(v)
                            if v == a:
                                new_cycle.append(w)
                        best_cycle = new_cycle
                        cycle_set.add(w)
                        improved = True
                        break
                if improved:
                    continue
                # Also try removing a vertex to allow a different extension? No, we want longer.
                # Try 2-opt: reverse a segment to enable new connections
                # This is complex; skip for now and rely on multiple random starts
                break

    if best_cycle is None:
        return None
    # Validate: all consecutive edges (including closing) must exist
    for i in range(len(best_cycle)):
        a = best_cycle[i]
        b = best_cycle[(i + 1) % len(best_cycle)]
        e = (min(a, b), max(a, b))
        if e not in edges_set:
            # Invalid cycle, discard
            return None
    return best_cycle


def _decompose_one(edges, time_limit=45.0):
    """Decompose edges into cycles + single edges, minimizing pieces. Returns list of pieces."""
    if not edges:
        return []
    remaining = set(tuple(e) for e in edges)
    n = max(max(e) for e in edges) + 1
    best_pieces = None
    best_count = len(remaining)  # baseline: all single edges
    rng = Random(42)
    deadline = _START + time_limit

    # Greedy: repeatedly find longest cycle
    def greedy_once(rng_seed):
        local_rng = Random(rng_seed)
        rem = set(remaining)
        pieces = []
        while rem:
            adj = _build_adj(rem)
            # Find longest cycle via multiple random attempts
            best_c = None
            best_clen = 0
            for _ in range(40):
                c = _find_long_cycle(adj, rem, local_rng, max_iter=25)
                if c and len(c) > best_clen:
                    best_clen = len(c)
                    best_c = c
            if best_c and best_clen >= 3:
                # Remove cycle edges from rem
                for i in range(len(best_c)):
                    a = best_c[i]
                    b = best_c[(i + 1) % len(best_c)]
                    e = (min(a, b), max(a, b))
                    rem.discard(e)
                pieces.append(best_c[:])
            else:
                # No cycle found; take a single edge
                e = next(iter(rem))
                rem.discard(e)
                pieces.append([e[0], e[1]])
        return pieces

    # Try multiple random seeds
    for seed in range(30):
        if time.time() > deadline:
            break
        pieces = greedy_once(seed)
        cnt = len(pieces)
        if cnt < best_count:
            best_count = cnt
            best_pieces = pieces
        # Early stop if we've done well
        if best_count <= best_count * 0.85 and seed > 5:
            # Some improvement threshold; continue a bit more
            pass

    if best_pieces is None:
        # Fallback: all single edges
        return [[u, v] for u, v in sorted(remaining)]
    return best_pieces


def construct():
    global _START
    _START = time.time()
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    total_budget = 140.0  # safety margin before 150s kill
    per_graph = total_budget / len(specs)
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        # Give each graph its own time slice, but allow borrowing if fast
        t0 = time.time()
        pieces = _decompose_one(edges, time_limit=per_graph - 2.0)
        results.append(pieces)
        # If we're running out of time, fall back to single edges for remaining
        if time.time() > _START + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nj, pj, sj = specs[j]
                ej = _instance(nj, pj, sj)
                results.append([[u, v] for u, v in ej])
            break
    return results
