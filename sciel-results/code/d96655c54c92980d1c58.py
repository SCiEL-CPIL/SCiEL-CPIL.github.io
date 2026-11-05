"""Decompose three fixed instance graphs into few cycles + single edges."""
import time
import sys
from random import Random
from collections import defaultdict, deque

_DEADLINE = 150.0
_START = None

def _elapsed():
    return time.time() - _START

def _time_left():
    return _DEADLINE - _elapsed() - 3.0  # safety margin

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _adjacency(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_longest_cycle(adj, edge_set, start, time_budget):
    """
    Find a long simple cycle in the graph (adj/edge_set).
    Use DFS with backtracking, but bounded by time_budget seconds.
    Returns a list of vertices in cycle order, or None.
    """
    best_cycle = None
    best_len = 0
    deadline = time.time() + time_budget

    # Try a randomized DFS from start, with pruning by best_len.
    # We'll do multiple randomized DFS attempts.
    rng = Random(0)

    def dfs(path, used_edges, cur):
        nonlocal best_cycle, best_len
        if time.time() > deadline:
            return
        # If we can close the cycle back to start with a remaining edge
        if len(path) >= 3 and (cur, start) in edge_set and (cur, start) not in used_edges:
            if len(path) > best_len:
                best_len = len(path)
                best_cycle = list(path)
        # Try neighbors in random order
        nbrs = list(adj[cur])
        rng.shuffle(nbrs)
        for nxt in nbrs:
            e = tuple(sorted((cur, nxt)))
            if e in edge_set and e not in used_edges and nxt not in path_set:
                if len(path) + 1 < best_len + 1:  # prune: even closing gives len+1, need > best
                    # Actually, max possible = len(path)+1 (close) or more. We can still extend.
                    pass
                path_set.add(nxt)
                used_edges.add(e)
                path.append(nxt)
                dfs(path, used_edges, nxt)
                path.pop()
                used_edges.remove(e)
                path_set.remove(nxt)
                if time.time() > deadline:
                    return

    # Start DFS
    path = [start]
    path_set = {start}
    used_edges = set()
    dfs(path, used_edges, start)

    return best_cycle


def _find_cycle_greedy(adj, edge_set, start, time_budget):
    """
    Greedy: walk using a heuristic that prefers to close the cycle early
    but also tries to extend. Use a bounded DFS with a soft deadline.
    """
    deadline = time.time() + time_budget
    best_cycle = None
    best_len = 0

    # Use a simple heuristic: try to find a cycle by DFS with random restarts
    rng = Random(42)

    def try_dfs(start_v):
        nonlocal best_cycle, best_len
        if time.time() > deadline:
            return
        # Iterative DFS with backtracking
        # State: (path, used_edges)
        # We'll do a recursive DFS but with a depth limit and time check
        sys.setrecursionlimit(10000)

        def rec(path, used_edges, cur):
            nonlocal best_cycle, best_len
            if time.time() > deadline:
                return
            if len(path) >= 3 and (cur, start_v) in edge_set and tuple(sorted((cur, start_v))) not in used_edges:
                if len(path) > best_len:
                    best_len = len(path)
                    best_cycle = list(path)
            nbrs = list(adj[cur])
            rng.shuffle(nbrs)
            for nxt in nbrs:
                e = tuple(sorted((cur, nxt)))
                if e in edge_set and e not in used_edges and nxt not in path:
                    path.append(nxt)
                    used_edges.add(e)
                    rec(path, used_edges, nxt)
                    path.pop()
                    used_edges.discard(e)
                    if time.time() > deadline:
                        return

        path = [start_v]
        used_edges = set()
        rec(path, used_edges, start_v)

    # Try from start
    try_dfs(start)
    return best_cycle


def _decompose(edges):
    """
    Decompose edges into cycles (as vertex lists) and single edges.
    Returns a list of pieces: each piece is either [u, v] (edge) or [v0, v1, ..., vk] (cycle, k>=2).
    """
    edge_set = set(tuple(sorted(e)) for e in edges)
    adj = _adjacency(edges)
    pieces = []

    # Iteratively find longest cycles
    while edge_set and _time_left() > 1.0:
        # Pick a random start vertex with degree >= 2
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            break

        rng = Random(len(edge_set))  # deterministic per iteration
        start = rng.choice(candidates)

        # Allocate a small time budget per cycle attempt
        budget = min(2.0, _time_left() * 0.5)
        cycle = _find_cycle_greedy(adj, edge_set, start, budget)

        if cycle is None:
            # Try a few more starts
            found = False
            for _ in range(3):
                if _time_left() < 0.5:
                    break
                s2 = rng.choice(candidates)
                c2 = _find_cycle_greedy(adj, edge_set, s2, min(1.0, _time_left() * 0.3))
                if c2 is not None:
                    cycle = c2
                    found = True
                    break
            if not found:
                break

        # Remove cycle edges from edge_set
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((u, v)))
            if e in edge_set:
                edge_set.remove(e)
                adj[u].discard(v)
                adj[v].discard(u)

        pieces.append(cycle)

    # Remaining edges as single-edge pieces
    for e in sorted(edge_set):
        pieces.append(list(e))

    return pieces


def _improve_pieces(pieces, edges, time_budget):
    """
    Try to reduce the number of pieces by merging adjacent single edges into cycles,
    or by finding cycles in the residual graph that were missed.
    This is a local search / polish pass.
    """
    if _time_left() < 1.0 or time_budget < 0.5:
        return pieces

    # Build residual edge set from single-edge pieces
    single_edges = [p for p in pieces if len(p) == 2]
    cycle_pieces = [p for p in pieces if len(p) >= 3]

    if not single_edges:
        return pieces

    edge_set = set(tuple(sorted(e)) for e in single_edges)
    adj = _adjacency(single_edges)

    # Try to find cycles in the residual single-edge graph
    improved = False
    deadline = time.time() + time_budget

    while edge_set and time.time() < deadline:
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            break
        rng = Random(len(edge_set))
        start = rng.choice(candidates)
        cycle = _find_cycle_greedy(adj, edge_set, start, min(1.0, deadline - time.time()))
        if cycle is None:
            break
        # Remove cycle edges
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((u, v)))
            if e in edge_set:
                edge_set.remove(e)
                adj[u].discard(v)
                adj[v].discard(u)
        # Replace single edges with cycle
        new_singles = [list(e) for e in sorted(edge_set)]
        pieces = cycle_pieces + [cycle] + new_singles
        improved = True

    return pieces


def construct():
    global _START
    _START = time.time()

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    all_pieces = []

    for spec in specs:
        if _time_left() < 2.0:
            # Fallback: return all edges as single pieces
            edges = _instance(*spec)
            all_pieces.append([list(e) for e in edges])
            continue

        edges = _instance(*spec)
        pieces = _decompose(edges)

        # Polish pass
        if _time_left() > 2.0:
            pieces = _improve_pieces(pieces, edges, min(5.0, _time_left() * 0.3))

        all_pieces.append(pieces)

    return all_pieces
