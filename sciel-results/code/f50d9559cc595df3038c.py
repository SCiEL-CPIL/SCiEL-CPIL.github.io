"""Path decomposition of three fixed connected graphs (Gallai's conjecture)."""
import time
import heapq
from random import Random
from collections import deque

_START = time.time()
_DEADLINE = 120.0
_MARGIN = 8.0


def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _START)


def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _solve_instance(n, p, seed):
    edges = _instance(n, p, seed)
    adj = {i: [] for i in range(n)}
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Convert to set for fast lookup
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    remaining = set(edge_set)
    paths = []
    
    # Greedy decomposition
    while remaining:
        # Pick an arbitrary edge to start the path
        start_edge = next(iter(remaining))
        u, v = start_edge
        
        # Initialize path
        path = [u, v]
        used_edges = {start_edge}
        visited = {u, v}
        
        # Extend from u (backwards)
        curr = u
        while True:
            extended = False
            for w in adj[curr]:
                e = (min(curr, w), max(curr, w))
                if e in remaining and e not in used_edges and w not in visited:
                    # Add w to the front
                    path.insert(0, w)
                    visited.add(w)
                    used_edges.add(e)
                    curr = w
                    extended = True
                    break
            if not extended:
                break
                
        # Extend from v (forwards)
        curr = v
        while True:
            extended = False
            for w in adj[curr]:
                e = (min(curr, w), max(curr, w))
                if e in remaining and e not in used_edges and w not in visited:
                    # Add w to the back
                    path.append(w)
                    visited.add(w)
                    used_edges.add(e)
                    curr = w
                    extended = True
                    break
            if not extended:
                break
        
        # Remove used edges from remaining
        for e in used_edges:
            remaining.discard(e)
            
        paths.append(path)
        
    return paths


def _is_valid(paths, edges, n):
    seen = set()
    for path in paths:
        if len(path) < 2:
            return False
        for i in range(len(path)):
            if not (0 <= path[i] < n):
                return False
        for u, v in zip(path, path[1:]):
            e = (min(u, v), max(u, v))
            if e in seen:
                return False
            if e not in edges:
                return False
            seen.add(e)
        if len(set(path)) != len(path):
            return False
    return seen == edges


def construct():
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    result = []
    for (n, p, seed) in specs:
        if _time_left() <= 0:
            # Fallback: single-edge paths (always valid)
            edges = _instance(n, p, seed)
            result.append([[list(e)] for e in edges])
            continue
        paths = _solve_instance(n, p, seed)
        edges = _instance(n, p, seed)
        if not _is_valid(paths, edges, n):
            paths = [[list(e)] for e in edges]
        result.append(paths)
    return result
