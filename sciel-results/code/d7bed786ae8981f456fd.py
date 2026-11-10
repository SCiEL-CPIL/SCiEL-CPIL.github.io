"""Decompose three fixed Erdos-Renyi-ish graphs into few cycles + edges.

Strategy:
  * Minimizing pieces == maximizing number of edge-disjoint cycles.
  * Use randomized local search: start with all edges as single pieces,
    then iteratively try to merge pieces into valid cycles.
  * A cycle is valid if it has >= 3 distinct vertices and all consecutive
    edges (including closing edge) exist in the graph.
  * We prioritize forming longer cycles since they reduce piece count more.
  * Use multiple restarts with different random seeds for robustness.
"""

import time
import random
from collections import deque

_DEADLINE = 140.0          # hard wall-clock cap (evaluator kills at 150s)
_SAFE = 3.0                # stop searching this many seconds before the cap
_START = time.monotonic()


def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _is_valid_cycle(cycle, adj):
    """Check if cycle (list of vertices) is valid: >=3 distinct vertices, all edges exist."""
    if len(cycle) < 3:
        return False
    # Check all vertices distinct
    if len(set(cycle)) != len(cycle):
        return False
    # Check all consecutive edges exist
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        if v not in adj[u]:
            return False
    return True


def _try_form_cycle_from_pieces(pieces, adj, rng, max_attempts=1000):
    """
    Try to form a cycle by merging existing pieces.
    Returns (cycle_vertices, piece_indices_used) or None.
    """
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        
        # Pick a random starting edge
        if not pieces:
            return None
        
        start_idx = rng.randint(0, len(pieces) - 1)
        piece = pieces[start_idx]
        
        if len(piece) < 2:
            continue
        
        # Try to extend this edge into a cycle
        # Start with the two vertices of this edge
        u, v = piece[0], piece[1]
        
        # We need to find a path from v back to u with at least one intermediate vertex
        # Use BFS/DFS to find such a path, avoiding using edges already in other pieces
        
        # Build a set of edges that are "used" by other pieces (excluding start piece)
        used_edges = set()
        for i, p in enumerate(pieces):
            if i == start_idx:
                continue
            if len(p) >= 2:
                for j in range(len(p)):
                    a = p[j]
                    b = p[(j + 1) % len(p)]
                    used_edges.add((min(a, b), max(a, b)))
        
        # The start edge itself is available
        start_edge = (min(u, v), max(u, v))
        
        # Find a path from v to u that doesn't use used_edges
        # and has at least 2 edges (so total cycle has >= 3 vertices)
        found_path = _find_path_avoiding(v, u, adj, used_edges, max_len=20, rng=rng)
        
        if found_path and len(found_path) >= 2:
            # found_path is [v, ..., u] or similar - reconstruct cycle
            # The cycle would be: u, v, ..., u
            # Actually found_path should be a sequence from v to u
            # Let's say found_path = [v, w1, w2, ..., u]
            # Then cycle is [u, v, w1, w2, ..., u] but we don't repeat u
            # Cycle vertices: [u] + found_path[:-1] if found_path ends with u
            # Or just found_path if it starts with v and ends with u
            
            cycle = [u] + found_path[:-1] if found_path[-1] == u else [u] + found_path
            # Remove duplicate u at end if present
            if cycle[0] == cycle[-1]:
                cycle = cycle[:-1]
            
            if _is_valid_cycle(cycle, adj):
                return (cycle, [start_idx])
        
        # Try another random start
    return None


def _find_path_avoiding(start, end, adj, forbidden_edges, max_len=20, rng=None):
    """Find a path from start to end avoiding forbidden edges."""
    if start == end:
        return [start]
    
    # BFS
    visited = {start}
    queue = deque([(start, [start])])
    
    while queue:
        node, path = queue.popleft()
        
        if len(path) > max_len:
            continue
        
        if node == end:
            return path
        
        neighbors = sorted(adj[node])
        if rng:
            rng.shuffle(neighbors)
        
        for nb in neighbors:
            if nb in visited:
                continue
            
            edge = (min(node, nb), max(node, nb))
            if edge in forbidden_edges:
                continue
            
            visited.add(nb)
            queue.append((nb, path + [nb]))
    
    return None


def _decompose(n, edges):
    adj = _build_adjacency(n, edges)
    
    # Start with all edges as single pieces
    pieces = [list(e) for e in edges]
    
    # Track which edges are covered by which pieces
    # We'll iteratively try to merge pieces into cycles
    
    rng = random.Random(42)
    t0 = time.monotonic()
    max_time = 110.0
    
    # First pass: find all simple cycles in the graph and use them
    # This is more effective than trying to merge single edges
    
    # Find cycles using a simple approach: for each edge, try to extend it into a cycle
    cycles_found = []
    used_edges = set()
    
    # Try to find cycles greedily
    for u, v in edges:
        if (u, v) in used_edges or (v, u) in used_edges:
            continue
        
        # Try to find a cycle containing edge (u, v)
        # Look for a path from v to u that doesn't use (u, v)
        path = _find_path_avoiding(v, u, adj, set([(u, v)]), max_len=n, rng=rng)
        
        if path and len(path) >= 2:
            # Cycle is: u, v, ..., u
            # path is [v, ..., u]
            cycle = [u] + path[:-1] if path[-1] == u else [u] + path
            
            if _is_valid_cycle(cycle, adj):
                cycles_found.append(cycle)
                # Mark all edges in cycle as used
                for i in range(len(cycle)):
                    a = cycle[i]
                    b = cycle[(i + 1) % len(cycle)]
                    used_edges.add((min(a, b), max(a, b)))
    
    # Build final pieces: cycles + remaining edges
    final_pieces = []
    for cycle in cycles_found:
        final_pieces.append(list(cycle))
    
    # Add edges not in any cycle
    for u, v in edges:
        edge = (min(u, v), max(u, v))
        if edge not in used_edges:
            final_pieces.append([u, v])
    
    return final_pieces


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    result = []
    for spec in specs:
        n, p, seed = spec
        edges = _instance(n, p, seed)
        if time.monotonic() - _START > _DEADLINE - _SAFE:
            # no time left: fall back to single edges
            result.append([list(e) for e in edges])
            continue
        pieces = _decompose(n, edges)
        result.append(pieces)
    return result
