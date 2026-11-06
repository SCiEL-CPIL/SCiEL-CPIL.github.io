"""Decompose three fixed graphs into few cycles and edges (Erdős-Gallai)."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edges, n):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_cycle_greedy(adj, start, n):
    """
    Find a cycle starting from 'start' using greedy DFS with degree-based ordering.
    Returns a list of vertices in cycle order, or None.
    """
    if not adj[start]:
        return None
    
    path = [start]
    visited = {start}
    
    def dfs(u):
        if not adj[u]:
            return False
        # Sort neighbors by degree (prefer lower degree first for longer cycles)
        neighbors = sorted(adj[u], key=lambda x: len(adj[x]))
        for v in neighbors:
            if v == start and len(path) >= 3:
                return True
            if v not in visited:
                visited.add(v)
                path.append(v)
                if dfs(v):
                    return True
                path.pop()
                visited.discard(v)
        return False
    
    if dfs(start):
        return path
    return None


def _find_longest_cycle(adj, n, max_starts=20):
    """
    Try multiple random starts to find a long cycle.
    Returns the longest cycle found, or None.
    """
    best_cycle = None
    best_len = 0
    
    # Try all vertices as starts, but limit for performance
    candidates = list(range(n))
    random.shuffle(candidates)
    
    for start in candidates[:max_starts]:
        if not adj[start]:
            continue
        cycle = _find_cycle_greedy(adj, start, n)
        if cycle and len(cycle) > best_len:
            best_len = len(cycle)
            best_cycle = cycle
    
    return best_cycle


def _decompose_graph(edges, n, deadline):
    """
    Decompose the graph into cycles and single edges.
    Returns a list of pieces.
    """
    if not edges:
        return []
    
    # Build adjacency
    adj = _build_adj(edges, n)
    
    # Track remaining edges
    remaining = set(tuple(sorted(e)) for e in edges)
    pieces = []
    
    # Strategy: repeatedly find long cycles, then handle leftovers
    # Use a greedy approach that prioritizes longer cycles
    
    # First, try to find as many long cycles as possible
    max_cycles = 200  # Safety limit
    for _ in range(max_cycles):
        if time.time() > deadline:
            break
        if not remaining:
            break
        
        # Find a long cycle in the remaining graph
        cycle = None
        # Try to find the longest cycle
        cycle = _find_longest_cycle(adj, n, max_starts=30)
        
        if cycle:
            # Add the cycle as a piece
            pieces.append(list(cycle))
            # Remove the cycle's edges from remaining
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                remaining.discard(e)
                # Update adjacency
                adj[u].discard(v)
                adj[v].discard(u)
        else:
            # No more cycles can be found
            break
    
    # Handle remaining edges as single-edge pieces
    for e in sorted(remaining):
        if time.time() > deadline:
            break
        pieces.append(list(e))
    
    return pieces


def _decompose_with_backtracking(edges, n, deadline):
    """
    Alternative decomposition using a more sophisticated approach.
    Try to maximize cycle lengths by considering edge removal order.
    """
    if not edges:
        return []
    
    adj = _build_adj(edges, n)
    remaining = set(tuple(sorted(e)) for e in edges)
    pieces = []
    
    # Use a priority queue approach: always extract the longest available cycle
    # but also consider that removing certain edges might prevent longer cycles later
    
    max_cycles = 200
    for _ in range(max_cycles):
        if time.time() > deadline:
            break
        if not remaining:
            break
        
        # Find the longest cycle
        cycle = _find_longest_cycle(adj, n, max_starts=50)
        
        if cycle:
            pieces.append(list(cycle))
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                remaining.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
        else:
            break
    
    for e in sorted(remaining):
        if time.time() > deadline:
            break
        pieces.append(list(e))
    
    return pieces


def construct():
    """
    Decompose three fixed graphs into few cycles and edges.
    Returns three lists of pieces, one per fixed instance graph.
    """
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = _instance(n, p, seed)
        
        if time.time() > deadline:
            # Fallback: all edges as single pieces
            results.append([list(e) for e in edges])
            continue
        
        # Try the main decomposition
        pieces = _decompose_graph(edges, n, deadline)
        
        # If we have time, try an alternative approach and keep the better one
        if time.time() < deadline - 20:
            pieces_alt = _decompose_with_backtracking(edges, n, deadline)
            if len(pieces_alt) < len(pieces):
                pieces = pieces_alt
        
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
