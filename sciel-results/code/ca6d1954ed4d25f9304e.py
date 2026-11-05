import time
import random
import sys
import heapq
from collections import defaultdict

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edge_set, n):
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _canonical(cycle):
    """Return canonical form of a cycle (tuple) for deduplication."""
    if len(cycle) < 3:
        return None
    min_val = min(cycle)
    idxs = [i for i, v in enumerate(cycle) if v == min_val]
    candidates = []
    for idx in idxs:
        c1 = cycle[idx:] + cycle[:idx]
        c2 = list(reversed(c1))
        candidates.append(tuple(c1) if c1 < c2 else tuple(c2))
    return min(candidates)


def _find_cycles_random(adj, n, edge_set, num_trials=200, rng=None, time_limit=1.0):
    """Find multiple random simple cycles in the graph."""
    if rng is None:
        rng = random.Random()
    
    start_time = time.time()
    cycles = set()
    
    for _ in range(num_trials):
        if time.time() - start_time > time_limit:
            break
        
        start = rng.randrange(n)
        if not adj[start]:
            continue
        
        visited = [False] * n
        path = [start]
        visited[start] = True
        
        # Random DFS to find a cycle
        stack = [(start, iter(adj[start]))]
        
        while stack:
            if time.time() - start_time > time_limit:
                return list(cycles)
            node, it = stack[-1]
            advanced = False
            for nb in it:
                if nb == start and len(path) >= 3:
                    # Found a cycle
                    cycle = path[:]
                    key = _canonical(cycle)
                    if key:
                        cycles.add(key)
                    break
                elif not visited[nb]:
                    visited[nb] = True
                    path.append(nb)
                    stack.append((nb, iter(adj[nb])))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
                if path:
                    path.pop()
                    visited[node] = False
    
    return list(cycles)


def _cycle_edges(cycle):
    """Return set of edges in a cycle."""
    edges = set()
    L = len(cycle)
    for i in range(L):
        u = cycle[i]
        v = cycle[(i + 1) % L]
        edges.add(tuple(sorted((u, v))))
    return edges


def _greedy_select(cycles, edge_set):
    """
    Greedy set-packing: select cycles that don't conflict with already-selected edges.
    Returns (selected_cycles, covered_edges).
    """
    # Sort cycles by length descending (longer cycles are more valuable)
    cycles_sorted = sorted(cycles, key=lambda c: -len(c))
    
    selected = []
    covered_edges = set()
    
    for cycle in cycles_sorted:
        ce = _cycle_edges(cycle)
        if ce.isdisjoint(covered_edges):
            selected.append(cycle)
            covered_edges |= ce
    
    return selected, covered_edges


def _local_search_improve(selected, edge_set, adj, n, rng, time_limit=3.0):
    """
    Try to improve the solution by swapping: remove a short cycle, add 2+ longer cycles
    that cover its edges plus some new edges.
    """
    start_time = time.time()
    current_covered = set()
    for c in selected:
        current_covered |= _cycle_edges(c)
    
    current_cost = len(selected)
    
    while time.time() - start_time < time_limit:
        # Try removing each cycle and seeing if we can replace it
        improved = False
        for i in range(len(selected)):
            if time.time() - start_time > time_limit:
                break
            
            removed = selected[i]
            removed_edges = _cycle_edges(removed)
            
            # Temporarily remove
            remaining = [c for j, c in enumerate(selected) if j != i]
            remaining_covered = set()
            for c in remaining:
                remaining_covered |= _cycle_edges(c)
            
            # Edges that need to be covered
            to_cover = removed_edges - remaining_covered
            
            # Find cycles from the pool that help cover to_cover
            # We'll generate new cycles from the current graph
            # (but for now, this is a stub for future use)
            # For simplicity, just try to add any cycle that covers some of to_cover
            # without conflicting with remaining_covered
            
            # This is complex; skip for now and rely on better initial selection
        
        if not improved:
            break
    
    return selected, current_covered


def _solve_one(edges, n, deadline):
    if not edges:
        return []
    
    m = len(edges)
    edge_set = set(tuple(sorted(e)) for e in edges)
    adj = _build_adj(edge_set, n)
    
    if m < 3:
        return [list(e) for e in edges]
    
    rng = random.Random(42)
    pieces = []
    
    # Phase 1: Generate a pool of cycles
    all_cycles = set()
    pool_time = 2.0
    pool_start = time.time()
    
    while time.time() - pool_start < pool_time:
        cycles = _find_cycles_random(adj, n, edge_set, num_trials=50, rng=rng, time_limit=0.5)
        for c in cycles:
            all_cycles.add(c)
        if len(all_cycles) > 500:
            break
    
    # Phase 2: Greedy set-packing on the pool
    selected, covered_edges = _greedy_select(list(all_cycles), edge_set)
    pieces.extend(selected)
    
    # Phase 3: If time remains, try to find more cycles to cover remaining edges
    remaining = edge_set - covered_edges
    if remaining:
        # Build subgraph of remaining edges
        rem_adj = [set() for _ in range(n)]
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Try to find cycles in the remaining graph
        while time.time() < deadline - 1.0 and remaining:
            cycles = _find_cycles_random(rem_adj, n, remaining, num_trials=20, rng=rng, time_limit=0.5)
            found_new = False
            for c in cycles:
                ce = _cycle_edges(c)
                if ce.issubset(remaining) and ce.isdisjoint(covered_edges):
                    # Add this cycle
                    pieces.append(c)
                    covered_edges |= ce
                    remaining -= ce
                    found_new = True
            if not found_new:
                break
    
    # Remaining edges as single pieces
    for e in remaining:
        pieces.append(list(e))
    
    return pieces


def construct():
    start_time = time.time()
    total_budget = 145.0
    per_graph_budget = total_budget / 3.0

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        graph_deadline = start_time + per_graph_budget * (i + 1)

        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0

        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)

        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break

    return results
