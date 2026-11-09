"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_2factor(adj, n):
    """Find a maximal 2-factor (set of vertex-disjoint cycles covering max edges).
    
    Uses a greedy approach: repeatedly find the longest cycle in remaining graph
    among vertices not yet used, building up a 2-factor.
    """
    used_vertices = set()
    cycles = []
    
    while True:
        # Build subgraph of unused vertices with degree >= 2
        avail = [v for v in range(n) if v not in used_vertices and len(adj[v]) >= 2]
        if not avail:
            break
        
        best_cycle = None
        best_len = 0
        
        # Try multiple random starts to find a long cycle
        for _ in range(50):
            start = avail[Random(42 + len(cycles) + _).randint(0, len(avail) - 1)]
            
            # DFS to find a cycle back to start
            visited = {start}
            path = [start]
            stack = [(start, list(adj[start]))]
            found = False
            
            while stack and not found:
                node, neighbors = stack[-1]
                advanced = False
                for nxt in neighbors:
                    if nxt == start and len(path) >= 3:
                        found = True
                        break
                    if nxt not in visited and nxt not in used_vertices:
                        visited.add(nxt)
                        path.append(nxt)
                        stack.append((nxt, list(adj[nxt])))
                        advanced = True
                        break
                if not advanced and not found:
                    stack.pop()
                    path.pop()
            
            if found and len(path) > best_len:
                best_len = len(path)
                best_cycle = list(path)
        
        if best_cycle is None:
            break
        
        # Mark vertices as used
        for v in best_cycle:
            used_vertices.add(v)
        cycles.append(best_cycle)
    
    return cycles

def _decompose(edges, n):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    pieces = []
    remaining = set(tuple(sorted(e)) for e in edges)
    adj = {i: set() for i in range(n)}
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    
    max_iterations = 200
    iters = 0
    while remaining and iters < max_iterations:
        # Find a 2-factor (set of vertex-disjoint cycles)
        cycles = _find_2factor(adj, n)
        
        if not cycles:
            break
        
        # Extract all cycles from the 2-factor
        valid_cycles = []
        for cycle in cycles:
            cycle_edges = set()
            valid = True
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = tuple(sorted((u, v)))
                if edge in remaining:
                    cycle_edges.add(edge)
                else:
                    valid = False
                    break
            if valid and len(cycle) >= 3:
                valid_cycles.append((cycle, cycle_edges))
        
        if not valid_cycles:
            break
        
        # Remove all cycle edges from remaining
        for cycle, cycle_edges in valid_cycles:
            for edge in cycle_edges:
                remaining.discard(edge)
                u, v = edge
                adj[u].discard(v)
                adj[v].discard(u)
            pieces.append(cycle[:])
        
        iters += 1
    
    # Remaining edges become single-edge pieces
    for u, v in remaining:
        pieces.append([u, v])
    
    return pieces

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            # Return what we have so far, fill rest with singletons
            while len(results) < 3:
                s = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))[len(results)]
                results.append([list(e) for e in _instance(*s)])
            break
        
        edges = _instance(*spec)
        n = spec[0]
        pieces = _decompose(edges, n)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
