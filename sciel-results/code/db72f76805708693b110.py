"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _remove_bridges_and_extract_cycles(edges_set, deadline):
    """
    Iteratively remove bridges (which become single-edge pieces) and
    extract cycles from the remaining 2-edge-connected components.
    Returns list of pieces (cycles as lists of vertices, single edges as [u,v]).
    """
    pieces = []
    remaining = set(edges_set)
    
    while remaining and time.time() < deadline:
        adj = _build_adj(remaining)
        
        # Find bridges in the current graph
        bridges = set()
        visited = set()
        
        def _find_bridges_dfs(u, parent):
            visited.add(u)
            for v in sorted(adj[u]):
                if v == parent:
                    continue
                if v not in visited:
                    # Edge (u,v) is a bridge if there's no back edge from v's subtree to u
                    # We'll use a simpler approach: count edges in biconnected components
                    pass
        
        # Simpler approach: use networkx-style bridge finding via DFS
        # Since we can't use networkx, let's use a different strategy:
        # Find all cycles via DFS and remove them, then what's left are bridges
        
        # Strategy: Find a cycle in the remaining graph
        cycle = _find_cycle(remaining, adj)
        
        if cycle and len(cycle) >= 3:
            pieces.append(cycle)
            # Remove cycle edges
            for i in range(len(cycle)):
                a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((a, b)))
                remaining.discard(e)
        else:
            # No more cycles; all remaining edges are bridges
            for e in remaining:
                pieces.append([e[0], e[1]])
            remaining.clear()
    
    return pieces

def _find_cycle(remaining, adj):
    """Find a cycle in the graph defined by remaining edges."""
    # Try to find a long cycle using DFS
    for start in sorted(adj.keys()):
        if len(adj[start]) < 2:
            continue
        
        cycle = _dfs_find_cycle(start, adj, remaining)
        if cycle:
            return cycle
    return None

def _dfs_find_cycle(start, adj, remaining):
    """DFS to find a cycle starting from start vertex."""
    visited = {start}
    path = [start]
    
    def dfs(node, depth):
        if depth > 15:  # Limit depth to avoid exponential blowup
            return None
        for nb in sorted(adj[node]):
            e = tuple(sorted((node, nb)))
            if e not in remaining:
                continue
            if nb == start and len(path) >= 3:
                return path[:]
            if nb not in visited:
                visited.add(nb)
                path.append(nb)
                result = dfs(nb, depth + 1)
                if result:
                    return result
                path.pop()
                visited.discard(nb)
        return None
    
    return dfs(start, 1)

def _decompose_graph(edges, deadline):
    """Decompose edges into cycles and single edges."""
    if not edges:
        return []
    
    edges_set = set(tuple(sorted(e)) for e in edges)
    adj = _build_adj(edges_set)
    
    # First, identify and remove bridges iteratively while extracting cycles
    pieces = _remove_bridges_and_extract_cycles(edges_set, deadline)
    
    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    deadline = time.time() + 140.0
    
    for i, spec in enumerate(specs):
        graph_deadline = deadline - (len(specs) - 1 - i) * 5.0
        
        edges = _instance(*spec)
        pieces = _decompose_graph(edges, graph_deadline)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
