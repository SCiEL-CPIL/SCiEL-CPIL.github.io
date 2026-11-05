"""Decompose three fixed graphs into few cycles and edges (Erdős-Gallai style)."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

_DEADLINE = 135.0  # safety margin under 150s wall-clock kill

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

def _find_longest_cycle(adj, edges_set, deadline):
    """Find a long cycle via randomized DFS with backtracking, trying to maximize length."""
    n = max(adj.keys()) + 1 if adj else 0
    if n == 0:
        return None
    
    best_cycle = []
    best_len = 0
    
    # Try multiple starting points and random orderings
    import random
    rng = random.Random(42)
    
    vertices = list(adj.keys())
    if not vertices:
        return None
    
    # Limit search effort
    max_attempts = 50
    max_depth = 60
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        start = rng.choice(vertices)
        
        # Randomize neighbor order
        adj_copy = {v: list(neighs) for v, neighs in adj.items()}
        for v in adj_copy:
            rng.shuffle(adj_copy[v])
        
        # DFS to find a cycle back to start
        visited = {start}
        path = [start]
        found = False
        
        # Iterative DFS with explicit stack
        stack = [(start, iter(adj_copy[start]))]
        
        while stack:
            if time.time() > deadline:
                return best_cycle if best_cycle else None
            
            node, neighbors = stack[-1]
            advanced = False
            for nb in neighbors:
                if nb == start and len(path) >= 3:
                    # Found a cycle
                    cycle = path[:]
                    if len(cycle) > best_len:
                        best_len = len(cycle)
                        best_cycle = cycle
                    found = True
                    break
                elif nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    stack.append((nb, iter(adj_copy[nb])))
                    advanced = True
                    break
            
            if not advanced:
                # Backtrack
                visited.discard(node)
                path.pop()
                stack.pop()
        
        if found:
            # Try to extend: not straightforward, but we have a cycle
            pass
    
    return best_cycle if best_cycle else None

def _find_cycle_greedy(adj, edges_set):
    """Find any cycle using BFS/DFS from each vertex."""
    for start in sorted(adj.keys()):
        visited = {start}
        queue = deque([(start, [start])])
        while queue:
            node, path = queue.popleft()
            for nb in sorted(adj[node]):
                if nb == start and len(path) >= 3:
                    return path
                elif nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
    return None

def _eulerian_decompose(subgraph_edges, adj):
    """Decompose an Eulerian subgraph (all vertices even degree) into cycles."""
    cycles = []
    remaining = set(subgraph_edges)
    
    if not remaining:
        return cycles
    
    # Build adjacency for remaining edges
    rem_adj = defaultdict(list)
    for u, v in remaining:
        rem_adj[u].append(v)
        rem_adj[v].append(u)
    
    while remaining:
        # Find a vertex with non-zero degree
        start = None
        for v in rem_adj:
            if rem_adj[v]:
                start = v
                break
        if start is None:
            break
        
        # Find an Eulerian trail/cycle
        cycle = []
        current = start
        visited_edges = set()
        
        while True:
            cycle.append(current)
            # Find an unused edge from current
            next_v = None
            for nb in rem_adj[current]:
                edge = tuple(sorted((current, nb)))
                if edge in remaining and edge not in visited_edges:
                    next_v = nb
                    visited_edges.add(edge)
                    break
            
            if next_v is None:
                break
            
            # Remove the edge
            edge = tuple(sorted((current, next_v)))
            remaining.discard(edge)
            # Remove from adjacency
            rem_adj[current].remove(next_v)
            rem_adj[next_v].remove(current)
            
            current = next_v
        
        if len(cycle) >= 2:
            # Check if it's a cycle (start == end)
            if cycle[0] == current:
                cycle = cycle[:-1]  # remove duplicate end
            if len(cycle) >= 3:
                cycles.append(cycle)
            elif len(cycle) == 2:
                # It's just an edge, not a valid cycle
                pass
    
    return cycles

def _decompose_one(edges, deadline):
    """Decompose a single graph into cycles and single edges."""
    if not edges:
        return []
    
    edges_set = set(tuple(sorted(e)) for e in edges)
    adj = _build_adj(edges)
    
    pieces = []
    remaining = set(edges_set)
    
    # Phase 1: Extract cycles greedily, preferring longer ones
    while remaining and time.time() < deadline:
        # Rebuild adjacency for remaining
        rem_adj = defaultdict(set)
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Find a cycle
        cycle = _find_cycle_greedy(rem_adj, remaining)
        
        if cycle is None:
            break
        
        # Verify cycle edges are all in remaining
        valid = True
        cycle_edges = []
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((u, v)))
            if e not in remaining:
                valid = False
                break
            cycle_edges.append(e)
        
        if not valid or len(cycle) < 3:
            break
        
        # Remove cycle edges from remaining
        for e in cycle_edges:
            remaining.discard(e)
        pieces.append(cycle)
    
    # Phase 2: Remaining edges as single-edge pieces
    for u, v in sorted(remaining):
        pieces.append([u, v])
    
    return pieces

def _try_improve_with_eulerian(edges, pieces, deadline):
    """Try to improve decomposition by finding Eulerian subgraphs."""
    if not edges or time.time() > deadline:
        return pieces
    
    edges_set = set(tuple(sorted(e)) for e in edges)
    
    # Count current number of pieces
    current_count = len(pieces)
    
    # Try to find larger cycles by looking at the structure
    # For now, just return what we have
    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        pieces = _decompose_one(edges, _DEADLINE)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
