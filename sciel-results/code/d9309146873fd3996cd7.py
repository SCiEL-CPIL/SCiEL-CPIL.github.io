"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency(edges, n):
    """Build adjacency list and edge set from edge list."""
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add(frozenset((u, v)))
    return adj, edge_set

def _find_hamiltonian_cycle(adj, n, forbidden_edges=None, start=0):
    """
    Find a Hamiltonian cycle in the graph defined by adj.
    forbidden_edges: set of frozenset pairs that cannot be used.
    Returns a list of vertices (the cycle) or None.
    """
    if forbidden_edges is None:
        forbidden_edges = set()
    
    visited = [False] * n
    path = [start]
    visited[start] = True
    
    # Pre-compute available neighbors
    def dfs(current, depth):
        if depth == n:
            # Check if we can close the cycle
            if frozenset((current, start)) not in forbidden_edges:
                if start in adj[current]:
                    return list(path)
            return None
        
        # Sort neighbors for determinism and potential pruning
        for neighbor in adj[current]:
            if not visited[neighbor]:
                edge = frozenset((current, neighbor))
                if edge not in forbidden_edges:
                    visited[neighbor] = True
                    path.append(neighbor)
                    result = dfs(neighbor, depth + 1)
                    if result is not None:
                        return result
                    path.pop()
                    visited[neighbor] = False
        return None
    
    return dfs(start, 1)

def _find_two_hamiltonian_cycles():
    """
    Find two edge-disjoint Hamiltonian cycles.
    Returns a list of two cycles (each a list of vertices) or None.
    """
    n = P['n']
    edges = P['edges']
    adj, edge_set = _build_adjacency(edges, n)
    
    # Get all edges as frozensets
    all_edges = [frozenset(e) for e in edges]
    
    # Strategy: try to find first cycle, then second in remaining graph
    # We'll try multiple starting points and orderings
    
    # First, let's find one Hamiltonian cycle
    cycle1 = _find_hamiltonian_cycle(adj, n, start=0)
    if cycle1 is None:
        return None
    
    # Extract edges from cycle1
    def cycle_edges(cycle):
        edges = set()
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edges.add(frozenset((u, v)))
        return edges
    
    c1_edges = cycle_edges(cycle1)
    
    # Remove cycle1 edges and find second cycle
    forbidden = c1_edges
    cycle2 = _find_hamiltonian_cycle(adj, n, forbidden_edges=forbidden, start=0)
    
    if cycle2 is not None:
        return [cycle1, cycle2]
    
    # If direct approach fails, try different first cycles
    # We can try to find a different first cycle by forbidding some edges
    # Let's try a few different strategies
    
    # Strategy 2: Try with different start vertices
    for start in range(1, n):
        cycle1_alt = _find_hamiltonian_cycle(adj, n, start=start)
        if cycle1_alt is None:
            continue
        c1_alt_edges = cycle_edges(cycle1_alt)
        cycle2_alt = _find_hamiltonian_cycle(adj, n, forbidden_edges=c1_alt_edges, start=0)
        if cycle2_alt is not None:
            return [cycle1_alt, cycle2_alt]
    
    # Strategy 3: Try to find cycle1, then for cycle2 try different starts
    cycle1 = _find_hamiltonian_cycle(adj, n, start=0)
    if cycle1 is not None:
        c1_edges = cycle_edges(cycle1)
        for start in range(1, n):
            cycle2_alt = _find_hamiltonian_cycle(adj, n, forbidden_edges=c1_edges, start=start)
            if cycle2_alt is not None:
                return [cycle1, cycle2_alt]
    
    return None

def construct():
    deadline = time.time() + 50  # Safety margin
    
    result = _find_two_hamiltonian_cycles()
    
    if result is not None:
        # Ensure types are correct: Python int, list
        cycles = []
        for cycle in result:
            # Ensure all elements are Python int
            clean_cycle = [int(v) for v in cycle]
            cycles.append(clean_cycle)
        return cycles
    
    # Fallback: return just one Hamiltonian cycle
    n = P['n']
    edges = P['edges']
    adj, edge_set = _build_adjacency(edges, n)
    cycle = _find_hamiltonian_cycle(adj, n, start=0)
    if cycle is not None:
        return [[int(v) for v in cycle]]
    
    # Last resort: return a trivial cycle if possible
    # Find any 3-cycle
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if frozenset((i,j)) in edge_set and frozenset((j,k)) in edge_set and frozenset((i,k)) in edge_set:
                    return [[i, j, k]]
    
    # Should not reach here
    return [[0, 1, 2]]
# EVOLVE-BLOCK-END
