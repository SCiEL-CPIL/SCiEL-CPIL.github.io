"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def _build_adjacency(edges):
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamiltonian_cycle(adj, start=0, deadline=None):
    """Find a Hamiltonian cycle in the graph defined by adj."""
    n = P['n']
    # adj is a list of sets of available neighbors
    
    # We'll do backtracking. Fix start vertex to 0 to avoid symmetry.
    # Path starts at 0, must visit all n vertices, return to 0.
    
    def backtrack(path, visited):
        if deadline and time.time() > deadline:
            return None
        if len(path) == n:
            # Check if last vertex connects back to start
            if path[0] in adj[path[-1]]:
                return list(path)
            return None
        
        last = path[-1]
        # Try neighbors in a deterministic order for reproducibility
        for nxt in sorted(adj[last]):
            if not visited[nxt]:
                # Pruning: check if remaining unvisited vertices are reachable
                # Simple pruning: if nxt is the only unvisited vertex and it doesn't connect to start, skip
                visited[nxt] = True
                path.append(nxt)
                result = backtrack(path, visited)
                if result is not None:
                    return result
                path.pop()
                visited[nxt] = False
        return None
    
    visited = [False] * n
    visited[0] = True
    result = backtrack([0], visited)
    return result

def _remove_cycle_edges(edges, cycle):
    """Remove edges used by cycle from edge list."""
    cycle_edges = set()
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        cycle_edges.add(_edge_key(u, v))
    
    remaining = []
    for u, v in edges:
        if _edge_key(u, v) not in cycle_edges:
            remaining.append([u, v])
    return remaining

def construct():
    deadline = time.time() + 50  # Safety margin before 60s limit
    
    edges = [list(e) for e in P['edges']]
    n = P['n']
    
    cycles = []
    current_edges = edges[:]
    
    for _ in range(3):  # Max 3 cycles possible (6-regular, 63 edges)
        if time.time() > deadline:
            break
        if not current_edges:
            break
        
        adj = _build_adjacency(current_edges)
        
        # Check that every vertex has degree >= 2 (needed for Hamiltonian cycle)
        if any(len(adj[v]) < 2 for v in range(n)):
            break
        
        cycle = _find_hamiltonian_cycle(adj, start=0, deadline=deadline)
        
        if cycle is None:
            break
        
        # Verify the cycle is valid
        valid = True
        cycle_edge_set = set()
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            ek = _edge_key(u, v)
            if ek in cycle_edge_set:
                valid = False
                break
            cycle_edge_set.add(ek)
        
        # Verify all edges in cycle exist in current_edges
        current_edge_set = set(_edge_key(u, v) for u, v in current_edges)
        if not valid or not cycle_edge_set.issubset(current_edge_set):
            break
        
        cycles.append([int(v) for v in cycle])
        current_edges = _remove_cycle_edges(current_edges, cycle)
    
    # Ensure we return at least 1 cycle
    if not cycles:
        # Fallback: just return the trivial full cycle (may not be valid but at least returns something)
        # Actually, let's try harder with a different starting approach
        adj = _build_adjacency(edges)
        cycle = _find_hamiltonian_cycle(adj, start=0, deadline=deadline)
        if cycle:
            cycles.append([int(v) for v in cycle])
    
    # Convert to proper types
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    
    return result
# EVOLVE-BLOCK-END
