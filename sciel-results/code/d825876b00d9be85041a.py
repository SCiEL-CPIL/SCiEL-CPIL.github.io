"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _build_adjacency(edges):
    n = P['n']
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamiltonian_cycle(adj, start=0, deadline=None):
    """Backtracking to find a Hamiltonian cycle starting and ending at 'start'."""
    n = P['n']
    all_vertices = set(range(n))
    
    # Precompute adjacency as list for speed
    adj_list = {v: list(adj[v]) for v in range(n)}
    
    def backtrack(path, visited):
        if deadline is not None and time.time() > deadline:
            return None
        current = path[-1]
        if len(path) == n:
            # Check if we can close the cycle
            if start in adj[current]:
                return list(path)
            return None
        
        # Order neighbors: prefer those with fewer unvisited neighbors (Warnsdorff)
        neighbors = []
        for nb in adj_list[current]:
            if nb not in visited:
                # Count unvisited neighbors of nb
                unvisited_count = 0
                for nn in adj_list[nb]:
                    if nn not in visited:
                        unvisited_count += 1
                neighbors.append((unvisited_count, nb))
        
        neighbors.sort()
        
        for _, nb in neighbors:
            new_path = path + [nb]
            new_visited = visited | {nb}
            result = backtrack(new_path, new_visited)
            if result is not None:
                return result
        return None
    
    path = [start]
    visited = {start}
    result = backtrack(path, visited)
    return result

def _remove_cycle_edges(edges, cycle):
    """Remove edges used by the cycle from the edge list."""
    n = P['n']
    cycle_edges = set()
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        cycle_edges.add(_edge_key(u, v))
    remaining = []
    for u, v in edges:
        if _edge_key(u, v) not in cycle_edges:
            remaining.append([u, v])
    return remaining

def _try_decomposition(edges, max_cycles=3, deadline=None):
    """Try to find edge-disjoint Hamiltonian cycles."""
    current_edges = list(edges)
    cycles = []
    
    for i in range(max_cycles):
        if deadline is not None and time.time() > deadline - 5:
            break
        adj = _build_adjacency(current_edges)
        
        # Check all vertices have degree >= 2
        valid = True
        for v in range(P['n']):
            if len(adj[v]) < 2:
                valid = False
                break
        if not valid:
            break
        
        # Try multiple starting vertices and random orderings
        found = False
        starts = list(range(P['n']))
        random.shuffle(starts)
        
        for start in starts:
            if deadline is not None and time.time() > deadline - 5:
                break
            # Shuffle adjacency for diversity
            adj_shuffled = {v: list(adj[v]) for v in range(P['n'])}
            for v in adj_shuffled:
                random.shuffle(adj_shuffled[v])
            
            cycle = _find_hamiltonian_cycle(adj_shuffled, start=start, deadline=deadline)
            if cycle is not None:
                cycles.append(cycle)
                current_edges = _remove_cycle_edges(current_edges, cycle)
                found = True
                break
        
        if not found:
            break
    
    return cycles

def construct():
    deadline = time.time() + 75  # Safety margin
    
    # Try to find 3 edge-disjoint Hamiltonian cycles
    best_cycles = []
    
    # Multiple attempts with different random seeds
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        random.seed(attempt * 12345 + 67890)
        cycles = _try_decomposition(P['edges'], max_cycles=3, deadline=deadline)
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
        if len(best_cycles) >= 3:
            break
    
    if not best_cycles:
        # Fallback: return a single Hamiltonian cycle
        adj = _build_adjacency(P['edges'])
        cycle = _find_hamiltonian_cycle(adj, start=0, deadline=deadline)
        if cycle is not None:
            best_cycles = [cycle]
    
    # Ensure all are lists of Python ints
    result = []
    for c in best_cycles:
        result.append([int(v) for v in c])
    
    return result if result else [list(range(P['n']))]
# EVOLVE-BLOCK-END
