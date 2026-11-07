"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    # Build adjacency structure
    adj = {i: [] for i in range(n)}
    edge_set = set()
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((min(u,v), max(u,v)))
    
    # Sort adjacency lists for determinism
    for i in range(n):
        adj[i].sort()
    
    start_time = time.time()
    deadline = start_time + 50  # safety margin
    
    best_result = [list(range(n))]  # fallback: single Hamiltonian cycle
    
    def find_hamiltonian_cycle(used_edges, current_cycle, visited, start):
        """Find a Hamiltonian cycle starting from 'start' using only unused edges."""
        if time.time() > deadline:
            return None
        
        if len(current_cycle) == n:
            # Check if we can close the cycle
            last = current_cycle[-1]
            if (min(last, start), max(last, start)) not in used_edges:
                return current_cycle[:]
            return None
        
        last = current_cycle[-1]
        for neighbor in adj[last]:
            if neighbor in visited:
                continue
            if neighbor == start and len(current_cycle) < n:
                continue
            edge_key = (min(last, neighbor), max(last, neighbor))
            if edge_key in used_edges:
                continue
            
            current_cycle.append(neighbor)
            visited.add(neighbor)
            used_edges.add(edge_key)
            
            result = find_hamiltonian_cycle(used_edges, current_cycle, visited, start)
            if result is not None:
                return result
            
            used_edges.remove(edge_key)
            visited.remove(neighbor)
            current_cycle.pop()
        
        return None
    
    def try_pack_cycles(cycles_found, used_edges):
        """Try to find more edge-disjoint Hamiltonian cycles."""
        if time.time() > deadline:
            return cycles_found
        
        if len(cycles_found) >= 10:
            return cycles_found
        
        # Try to find another Hamiltonian cycle
        # Start from vertex 0 for determinism
        start = 0
        current_cycle = [start]
        visited = {start}
        
        result = find_hamiltonian_cycle(used_edges, current_cycle, visited, start)
        if result is not None:
            cycles_found.append(result)
            # Remove edges of this cycle from available
            for i in range(len(result)):
                u = result[i]
                v = result[(i+1) % len(result)]
                edge_key = (min(u,v), max(u,v))
                used_edges.add(edge_key)
            return try_pack_cycles(cycles_found, used_edges)
        
        return cycles_found
    
    # First, verify we can find at least one Hamiltonian cycle
    # Start with vertex 0
    start = 0
    current_cycle = [start]
    visited = {start}
    used_edges = set()
    
    result = find_hamiltonian_cycle(used_edges, current_cycle, visited, start)
    if result is None:
        # Try different starting vertices
        for s in range(1, n):
            current_cycle = [s]
            visited = {s}
            used_edges = set()
            result = find_hamiltonian_cycle(used_edges, current_cycle, visited, s)
            if result is not None:
                break
    
    if result is None:
        return best_result
    
    cycles = [result]
    
    # Remove edges of first cycle
    for i in range(len(result)):
        u = result[i]
        v = result[(i+1) % len(result)]
        edge_key = (min(u,v), max(u,v))
        used_edges.add(edge_key)
    
    # Try to find more cycles
    cycles = try_pack_cycles(cycles, used_edges)
    
    # Convert to proper format
    final_cycles = []
    for cycle in cycles:
        final_cycles.append([int(v) for v in cycle])
    
    if not final_cycles:
        return best_result
    
    return final_cycles
# EVOLVE-BLOCK-END
