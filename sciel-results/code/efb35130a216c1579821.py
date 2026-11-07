"""Packing edge-disjoint Hamiltonian cycles on 21 vertices."""
import time

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list with edge sets for fast lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Track available edges
    available_edges = set()
    for u, v in edges:
        available_edges.add(frozenset((u, v)))
    
    cycles_found = []
    deadline = time.time() + 55  # Safety margin before 60s limit
    
    def find_hamiltonian_cycle(start, used_edges, remaining_verts):
        """Find a Hamiltonian cycle starting from 'start' using only unused edges."""
        if len(used_edges) > 0 and time.time() > deadline:
            return None
        
        path = [start]
        visited = {start}
        
        def dfs(current, path, visited, used_edges):
            if len(path) == n:
                # Check if we can close the cycle
                if frozenset((current, start)) in used_edges:
                    return list(path)
                return None
            
            # Warnsdorff's heuristic: sort neighbors by number of available unvisited neighbors
            neighbors = []
            for nb in adj[current]:
                if nb not in visited and frozenset((current, nb)) not in used_edges:
                    # Count how many unvisited neighbors this neighbor has (excluding current)
                    count = 0
                    for nb2 in adj[nb]:
                        if nb2 not in visited and nb2 != current and frozenset((nb, nb2)) not in used_edges:
                            count += 1
                    neighbors.append((count, nb))
            
            neighbors.sort()
            
            for _, nb in neighbors:
                edge = frozenset((current, nb))
                if edge not in used_edges:
                    new_used = used_edges | {edge}
                    result = dfs(nb, path + [nb], visited | {nb}, new_used)
                    if result is not None:
                        return result
            return None
        
        return dfs(start, path, visited, used_edges)
    
    # Try to find up to 3 Hamiltonian cycles
    for _ in range(3):
        if time.time() > deadline:
            break
        
        # Try different starting vertices for diversity
        found = False
        for start in range(n):
            if time.time() > deadline:
                break
            cycle = find_hamiltonian_cycle(start, set(), set())
            if cycle is not None:
                # Verify the cycle is valid
                valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge = frozenset((u, v))
                    if edge not in available_edges:
                        valid = False
                        break
                if valid:
                    # Remove edges from available
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        available_edges.discard(frozenset((u, v)))
                    cycles_found.append(cycle)
                    found = True
                    break
        
        if not found:
            break
    
    # Ensure we have at least one cycle; if none found, return a valid fallback
    if not cycles_found:
        # Fallback: return a single Hamiltonian cycle using all edges
        # This should always be findable given the graph structure
        for start in range(n):
            cycle = find_hamiltonian_cycle(start, set(), set())
            if cycle is not None:
                cycles_found.append(cycle)
                break
    
    # Return as list of lists of Python ints
    return [list(cycle) for cycle in cycles_found]
# EVOLVE-BLOCK-END
