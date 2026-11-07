"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Build edge set for O(1) lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    def is_edge(u, v):
        return (min(u,v), max(u,v)) in edge_set
    
    def find_hamiltonian_cycle(used_edges, start_vertex=0):
        """Find a Hamiltonian cycle using DFS with pruning."""
        n = P['n']
        
        # Fix start vertex to break symmetry
        path = [start_vertex]
        visited = {start_vertex}
        
        def dfs():
            if time.time() > deadline:
                return None
            
            if len(path) == n:
                # Check if we can close the cycle
                last = path[-1]
                if is_edge(last, start_vertex) and (min(last, start_vertex), max(last, start_vertex)) not in used_edges:
                    return path[:]
                return None
            
            last = path[-1]
            # Try neighbors in sorted order for determinism
            for neighbor in sorted(adj[last]):
                if neighbor in visited:
                    continue
                # Check edge not used
                e = (min(last, neighbor), max(last, neighbor))
                if e in used_edges:
                    continue
                # Pruning: check if neighbor can still reach all unvisited vertices
                # Simple pruning: if neighbor is the only unvisited vertex's neighbor, skip
                path.append(neighbor)
                visited.add(neighbor)
                
                result = dfs()
                if result is not None:
                    return result
                
                path.pop()
                visited.remove(neighbor)
            
            return None
        
        return dfs()
    
    # Try to find 3 edge-disjoint Hamiltonian cycles
    used_edges = set()
    cycles = []
    
    for attempt in range(3):
        if time.time() > deadline:
            break
        cycle = find_hamiltonian_cycle(used_edges)
        if cycle is None:
            break
        # Mark edges as used
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            e = (min(u,v), max(u,v))
            used_edges.add(e)
        cycles.append(cycle)
    
    if not cycles:
        # Fallback: return the seed solution
        return [list(range(n))]
    
    # Convert to proper int lists
    result = []
    for cycle in cycles:
        result.append([int(v) for v in cycle])
    
    return result
# EVOLVE-BLOCK-END
