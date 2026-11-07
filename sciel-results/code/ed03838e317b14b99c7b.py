"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

import random

def construct():
    n = P['n']
    all_edges = set()
    for e in P['edges']:
        all_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in all_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def find_hamiltonian_cycle(available_edges, randomize=False):
        """Find a Hamiltonian cycle using only edges in available_edges."""
        # Build adjacency from available edges
        adj_avail = [[] for _ in range(n)]
        for (u, v) in available_edges:
            adj_avail[u].append(v)
            adj_avail[v].append(u)
        
        if randomize:
            for i in range(n):
                random.shuffle(adj_avail[i])

        # Try starting from vertex 0
        start = 0
        path = [start]
        visited = {start}
        
        def dfs(current, path, visited):
            if len(path) == n:
                # Check if we can close the cycle
                if (min(current, start), max(current, start)) in available_edges:
                    return path[:]
                return None
            
            for neighbor in adj_avail[current]:
                if neighbor not in visited:
                    # Check edge exists in available_edges
                    if (min(current, neighbor), max(current, neighbor)) in available_edges:
                        visited.add(neighbor)
                        path.append(neighbor)
                        result = dfs(neighbor, path, visited)
                        if result is not None:
                            return result
                        path.pop()
                        visited.remove(neighbor)
            return None
        
        result = dfs(start, path, visited)
        if result is not None:
            return result
        return None
    
    # Try to find 3 cycles with some randomization if the first attempt fails
    for attempt in range(10):
        cycles = []
        remaining_edges = set(all_edges)
        success = True
        for i in range(3):
            cycle = find_hamiltonian_cycle(remaining_edges, randomize=(attempt > 0))
            if cycle is None:
                success = False
                break
            # Remove edges used in this cycle
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                edge = (min(u, v), max(u, v))
                if edge in remaining_edges:
                    remaining_edges.remove(edge)
            cycles.append(cycle)
        
        if success:
            return cycles

    # Fallback: return a single cycle if we can find one
    cycle = find_hamiltonian_cycle(set(all_edges))
    if cycle is not None:
        return [cycle]
    return [list(range(n))]
# EVOLVE-BLOCK-END
