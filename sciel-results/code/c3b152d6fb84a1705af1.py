import time
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    deadline = time.time() + 55.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def edge_key(u, v):
        return (min(u, v), max(u, v))

    def find_hamiltonian_cycle(adj, available_edges, start=0):
        """
        Finds a Hamiltonian cycle using backtracking with Warnsdorff's heuristic.
        adj: adjacency list of the full graph
        available_edges: set of edges that can be used
        start: starting vertex
        """
        # Pre-compute degrees in available graph for heuristic
        # We update this dynamically or use a static approximation?
        # Static approximation based on initial available edges is faster but less accurate.
        # Dynamic is safer.
        
        path = [start]
        visited = {start}
        
        # To speed up, we can pre-sort neighbors
        # But Warnsdorff needs current degree.
        
        def backtrack():
            if time.time() > deadline:
                return None
            
            if len(path) == n:
                last = path[-1]
                if edge_key(last, start) in available_edges:
                    return list(path)
                return None
            
            last = path[-1]
            
            # Get valid neighbors
            candidates = []
            for nb in adj[last]:
                if nb not in visited and edge_key(last, nb) in available_edges:
                    candidates.append(nb)
            
            if not candidates:
                return None
                
            # Warnsdorff's heuristic: sort by number of available neighbors (degree in remaining graph)
            # We need to count how many unvisited neighbors each candidate has in available_edges
            def get_degree(v):
                count = 0
                for nb in adj[v]:
                    if nb not in visited and edge_key(v, nb) in available_edges:
                        count += 1
                return count
            
            candidates.sort(key=get_degree)
            
            for nb in candidates:
                # Optimization: If the last vertex (before nb) has only 1 available edge left 
                # (the one we are taking), we must take it. 
                # This is implicit in the "dead end" check, but sorting helps.
                
                # Tentative move
                path.append(nb)
                visited.add(nb)
                
                # Check if this move creates a dead end for the remaining path
                # Specifically, if any unvisited vertex now has 0 available edges, fail.
                # This is expensive to check fully every step.
                # Warnsdorff usually handles this well enough.
                
                result = backtrack()
                if result:
                    return result
                
                path.pop()
                visited.remove(nb)
                
            return None
            
        return backtrack()

    cycles = []
    available = set(edge_key(u, v) for u, v in edges)
    
    # We need 3 cycles
    for i in range(3):
        found = None
        # Try different starts if the first one fails
        # But with good heuristics, start 0 usually works.
        # To be safe, try a few starts.
        for start in range(n):
            found = find_hamiltonian_cycle(adj, available, start)
            if found:
                break
            # If timeout or fail, try next start
            if time.time() > deadline:
                break
            
        if not found:
            break
            
        # Remove edges of this cycle
        for j in range(len(found)):
            u = found[j]
            v = found[(j + 1) % len(found)]
            available.discard(edge_key(u, v))
            
        cycles.append(found)
        
        # If we have fewer edges than needed for a cycle, stop
        if len(available) < n:
            break

    if len(cycles) > 0:
        return [list(c) for c in cycles]
    else:
        return []

# EVOLVE-BLOCK-END
