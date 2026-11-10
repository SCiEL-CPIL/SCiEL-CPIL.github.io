import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    # Build adjacency list as sets of edges for quick removal
    # edges is a set of frozensets or tuples (u, v) with u < v
    remaining_edges = set()
    adj = {i: set() for i in range(n)}
    for u, v in P['edges']:
        if u > v: u, v = v, u
        remaining_edges.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
    
    cycles_found = []
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    def find_hamiltonian_cycle():
        # Returns a list of vertices representing a Hamiltonian cycle, or None
        # Uses backtracking
        # Fix start vertex to 0 to break symmetry
        path = [0]
        visited = {0}
        
        def backtrack():
            if time.time() > deadline:
                return None
            
            if len(path) == n:
                # Check if edge from last to 0 exists
                last = path[-1]
                if (0, last) in remaining_edges or (last, 0) in remaining_edges:
                    # Normalize edge key
                    if last > 0:
                        if (0, last) in remaining_edges: return path[:]
                        if (last, 0) in remaining_edges: return path[:]
                return None
            
            curr = path[-1]
            # Sort neighbors by degree in remaining graph (heuristic: least constraining value or most constrained)
            # Let's try neighbors with smallest remaining degree first (Warnsdorff's rule)
            neighbors = list(adj[curr])
            neighbors.sort(key=lambda x: len(adj[x]))
            
            for nxt in neighbors:
                if nxt not in visited:
                    # Check if edge exists
                    edge_key = (curr, nxt) if curr < nxt else (nxt, curr)
                    if edge_key in remaining_edges:
                        # Try this move
                        path.append(nxt)
                        visited.add(nxt)
                        
                        # Pruning: Check if any unvisited vertex is isolated from the rest
                        # Simple check: if an unvisited vertex has no neighbors in unvisited and no edge to 0 (if it's the last), prune
                        # This is expensive, so maybe just do a basic check:
                        # If len(unvisited) > 1, every unvisited vertex must have at least one neighbor in unvisited or be 0? No, 0 is visited.
                        # Actually, if an unvisited vertex has no edges to other unvisited vertices, it can only be the last one.
                        # If there are > 1 unvisited vertices, and one of them has no unvisited neighbors, it's a dead end.
                        dead_end = False
                        unvisited = [v for v in range(n) if v not in visited]
                        if len(unvisited) > 1:
                            for v in unvisited:
                                # Check if v has any neighbor in unvisited
                                has_unvisited_neighbor = False
                                for u in adj[v]:
                                    if u in unvisited:
                                        has_unvisited_neighbor = True
                                        break
                                if not has_unvisited_neighbor:
                                    dead_end = True
                                    break
                        
                        if not dead_end:
                            res = backtrack()
                            if res is not None:
                                return res
                        
                        path.pop()
                        visited.remove(nxt)
            
            return None
        
        return backtrack()

    while True:
        if time.time() > deadline:
            break
        
        cycle = find_hamiltonian_cycle()
        if cycle is None:
            break
        
        # Remove edges from cycle
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge_key = (u, v) if u < v else (v, u)
            if edge_key in remaining_edges:
                remaining_edges.remove(edge_key)
                adj[u].remove(v)
                adj[v].remove(u)
            else:
                # This shouldn't happen if find_hamiltonian_cycle is correct
                break
        
        cycles_found.append(cycle)
        
        # If we found 3 cycles, we hit the theoretical max (72 // 21 = 3)
        if len(cycles_found) >= 3:
            break

    if not cycles_found:
        # Fallback to seed solution
        return [list(range(n))]
        
    # Ensure types are correct
    result = []
    for c in cycles_found:
        result.append([int(v) for v in c])
        
    return result
# EVOLVE-BLOCK-END
