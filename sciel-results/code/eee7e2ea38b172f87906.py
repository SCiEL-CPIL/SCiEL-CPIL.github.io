"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}
    n = P['n']
    edges = [tuple(sorted(e)) for e in P['edges']]
    edge_set = set(edges)
    
    # Adjacency list for original graph
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find a Hamiltonian cycle as the initial solution
    def find_hamiltonian():
        path = [0]
        visited = {0}
        
        def backtrack(u):
            if len(path) == n:
                if 0 in adj[u]:
                    return True
                return False
            for v in sorted(adj[u] - visited):
                path.append(v)
                visited.add(v)
                if backtrack(v):
                    return True
                visited.remove(v)
                path.pop()
            return False
        
        if backtrack(0):
            return path
        return None
    
    seed_cycle = find_hamiltonian()
    best_sol = [seed_cycle] if seed_cycle else []
    
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    def find_cycle(rem_edge_set, used_cycles):
        """Find a Hamiltonian cycle in the remaining graph that doesn't conflict with used cycles."""
        if not rem_edge_set:
            return None
            
        # Build adjacency for remaining edges
        rem_adj = {i: set() for i in range(n)}
        for u, v in rem_edge_set:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
            
        # Check connectivity
        visited = {0}
        stack = [0]
        while stack:
            u = stack.pop()
            for v in rem_adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        if len(visited) < n:
            return None
        
        # Check degree: every vertex must have degree >= 2 in remaining graph
        for i in range(n):
            if len(rem_adj[i]) < 2:
                return None
        
        # Randomized backtracking with multiple restarts
        for attempt in range(50):
            random.seed(random.randint(0, 2**31))
            path = [0]
            visited = {0}
            
            def backtrack(u):
                if len(path) == n:
                    if 0 in rem_adj[u]:
                        return True
                    return False
                
                # Get unvisited neighbors
                candidates = [v for v in rem_adj[u] if v not in visited]
                if not candidates:
                    return False
                
                # Warnsdorff's rule: prefer vertices with fewer onward moves
                candidates.sort(key=lambda x: len(rem_adj[x] - visited))
                
                # Randomize slightly to escape local optima
                if random.random() < 0.3 and len(candidates) > 1:
                    random.shuffle(candidates)
                
                for v in candidates:
                    path.append(v)
                    visited.add(v)
                    if backtrack(v):
                        return True
                    visited.remove(v)
                    path.pop()
                return False
            
            if backtrack(0):
                # Verify the cycle is valid and edge-disjoint from used cycles
                cycle_edges = set()
                for i in range(n):
                    u = path[i]
                    v = path[(i + 1) % n]
                    e = tuple(sorted((u, v)))
                    if e not in rem_edge_set:
                        return None
                    cycle_edges.add(e)
                
                # Check no edge conflicts with previous cycles
                for c in used_cycles:
                    for i in range(len(c)):
                        u = c[i]
                        v = c[(i + 1) % len(c)]
                        e = tuple(sorted((u, v)))
                        if e in cycle_edges:
                            return None
                
                return path
        return None
    
    # Search for maximum number of edge-disjoint cycles
    while time.time() < deadline:
        rem_edges = set(edge_set)
        cycles = []
        
        while time.time() < deadline:
            c = find_cycle(rem_edges, cycles)
            if c is None:
                break
            cycles.append(c)
            
            # Remove edges of this cycle from remaining edges
            for i in range(n):
                u = c[i]
                v = c[(i + 1) % n]
                e = tuple(sorted((u, v)))
                if e in rem_edges:
                    rem_edges.remove(e)
            
            # Update best solution
            if len(cycles) > len(best_sol):
                best_sol = [list(c) for c in cycles]
            
            # If we've reached the theoretical maximum (3 for 6-regular graph), stop
            if len(best_sol) == 3:
                return best_sol
    
    # Return best solution found, ensuring proper types
    result = []
    for c in best_sol:
        result.append([int(v) for v in c])
    
    if not result:
        # Fallback: return the seed cycle if we found one
        if seed_cycle:
            result = [[int(v) for v in seed_cycle]]
        else:
            # Last resort: construct a trivial valid cycle if possible
            # Find any 3-cycle
            for e1 in edges:
                for e2 in edges:
                    if e1[1] == e2[0] and e1 != e2:
                        if (e1[0], e2[1]) in edge_set or (e2[1], e1[0]) in edge_set:
                            result = [[e1[0], e1[1], e2[1]]]
                            break
                if result:
                    break
            if not result:
                # Return the Hamiltonian cycle as fallback
                if seed_cycle:
                    result = [[int(v) for v in seed_cycle]]
                else:
                    result = [[0, 1, 2]]  # Should not happen
    
    return result
# EVOLVE-BLOCK-END
