"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    deadline = time.time() + 80  # Safety margin before 90s kill
    n = P['n']
    all_edges = set()
    for e in P['edges']:
        all_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    def build_adj(edge_set):
        adj = {i: set() for i in range(n)}
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def find_ham_cycle(adj, start=None, max_time=5.0):
        """Find a Hamiltonian cycle using backtracking with heuristics."""
        t0 = time.time()
        if start is None:
            start = 0
        
        # Use DFS with neighbor ordering by degree (Warnsdorff-like)
        def dfs(path, visited, current):
            if time.time() - t0 > max_time:
                return None
            if len(path) == n:
                if current == start:
                    return path
                return None
            # Order neighbors by their degree in remaining graph (prefer low degree)
            neighbors = list(adj[current])
            neighbors.sort(key=lambda x: len(adj[x]))
            for nb in neighbors:
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    result = dfs(path, visited, nb)
                    if result is not None:
                        return result
                    path.pop()
                    visited.remove(nb)
            return None
        
        visited = {start}
        result = dfs([start], visited, start)
        return result
    
    def find_ham_cycle_rand(adj, max_time=3.0):
        """Find a Hamiltonian cycle with randomized neighbor ordering."""
        t0 = time.time()
        start = random.randint(0, n - 1)
        
        def dfs(path, visited, current):
            if time.time() - t0 > max_time:
                return None
            if len(path) == n:
                if current == start:
                    return path
                return None
            neighbors = list(adj[current])
            random.shuffle(neighbors)
            # Sort by degree as secondary criterion
            neighbors.sort(key=lambda x: (len(adj[x]), random.random()))
            for nb in neighbors:
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    result = dfs(path, visited, nb)
                    if result is not None:
                        return result
                    path.pop()
                    visited.remove(nb)
            return None
        
        visited = {start}
        result = dfs([start], visited, start)
        return result
    
    def cycle_edges(cycle):
        """Get set of edges from a cycle."""
        edges = set()
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edges.add((min(u, v), max(u, v)))
        return edges
    
    # Try to find 3 edge-disjoint Hamiltonian cycles
    best_cycles = []
    
    # Strategy 1: Sequential backtracking
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        remaining_edges = set(all_edges)
        cycles = []
        
        for c in range(3):
            adj = build_adj(remaining_edges)
            # Check if graph is still 2-regular or higher for all vertices
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg < 2:
                break
            
            found = None
            # Try multiple random starts
            for trial in range(50):
                if time.time() > deadline:
                    break
                if c == 0 and attempt % 2 == 0:
                    found = find_ham_cycle(adj, start=0, max_time=2.0)
                else:
                    found = find_ham_cycle_rand(adj, max_time=1.5)
                if found is not None:
                    break
            
            if found is None:
                break
            
            ce = cycle_edges(found)
            # Verify all edges are in remaining
            if not ce.issubset(remaining_edges):
                break
            
            cycles.append(found)
            remaining_edges -= ce
        
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
        
        if len(best_cycles) == 3:
            break
    
    # If we didn't get 3, try a different strategy: 2-factor decomposition
    if len(best_cycles) < 3:
        # Try to decompose into 3 2-factors, then merge cycles in each
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Find 3 perfect matchings whose union gives 3 2-factors
            # Actually, let's try: find 2 edge-disjoint Hamiltonian cycles,
            # then check if remaining edges form a Hamiltonian cycle
            remaining_edges = set(all_edges)
            cycles = []
            
            for c in range(3):
                adj = build_adj(remaining_edges)
                min_deg = min(len(adj[i]) for i in range(n))
                if min_deg < 2:
                    break
                
                found = None
                for trial in range(100):
                    if time.time() > deadline:
                        break
                    found = find_ham_cycle_rand(adj, max_time=1.0)
                    if found is not None:
                        break
                
                if found is None:
                    break
                
                ce = cycle_edges(found)
                if not ce.issubset(remaining_edges):
                    break
                
                cycles.append(found)
                remaining_edges -= ce
            
            if len(cycles) > len(best_cycles):
                best_cycles = cycles
            
            if len(best_cycles) == 3:
                break
    
    # Final verification and return
    if not best_cycles:
        # Fallback: return a single Hamiltonian cycle
        adj = build_adj(all_edges)
        cycle = find_ham_cycle(adj, start=0, max_time=10.0)
        if cycle is None:
            cycle = find_ham_cycle_rand(adj, max_time=10.0)
        if cycle is not None:
            best_cycles = [cycle]
        else:
            # Last resort: return a trivial valid cycle if possible
            # Find any cycle of length >= 3
            for e in P['edges']:
                pass
            # Return empty list would be invalid, so we need at least one cycle
            # Let's find any 3-cycle or longer
            adj = build_adj(all_edges)
            for u in range(n):
                for v in adj[u]:
                    for w in adj[v]:
                        if w != u and w in adj[u] and u < v < w:
                            best_cycles = [[u, v, w]]
                            break
                if best_cycles:
                    break
    
    # Validate and convert
    result = []
    used_edges = set()
    vertex_set = None
    
    for cycle in best_cycles:
        if not isinstance(cycle, list):
            cycle = list(cycle)
        # Ensure all vertices are int
        cycle = [int(v) for v in cycle]
        # Check distinct vertices
        if len(set(cycle)) != len(cycle):
            continue
        if len(cycle) < 3:
            continue
        # Check all edges exist in host
        valid = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge = (min(u, v), max(u, v))
            if edge not in all_edges:
                valid = False
                break
            if edge in used_edges:
                valid = False
                break
            used_edges.add(edge)
        if not valid:
            continue
        # Check vertex set
        cv = frozenset(cycle)
        if vertex_set is None:
            vertex_set = cv
        elif cv != vertex_set:
            continue
        if len(cycle) != n:
            continue
        result.append(cycle)
    
    if not result:
        # Absolute fallback
        adj = build_adj(all_edges)
        cycle = find_ham_cycle(adj, start=0, max_time=30.0)
        if cycle is not None:
            result = [[int(v) for v in cycle]]
    
    return result
# EVOLVE-BLOCK-END
