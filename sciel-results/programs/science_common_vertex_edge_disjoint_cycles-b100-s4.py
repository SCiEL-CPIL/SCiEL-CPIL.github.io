import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    
    def edge_key(u, v):
        return (min(u, v), max(u, v))
    
    def find_hamiltonian_cycles(used_edges, cycles_found, deadline):
        """Find edge-disjoint Hamiltonian cycles using backtracking."""
        if time.time() > deadline:
            return cycles_found
        
        if len(cycles_found) >= 10:
            return cycles_found
        
        # Try to find a Hamiltonian cycle in remaining edges
        cycle = _find_one_hc(adj, used_edges, n, deadline)
        if cycle is not None:
            cycles_found.append(cycle)
            # Mark edges as used
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                used_edges.add(edge_key(u, v))
            # Recurse to find more
            find_hamiltonian_cycles(used_edges, cycles_found, deadline)
        
        return cycles_found
    
    def _find_one_hc(adj, used_edges, n, deadline, start=None, path=None):
        """Find a single Hamiltonian cycle using available edges."""
        if time.time() > deadline:
            return None
        
        if path is None:
            path = [0]
            start = 0
        else:
            start = path[0]
        
        if len(path) == n:
            # Check if we can close the cycle
            last = path[-1]
            ek = edge_key(last, start)
            if ek not in used_edges and start in adj[last]:
                return list(path)
            return None
        
        last = path[-1]
        visited = set(path)
        
        # Get neighbors not yet visited and with available edges
        candidates = []
        for nb in adj[last]:
            if nb not in visited:
                ek = edge_key(last, nb)
                if ek not in used_edges:
                    candidates.append(nb)
        
        # Heuristic: prefer vertices with fewer remaining options
        candidates.sort(key=lambda v: len([x for x in adj[v] if x not in visited and edge_key(v, x) not in used_edges]))
        
        for nb in candidates:
            result = _find_one_hc(adj, used_edges, n, deadline, start, path + [nb])
            if result is not None:
                return result
        
        return None
    
    # Main search with restarts
    deadline = time.time() + 80  # 80s safety margin
    
    best_result = []
    
    # Try multiple random orderings / restarts
    random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        used_edges = set()
        cycles = []
        
        # Try to find up to 3 Hamiltonian cycles
        for _ in range(3):
            if time.time() > deadline:
                break
            
            # Shuffle adjacency for diversity
            shuffled_adj = [list(s) for s in adj]
            for lst in shuffled_adj:
                random.shuffle(lst)
            
            cycle = _find_one_hc(shuffled_adj, used_edges, n, deadline)
            if cycle is None:
                break
            
            cycles.append(cycle)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                used_edges.add(edge_key(u, v))
        
        if len(cycles) > len(best_result):
            best_result = cycles
            if len(best_result) >= 3:
                break
    
    # Ensure types are correct
    result = []
    for cycle in best_result:
        result.append([int(v) for v in cycle])
    
    # If we couldn't find any, return a valid single cycle
    if not result:
        # Fallback: just return a single Hamiltonian cycle
        used_edges = set()
        cycle = _find_one_hc(adj, used_edges, n, time.time() + 5)
        if cycle:
            result = [[int(v) for v in cycle]]
    
    return result
# EVOLVE-BLOCK-END
