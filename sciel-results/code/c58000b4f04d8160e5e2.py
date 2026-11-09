"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _build_adjacency(edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _find_hc_fast(adj, used_edges, n, deadline, rng):
    """Fast Hamiltonian cycle finder with multiple strategies."""
    # Strategy 1: Warnsdorff with randomization
    def _warnsdorff(start):
        visited = {start}
        path = [start]
        
        def bt():
            if time.time() > deadline:
                return None
            if len(path) == n:
                last = path[-1]
                if _edge_key(last, start) not in used_edges:
                    return list(path)
                return None
            
            cur = path[-1]
            neighbors = []
            for nb in adj[cur]:
                if nb not in visited and _edge_key(cur, nb) not in used_edges:
                    fc = 0
                    for nb2 in adj[nb]:
                        if nb2 not in visited and nb2 != cur and _edge_key(nb, nb2) not in used_edges:
                            fc += 1
                    neighbors.append((fc, rng.random(), nb))
            
            if not neighbors:
                return None
            
            neighbors.sort()
            for _, _, nb in neighbors:
                ek = _edge_key(cur, nb)
                if ek in used_edges:
                    continue
                visited.add(nb)
                path.append(nb)
                res = bt()
                if res is not None:
                    return res
                path.pop()
                visited.remove(nb)
            return None
        
        return bt()
    
    # Try multiple starts and strategies
    starts = list(range(n))
    rng.shuffle(starts)
    
    for start in starts:
        if time.time() > deadline:
            break
        result = _warnsdorff(start)
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = P['n']
    adj = _build_adjacency(P['edges'])
    
    best_cycles = []
    
    # Try many random configurations to find 3 edge-disjoint Hamiltonian cycles
    max_attempts = 200
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        rng = random.Random(attempt * 12345 + 67890)
        
        # Shuffle adjacency for diversity
        adj_shuffled = {v: list(neighbors) for v, neighbors in adj.items()}
        for v in adj_shuffled:
            rng.shuffle(adj_shuffled[v])
        
        used_edges = set()
        cycles = []
        
        for cycle_idx in range(3):
            if time.time() > deadline:
                break
            
            # Try to find Hamiltonian cycle avoiding used edges
            cycle = _find_hc_fast(adj_shuffled, used_edges, n, deadline, rng)
            
            if cycle is None:
                break
            
            # Mark edges as used
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                used_edges.add(_edge_key(u, v))
            
            cycles.append(cycle)
        
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
            if len(cycles) == 3:
                break
    
    # Validate and convert to proper types
    if not best_cycles:
        return []
    
    valid_cycles = []
    seen_edges = set()
    
    for cycle in best_cycles:
        if len(cycle) < 3:
            continue
        if len(set(cycle)) != len(cycle):
            continue
        
        edges_in_cycle = set()
        valid = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            ek = _edge_key(u, v)
            if ek in seen_edges or ek in edges_in_cycle:
                valid = False
                break
            edges_in_cycle.add(ek)
        
        if valid:
            seen_edges.update(edges_in_cycle)
            valid_cycles.append([int(v) for v in cycle])
    
    return valid_cycles
# EVOLVE-BLOCK-END
