import numpy as np

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem by exhaustive search with pruning over all 
    2^15 routing combinations to find the one that minimizes the maximum edge load.
    """
    # Demand definitions: (source, target, weight)
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Pre-compute load vectors for clockwise and counter-clockwise routing
    cw_loads = np.zeros((n_demands, n_nodes), dtype=np.int32)
    ccw_loads = np.zeros((n_demands, n_nodes), dtype=np.int32)
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise: edges s, s+1, ..., t-1 (mod n_nodes)
        current = s
        while current != t:
            cw_loads[i, current] += w
            current = (current + 1) % n_nodes
        
        # Counter-clockwise: all other edges
        ccw_loads[i, :] = 0
        for e in range(n_nodes):
            if cw_loads[i, e] == 0:
                ccw_loads[i, e] = w

    best_max_load = float('inf')
    best_routing = np.zeros(n_demands)
    
    # Exhaustive search with pruning
    for mask in range(1 << n_demands):
        loads = np.zeros(n_nodes, dtype=np.int32)
        pruned = False
        
        for i in range(n_demands):
            if (mask >> i) & 1:
                loads += ccw_loads[i]
            else:
                loads += cw_loads[i]
            
            # Pruning: if current max load already exceeds or equals best, 
            # no need to add more demands (loads only increase)
            if np.max(loads) >= best_max_load:
                pruned = True
                break
        
        if not pruned:
            current_max_load = np.max(loads)
            if current_max_load < best_max_load:
                best_max_load = current_max_load
                best_routing = np.array([1.0 if (mask >> i) & 1 else 0.0 for i in range(n_demands)])
                
                # Early termination if we hit the theoretical lower bound.
                if best_max_load == 4:
                    break

    return best_routing
