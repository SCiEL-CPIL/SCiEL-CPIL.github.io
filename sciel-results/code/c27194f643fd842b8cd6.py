import numpy as np
import time
from scipy.optimize import differential_evolution

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem using differential evolution on the 
    continuous relaxation of the binary routing decisions, followed by 
    local search refinement.
    """
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
    
    # Pre-compute edge masks for CW and CCW
    cw_masks = []
    ccw_masks = []
    for s, t, w in demands:
        cw_set = set()
        current = s
        while current != t:
            cw_set.add(current)
            current = (current + 1) % n_nodes
        
        cw_mask = np.zeros(n_nodes, dtype=bool)
        for e in cw_set:
            cw_mask[e] = True
            
        ccw_mask = ~cw_mask
        
        cw_masks.append(cw_mask)
        ccw_masks.append(ccw_mask)
        
    cw_masks = np.array(cw_masks) # shape (15, 10)
    ccw_masks = np.array(ccw_masks) # shape (15, 10)
    
    weights = np.array([w for s, t, w in demands])
    
    def objective(x):
        # x is a vector of length n_demands in [0, 1]
        # Threshold at 0.5 to determine routing
        is_ccw = (x >= 0.5).astype(float)
        edge_selection = is_ccw[:, None] * ccw_masks + (1.0 - is_ccw[:, None]) * cw_masks
        loads = (weights[:, None] * edge_selection).sum(axis=0)
        
        return np.max(loads)

    start_time = time.time()
    deadline = start_time + 590.0
    
    # Differential Evolution
    bounds = [(0.0, 1.0) for _ in range(n_demands)]
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=500, 
            popsize=15, 
            seed=42,
            tol=0.01,
            init='latinhypercube'
        )
    except Exception:
        # Fallback to simple heuristic if DE fails
        return np.zeros(n_demands)
        
    best_x = result.x
    current_routing = (best_x >= 0.5).astype(float)
    current_load = objective(current_routing)
    
    # Local Search (Hill Climbing)
    improved = True
    while improved:
        improved = False
        for i in range(n_demands):
            if time.time() > deadline:
                break
            neighbor = current_routing.copy()
            neighbor[i] = 1.0 - neighbor[i]
            new_load = objective(neighbor)
            if new_load < current_load:
                current_routing = neighbor
                current_load = new_load
                improved = True
        if time.time() > deadline:
            break

    return current_routing
