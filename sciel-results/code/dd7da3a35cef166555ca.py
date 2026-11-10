# EVOLVE-BLOCK-START
import numpy as np

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem using Local Search (Hill Climbing with Restarts).
    """
    n_nodes = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1),
        (0, 4, 5), (2, 8, 2), (5, 9, 3), (1, 4, 2),
        (6, 0, 4), (3, 8, 1), (7, 2, 3), (4, 9, 2),
        (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute edge masks
    # masks[d][0] is boolean array for CW
    # masks[d][1] is boolean array for CCW
    masks = np.zeros((D, 2, n_nodes), dtype=np.int8)
    weights = np.zeros(D, dtype=np.float64)
    
    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        # CW edges
        e = s
        while e != t:
            masks[d, 0, e] = 1
            e = (e + 1) % n_nodes
        # CCW edges
        for e in range(n_nodes):
            if masks[d, 0, e] == 0:
                masks[d, 1, e] = 1

    def get_loads(dirs):
        """
        Compute edge loads given direction array (0=CW, 1=CCW).
        """
        loads = np.zeros(n_nodes, dtype=np.float64)
        for d in range(D):
            dir_idx = dirs[d]
            loads += weights[d] * masks[d, dir_idx]
        return loads

    def improve(dirs):
        """
        Hill climbing: flip directions if it reduces max load.
        Returns improved dirs and best max load.
        """
        dirs = dirs.copy()
        loads = get_loads(dirs)
        current_max = np.max(loads)
        
        improved = True
        while improved:
            improved = False
            for d in range(D):
                # Try flipping demand d
                old_dir = dirs[d]
                new_dir = 1 - old_dir
                
                # Calculate delta loads
                # Load change = -w * mask[old] + w * mask[new]
                delta = weights[d] * (masks[d, new_dir] - masks[d, old_dir])
                new_loads = loads + delta
                new_max = np.max(new_loads)
                
                if new_max < current_max:
                    # Accept flip
                    dirs[d] = new_dir
                    loads = new_loads
                    current_max = new_max
                    improved = True
        
        return dirs, current_max

    best_dirs = None
    best_max_load = np.inf
    
    # Multiple random restarts
    np.random.seed(42) # For reproducibility in development, though evaluator might not care
    for restart in range(200):
        # Random initialization
        init_dirs = np.random.randint(0, 2, D)
        improved_dirs, max_load = improve(init_dirs)
        
        if max_load < best_max_load:
            best_max_load = max_load
            best_dirs = improved_dirs.copy()
            
        # Early termination if we hit a very low load (theoretical min is sum(weights)/n_nodes? No, that's for uniform distribution. 
        # Sum weights = 3+2+4+1+5+2+3+2+4+1+3+2+5+1+2 = 40.
        # Avg load per edge if perfectly balanced = 40/10 = 4.
        # So 4 is a lower bound. If we reach 4, it's optimal.
        if best_max_load <= 4.0:
            break

    return best_dirs.astype(np.float64)

# EVOLVE-BLOCK-END
