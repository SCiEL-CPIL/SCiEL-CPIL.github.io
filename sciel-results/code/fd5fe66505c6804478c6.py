# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem using Local Search with Random Restarts.
    """
    n_nodes = 10
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
    D = len(demands)
    
    # Precompute edge sets for each demand in each direction
    # cw_matrix[i, e] = weight if demand i is routed clockwise, 0 otherwise
    # ccw_matrix[i, e] = weight if demand i is routed counter-clockwise, 0 otherwise
    cw_matrix = np.zeros((D, n_nodes))
    ccw_matrix = np.zeros((D, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise edges: s, s+1, ..., t-1 (mod 10)
        if s <= t:
            cw_edges = range(s, t)
        else:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        
        for e in cw_edges:
            cw_matrix[i, e] = w
            
        # Counter-clockwise edges: all edges NOT in clockwise path
        # Since it's a ring, CCW path is the complement of CW path
        for e in range(n_nodes):
            if cw_matrix[i, e] == 0:
                ccw_matrix[i, e] = w

    def get_loads(routing):
        """
        Calculate edge loads for a given routing vector.
        routing: np.array of shape (D,), 0 for CW, 1 for CCW
        """
        # loads = sum over demands of (routing[i] * ccw_matrix[i] + (1-routing[i]) * cw_matrix[i])
        # loads = cw_matrix.sum(axis=0) + routing @ (ccw_matrix - cw_matrix)
        return cw_matrix.sum(axis=0) + routing @ (ccw_matrix - cw_matrix)

    def get_max_load(routing):
        return np.max(get_loads(routing))

    # Local Search with Random Restarts
    best_routing = np.zeros(D)
    best_score = float('inf')
    
    # Set random seed for reproducibility if needed, but random restarts are better
    # We will use a fixed number of restarts and iterations
    num_restarts = 50
    max_iterations = 1000
    deadline = time.time() + 500.0 # Leave 100s margin
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize routing: random 0s and 1s
        current_routing = np.random.randint(0, 2, D).astype(np.float64)
        current_max_load = get_max_load(current_routing)
        
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try flipping each single demand
            best_flip_idx = -1
            best_flip_load = current_max_load
            
            # Vectorized single-flip evaluation
            # For each i, new_routing = current_routing with i flipped
            # We can compute loads for all single flips at once?
            # loads_i = current_loads - cw_matrix[i] + ccw_matrix[i] (if current is 0)
            #          = current_loads - ccw_matrix[i] + cw_matrix[i] (if current is 1)
            
            current_loads = get_loads(current_routing)
            
            # Create a matrix of potential new loads for all single flips
            # shape (D, n_nodes)
            # If current_routing[i] == 0, flip to 1: add (ccw[i] - cw[i])
            # If current_routing[i] == 1, flip to 0: add (cw[i] - ccw[i])
            
            delta_single = np.where(current_routing[:, np.newaxis] == 0, 
                                    ccw_matrix - cw_matrix, 
                                    cw_matrix - ccw_matrix)
            
            potential_loads = current_loads[np.newaxis, :] + delta_single
            potential_max_loads = np.max(potential_loads, axis=1)
            
            min_potential_load = np.min(potential_max_loads)
            
            if min_potential_load < current_max_load - 1e-9:
                # Found an improving flip
                best_flip_idx = np.argmin(potential_max_loads)
                current_routing[best_flip_idx] = 1 - current_routing[best_flip_idx]
                current_max_load = min_potential_load
                improved = True
                continue
            
            # If no single flip improves, try 2-flips (optional, but helps escape local minima)
            # To keep it simple and fast, we might skip 2-flips or do a limited search.
            # Given D=15, there are 105 pairs.
            # Let's try all pairs if we are stuck.
            
            # Vectorized 2-flip evaluation is trickier.
            # Let's just do a simple loop for pairs if stuck.
            # This might be slow if done every iteration, so we only do it when stuck.
            
            found_pair = False
            # Randomly sample some pairs to avoid O(D^2) every time? 
            # Or just do all pairs since D=15 is small.
            for i in range(D):
                for j in range(i+1, D):
                    # Flip i and j
                    orig_i = current_routing[i]
                    orig_j = current_routing[j]
                    
                    current_routing[i] = 1 - orig_i
                    current_routing[j] = 1 - orig_j
                    
                    new_max_load = get_max_load(current_routing)
                    
                    if new_max_load < current_max_load - 1e-9:
                        current_max_load = new_max_load
                        improved = True
                        found_pair = True
                        break
                    else:
                        # Revert
                        current_routing[i] = orig_i
                        current_routing[j] = orig_j
                
                if found_pair:
                    break
            
            if not found_pair and not improved:
                break # Local minimum reached
        
        if current_max_load < best_score:
            best_score = current_max_load
            best_routing = current_routing.copy()
            
    return best_routing

# EVOLVE-BLOCK-END
