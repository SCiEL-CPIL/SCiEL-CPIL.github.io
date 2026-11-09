import numpy as np
import time

def ring_routing() -> np.ndarray:
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)
    
    # Precompute edge masks
    # mask[d, e] is 1 if edge e is used by demand d in CW direction
    mask_cw = np.zeros((D, n), dtype=int)
    for d, (s, t, w) in enumerate(demands):
        e = s
        while e != t:
            mask_cw[d, e % n] = 1
            e = (e + 1) % n
            
    weights = np.array([w for _, _, w in demands], dtype=float)
    
    # Load calculation:
    # If routing vector x is 0 (CW), load = sum(weights[d] * mask_cw[d])
    # If routing vector x is 1 (CCW), load = sum(weights[d] * (1 - mask_cw[d]))
    # We can precompute base_cw_load and delta_load
    base_cw_load = (weights[:, None] * mask_cw).sum(axis=0)
    # delta[d] = load contribution if CCW - load contribution if CW
    # CCW uses edges not in CW. So CCW contribution is weights[d] * (1 - mask_cw[d])
    # delta[d] = weights[d] * (1 - mask_cw[d]) - weights[d] * mask_cw[d]
    delta = weights[:, None] * (1 - 2 * mask_cw)
    
    def get_loads(x):
        # x is array of 0s and 1s
        return base_cw_load + delta[x.astype(bool)].sum(axis=0)
        
    def get_max_load(x):
        return np.max(get_loads(x))

    # Simulated Annealing
    start_time = time.time()
    deadline = start_time + 5.0 # Small budget, problem is tiny
    best_x = np.zeros(D, dtype=int)
    best_score = get_max_load(best_x)
    
    # Try a few random restarts
    rng = np.random.default_rng(42)
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Initialize randomly
        x = rng.integers(0, 2, size=D)
        current_score = get_max_load(x)
        
        if current_score < best_score:
            best_score = current_score
            best_x = x.copy()
            
        temp = 1.0
        min_temp = 0.01
        cooling = 0.95
        
        while temp > min_temp:
            if time.time() > deadline:
                break
                
            # Pick a random demand to flip
            d = rng.integers(0, D)
            
            # Calculate new score
            x[d] = 1 - x[d]
            new_score = get_max_load(x)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_x = x.copy()
            else:
                # Accept with probability exp(-(new_score - current_score) / temp)
                # Note: scores are integers or simple floats, so diff is small
                prob = np.exp(-(new_score - current_score) / temp)
                if rng.random() < prob:
                    current_score = new_score
                else:
                    x[d] = 1 - x[d] # revert
                    
            temp *= cooling
            
    # Convert to required format: >= 0.5 for CCW, < 0.5 for CW
    result = best_x.astype(float)
    # Ensure strict compliance: 1.0 for CCW, 0.0 for CW
    return result
