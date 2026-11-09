# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    N = 10
    D = 15
    demands = [
        (0,5,3), (1,7,2), (2,9,4), (3,6,1), (0,4,5),
        (2,8,2), (5,9,3), (1,4,2), (6,0,4), (3,8,1),
        (7,2,3), (4,9,2), (8,1,5), (0,6,1), (5,2,2)
    ]
    
    # Precompute edge masks for CW and CCW routing for each demand
    # CW from s to t: edges s, s+1, ..., t-1 (mod N)
    # CCW from s to t: all other edges
    cw_masks = []
    ccw_masks = []
    for s, t, w in demands:
        cw = np.zeros(N)
        if s <= t:
            edges = list(range(s, t))
        else:
            edges = list(range(s, N)) + list(range(0, t))
        for e in edges:
            cw[e] = 1
        cw_masks.append(cw)
        ccw = 1.0 - cw
        ccw_masks.append(ccw)
    
    cw_masks = np.array(cw_masks)  # shape (D, N)
    ccw_masks = np.array(ccw_masks)  # shape (D, N)
    weights = np.array([d[2] for d in demands])  # shape (D,)
    
    def evaluate(routing):
        """Compute max edge load given routing vector (0=CW, 1=CCW)."""
        # routing: shape (D,), values 0 or 1
        # For each demand, if routing[d]==0 use cw_masks[d], else ccw_masks[d]
        # load on edge e = sum over d of weight[d] * mask[d][e]
        # mask[d][e] = cw_masks[d][e] if routing[d]==0 else ccw_masks[d][e]
        # = cw_masks[d][e] * (1 - routing[d]) + ccw_masks[d][e] * routing[d]
        # = cw_masks[d][e] + (ccw_masks[d][e] - cw_masks[d][e]) * routing[d]
        # = cw_masks[d][e] + (1 - 2*cw_masks[d][e]) * routing[d]
        
        # Efficient computation:
        # base_load = sum_d weight[d] * cw_masks[d]  (all CW)
        # delta_load = sum_d weight[d] * (ccw_masks[d] - cw_masks[d]) * routing[d]
        # total_load = base_load + delta_load
        
        # delta_masks[d] = ccw_masks[d] - cw_masks[d] = 1 - 2*cw_masks[d]
        delta_masks = ccw_masks - cw_masks  # shape (D, N)
        base_load = (cw_masks * weights[:, None]).sum(axis=0)  # shape (N,)
        # delta_load = (delta_masks * weights[:, None] * routing[:, None]).sum(axis=0)
        weighted_delta = delta_masks * weights[:, None]  # shape (D, N)
        delta_load = (weighted_delta * routing[:, None]).sum(axis=0)  # shape (N,)
        total_load = base_load + delta_load
        return np.max(total_load)
    
    def evaluate_loads(routing):
        """Return full load vector."""
        delta_masks = ccw_masks - cw_masks
        base_load = (cw_masks * weights[:, None]).sum(axis=0)
        weighted_delta = delta_masks * weights[:, None]
        delta_load = (weighted_delta * routing[:, None]).sum(axis=0)
        return base_load + delta_load
    
    # Simulated annealing
    best_routing = np.zeros(D, dtype=np.float64)
    best_score = evaluate(best_routing)
    
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds budget
    
    # Multiple restarts with simulated annealing
    rng = np.random.default_rng(42)
    
    for restart in range(500):
        if time.time() > deadline - 10:
            break
        
        # Start from best found so far or random
        if restart == 0:
            routing = np.zeros(D, dtype=np.float64)
        else:
            routing = rng.random(D).round()
        
        current_score = evaluate(routing)
        if current_score < best_score:
            best_score = current_score
            best_routing = routing.copy()
        
        # Simulated annealing
        T = 5.0
        T_min = 0.01
        alpha = 0.9995
        steps_per_iter = 5000
        
        for step in range(steps_per_iter):
            if time.time() > deadline - 10:
                break
            
            # Pick a random demand to flip
            d = rng.integers(0, D)
            old_val = routing[d]
            routing[d] = 1.0 - old_val
            
            new_score = evaluate(routing)
            delta = new_score - current_score
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_routing = routing.copy()
            else:
                routing[d] = old_val
            
            T *= alpha
            if T < T_min:
                break
    
    # Also try exhaustive search on subsets for final polish
    # Try flipping each single demand from best
    for i in range(D):
        if time.time() > deadline - 5:
            break
        trial = best_routing.copy()
        trial[i] = 1.0 - trial[i]
        s = evaluate(trial)
        if s < best_score:
            best_score = s
            best_routing = trial
    
    # Try pairs
    for i in range(D):
        if time.time() > deadline - 5:
            break
        for j in range(i+1, D):
            trial = best_routing.copy()
            trial[i] = 1.0 - trial[i]
            trial[j] = 1.0 - trial[j]
            s = evaluate(trial)
            if s < best_score:
                best_score = s
                best_routing = trial
    
    # Return: >= 0.5 means CCW, < 0.5 means CW
    return best_routing

# EVOLVE-BLOCK-END
