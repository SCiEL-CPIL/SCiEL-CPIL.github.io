# EVOLVE-BLOCK-START
import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Use a 3-point support {0, 1, M} with optimized weights and M.
    We perform a fine grid search over the simplex of weights and a range of M values,
    followed by local refinement.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin

    def calc_prob(support, weights):
        """
        Compute P[X1 + X2 + X3 < 2*X4] exactly in float64.
        P = sum_{k,l,i,j} w_k w_l w_i w_j * 1[s_i + s_j < 2*s_k - s_l]
        where k is index for X4, l for X1, i for X2, j for X3.
        """
        n = len(support)
        # Precompute pairwise sums S[i,j] = s_i + s_j
        S = support[:, None] + support[None, :]
        
        total_prob = 0.0
        # Vectorized computation over k and l
        # For each (k, l), threshold T = 2*s_k - s_l
        # We need sum_{i,j} w_i w_j * 1[S[i,j] < T]
        
        # We can compute the inner sum for all (k,l) at once if we want, but n is small.
        # Let's stick to loops for clarity and speed for small n, or vectorize if n is larger.
        # With n=3, n^4=81, so loops are fine.
        
        # Optimization: Precompute the "CDF" of the sum distribution?
        # The sum X2+X3 takes values in {s_i+s_j}.
        # Let's just do the double loop over k,l and inner double loop over i,j.
        
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * support[k] - support[l]
                # Sum w_i * w_j for s_i + s_j < threshold
                inner_sum = 0.0
                for i in range(n):
                    for j in range(n):
                        if S[i, j] < threshold:
                            inner_sum += weights[i] * weights[j]
                total_prob += weights[k] * weights[l] * inner_sum
        return total_prob

    best_prob = -1.0
    best_weights = None
    best_support = None

    # Support points: 0, 1, M
    # Weights: p0, p1, pM. p0+p1+pM = 1.
    
    # Grid search parameters
    # M range: [1.5, 1000]
    # Weights: p0, p1 in [0, 1], pM = 1 - p0 - p1 >= 0.
    
    # Let's do a coarse-to-fine search.
    
    # Phase 1: Coarse Grid
    M_values = np.logspace(np.log10(1.5), np.log10(2000), 40)
    # For each M, grid search weights.
    # Simplex grid: p0, p1.
    # Let's use a resolution of 20x20 for the simplex initially.
    
    grid_size = 30
    
    for M in M_values:
        if time.time() > deadline:
            break
        support = np.array([0.0, 1.0, M], dtype=np.float64)
        
        for i in range(grid_size + 1):
            p0 = i / grid_size
            for j in range(grid_size + 1 - i):
                p1 = j / grid_size
                pM = 1.0 - p0 - p1
                if pM < -1e-9:
                    continue
                weights = np.array([p0, p1, pM], dtype=np.float64)
                prob = calc_prob(support, weights)
                if prob > best_prob:
                    best_prob = prob
                    best_weights = weights.copy()
                    best_support = support.copy()

    # Phase 2: Local Refinement
    if best_weights is not None:
        current_weights = best_weights.copy()
        current_support = best_support.copy()
        current_prob = best_prob
        
        # Refine M and weights simultaneously using a simple hill climbing / Nelder-Mead-like approach
        # or just coordinate descent with adaptive step size.
        
        step_w = 0.05
        step_M = 1.0
        
        for _ in range(500):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adjusting weights
            for dim in range(3):
                for direction in [-1, 1]:
                    new_weights = current_weights.copy()
                    new_weights[dim] += direction * step_w
                    # Renormalize to keep sum=1 and non-negative
                    if np.any(new_weights < 0):
                        # Project back? Or just skip if too far.
                        # Simple projection: clip to 0 and renormalize
                        new_weights = np.maximum(new_weights, 0)
                        s = new_weights.sum()
                        if s <= 1e-9:
                            continue
                        new_weights = new_weights / s
                    
                    new_prob = calc_prob(current_support, new_weights)
                    if new_prob > current_prob:
                        current_weights = new_weights
                        current_prob = new_prob
                        improved = True
                        # Don't break, try other dims too? Or break to accept move?
                        # Usually accept one move at a time.
                        break
                if improved:
                    break
            
            # Try adjusting M
            if not improved:
                for direction in [-1, 1]:
                    new_M = current_support[2] + direction * step_M
                    if new_M <= 1.0:  # Must be > 1 to be distinct
                        continue
                    new_support = np.array([0.0, 1.0, new_M], dtype=np.float64)
                    new_prob = calc_prob(new_support, current_weights)
                    if new_prob > current_prob:
                        current_support = new_support
                        current_prob = new_prob
                        improved = True
                        break
            
            if not improved:
                step_w *= 0.8
                step_M *= 0.8
                if step_w < 1e-5 and step_M < 1e-3:
                    break
            
            if current_prob > best_prob:
                best_prob = current_prob
                best_weights = current_weights.copy()
                best_support = current_support.copy()

    # Fallback
    if best_weights is None:
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights

    # Final sanity check
    if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9) or best_weights.sum() <= 1e-9:
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights

    return best_support, best_weights

# EVOLVE-BLOCK-END
