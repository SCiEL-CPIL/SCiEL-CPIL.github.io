import numpy as np
from scipy.optimize import minimize

N = 14

def construct():
    # We optimize the angular gaps between consecutive points on a circle.
    # There are N gaps, but they sum to 2*pi, so we have N-1 free variables.
    
    def compute_ratio(deltas):
        """
        Compute the ratio of min triangle area to hull area for given gaps.
        deltas: array of length N, positive, summing to 2*pi.
        """
        if np.any(deltas <= 0) or np.abs(np.sum(deltas) - 2 * np.pi) > 1e-6:
            return 0.0
            
        # Calculate cumulative angles
        phi = np.zeros(N)
        phi[1:] = np.cumsum(deltas[:N-1])
        
        # Calculate the minimum triangle area numerator
        # Area ~ sin(a) + sin(b) + sin(c)
        min_num = float('inf')
        
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    a = phi[j] - phi[i]
                    b = phi[k] - phi[j]
                    c = 2 * np.pi - a - b
                    val = np.sin(a) + np.sin(b) + np.sin(c)
                    if val < min_num:
                        min_num = val
        
        # Calculate hull area denominator
        den = np.sum(np.sin(deltas))
        
        if den <= 0:
            return 0.0
            
        return min_num / den

    def objective(x):
        # x: array of first N-1 gaps
        if np.any(x <= 0):
            return 0.0
            
        deltas = np.empty(N)
        deltas[:N-1] = x
        deltas[N-1] = 2 * np.pi - np.sum(x)
        
        if deltas[N-1] <= 0:
            return 0.0
            
        ratio = compute_ratio(deltas)
        return -ratio # Minimize negative ratio

    # Bounds for each gap
    bounds = [(1e-5, 2 * np.pi - 1e-5)] * (N-1)
    
    best_result = None
    best_score = 0.0
    
    # Multi-start optimization
    # 1. Uniform start
    # 2. Random starts
    
    starts = []
    # Uniform
    starts.append(np.full(N-1, 2 * np.pi / N))
    
    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(10):
        # Generate random positive numbers that sum to 2*pi
        # Use Dirichlet distribution for uniformity on the simplex
        # Or just random gaps and normalize
        raw = rng.random(N)
        raw = raw / raw.sum() * 2 * np.pi
        # Ensure no gap is too small to cause numerical issues in optimizer bounds
        # But L-BFGS-B handles bounds.
        # We take first N-1 as x
        x0 = raw[:N-1]
        # Clip to bounds to be safe for initialization
        x0 = np.clip(x0, 1e-5, 2 * np.pi - 1e-5)
        starts.append(x0)
        
    for x0 in starts:
        # Ensure x0 is valid
        if np.any(x0 <= 0) or np.sum(x0) >= 2 * np.pi:
            continue
            
        try:
            result = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={
                    'maxiter': 500, 
                    'ftol': 1e-12,
                    'gtol': 1e-8
                }
            )
            
            if result.success:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_result = result.x
        except Exception:
            continue
            
    if best_result is None:
        # Fallback to uniform
        opt_deltas = np.full(N, 2 * np.pi / N)
    else:
        opt_deltas = np.empty(N)
        opt_deltas[:N-1] = best_result
        opt_deltas[N-1] = 2 * np.pi - np.sum(best_result)
        
    # Construct points on the unit circle
    angles = np.zeros(N)
    angles[1:] = np.cumsum(opt_deltas[:N-1])
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
