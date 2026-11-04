import numpy as np
from scipy.optimize import minimize

N = 14

def construct():
    # Define the objective function
    def objective(gaps):
        # gaps: array of N-1 values
        if np.any(gaps <= 0):
            return 0.0
            
        deltas = np.empty(N)
        deltas[:N-1] = gaps
        deltas[N-1] = 2 * np.pi - np.sum(gaps)
        
        if deltas[N-1] <= 0:
            return 0.0
            
        # Compute angles
        phi = np.zeros(N)
        phi[1:] = np.cumsum(deltas[:N-1])
        
        # Compute min triangle area numerator
        min_area_num = np.inf
        
        # Vectorized-ish approach for inner loops
        # For each i, we can vectorize j and k?
        # It's complex. Let's stick to loops but optimize them.
        
        for i in range(N):
            # j > i
            for j in range(i + 1, N):
                a = phi[j] - phi[i]
                # k > j
                # We need min over k of sin(a) + sin(phi[k]-phi[j]) + sin(2*pi - a - (phi[k]-phi[j]))
                # Let b = phi[k] - phi[j]. c = 2*pi - a - b.
                # val = sin(a) + sin(b) + sin(2*pi - a - b)
                # sin(2*pi - x) = -sin(x). So sin(c) = -sin(a+b).
                # val = sin(a) + sin(b) - sin(a+b)
                
                # We want to minimize this over k in j+1..N-1.
                # We can compute this for all valid k at once?
                # k_indices = np.arange(j+1, N)
                # b_vals = phi[k_indices] - phi[j]
                # vals = np.sin(a) + np.sin(b_vals) - np.sin(a + b_vals)
                # min_val = np.min(vals)
                # if min_val < min_area_num: min_area_num = min_val
                
                # This reduces the inner loop from O(N) to O(1) (vectorized).
                k_indices = np.arange(j + 1, N)
                if len(k_indices) == 0:
                    continue
                b_vals = phi[k_indices] - phi[j]
                vals = np.sin(a) + np.sin(b_vals) - np.sin(a + b_vals)
                current_min = np.min(vals)
                if current_min < min_area_num:
                    min_area_num = current_min
        
        # Compute hull area denominator
        den = np.sum(np.sin(deltas))
        
        if den <= 0:
            return 0.0
            
        ratio = min_area_num / den
        return -ratio

    # Initial guess: uniform
    x0 = np.full(N-1, 2 * np.pi / N)
    
    # Bounds
    bounds = [(1e-8, 2 * np.pi - 1e-8)] * (N-1)
    
    # Use Powell method
    result = minimize(
        objective, 
        x0, 
        method='Powell',
        bounds=bounds,
        options={
            'xtol': 1e-12, 
            'ftol': 1e-12, 
            'maxiter': 5000, 
            'maxfev': 50000
        }
    )
    
    if result.success:
        opt_gaps = result.x
    else:
        opt_gaps = x0
        
    deltas = np.empty(N)
    deltas[:N-1] = opt_gaps
    deltas[N-1] = 2 * np.pi - np.sum(opt_gaps)
    
    # Safety check
    if np.any(deltas <= 0):
        deltas = np.full(N, 2 * np.pi / N)

    angles = np.zeros(N)
    angles[1:] = np.cumsum(deltas[:N-1])
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
