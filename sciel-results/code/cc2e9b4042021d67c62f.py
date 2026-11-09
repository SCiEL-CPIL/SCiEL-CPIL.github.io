import numpy as np

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    best_P = -1.0
    best_a, best_b, best_c = 0.0, 0.0, 0.0
    best_M = 2.0
    
    # M grid
    # Focus on regions where jumps occur and potential maxima
    # M > 1
    M_vals = np.concatenate([
        np.linspace(1.0001, 1.5, 100),
        np.linspace(1.5001, 2.0, 100),
        np.linspace(2.0001, 5.0, 200),
        np.linspace(5.0, 20.0, 200),
        np.linspace(20.0, 100.0, 200)
    ])
    
    # b, c grid
    # b in [0, 1], c in [0, 1-b]
    # We can use a square grid and mask
    n_grid = 800
    b_vals = np.linspace(0.0, 1.0, n_grid)
    c_vals = np.linspace(0.0, 1.0, n_grid)
    B, C = np.meshgrid(b_vals, c_vals, indexing='ij')
    A = 1.0 - B - C
    
    # Mask valid points
    valid_mask = (A >= 0) & (B >= 0) & (C >= 0)
    
    # We will iterate over M
    for M in M_vals:
        # Determine formula
        if M <= 1.5:
            # Case 1
            P2 = A**2 * (3.0 - 2.0*A)
            P1 = 3.0*C*A*(1.0-C) + (1.0-C)**3 - B**3
            P = C * P1 + B * P2
        elif M <= 2.0:
            # Case 2
            P2 = A**2 * (3.0 - 2.0*A)
            P1 = 3.0*C*A*(1.0-C) + (1.0-C)**3
            P = C * P1 + B * P2
        else:
            # Case 3
            P2 = A**2 * (A + 3.0*B)
            P1 = 3.0*C*(1.0-C)**2 + (1.0-C)**3
            P = C * P1 + B * P2
            
        # Apply mask
        P[~valid_mask] = -np.inf
        
        max_idx = np.unravel_index(np.argmax(P), P.shape)
        max_P = P[max_idx]
        
        if max_P > best_P:
            best_P = max_P
            best_b = B[max_idx]
            best_c = C[max_idx]
            best_a = A[max_idx]
            best_M = M
            
    # Ensure weights are non-negative and sum to 1
    # Due to floating point, A might be slightly negative or > 1
    best_a = np.clip(best_a, 0.0, 1.0)
    best_b = np.clip(best_b, 0.0, 1.0)
    best_c = np.clip(best_c, 0.0, 1.0)
    
    # Normalize just in case
    s = best_a + best_b + best_c
    if s < 1e-9:
        # Fallback
        return np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.0, 0.0])
        
    best_a /= s
    best_b /= s
    best_c /= s
    
    support = np.array([0.0, 1.0, best_M], dtype=np.float64)
    weights = np.array([best_a, best_b, best_c], dtype=np.float64)
    
    return support, weights
