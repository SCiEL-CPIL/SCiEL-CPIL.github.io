import numpy as np

def _compute_P_grid(s, w_grid):
    """
    Compute P for a set of weight vectors w_grid (shape M, n) and fixed support s (shape n).
    Returns array of P values (shape M,).
    """
    n = len(s)
    M = len(w_grid)
    
    # Precompute the condition matrix for each (k, l)
    # C[k, l, i, j] = 1 if s[i] + s[j] < 2*s[l] - s[k]
    # Shape: (n, n, n, n)
    # This might be large for n=4 (4^4=256), n=5 (5^4=625). Manageable.
    
    s_i = s[:, None, None, None] # (n, 1, 1, 1)
    s_j = s[None, :, None, None] # (1, n, 1, 1)
    s_k = s[None, None, :, None] # (1, 1, n, 1)
    s_l = s[None, None, None, :] # (1, 1, 1, n)
    
    # Condition: s_i + s_j < 2*s_l - s_k
    # Reshape to broadcast: (n, n, 1, 1) + (1, 1, n, 1) vs 2*(1, 1, 1, n) - (1, 1, n, 1)
    # Let's do it carefully.
    # i, j vary in first two dims. k, l vary in last two dims?
    # Let's define C as (n_k, n_l, n_i, n_j)
    
    # s_i: (n_i, 1, 1, 1) -> (1, 1, n_i, 1) ? No, let's align axes.
    # Axis 0: k
    # Axis 1: l
    # Axis 2: i
    # Axis 3: j
    
    s_k_arr = s[:, None, None, None] # (n, 1, 1, 1) -> k
    s_l_arr = s[None, :, None, None] # (1, n, 1, 1) -> l
    s_i_arr = s[None, None, :, None] # (1, 1, n, 1) -> i
    s_j_arr = s[None, None, None, :] # (1, 1, 1, n) -> j
    
    # Threshold for (k, l) is 2*s_l - s_k
    # Shape (n, n, 1, 1)
    threshold = 2.0 * s_l_arr - s_k_arr
    
    # Sum s_i + s_j
    # Shape (1, 1, n, n)
    pair_sum = s_i_arr + s_j_arr
    
    # Broadcast to (n, n, n, n)
    # C[k, l, i, j] = (pair_sum[i, j] < threshold[k, l])
    C = (pair_sum < threshold) # Boolean array (n, n, n, n)
    
    # Now compute P for each w in w_grid
    # P_m = sum_{k,l} w_m[k] w_m[l] * (sum_{i,j} C[k,l,i,j] w_m[i] w_m[j])
    
    # Let Q_m[k, l] = sum_{i,j} C[k,l,i,j] w_m[i] w_m[j]
    # This is a quadratic form.
    # w_m shape (M, n).
    # w_i term: w_m[:, i] -> (M, 1, 1, 1) ? No.
    
    # Let's compute the inner sum first.
    # For each (k, l), we have a matrix C_kl of size n x n.
    # Inner sum for weight vector w is w^T C_kl w.
    
    # We can do this by summing over i and j.
    # w_m[i] is w_grid[:, i] shape (M,)
    # w_m[j] is w_grid[:, j] shape (M,)
    
    # Let's expand C to (n, n, n, n) and use einsum or broadcasting.
    # Term = w_grid[:, k] * w_grid[:, l] * w_grid[:, i] * w_grid[:, j] * C[k, l, i, j]
    
    # This is a 4-way contraction.
    # Sum over k, l, i, j.
    # Result shape (M,)
    
    # Using einsum:
    # 'm k, m l, m i, m j, k l i j -> m'
    # w_grid: (M, n) -> 'm k'
    # C: (n, n, n, n) -> 'k l i j'
    
    P_vals = np.einsum('mk,ml,mi,mj,klij->m', w_grid, w_grid, w_grid, w_grid, C, optimize=True)
    
    return P_vals

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    best_P = -1
    best_support = None
    best_weights = None
    
    # Strategy: Grid search over structured supports and weights.
    
    # Case n=2
    # Support [0, 1]
    s2 = np.array([0.0, 1.0])
    # Optimal p for n=2 is root of 8p^3 - 9p^2 + 1 = 0
    # p = (1 + sqrt(33))/16
    p_opt = (1 + np.sqrt(33)) / 16
    w2 = np.array([1 - p_opt, p_opt])
    P2 = _compute_P_grid(s2, w2[None, :])[0]
    if P2 > best_P:
        best_P = P2
        best_support = s2.copy()
        best_weights = w2.copy()
        
    # Case n=3
    # Support [0, x, 1]
    # Weights (w0, w1, w2)
    # Grid for x
    x_grid = np.linspace(0.1, 0.9, 50)
    # Grid for weights on simplex
    # Parameterize w1 = u * (1-w0), w2 = (1-u) * (1-w0) ? No.
    # Let w0 = 1 - v. Then w1 + w2 = v.
    # Let w1 = v * u, w2 = v * (1-u).
    # u in [0, 1], v in [0, 1].
    u_grid = np.linspace(0.0, 1.0, 50)
    v_grid = np.linspace(0.0, 1.0, 50)
    
    U, V = np.meshgrid(u_grid, v_grid, indexing='ij')
    U_flat = U.flatten()
    V_flat = V.flatten()
    
    # Weights for all points in grid
    # w0 = 1 - V
    # w1 = V * U
    # w2 = V * (1 - U)
    W_grid = np.column_stack([1.0 - V_flat, V_flat * U_flat, V_flat * (1.0 - U_flat)])
    
    # Evaluate for each x
    for x in x_grid:
        s3 = np.array([0.0, x, 1.0])
        P_vals = _compute_P_grid(s3, W_grid)
        max_idx = np.argmax(P_vals)
        if P_vals[max_idx] > best_P:
            best_P = P_vals[max_idx]
            best_support = s3.copy()
            best_weights = W_grid[max_idx].copy()
            
    # Case n=4
    # Support [0, a, b, 1] with 0 < a < b < 1
    # This is a 3D grid for supports (a, b) and 3D grid for weights.
    # Too large for full grid. 
    # We can fix supports to a coarse grid and optimize weights locally or with a finer weight grid.
    
    a_grid = np.linspace(0.1, 0.4, 10)
    b_grid = np.linspace(0.6, 0.9, 10)
    
    # Weight grid for n=4: Simpler to use a smaller grid or random sampling?
    # Let's use a coarser weight grid for n=4 to keep it fast.
    u1_grid = np.linspace(0.0, 1.0, 20)
    u2_grid = np.linspace(0.0, 1.0, 20)
    u3_grid = np.linspace(0.0, 1.0, 20)
    
    # Dirichlet sampling or grid? Grid is deterministic.
    # w0 = 1 - w1 - w2 - w3
    # w1 = t1
    # w2 = (1-t1)*t2
    # w3 = (1-t1)*(1-t2)*t3
    # t1, t2, t3 in [0, 1]
    
    T1, T2, T3 = np.meshgrid(u1_grid, u2_grid, u3_grid, indexing='ij')
    T1_flat = T1.flatten()
    T2_flat = T2.flatten()
    T3_flat = T3.flatten()
    
    W4_grid = np.column_stack([
        1.0 - T1_flat - (1.0-T1_flat)*T2_flat - (1.0-T1_flat)*(1.0-T2_flat)*T3_flat,
        T1_flat,
        (1.0-T1_flat)*T2_flat,
        (1.0-T1_flat)*(1.0-T2_flat)*T3_flat
    ])
    # Ensure non-negative (they are by construction) and sum to 1
    
    for a in a_grid:
        for b in b_grid:
            if a >= b: continue
            s4 = np.array([0.0, a, b, 1.0])
            P_vals = _compute_P_grid(s4, W4_grid)
            max_idx = np.argmax(P_vals)
            if P_vals[max_idx] > best_P:
                best_P = P_vals[max_idx]
                best_support = s4.copy()
                best_weights = W4_grid[max_idx].copy()
                
    # Final checks
    support = np.asarray(best_support, dtype=np.float64).flatten()
    weights = np.asarray(best_weights, dtype=np.float64).flatten()
    
    if len(support) != len(weights) or len(support) < 1:
        support = np.array([0.0, 1.0])
        weights = np.array([0.5, 0.5])
        
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        support = np.array([0.0, 1.0])
        weights = np.array([0.5, 0.5])
        
    if weights.sum() <= 1e-9:
        support = np.array([0.0, 1.0])
        weights = np.array([0.5, 0.5])
        
    return support, weights
