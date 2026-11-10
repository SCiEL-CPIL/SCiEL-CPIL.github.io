import numpy as np

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    N = 256
    support = np.arange(N, dtype=np.float64)
    
    def compute_gradient(weights):
        w = weights
        P_prefix = np.cumsum(w)
        total_w = P_prefix[-1]
        
        # Compute Q matrix: Q[k, l] = sum_{j < 2l - k} w_j
        # D = 2l - k
        # If D <= 0, sum is 0
        # If D >= N, sum is total_w
        # Else sum is P_prefix[D - 1]
        
        k_idx = np.arange(N)[:, None] # N x 1
        l_idx = np.arange(N)[None, :] # 1 x N
        D = 2 * l_idx - k_idx # N x N
        
        Q = np.zeros((N, N))
        
        # D <= 0
        mask_neg = D <= 0
        # D >= N
        mask_large = D >= N
        # 0 < D < N
        mask_mid = ~(mask_neg | mask_large)
        
        Q[mask_large] = total_w
        
        # For mask_mid, index is D - 1
        # D is integer array
        indices = (D[mask_mid] - 1).astype(int)
        # We need to map these indices to P_prefix
        # But D[mask_mid] is a 1D array of the valid entries
        # We can just use P_prefix[indices]
        Q[mask_mid] = P_prefix[indices]
        
        # A_m = sum_l w_l Q[m, l]
        A = Q.T @ w
        
        # B_m = sum_k w_k Q[k, m]
        B = Q @ w
        
        # D_m = 2 * sum_{k,l} w_k w_l C[2l - k - m - 1]
        # Where C[d] = sum_{j < d} w_j = P_prefix[d-1] (with bounds)
        # Let K = 2l - k - m - 1
        # If K <= 0, term is 0
        # If K >= N, term is total_w
        # Else term is P_prefix[K-1]
        
        # We need to compute D_grad[m] for all m
        # D_grad = 2 * sum_{k,l} w_k w_l Val(2l - k - m - 1)
        
        # Create 3D index array
        # m: 0..N-1
        # k: 0..N-1
        # l: 0..N-1
        # K = 2l - k - m - 1
        
        m_idx = np.arange(N)[:, None, None]
        k_idx3 = np.arange(N)[None, :, None]
        l_idx3 = np.arange(N)[None, None, :]
        
        K = 2 * l_idx3 - k_idx3 - m_idx - 1
        
        # Val(K)
        # We can compute this vectorized
        # But K is N x N x N
        # We need to apply the piecewise function
        
        # Instead of creating huge arrays, we can loop over m?
        # N=256. Looping 256 times, each doing N^2=65k ops. Total 16M ops.
        # This is fine in Python if we vectorize the inner loop.
        
        D_grad = np.zeros(N)
        
        # Precompute w_k w_l matrix
        WW = np.outer(w, w)
        
        for m in range(N):
            # K for this m: 2l - k - m - 1
            # Shape N x N (k, l)
            K_m = 2 * l_idx - k_idx - m - 1
            
            # Val
            Val = np.zeros((N, N))
            mask_neg_m = K_m <= 0
            mask_large_m = K_m >= N
            mask_mid_m = ~(mask_neg_m | mask_large_m)
            
            Val[mask_large_m] = total_w
            indices_m = (K_m[mask_mid_m] - 1).astype(int)
            Val[mask_mid_m] = P_prefix[indices_m]
            
            D_grad[m] = 2.0 * np.sum(WW * Val)
            
        return A + B + D_grad

    def project_simplex(w):
        w = np.clip(w, 0, None)
        s = np.sum(w)
        if s > 0:
            w = w / s
        else:
            w = np.ones(N) / N
        return w

    def compute_P(weights):
        # Use the same logic as gradient but just return the scalar
        # P = sum_{k,l} w_k w_l Q[k,l]
        w = weights
        P_prefix = np.cumsum(w)
        total_w = P_prefix[-1]
        
        k_idx = np.arange(N)[:, None]
        l_idx = np.arange(N)[None, :]
        D = 2 * l_idx - k_idx
        
        Q = np.zeros((N, N))
        mask_neg = D <= 0
        mask_large = D >= N
        mask_mid = ~(mask_neg | mask_large)
        
        Q[mask_large] = total_w
        indices = (D[mask_mid] - 1).astype(int)
        Q[mask_mid] = P_prefix[indices]
        
        P_val = np.sum(w[:, None] * w[None, :] * Q)
        return P_val

    best_weights = None
    best_P = -1.0
    
    # Initializations
    initializations = [
        np.ones(N) / N,
        np.concatenate([np.ones(64) / 64, np.zeros(N-64)]),
        np.concatenate([np.zeros(N-64), np.ones(64) / 64]),
        np.array([1.0] + [0.0]*(N-1)),
        np.array([0.0]*(N-1) + [1.0]),
    ]
    
    for init_w in initializations:
        weights = project_simplex(init_w.copy())
        step = 0.1
        for it in range(1000):
            grad = compute_gradient(weights)
            P_old = compute_P(weights)
            
            alpha = step
            improved = False
            for _ in range(20):
                new_w = project_simplex(weights + alpha * grad)
                P_new = compute_P(new_w)
                if P_new > P_old:
                    weights = new_w
                    improved = True
                    break
                alpha *= 0.5
            
            if improved:
                P_cur = compute_P(weights)
                if P_cur > best_P:
                    best_P = P_cur
                    best_weights = weights.copy()
                step = min(step * 1.2, 0.5)
            else:
                step = max(step * 0.5, 0.001)
                if step < 0.005:
                    break

    return support, best_weights
