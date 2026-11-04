import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def _build_tensor():
    n, m, p = DIMENSIONS
    # Flatten indices:
    # a in [0, n*m)
    # b in [0, m*p)
    # c in [0, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _cp_als(T, r, n_iter=100, tol=1e-9, seed=0):
    """
    Perform CP-ALS on tensor T with rank r.
    T is shape (n_m, m_p, p_n).
    Returns U, V, W of shapes (n_m, r), (m_p, r), (p_n, r).
    """
    np.random.seed(seed)
    n_m, m_p, p_n = T.shape
    
    # Initialize
    U = np.random.randn(n_m, r)
    V = np.random.randn(m_p, r)
    W = np.random.randn(p_n, r)
    
    prev_residual = np.inf
    
    for it in range(n_iter):
        # Solve for U
        # T_kr = sum_c T[:,:,c] * W[c,:]
        # U_new = T_kr * inv((V^T V) * (W^T W)) ? 
        # Actually, standard ALS:
        # U = T_kr * (V^T V)^-1 * (W^T W)^-1
        # Let's use the mode-2 and mode-3 unfoldings.
        
        # T_kr is matricization along mode 1? No, we want to solve for U (mode 1).
        # T_{a,b,c} = sum_t U_{a,t} V_{b,t} W_{c,t}
        # Fix V, W. For each a:
        # T_{a,: ,:} = U_{a,:} * (V^T W elementwise? No)
        # T_{a,b,c} = sum_t U_{a,t} (V_{b,t} W_{c,t})
        # Let Z_t = V[:,t] * W[:,t] (outer product vectorized? No, Z is matrix same shape as T? No)
        # Z_{b,c} = sum_t V_{b,t} W_{c,t} is not right for single t.
        
        # Better:
        # T_kr_{a, (b,c)} = sum_t U_{a,t} (V W)^t_{b,c}
        # Where (V W)^t is the outer product of col t of V and col t of W.
        # This is equivalent to:
        # T_kr = U * (V^T * W^T elementwise summed over t? No)
        
        # Let's use the standard formula:
        # U = T_kr * ( (V^T V) * (W^T W) )^{-1}
        # Where T_kr is the mode-1 matricization of T.
        # T_kr_{a, b*r + c} = T[a,b,c]
        # This works if we treat the index pair (b,c) as a single index.
        
        T_kr = T.reshape(n_m, m_p * p_n)
        
        # We need to solve: T_kr approx U * M
        # Where M_{t, (b,c)} = V_{b,t} * W_{c,t}
        # M is shape (r, m_p * p_n)
        
        M = np.zeros((r, m_p * p_n))
        for t in range(r):
            M[t, :] = (V[:, t].reshape(-1, 1) * W[:, t].reshape(1, -1)).flatten()
            
        # Least squares: U = T_kr @ M^T @ pinv(M @ M^T) ?
        # Minimize ||T_kr - U M||_F
        # U = T_kr M^T (M M^T)^{-1}
        
        MMt = M @ M.T
        try:
            U_new = T_kr @ M.T @ np.linalg.inv(MMt)
        except np.linalg.LinAlgError:
            U_new = T_kr @ M.T @ np.linalg.pinv(MMt)
            
        # Solve for V
        T_kc = T.reshape(m_p, n_m * p_n)
        # M_v shape (r, n_m * p_n)
        M_v = np.zeros((r, n_m * p_n))
        for t in range(r):
            M_v[t, :] = (U_new[:, t].reshape(-1, 1) * W[:, t].reshape(1, -1)).flatten()
            
        MvMvt = M_v @ M_v.T
        try:
            V_new = T_kc @ M_v.T @ np.linalg.inv(MvMvt)
        except np.linalg.LinAlgError:
            V_new = T_kc @ M_v.T @ np.linalg.pinv(MvMvt)
            
        # Solve for W
        T_kb = T.reshape(p_n, n_m * m_p)
        # M_w shape (r, n_m * m_p)
        M_w = np.zeros((r, n_m * m_p))
        for t in range(r):
            M_w[t, :] = (U_new[:, t].reshape(-1, 1) * V_new[:, t].reshape(1, -1)).flatten()
            
        MwMwt = M_w @ M_w.T
        try:
            W_new = T_kb @ M_w.T @ np.linalg.inv(MwMwt)
        except np.linalg.LinAlgError:
            W_new = T_kb @ M_w.T @ np.linalg.pinv(MwMwt)
            
        # Calculate residual
        residual = np.linalg.norm(T - np.einsum('at,bt,ct->abc', U_new, V_new, W_new))
        
        if abs(prev_residual - residual) < tol:
            break
            
        prev_residual = residual
        U, V, W = U_new, V_new, W_new
        
    return U, V, W

def _verify_decomposition(U, V, W, T, tol=1e-6):
    """
    Check if U, V, W exactly decompose T (within tol).
    U, V, W should have integer or half-integer entries.
    """
    # Reconstruct
    T_recon = np.einsum('at,bt,ct->abc', U, V, W)
    diff = np.linalg.norm(T - T_recon)
    return diff < tol

def _round_to_half_int(M):
    """Round matrix entries to nearest 0.5"""
    return np.round(M * 2) / 2

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Lower bound for rank is max(n*m, m*p, p*n)
    lb = max(n * m, m * p, p * n)
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p # Fallback to schoolbook
    
    # Generate schoolbook as fallback
    U_school = np.zeros((n * m, n * m * p))
    V_school = np.zeros((m * p, n * m * p))
    W_school = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p

    # Try ranks from lower bound upwards
    for r in range(lb, best_rank + 1):
        if time.time() > deadline - 20:
            break
            
        # Try multiple restarts
        num_restarts = 5
        found = False
        
        for restart in range(num_restarts):
            if time.time() > deadline - 20:
                break
                
            U, V, W = _cp_als(T, r, n_iter=50, seed=restart)
            
            # Round to half-integers
            U_r = _round_to_half_int(U)
            V_r = _round_to_half_int(V)
            W_r = _round_to_half_int(W)
            
            if _verify_decomposition(U_r, V_r, W_r, T, tol=1e-9):
                # Found a valid decomposition with rank r
                # Convert to lists for output
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
        
    # If no better rank found, return schoolbook
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
