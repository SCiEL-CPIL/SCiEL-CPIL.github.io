import numpy as np
import time
import random

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

def _cp_als(T, r, n_iter=100, tol=1e-9, seed=0, init_type='random'):
    """
    Perform CP-ALS on tensor T with rank r.
    T is shape (n_m, m_p, p_n).
    Returns U, V, W of shapes (n_m, r), (m_p, r), (p_n, r).
    """
    np.random.seed(seed)
    n_m, m_p, p_n = T.shape
    
    # Initialize
    if init_type == 'random':
        U = np.random.randn(n_m, r)
        V = np.random.randn(m_p, r)
        W = np.random.randn(p_n, r)
    elif init_type == 'structured':
        # Use the tensor's support for initialization
        U = np.zeros((n_m, r))
        V = np.zeros((m_p, r))
        W = np.zeros((p_n, r))
        # Assign each non-zero entry to a random rank component
        nz_a, nz_b, nz_c = np.where(T > 0)
        for idx in range(len(nz_a)):
            t = idx % r
            U[nz_a[idx], t] = 1.0
            V[nz_b[idx], t] = 1.0
            W[nz_c[idx], t] = 1.0
        # Add small random perturbation to break symmetry
        U += 0.01 * np.random.randn(n_m, r)
        V += 0.01 * np.random.randn(m_p, r)
        W += 0.01 * np.random.randn(p_n, r)
    else:
        U = np.random.randn(n_m, r)
        V = np.random.randn(m_p, r)
        W = np.random.randn(p_n, r)
    
    prev_residual = np.inf
    
    for it in range(n_iter):
        # Solve for U
        T_kr = T.reshape(n_m, m_p * p_n)
        
        M = np.zeros((r, m_p * p_n))
        for t in range(r):
            M[t, :] = (V[:, t].reshape(-1, 1) * W[:, t].reshape(1, -1)).flatten()
            
        MMt = M @ M.T
        try:
            U_new = T_kr @ M.T @ np.linalg.inv(MMt)
        except np.linalg.LinAlgError:
            U_new = T_kr @ M.T @ np.linalg.pinv(MMt)
            
        # Solve for V
        T_kc = T.reshape(m_p, n_m * p_n)
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

def _greedy_rank1(T, r, seed=0):
    """
    Greedy algorithm: pick r rank-1 components that best explain T.
    """
    np.random.seed(seed)
    n_m, m_p, p_n = T.shape
    
    U = np.zeros((n_m, r))
    V = np.zeros((m_p, r))
    W = np.zeros((p_n, r))
    
    residual = T.copy()
    
    for t in range(r):
        # Find the best rank-1 approximation to residual
        # Use truncated SVD on mode-1 unfolding
        best_score = -np.inf
        best_a, best_b, best_c = None, None, None
        
        # Sample some candidate rank-1 components
        n_samples = 200
        for _ in range(n_samples):
            a = np.random.randn(n_m)
            b = np.random.randn(m_p)
            c = np.random.randn(p_n)
            
            # Normalize
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            c_norm = np.linalg.norm(c)
            if a_norm < 1e-10 or b_norm < 1e-10 or c_norm < 1e-10:
                continue
            a /= a_norm
            b /= b_norm
            c /= c_norm
            
            # Compute inner product with residual
            # <residual, a ⊗ b ⊗ c>
            score = np.einsum('abc,a,b,c->', residual, a, b, c)
            
            if score > best_score:
                best_score = score
                best_a, best_b, best_c = a, b, c
        
        if best_a is None:
            continue
            
        # Add this rank-1 component
        U[:, t] = best_a
        V[:, t] = best_b
        W[:, t] = best_c
        
        # Update residual
        residual -= np.einsum('a,b,c->abc', best_a, best_b, best_c)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    # Lower bound for rank is max(n*m, m*p, p*n)
    lb = max(n * m, m * p, p * n)
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # Fallback to schoolbook
    
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
        if time.time() > deadline - 30:
            break
            
        # Try multiple methods and restarts
        found = False
        
        # Method 1: CP-ALS with random initialization
        num_restarts = 30
        for restart in range(num_restarts):
            if time.time() > deadline - 30:
                break
                
            U, V, W = _cp_als(T, r, n_iter=80, seed=restart, init_type='random')
            
            # Round to half-integers
            U_r = _round_to_half_int(U)
            V_r = _round_to_half_int(V)
            W_r = _round_to_half_int(W)
            
            if _verify_decomposition(U_r, V_r, W_r, T, tol=1e-9):
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
        
        # Method 2: CP-ALS with structured initialization
        for restart in range(10):
            if time.time() > deadline - 30:
                break
                
            U, V, W = _cp_als(T, r, n_iter=80, seed=restart + 100, init_type='structured')
            
            U_r = _round_to_half_int(U)
            V_r = _round_to_half_int(V)
            W_r = _round_to_half_int(W)
            
            if _verify_decomposition(U_r, V_r, W_r, T, tol=1e-9):
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
        
        # Method 3: Greedy rank-1 selection
        for restart in range(5):
            if time.time() > deadline - 30:
                break
                
            U, V, W = _greedy_rank1(T, r, seed=restart + 200)
            
            U_r = _round_to_half_int(U)
            V_r = _round_to_half_int(V)
            W_r = _round_to_half_int(W)
            
            if _verify_decomposition(U_r, V_r, W_r, T, tol=1e-9):
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
        
        # If we found any valid decomposition (not necessarily at this rank), update best
        # But we're returning immediately when found, so this is handled above
        
    # If no better rank found, return schoolbook
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
