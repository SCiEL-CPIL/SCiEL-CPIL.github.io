import time
import numpy as np
import scipy.optimize

DIMENSIONS = [3, 3, 4]
_SEARCH_TIMEOUT = 140.0 # 10s margin from 150s limit

def _build_target_tensor(n, m, p):
    T = np.ones((n, m, p), dtype=float)
    return T

def _verify_exact(U, V, W, n, m, p, tol=1e-6):
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    W_perm = np.transpose(W3, (1, 0, 2))
    T_calc = np.einsum('ijr,jkr,ikr->ijk', U3, V3, W_perm, optimize=True)
    if not np.allclose(T_calc, np.ones((n, m, p)), atol=tol):
        return False
    def is_half_int(arr):
        return np.allclose(arr * 2, np.round(arr * 2), atol=1e-6)
    return is_half_int(U) and is_half_int(V) and is_half_int(W)

def _residual(U, V, W, n, m, p):
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    W_perm = np.transpose(W3, (1, 0, 2))
    T_calc = np.einsum('ijr,jkr,ikr->ijk', U3, V3, W_perm, optimize=True)
    return np.linalg.norm(T_calc - 1.0)

def _snap_to_half_int(arr):
    return np.round(arr * 2.0) / 2.0

def _search_rank_333(R, deadline):
    """Search for rank R decomposition of M_3,3,3"""
    n, m, p = 3, 3, 3
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    best_res = float('inf')
    best_params = None
    
    # Initialize with random sparse structure
    np.random.seed(0)
    for r in range(R):
        U[:, r] = np.random.choice([-1, 0, 1], size=n*m)
        V[:, r] = np.random.choice([-1, 0, 1], size=m*p)
        W[:, r] = np.random.choice([-1, 0, 1], size=p*n)
        
    def objective(params):
        U = params[:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R:].reshape(p*n, R)
        res = _residual(U, V, W, n, m, p)
        # Penalty for non-integer values to guide towards integer solutions
        pen = np.sum((U - np.round(U))**2) + np.sum((V - np.round(V))**2) + np.sum((W - np.round(W))**2)
        return res + 0.1 * pen
        
    params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        result = scipy.optimize.minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 5000, 'ftol': 1e-8, 'maxfun': 50000})
        if result.success or result.fun < best_res:
            best_res = result.fun
            best_params = result.x
    except Exception:
        pass
        
    if best_params is not None:
        U_opt = best_params[:n*m*R].reshape(n*m, R)
        V_opt = best_params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W_opt = best_params[n*m*R + m*p*R:].reshape(p*n, R)
        
        U_snap = _snap_to_half_int(U_opt)
        V_snap = _snap_to_half_int(V_opt)
        W_snap = _snap_to_half_int(W_opt)
        
        if _verify_exact(U_snap, V_snap, W_snap, n, m, p):
            return U_snap, V_snap, W_snap
            
    return None

def _schoolbook_333():
    n, m, p = 3, 3, 3
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _schoolbook_331():
    # M_3,3,1: C_i = sum_j A_ij B_j
    # This is 3 dot products of length 3.
    # Rank 9.
    n, m, p = 3, 3, 1
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    t0 = time.time()
    hard_deadline = t0 + _SEARCH_TIMEOUT
    n, m, p = DIMENSIONS # 3, 3, 4
    
    # Strategy:
    # 1. Try to find rank 23 for M_3,3,3. If found, total rank 32.
    # 2. Try to find rank 24 for M_3,3,3. If found, total rank 33.
    # 3. Try to find rank 25 for M_3,3,3. If found, total rank 34.
    # 4. Fallback to schoolbook M_3,3,3 (rank 27) + M_3,3,1 (rank 9) = 36.
    
    best_total_rank = 36
    best_U, best_V, best_W = None, None, None
    
    # Get schoolbook parts
    U_333_sb, V_333_sb, W_333_sb = _schoolbook_333()
    U_331_sb, V_331_sb, W_331_sb = _schoolbook_331()
    
    # Combine schoolbook for fallback
    # M_3,3,4 is M_3,3,3 (k=0,1,2) + M_3,3,1 (k=3)
    # We need to align the indices.
    # U_333: (9, 27), V_333: (9, 27), W_333: (9, 27)
    # U_331: (9, 9), V_331: (3, 9), W_331: (3, 9)
    # Wait, M_3,3,1 dimensions:
    # A is 3x3, B is 3x1, C is 3x1.
    # U shape (n*m, R) = (9, R)
    # V shape (m*p, R) = (3*1, R) = (3, R)
    # W shape (p*n, R) = (1*3, R) = (3, R)
    
    # To combine into M_3,3,4:
    # V_331 needs to be padded to (12, R) because m*p = 3*4 = 12.
    # The indices for V in M_3,3,4 are j*p+k.
    # For M_3,3,3, k=0,1,2. Indices 0..8.
    # For M_3,3,1, k=3. Indices 9,10,11 (j*4+3).
    # So V_331 (3x9) should be placed at rows 9,10,11.
    # W_331 needs to be padded to (12, R) because p*n = 4*3 = 12.
    # Indices for W in M_3,3,4 are k*n+i.
    # For M_3,3,3, k=0,1,2. Indices 0..8.
    # For M_3,3,1, k=3. Indices 9,10,11 (3*3+i).
    # So W_331 (3x9) should be placed at rows 9,10,11.
    
    def combine_333_331(U_333, V_333, W_333, U_331, V_331, W_331):
        R_333 = U_333.shape[1]
        R_331 = U_331.shape[1]
        R_total = R_333 + R_331
        
        U_total = np.zeros((9, R_total))
        V_total = np.zeros((12, R_total))
        W_total = np.zeros((12, R_total))
        
        # Copy 333 part
        U_total[:, :R_333] = U_333
        V_total[:9, :R_333] = V_333
        W_total[:9, :R_333] = W_333
        
        # Copy 331 part
        U_total[:, R_333:] = U_331
        # V_331 is (3, R_331). Place at rows 9,10,11.
        V_total[9:12, R_333:] = V_331
        # W_331 is (3, R_331). Place at rows 9,10,11.
        W_total[9:12, R_333:] = W_331
        
        return U_total, V_total, W_total, R_total

    # Try ranks for 333
    for R_333 in [23, 24, 25, 26]:
        if time.time() > hard_deadline:
            break
            
        # Time slice for this rank
        rank_deadline = time.time() + 20.0
        
        result = _search_rank_333(R_333, rank_deadline)
        if result is not None:
            U_333, V_333, W_333 = result
            U_tot, V_tot, W_tot, R_tot = combine_333_331(U_333, V_333, W_333, U_331_sb, V_331_sb, W_331_sb)
            if _verify_exact(U_tot, V_tot, W_tot, 3, 3, 4):
                best_total_rank = R_tot
                best_U, best_V, best_W = U_tot, V_tot, W_tot
                break # Found minimal for this 333 rank, and 333 rank is increasing, so this is best possible with this strategy?
                # Actually, if we find 23, we stop. If we find 24, we stop?
                # Yes, because we iterate 23, 24, 25. 23 is best.
                
    if best_U is None:
        # Fallback to schoolbook
        U_tot, V_tot, W_tot, R_tot = combine_333_331(U_333_sb, V_333_sb, W_333_sb, U_331_sb, V_331_sb, W_331_sb)
        best_U, best_V, best_W = U_tot, V_tot, W_tot
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
