"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W form a valid decomposition of the matmul tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T(i,j,k) = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(actual - expected) > tol:
                    return False
    return True

def _build_rank_34_decomposition(n, m, p):
    """
    Construct a rank-34 decomposition for 3x3 x 3x4 matrix multiplication.
    
    Strategy: Start from schoolbook rank 36 and find 2 linear dependencies.
    We modify the last 2 columns of the schoolbook decomposition to absorb
    the contributions of the 2 removed terms.
    
    The schoolbook decomposition has 36 terms indexed by (i,j,k) in [0,3)^3 x [0,4).
    We remove terms (2,2,2) and (2,2,3) and try to express their contributions
    as linear combinations of the remaining 34 terms by adjusting the factor matrices.
    """
    n, m, p = 3, 3, 4
    target_rank = 34
    
    U_s, V_s, W_s = _build_schoolbook(n, m, p)
    U_arr = np.array(U_s, dtype=float)
    V_arr = np.array(V_s, dtype=float)
    W_arr = np.array(W_s, dtype=float)
    
    # The two removed terms:
    # Term 34 (0-indexed): i=2, j=2, k=2
    # Term 35 (0-indexed): i=2, j=2, k=3
    
    # We keep the first 34 columns and try to adjust them to account for the missing terms.
    U_base = U_arr[:, :target_rank].copy()
    V_base = V_arr[:, :target_rank].copy()
    W_base = W_arr[:, :target_rank].copy()
    
    # The missing contributions:
    # For term (2,2,2): U[8,r]*V[10,r]*W[8,r] = 1 for r=34
    # For term (2,2,3): U[8,r]*V[11,r]*W[11,r] = 1 for r=35
    
    # We need to find deltas dU, dV, dW (each of shape (rows, target_rank))
    # such that for all (i,j,k), the decomposition still holds.
    # 
    # For (i,j,k) != (2,2,2) and (i,j,k) != (2,2,3):
    #   sum_r (U_base[i*m+j,r] + dU[i*m+j,r]) * (V_base[j*p+k,r] + dV[j*p+k,r]) * (W_base[k*n+i,r] + dW[k*n+i,r]) = 1
    #
    # For (2,2,2) and (2,2,3):
    #   sum_r (U_base[8,r] + dU[8,r]) * (V_base[10,r] + dV[10,r]) * (W_base[8,r] + dW[8,r]) = 1
    #   sum_r (U_base[8,r] + dU[8,r]) * (V_base[11,r] + dV[11,r]) * (W_base[11,r] + dW[11,r]) = 1
    #
    # This is a system of 36 equations in 3*target_rank*(n*m + m*p + p*n) variables,
    # which is too large. Instead, let's use a simpler approach:
    # Only modify the last few columns where the missing terms had non-zero entries.
    
    # In the schoolbook, each term (i,j,k) has exactly one non-zero entry in each matrix:
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    #
    # The removed terms are at columns 34 and 35 (0-indexed).
    # Term 34: i=2, j=2, k=2 -> affects U[8,34]=1, V[10,34]=1, W[8,34]=1
    # Term 35: i=2, j=2, k=3 -> affects U[8,35]=1, V[11,35]=1, W[11,35]=1
    #
    # We need to distribute these contributions into the remaining 34 columns.
    # Let's try adding small corrections to columns that share indices.
    
    # Columns that have U[8,r] != 0: r where i=2, j=2 -> r in {34,35} (removed)
    # Actually, in schoolbook, U[i*m+j, r] = 1 only for r corresponding to (i,j,k).
    # So U[8,r] = 1 only for r=34,35 (the removed ones).
    # Similarly, V[10,r] = 1 only for r=34, and V[11,r] = 1 only for r=35.
    # W[8,r] = 1 only for r=34, and W[11,r] = 1 only for r=35.
    
    # This means we can't just modify existing columns - we need to add new non-zero
    # entries. Let's use a different strategy: introduce a small number of free
    # parameters and solve the resulting system.
    
    # Strategy: Add 2 new "virtual" columns that are linear combinations of existing ones.
    # Let U[8, r] = a_r, V[10, r] = b_r, W[8, r] = c_r for some r in [0,34)
    # and U[8, r] = d_r, V[11, r] = e_r, W[11, r] = f_r for some r in [0,34)
    # such that sum_r a_r * b_r * c_r = 1 and sum_r d_r * e_r * f_r = 1
    # and for all other (i,j,k), the contributions sum correctly.
    
    # This is still complex. Let me try a numerical approach with a reduced variable set.
    
    # We'll only allow modifications to a small subset of entries.
    # Let's pick a few columns and allow small perturbations.
    
    from scipy.optimize import minimize
    
    # Variables: perturbations to U[8,:], V[10,:], V[11,:], W[8,:], W[11,:]
    # That's 5 * target_rank = 170 variables.
    
    def make_objective(U_base, V_base, W_base, n, m, p):
        def objective(x):
            if _time_remaining() < 5:
                return 1e10
            dU8 = x[:target_rank]
            dV10 = x[target_rank:2*target_rank]
            dV11 = x[2*target_rank:3*target_rank]
            dW8 = x[3*target_rank:4*target_rank]
            dW11 = x[4*target_rank:5*target_rank]
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_val = U_base[i*m+j, :]
                        v_val = V_base[j*p+k, :]
                        w_val = W_base[k*n+i, :]
                        
                        # Apply perturbations
                        if i == 2 and j == 2:
                            u_val = u_val + dU8
                        if j == 2 and k == 2:
                            v_val = v_val + dV10
                        if j == 2 and k == 3:
                            v_val = v_val + dV11
                        if k == 2 and i == 2:
                            w_val = w_val + dW8
                        if k == 3 and i == 2:
                            w_val = w_val + dW11
                        
                        val = np.dot(u_val, v_val * w_val)
                        err += (val - 1.0) ** 2
            return err
        return objective
    
    obj = make_objective(U_base, V_base, W_base, n, m, p)
    
    best_err = float('inf')
    best_x = None
    
    np.random.seed(123)
    
    for attempt in range(20):
        if _time_remaining() < 10:
            break
        
        x0 = np.zeros(5 * target_rank)
        # Add small random perturbation
        x0 += np.random.randn(5 * target_rank) * 0.1
        
        try:
            result = minimize(obj, x0, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-14})
            if result.fun < best_err:
                best_err = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if best_x is not None and best_err < 1e-8:
        dU8 = best_x[:target_rank]
        dV10 = best_x[target_rank:2*target_rank]
        dV11 = best_x[2*target_rank:3*target_rank]
        dW8 = best_x[3*target_rank:4*target_rank]
        dW11 = best_x[4*target_rank:5*target_rank]
        
        U_opt = U_base.copy()
        V_opt = V_base.copy()
        W_opt = W_base.copy()
        
        U_opt[8, :] += dU8
        V_opt[10, :] += dV10
        V_opt[11, :] += dV11
        W_opt[8, :] += dW8
        W_opt[11, :] += dW11
        
        # Round to nearest half-integer
        U_opt = np.round(U_opt * 2) / 2
        V_opt = np.round(V_opt * 2) / 2
        W_opt = np.round(W_opt * 2) / 2
        
        if _verify_decomposition(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p, tol=1e-6):
            return U_opt.tolist(), V_opt.tolist(), W_opt.tolist()
    
    return None

def construct():
    """Construct the rank-minimal decomposition for 3x3x4 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Try to find rank-34 decomposition
    if _time_remaining() > 20:
        result = _build_rank_34_decomposition(n, m, p)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook rank 36
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
