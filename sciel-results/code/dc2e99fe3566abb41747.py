import numpy as np
import time
import sys
from fractions import Fraction
import scipy.optimize

DIMENSIONS = [2, 2, 5]

def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
    return T

def _snap_to_rational(val, max_denom=16):
    """Snap a float value to the nearest rational with denominator <= max_denom if close enough."""
    for d in range(1, max_denom + 1):
        num = round(val * d)
        if abs(val - num / d) < 1e-9:
            return Fraction(num, d)
    return None

def _check_decomposition_exact_fast(U, V, W, T):
    """
    Fast exact check.
    1. Convert inputs to float.
    2. Compute tensor contraction.
    3. If close to T, try to snap all entries to rationals with small denominators.
    4. If snapping succeeds, verify exactness with Fractions.
    5. If snapping fails, return False (strict requirement for integer/half-integer).
    """
    n, m, p = DIMENSIONS
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    
    # Fast float check
    result = np.einsum('ur,vr,wr->uvw', U_a, V_a, W_a)
    
    # Check if float result matches T closely
    if not np.allclose(result, T.astype(np.float64), atol=1e-8, rtol=0):
        return False
    
    # Attempt to snap to exact rationals (integers or halves primarily, up to /16)
    U_frac = np.zeros_like(U_a, dtype=object)
    V_frac = np.zeros_like(V_a, dtype=object)
    W_frac = np.zeros_like(W_a, dtype=object)
    
    def try_snap_matrix(mat, mat_frac):
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                f = _snap_to_rational(mat[i, j], max_denom=8) # Denom up to 8 is usually safe for these problems
                if f is None:
                    return False
                mat_frac[i, j] = f
        return True

    if not try_snap_matrix(U_a, U_frac):
        return False
    if not try_snap_matrix(V_a, V_frac):
        return False
    if not try_snap_matrix(W_a, W_frac):
        return False
    
    # Exact verification with Fractions
    R = U_a.shape[1]
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                total = Fraction(0)
                for r in range(R):
                    total += U_frac[u, r] * V_frac[v, r] * W_frac[w, r]
                if total != T[u, v, w]:
                    return False
    return True

def _build_rank18(n, m, p):
    """Build the known rank-18 Strassen+schoolbook decomposition."""
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1

    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    return U, V, W

def _try_rank17_scipy(n, m, p, T, deadline):
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    T_a = np.array(T, dtype=np.float64)
    
    # Define the objective function for SciPy
    def objective(x):
        # x is a flattened vector of length 4*17 + 10*17 + 10*17
        U_flat = x[:4*17]
        V_flat = x[4*17:4*17+10*17]
        W_flat = x[4*17+10*17:]
        
        U = U_flat.reshape(4, 17)
        V = V_flat.reshape(10, 17)
        W = W_flat.reshape(10, 17)
        
        # Compute tensor
        result = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        # Squared error
        err = np.sum((result - T_a)**2)
        return err

    # Strategy: Remove one term from rank-18 and search
    for term_to_remove in range(18):
        if time.time() > deadline:
            break
        
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        # Flatten initial guess
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Try a few random perturbations to escape local minima
        for seed in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(term_to_remove * 1000 + seed)
            # Small perturbation
            scale = 0.1 * (seed % 5 + 1)
            x_pert = x0 + np.random.randn(len(x0)) * scale
            
            # Use Powell method, which is derivative-free and robust
            # maxfev is the max number of function evaluations
            # We have 150s total. Each eval is fast (~1ms). 
            # Let's allow 1000 evals per attempt. 20 attempts * 18 terms = 360 attempts.
            # 360 * 1000 = 360,000 evals. Might be too slow if each is 1ms.
            # Let's limit to 200 evals per attempt.
            try:
                res = scipy.optimize.minimize(objective, x_pert, method='Powell', options={'maxfev': 200, 'maxiter': 100})
            except Exception:
                continue
            
            if res.fun < 1e-8:
                # Extract solution
                U_sol = res.x[:4*17].reshape(4, 17)
                V_sol = res.x[4*17:4*17+10*17].reshape(10, 17)
                W_sol = res.x[4*17+10*17:].reshape(10, 17)
                
                if _check_decomposition_exact_fast(U_sol, V_sol, W_sol, T):
                    def snap_list(mat):
                        out = []
                        for row in mat:
                            r = []
                            for val in row:
                                f = _snap_to_rational(val, max_denom=8)
                                if f is not None:
                                    r.append(float(f))
                                else:
                                    r.append(val)
                            out.append(r)
                        return out
                    return snap_list(U_sol), snap_list(V_sol), snap_list(W_sol)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145 # 5s safety margin
    
    T = _build_target()
    
    # Verify rank-18 exists (it should)
    U18, V18, W18 = _build_rank18(n, m, p)
    
    # Try to find rank-17 using SciPy
    result17 = _try_rank17_scipy(n, m, p, T, deadline)
    if result17 is not None:
        U, V, W = result17
        # Final verification
        if _check_decomposition_exact_fast(U, V, W, T):
            return {"U": U, "V": V, "W": W}
            
    # Fallback to rank-18
    return {
        "U": U18,
        "V": V18,
        "W": W18
    }
