import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using rational arithmetic."""
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for r in range(R):
                    u_val = Fraction(U[i*m + j, r]).limit_denominator(10**6)
                    v_val = Fraction(V[j*p + k, r]).limit_denominator(10**6)
                    w_val = Fraction(W[k*n + i, r]).limit_denominator(10**6)
                    val += u_val * v_val * w_val
                if val != 1:
                    return False
    return True

def _try_rank_direct_enumeration(target_rank, n, m, p, deadline, start_time):
    """
    Try to find a rank decomposition by enumerating small integer/half-integer values.
    For small ranks, this can be exhaustive.
    """
    # For rank R, we need to assign values to U (n*m x R), V (m*p x R), W (p*n x R)
    # Each entry is in {-1, -0.5, 0, 0.5, 1} for small ranks
    # This is a constraint satisfaction problem
    
    # Build the tensor equations:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Number of unknowns: (n*m + m*p + p*n) * R
    # Number of equations: n*m*p
    
    # For small R, we can try random sampling with verification
    # or use a more systematic approach
    
    # Strategy: random sampling with intelligent initialization
    # Focus on values that are integers or half-integers
    
    candidate_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # Try random assignments from candidate values
    max_attempts = 5000
    
    for attempt in range(max_attempts):
        if time.time() - start_time > deadline:
            return None
        
        # Random assignment
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        for idx in range(n*m):
            for r in range(target_rank):
                U[idx, r] = candidate_values[np.random.randint(len(candidate_values))]
        
        for idx in range(m*p):
            for r in range(target_rank):
                V[idx, r] = candidate_values[np.random.randint(len(candidate_values))]
        
        for idx in range(p*n):
            for r in range(target_rank):
                W[idx, r] = candidate_values[np.random.randint(len(candidate_values))]
        
        # Quick numerical check
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(target_rank):
                        val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    error += (val - 1.0) ** 2
        
        if error < 1e-9:
            # Verify exactly
            if _verify_exact(U, V, W, n, m, p):
                return (U, V, W)
    
    return None

def _als_decomposition(target_rank, n, m, p, start_time, deadline, max_iters=500):
    """
    ALS-based decomposition with improved convergence.
    """
    best_result = None
    best_error = np.inf
    
    for restart in range(50):
        if time.time() - start_time > deadline:
            break
        
        np.random.seed(restart * 104729 + target_rank * 31)
        
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Normalize
        for r in range(target_rank):
            nu = np.linalg.norm(U[:, r])
            nv = np.linalg.norm(V[:, r])
            nw = np.linalg.norm(W[:, r])
            if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
                scale = (nu * nv * nw) ** (1/3)
                U[:, r] /= nu / scale
                V[:, r] /= nv / scale
                W[:, r] /= nw / scale
        
        for iteration in range(max_iters):
            if time.time() - start_time > deadline:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                    try:
                        U[i*m + j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, target_rank))
                    for i in range(n):
                        B[i, :] = U[i*m + j, :] * W[k*n + i, :]
                    try:
                        V[j*p + k, :] = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, target_rank))
                    for j in range(m):
                        C[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    try:
                        W[k*n + i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(target_rank):
                            val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                        error += (val - 1.0) ** 2
            
            if error < 1e-10:
                # Try to snap to nice values
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if _verify_exact(U_snap, V_snap, W_snap, n, m, p):
                    return (U_snap, V_snap, W_snap)
                
                # Also try integer snapping
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                
                if _verify_exact(U_int, V_int, W_int, n, m, p):
                    return (U_int, V_int, W_int)
                
                # Try mixed snapping
                for factor in [1, 2, 3, 4]:
                    U_f = np.round(U * factor) / factor
                    V_f = np.round(V * factor) / factor
                    W_f = np.round(W * factor) / factor
                    if _verify_exact(U_f, V_f, W_f, n, m, p):
                        return (U_f, V_f, W_f)
                
                # Keep as floating point if exact enough
                if error < 1e-12:
                    return (U, V, W)
            
            if error < best_error:
                best_error = error
                best_result = (U.copy(), V.copy(), W.copy())
        
        # After ALS, try to improve best result
        if best_result is not None:
            U_best, V_best, W_best = best_result
            for factor in [1, 2, 3, 4, 5, 6, 8, 10]:
                U_f = np.round(U_best * factor) / factor
                V_f = np.round(V_best * factor) / factor
                W_f = np.round(W_best * factor) / factor
                if _verify_exact(U_f, V_f, W_f, n, m, p):
                    return (U_f, V_f, W_f)
    
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3x3 matrix multiplication.
    Uses multiple strategies: direct enumeration for small ranks, ALS for larger ranks.
    Falls back to schoolbook rank 18 if no lower rank is found.
    """
    n, m, p = DIMENSIONS
    R_school = n * m * p  # 18
    
    # Build schoolbook solution as fallback
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school
    
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Try ranks from low to high
    # Based on literature, R(2,3,3) is likely around 14-16
    for target_rank in range(13, R_school):
        if time.time() - start_time > deadline:
            break
        
        found = False
        
        # Strategy 1: Direct enumeration for small ranks
        if target_rank <= 15:
            result = _try_rank_direct_enumeration(target_rank, n, m, p, deadline, start_time)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = target_rank
                found = True
                break
        
        # Strategy 2: ALS
        if not found:
            result = _als_decomposition(target_rank, n, m, p, start_time, deadline)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
