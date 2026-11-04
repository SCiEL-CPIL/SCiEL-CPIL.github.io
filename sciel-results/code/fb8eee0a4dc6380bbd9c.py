import time
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq
import random

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # First, build the known rank-56 Strassen decomposition as a fallback
    U56, V56, W56 = _build_strassen_56()
    
    # Verify the 56-rank decomposition
    if _verify_decomposition(U56, V56, W56):
        best_rank = 56
        best_U, best_V, best_W = U56.copy(), V56.copy(), W56.copy()
    else:
        # Fallback to naive 64
        best_rank = 64
        best_U = np.zeros((8, 64), dtype=float)
        best_V = np.zeros((32, 64), dtype=float)
        best_W = np.zeros((16, 64), dtype=float)
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    r = i*32 + j*8 + k
                    best_U[i*4+j, r] = 1
                    best_V[j*8+k, r] = 1
                    best_W[k*2+i, r] = 1
    
    # Try to find a lower-rank decomposition
    # Target: rank 48 (theoretical lower bound might be lower, but 48 is a good target)
    # Strategy: Use a structured random search + local optimization
    
    # For rank R, we need U (8,R), V (32,R), W (16,R)
    # The tensor equation is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # We'll try ranks from 48 up to 55, and for each, attempt to find a valid decomposition
    for target_rank in [48, 49, 50, 51, 52, 53, 54, 55]:
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            U, V, W = _try_find_decomposition(target_rank, n, m, p, start_time, deadline)
            if U is not None:
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break  # Found a valid decomposition for this rank
        
        if best_rank == target_rank:
            break  # Found the target rank, no need to try higher ranks
    
    # Convert to lists for output
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _build_strassen_56():
    """Build the standard 56-rank Strassen decomposition for 2x4x8."""
    U_local = np.zeros((4, 7), dtype=float)
    V_local = np.zeros((4, 7), dtype=float)
    W_local = np.zeros((4, 7), dtype=float)
    
    # Standard Strassen for 2x2x2
    U_local[0,0] = 1; U_local[3,0] = 1
    V_local[0,0] = 1; V_local[3,0] = 1
    W_local[0,0] = 1; W_local[3,0] = 1
    
    U_local[2,1] = 1; U_local[3,1] = 1
    V_local[0,1] = 1
    W_local[2,1] = 1; W_local[3,1] = -1
    
    U_local[0,2] = 1
    V_local[1,2] = 1; V_local[3,2] = -1
    W_local[1,2] = 1; W_local[3,2] = 1
    
    U_local[3,3] = 1
    V_local[2,3] = 1; V_local[0,3] = -1
    W_local[0,3] = 1; W_local[2,3] = 1
    
    U_local[0,4] = 1; U_local[1,4] = 1
    V_local[3,4] = 1
    W_local[0,4] = -1; W_local[1,4] = 1
    
    U_local[2,5] = 1; U_local[0,5] = -1
    V_local[0,5] = 1; V_local[1,5] = 1
    W_local[3,5] = 1
    
    U_local[1,6] = 1; U_local[3,6] = -1
    V_local[2,6] = 1; V_local[3,6] = 1
    W_local[0,6] = 1
    
    total_rank = 56
    U = np.zeros((8, total_rank), dtype=float)
    V = np.zeros((32, total_rank), dtype=float)
    W = np.zeros((16, total_rank), dtype=float)
    
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            a_map = [0, 1, 4, 5] if s == 0 else [2, 3, 6, 7]
            b_map = [0*8+2*j, 0*8+2*j+1, 1*8+2*j, 1*8+2*j+1] if s == 0 else \
                    [2*8+2*j, 2*8+2*j+1, 3*8+2*j, 3*8+2*j+1]
            c_map = [2*j*2+0, (2*j+1)*2+0, 2*j*2+1, (2*j+1)*2+1]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val
    
    return U, V, W

def _verify_decomposition(U, V, W, tol=1e-6):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = 2, 4, 8
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_find_decomposition(rank, n, m, p, start_time, deadline):
    """Try to find a valid rank decomposition using random initialization + local search."""
    # Time budget for this attempt
    time_budget = min(15, deadline - time.time() - 5)
    if time_budget <= 1:
        return None, None, None
    
    # Flatten the tensor: we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all (i,j,k)
    # This is a 3rd-order tensor equation
    
    # Strategy: Fix U and V randomly, then solve for W (or vice versa)
    # The equation is linear in W for fixed U, V:
    # For each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Let's try: fix U, V and solve for W
    # But this is nonlinear in general. Instead, use iterative approach.
    
    # Alternative: use a greedy construction approach
    # Start with a known rank-56 decomposition, then try to merge terms
    
    # For now, use a simple random search with verification
    # This is slow but might find something
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for trial in range(50):
        if time.time() > start_time + time_budget:
            break
        
        # Random initialization with some structure
        U = np.random.randn(8, rank) * 0.5
        V = np.random.randn(32, rank) * 0.5
        W = np.random.randn(16, rank) * 0.5
        
        # Try a few iterations of coordinate descent
        for coord_iter in range(10):
            if time.time() > start_time + time_budget:
                break
            
            # Update U: for each i,j, the equation is sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all k
            # This is linear in U[i*m+j,:] for fixed V, W
            # For fixed (i,j), we need: U[i*m+j,:] @ (V[j*p+k,:] * W[k*n+i,:]) = 1 for all k
            # This is: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for k=0..7
            # Let M[k,r] = V[j*p+k,r] * W[k*n+i,r]
            # Then we need: U[i*m+j,:] @ M^T = 1 (vector of ones)
            # If M has full row rank, we can solve for U[i*m+j,:]
            
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    
                    # Solve U[i*m+j,:] @ M^T = ones(p)
                    # i.e., M @ U[i*m+j,:].T = ones(p)
                    try:
                        sol, residuals, rank_m, s = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[i*m+j, :] = sol
                    except:
                        pass
            
            # Similarly update V and W
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n*m, rank))
                    for i in range(n):
                        for jj in range(m):
                            M[i*m+jj, :] = U[i*m+jj, :] * W[k*n+i, :]
                    
                    # We need: sum_r U[i*m+jj,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all (i,jj)
                    # For fixed (j,k): sum_r V[j*p+k,r] * (U[i*m+jj,r] * W[k*n+i,r]) = 1
                    # Let M2[imj, r] = U[imj, r] * W[k*n+i, r] where imj = i*m+jj
                    # Then V[j*p+k,:] @ M2^T = ones(n*m)
                    
                    M2 = np.zeros((n*m, rank))
                    for i in range(n):
                        for jj in range(m):
                            M2[i*m+jj, :] = U[i*m+jj, :] * W[k*n+i, :]
                    
                    try:
                        sol, residuals, rank_m, s = np.linalg.lstsq(M2, np.ones(n*m), rcond=None)
                        V[j*p+k, :] = sol
                    except:
                        pass
            
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m*p, rank))
                    for j in range(m):
                        for kk in range(p):
                            M[j*p+kk, :] = U[i*m+j, :] * V[j*p+kk, :]
                    
                    # We need: sum_r U[i*m+j,r] * V[j*p+kk,r] * W[kk*n+i,r] = 1 for all (j,kk)
                    # For fixed (k,i): sum_r W[k*n+i,r] * (U[i*m+j,r] * V[j*p+kk,r]) = 1
                    # Let M3[jpkk, r] = U[i*m+j, r] * V[j*p+kk, r]
                    # Then W[k*n+i,:] @ M3^T = ones(m*p)
                    
                    M3 = np.zeros((m*p, rank))
                    for j in range(m):
                        for kk in range(p):
                            M3[j*p+kk, :] = U[i*m+j, :] * V[j*p+kk, :]
                    
                    try:
                        sol, residuals, rank_m, s = np.linalg.lstsq(M3, np.ones(m*p), rcond=None)
                        W[k*n+i, :] = sol
                    except:
                        pass
        
        # Check error
        error = _compute_tensor_error(U, V, W, n, m, p)
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-6:
            return U, V, W
    
    # Check if we found a valid decomposition
    if best_U is not None and _verify_decomposition(best_U, best_V, best_W, tol=1e-4):
        return best_U, best_V, best_W
    
    return None, None, None

def _compute_tensor_error(U, V, W, n, m, p):
    """Compute the max error in the tensor equation."""
    R = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err
# EVOLVE-BLOCK-END

