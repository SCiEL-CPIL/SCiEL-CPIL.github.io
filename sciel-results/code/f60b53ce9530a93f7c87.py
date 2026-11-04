"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product
import scipy.optimize

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    Uses optimization to find a low-rank approximation, then snaps to half-integers
    and verifies exactly.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    def schoolbook_decomposition():
        rank_schoolbook = n * m * p
        U = np.zeros((n*m, rank_schoolbook))
        V = np.zeros((m*p, rank_schoolbook))
        W = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_school, V_school, W_school = schoolbook_decomposition()
    
    def verify_half_integer(U, V, W):
        """
        Verify that U, V, W with half-integer entries exactly decompose the tensor.
        We check if 8 * sum(U*V*W) == 8 for all i,j,k, i.e., sum(U*V*W) == 1.
        Since entries are half-integers, 2*U, 2*V, 2*W are integers.
        Let U' = 2U, V' = 2V, W' = 2W.
        Then U*V*W = U'V'W' / 8.
        We need sum(U'V'W') == 8.
        """
        U_int = (2 * U).astype(int)
        V_int = (2 * V).astype(int)
        W_int = (2 * W).astype(int)
        
        # Check if rounding was exact
        if not np.allclose(U * 2, U_int) or not np.allclose(V * 2, V_int) or not np.allclose(W * 2, W_int):
            return False
            
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(U.shape[1]):
                        val += U_int[i*m+j, r] * V_int[j*p+k, r] * W_int[k*n+i, r]
                    if val != 8:
                        return False
        return True

    def loss_function(x, R):
        """
        Loss function for optimization.
        x is flattened [U, V, W].
        """
        U_size = n * m * R
        V_size = m * p * R
        W_size = p * n * R
        
        U = x[:U_size].reshape(n*m, R)
        V = x[U_size:U_size+V_size].reshape(m*p, R)
        W = x[U_size+V_size:].reshape(p*n, R)
        
        # Compute tensor residual
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want T to be 1.
        
        # Vectorized computation is tricky due to index permutations, 
        # but for small dimensions, a loop is fast enough.
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum over r
                    val = np.dot(U[i*m+j, :], V[j*p+k, :]) # This is wrong, W is missing
                    # Correct: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # This is element-wise product sum
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    loss += (val - 1.0)**2
        return loss

    best_rank = n * m * p
    best_decomp = (U_school, V_school, W_school)
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound is max(nm, mp, pn) = 20
    candidate_ranks = [20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
    
    for target_rank in candidate_ranks:
        if target_rank >= best_rank:
            continue
        if time.time() > deadline - 20:
            break
            
        found = False
        # Try a few random initializations
        num_restarts = 10
        for restart in range(num_restarts):
            if time.time() > deadline - 20:
                break
                
            # Random initialization
            U_init = np.random.randn(n*m, target_rank)
            V_init = np.random.randn(m*p, target_rank)
            W_init = np.random.randn(p*n, target_rank)
            
            # Scale to help optimization
            U_init /= np.sqrt(target_rank)
            V_init /= np.sqrt(target_rank)
            W_init /= np.sqrt(target_rank)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = scipy.optimize.minimize(
                    loss_function, x0, args=(target_rank,), 
                    method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-8}
                )
            except Exception:
                continue
                
            if res.success or res.fun < 1e-5:
                # Extract U, V, W
                U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Snap to half-integers
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                if verify_half_integer(U_snap, V_snap, W_snap):
                    found = True
                    best_rank = target_rank
                    best_decomp = (U_snap, V_snap, W_snap)
                    break
        
        if found:
            break
    
    if best_rank == n * m * p:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    else:
        U_best, V_best, W_best = best_decomp
        return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
