import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    
    The tensor equation is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 112
    
    def compute_error(U_np, V_np, W_np):
        """
        Compute the sum of squared errors for all (i,j,k).
        Returns 0.0 if exact, otherwise positive value.
        """
        rank = U_np.shape[1]
        # Precompute indices for speed? No, vectorization is better.
        # We need to compute S[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This can be done by iterating over r and accumulating.
        S = np.zeros((n, m, p))
        for r in range(rank):
            u_r = U_np[:, r]
            v_r = V_np[:, r]
            w_r = W_np[:, r]
            
            # u_r is (n*m,), v_r is (m*p,), w_r is (p*n,)
            # We need to reshape them to multiply element-wise in the (i,j,k) space.
            # u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
            
            # This is a tensor contraction.
            # Let's just loop over i, j, k for simplicity if rank is small, 
            # or use einsum if possible.
            # Actually, for each r, we add u_r[i,j] * v_r[j,k] * w_r[k,i] to S[i,j,k].
            # u_r reshaped to (n, m)
            # v_r reshaped to (m, p)
            # w_r reshaped to (p, n)
            
            u_mat = u_r.reshape(n, m)
            v_mat = v_r.reshape(m, p)
            w_mat = w_r.reshape(p, n)
            
            # S[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # This is trace of (u_mat[:, j] * v_mat[j, :] * w_mat[:, i])? No.
            # It's a simple element-wise product sum over r.
            # We can compute the contribution of this r to S.
            # contrib[i,j,k] = u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            
            # Vectorized way:
            # u_mat[:, :, None] * v_mat[None, :, :] * w_mat[None, None, :] -> (n, m, p)
            # But w_mat is (p, n), so we need w_mat.T to be (n, p)? No.
            # w_mat[k, i]. So we need index k from v and i from u.
            # Let's just use einsum: 'ij,jk,ki->ijk'
            # u_mat is (n, m) -> 'ij'
            # v_mat is (m, p) -> 'jk'
            # w_mat is (p, n) -> 'ki'
            # Result is (n, m, p) -> 'ijk'
            
            contrib = np.einsum('ij,jk,ki->ijk', u_mat, v_mat, w_mat)
            S += contrib
            
        error = np.sum((S - 1.0) ** 2)
        return error

    def verify_fast(U_np, V_np, W_np, tol=1e-9):
        """Verify if the decomposition is exact."""
        S = np.zeros((n, m, p))
        rank = U_np.shape[1]
        for r in range(rank):
            u_mat = U_np[:, r].reshape(n, m)
            v_mat = V_np[:, r].reshape(m, p)
            w_mat = W_np[:, r].reshape(p, n)
            S += np.einsum('ij,jk,ki->ijk', u_mat, v_mat, w_mat)
        return np.allclose(S, 1.0, atol=tol)

    def quantize(arr):
        """Quantize to nearest half-integer."""
        return np.round(arr * 2) / 2

    best_U, best_V, best_W = None, None, None
    best_rank = schoolbook_rank
    
    # Schoolbook solution as fallback
    U_sch = np.zeros((n * m, schoolbook_rank))
    V_sch = np.zeros((m * p, schoolbook_rank))
    W_sch = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i * m + j, col] = 1
                V_sch[j * p + k, col] = 1
                W_sch[k * n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sch, V_sch, W_sch

    # Try to find lower rank
    random.seed(42)
    np.random.seed(42)
    
    # Search ranks from 30 to 111
    # We prioritize lower ranks.
    for target_rank in range(30, 112):
        if time.time() > deadline - 20:
            break
            
        if target_rank >= best_rank:
            continue
            
        # Number of restarts
        num_restarts = 5
        if target_rank < 50:
            num_restarts = 10
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            # Scale to avoid large numbers initially
            scale = 1.0
            U_r = np.random.randn(n * m, target_rank) * scale
            V_r = np.random.randn(m * p, target_rank) * scale
            W_r = np.random.randn(p * n, target_rank) * scale
            
            # Flatten for optimizer
            def flatten_params(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def unflatten_params(params):
                U_flat = params[:n*m*target_rank]
                V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank]
                W_flat = params[n*m*target_rank + m*p*target_rank:]
                return U_flat.reshape(n*m, target_rank), V_flat.reshape(m*p, target_rank), W_flat.reshape(p*n, target_rank)
            
            def objective(params):
                U, V, W = unflatten_params(params)
                return compute_error(U, V, W)
            
            initial_params = flatten_params(U_r, V_r, W_r)
            
            # Use L-BFGS-B
            try:
                res = minimize(objective, initial_params, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
                if res.success or res.fun < 1e-6:
                    U_opt, V_opt, W_opt = unflatten_params(res.x)
                    
                    # Snap to half-integers
                    U_q = quantize(U_opt)
                    V_q = quantize(V_opt)
                    W_q = quantize(W_opt)
                    
                    if verify_fast(U_q, V_q, W_q):
                        best_U, best_V, best_W = U_q, V_q, W_q
                        best_rank = target_rank
                        # Break out of restarts, try lower rank? 
                        # Since we are iterating from low to high, if we find one, 
                        # we should check if we can find a lower one. 
                        # But we are going up. So we should break and return?
                        # No, we want the MINIMAL rank. 
                        # So if we find rank 30, we should stop.
                        # But we start at 30. If we find 30, we are done.
                        # If we don't find 30, we try 31, etc.
                        # So if we find a solution, we can return immediately?
                        # Yes, because we are checking in increasing order of rank.
                        
                        # Convert to list and return
                        def to_list(arr):
                            return [[float(arr[i][j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]
                        
                        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
                        
            except Exception as e:
                pass

    # If no better solution found, return schoolbook
    def to_list(arr):
        return [[float(arr[i][j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
