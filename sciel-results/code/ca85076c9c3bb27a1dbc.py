"""Tensor decomposition search for 3x3x4 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W, n, m, p):
    """Check if U @ V^T reconstructs the matmul tensor exactly."""
    rank = U.shape[1]
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Build the target tensor: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition: C[i,k] = sum_r U[i*m+j_r? ...]
    # Actually: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should give the coefficient
    # For matmul: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i,j,k] = 1 (the coefficient of A[i,j]*B[j,k] in C[i,k])
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Compute sum_r U[i*m+j, r] * V[j*p+k, r]
    # For each (i,j,k), check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    
    # Efficient: for each r, outer product structure
    # T_approx[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's compute the full tensor
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    
    # T_approx[i,j,k] = sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
    # This is a 3-way contraction
    
    # Compute via einsum: sum over r of Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
    # First: sum_r Ur[i,j,r] * Vr[j,k,r] gives an (n,m,p) tensor
    # Then multiply by Wr[k,i,r]... wait, that's not right.
    
    # Actually the decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So for each r, we have a rank-1 tensor: U[:,r] (indexed by i*m+j) 
    #   * V[:,r] (indexed by j*p+k) * W[:,r] (indexed by k*n+i)
    
    # The product U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # is nonzero only when the indices are consistent: i from U, j from U and V, k from V and W, and the W index must be k*n+i.
    
    # So effectively, for each r, this contributes to T[i,j,k] where:
    # U has nonzero at (i*m+j), V has nonzero at (j*p+k), W has nonzero at (k*n+i)
    
    # Let's just compute it directly for small dimensions
    T_approx = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T_approx[i, j, k] = val
    
    # Target: T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_approx, T_target, atol=1e-10, rtol=1e-10), T_approx
    
def _round_to_half(x):
    """Round to nearest half-integer."""
    return np.round(2.0 * x) / 2.0

def _refine(U, V, W, n, m, p, max_iter=200, tol=1e-12):
    """Alternating least squares refinement."""
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank).copy()
    Vr = V.reshape(m, p, rank).copy()
    Wr = W.reshape(p, n, rank).copy()
    
    for it in range(max_iter):
        # Update U: for each (i,j), solve least squares
        # T[i,j,k] = sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
        # For fixed (i,j): T[i,j,:] = sum_r Ur[i,j,r] * (Vr[j,:,r] * Wr[:,i,r])
        # Let M[i,j] be a (p, rank) matrix where M[i,j][k,r] = Vr[j,k,r] * Wr[k,i,r]
        # Then T[i,j,:] = M[i,j] @ Ur[i,j,:]  (if we think of it as matrix-vector)
        # Actually: T[i,j,k] = sum_r Ur[i,j,r] * M[i,j][k,r]
        # So T[i,j] = M[i,j]^T @ Ur[i,j]  where M[i,j] is (p, rank)
        
        for i in range(n):
            for j in range(m):
                # Build M: (p, rank) where M[k, r] = Vr[j,k,r] * Wr[k,i,r]
                M = Vr[j, :, :] * Wr[:, i, :]  # (p, rank)
                # Solve: M^T @ x = T_target[i,j,:] = ones(p)
                # x = Ur[i,j,:]
                try:
                    sol, residuals, rank_s, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    Ur[i, j, :] = sol
                except:
                    pass
        
        # Update V: for each (j,k)
        # T[i,j,k] = sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
        # For fixed (j,k): T[:,j,k] = sum_r Vr[j,k,r] * (Ur[:,j,r] * Wr[k,:,r])
        # Let N[j,k] be (n, rank) where N[i,r] = Ur[i,j,r] * Wr[k,i,r]
        # T[:,j,k] = N[j,k]^T @ Vr[j,k,:]
        
        for j in range(m):
            for k in range(p):
                N = Ur[:, j, :] * Wr[k, :, :]  # (n, rank)
                try:
                    sol, residuals, rank_s, sv = np.linalg.lstsq(N.T, np.ones(n), rcond=None)
                    Vr[j, k, :] = sol
                except:
                    pass
        
        # Update W: for each (k,i)
        # T[i,j,k] = sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
        # For fixed (k,i): T[i,:,k] = sum_r Wr[k,i,r] * (Ur[i,:,r] * Vr[:,k,r])
        # Let P[k,i] be (m, rank) where P[j,r] = Ur[i,j,r] * Vr[j,k,r]
        # T[i,:,k] = P[k,i]^T @ Wr[k,i,:]
        
        for k in range(p):
            for i in range(n):
                P = Ur[i, :, :] * Vr[:, k, :]  # (m, rank)
                try:
                    sol, residuals, rank_s, sv = np.linalg.lstsq(P.T, np.ones(m), rcond=None)
                    Wr[k, i, :] = sol
                except:
                    pass
        
        # Check convergence
        U_flat = Ur.reshape(n*m, rank)
        V_flat = Vr.reshape(m*p, rank)
        W_flat = Wr.reshape(p*n, rank)
        T_approx = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_approx[i, j, k] = np.dot(Ur[i,j,:], Vr[j,k,:] * Wr[k,i,:])
        err = np.max(np.abs(T_approx - 1.0))
        if err < tol:
            break
    
    return Ur.reshape(n*m, rank), Vr.reshape(m*p, rank), Wr.reshape(p*n, rank), err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # First, return schoolbook as fallback
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Try ranks from low to high
    min_possible = 18  # lower bound for 3x3x4 is known to be around 19-23
    max_rank_to_try = 30
    
    for rank in range(min_possible, min(max_rank_to_try + 1, best_rank)):
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            random.seed(hash((rank, restart, n, m, p)) & 0xFFFFFFFF)
            np.random.seed(hash((rank, restart, n, m, p)) & 0xFFFFFFFF)
            
            # Random initialization
            U_init = np.random.randn(n*m, rank) * 0.1
            V_init = np.random.randn(m*p, rank) * 0.1
            W_init = np.random.randn(p*n, rank) * 0.1
            
            # Refine
            U_ref, V_ref, W_ref, err = _refine(U_init, V_init, W_init, n, m, p, max_iter=100)
            
            if err < 1e-8:
                # Try rounding to half-integers
                U_round = _round_to_half(U_ref)
                V_round = _round_to_half(V_ref)
                W_round = _round_to_half(W_ref)
                
                valid, _ = _verify(U_round, V_round, W_round, n, m, p)
                if valid:
                    best_rank = rank
                    best_result = {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                    break
                else:
                    # Try without rounding (check if already close to half-integers)
                    valid, _ = _verify(U_ref, V_ref, W_ref, n, m, p)
                    if valid:
                        best_rank = rank
                        best_result = {
                            "U": U_ref.tolist(),
                            "V": V_ref.tolist(),
                            "W": W_ref.tolist()
                        }
                        break
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        best_result = {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
    
    return best_result
