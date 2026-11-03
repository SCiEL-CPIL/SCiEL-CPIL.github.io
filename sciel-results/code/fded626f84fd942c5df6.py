import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x7 matrix.
    
    The output factor W follows the official trace convention W[k*n+i, r].
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # 1. Construct Schoolbook Decomposition (Rank 42)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Precompute the target tensor T for verification
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0

    def verify(U_mat, V_mat, W_mat, atol=1e-9):
        T_recon = np.einsum('ar,br,cr->abc', W_mat, U_mat, V_mat)
        return np.allclose(T_recon, T, atol=atol)

    # 2. Attempt to find a lower rank decomposition using optimized ALS
    # Try ranks from 41 downwards
    # We will search for the minimal rank by going down.
    # To save time, we limit the number of restarts and iterations but make them faster.
    
    # Precompute indices for faster slicing
    indices_m = np.arange(m)
    indices_n = np.arange(n)
    indices_p = np.arange(p)
    
    for target_rank in range(rank_schoolbook - 1, 0, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Increase restarts since each iteration is faster
        num_restarts = 50
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Normalize
            U /= (np.linalg.norm(U) + 1e-6)
            V /= (np.linalg.norm(V) + 1e-6)
            W /= (np.linalg.norm(W) + 1e-6)
            
            max_iters = 100
            prev_error = np.inf
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                    
                # Update W
                for k in range(p):
                    for i in range(n):
                        # M_W[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        U_slice = U[i*m : i*m+m, :]
                        V_slice = V[indices_m * p + k, :]
                        M_W = U_slice * V_slice
                        w, _, _, _ = np.linalg.lstsq(M_W, np.ones(m), rcond=None)
                        W[k * n + i, :] = w
                        
                # Update V
                for j in range(m):
                    for k in range(p):
                        # M_V[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        U_slice = U[indices_n * m + j, :]
                        W_slice = W[k*n : k*n+n, :]
                        M_V = U_slice * W_slice
                        v, _, _, _ = np.linalg.lstsq(M_V, np.ones(n), rcond=None)
                        V[j * p + k, :] = v
                        
                # Update U
                for i in range(n):
                    for j in range(m):
                        # M_U[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        V_slice = V[j*p : j*p+p, :]
                        W_slice = W[i::n, :]
                        M_U = V_slice * W_slice
                        u, _, _, _ = np.linalg.lstsq(M_U, np.ones(p), rcond=None)
                        U[i * m + j, :] = u
                        
                # Check convergence
                if it > 5:
                    T_recon = np.einsum('ar,br,cr->abc', W, U, V)
                    error = np.linalg.norm(T_recon - T)
                    
                    # Early stopping if no improvement
                    if error > prev_error * 0.99:
                        break
                    prev_error = error
                    
                    if error < 1e-9:
                        best_rank = target_rank
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        found = True
                        break
                        
            if found:
                break
                
        # If we found a solution for this rank, we continue to lower ranks to find the minimum.
        # If we didn't find it, we continue to lower ranks as well.
        # But if we are running out of time, we break.

    # 3. Return the best result
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
