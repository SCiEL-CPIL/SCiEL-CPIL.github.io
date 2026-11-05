import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS
    deadline = t_start + 140  # safety margin before 150s kill

    def schoolbook():
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
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    best_rank = n * m * p

    def check_decomposition(U, V, W, tol=1e-10):
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False, abs(val - 1.0)
        return True, 0.0

    def snap_to_half_integers(U, V, W):
        U_snap = np.round(np.array(U, dtype=np.float64) * 2) / 2
        V_snap = np.round(np.array(V, dtype=np.float64) * 2) / 2
        W_snap = np.round(np.array(W, dtype=np.float64) * 2) / 2
        return U_snap, V_snap, W_snap

    def try_als(target_rank, max_iters=500, seed=0):
        """ALS-style optimization for tensor decomposition."""
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        # Precompute index mappings
        # For each (i,j,k), we need: U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:]
        # We'll solve for each matrix in turn using least squares
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Solve for U: fix V, W
            # For each row of U (indexed by i*m+j), we need:
            # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
            # This is: U[i*m+j, :] @ diag(V[j*p+k,:] * W[k*n+i,:]) = 1
            # Actually, let's think of it as a linear system.
            # For fixed (i,j), and for each k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # Let A_k = V[j*p+k,:] * W[k*n+i,:] (element-wise product)
            # Then sum_r U[i*m+j,r] * A_k[r] = 1, for k = 0,...,p-1
            # This is: U[i*m+j,:] @ A^T = 1 (vector of ones)
            # Where A is p x target_rank with A[k,r] = V[j*p+k,r] * W[k*n+i,r]
            
            for i in range(n):
                for j in range(m):
                    # Build system: for each k, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve: U[i*m+j,:] @ A.T = ones(p)
                    # i.e., A @ U[i*m+j,:].T = ones(p)
                    try:
                        sol = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        U[i*m+j, :] = sol
                    except:
                        pass
            
            # Solve for V: fix U, W
            for j in range(m):
                for k in range(p):
                    # For fixed (j,k), sum_r V[j*p+k,r] * U[i*m+j,r] * W[k*n+i,r] = 1 for all i
                    # Let B_i = U[i*m+j,:] * W[k*n+i,:]
                    # Then sum_r V[j*p+k,r] * B_i[r] = 1 for i = 0,...,n-1
                    # V[j*p+k,:] @ B^T = ones(n)
                    B = np.zeros((n, target_rank))
                    for i in range(n):
                        B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        sol = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                        V[j*p+k, :] = sol
                    except:
                        pass
            
            # Solve for W: fix U, V
            for k in range(p):
                for i in range(n):
                    # For fixed (k,i), sum_r W[k*n+i,r] * U[i*m+j,r] * V[j*p+k,r] = 1 for all j
                    # Let C_j = U[i*m+j,:] * V[j*p+k,:]
                    # Then sum_r W[k*n+i,r] * C_j[r] = 1 for j = 0,...,m-1
                    # W[k*n+i,:] @ C^T = ones(m)
                    C = np.zeros((m, target_rank))
                    for j in range(m):
                        C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        sol = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                        W[k*n+i, :] = sol
                    except:
                        pass
            
            # Check convergence
            if iteration % 50 == 0 or iteration == max_iters - 1:
                valid, err = check_decomposition(U, V, W, tol=1e-8)
                if valid:
                    U_snap, V_snap, W_snap = snap_to_half_integers(U, V, W)
                    valid_snap, _ = check_decomposition(U_snap, V_snap, W_snap, tol=1e-6)
                    if valid_snap:
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    else:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # Try ranks from low to high
    for target_rank in range(24, best_rank):
        if time.time() > deadline - 10:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            result = try_als(target_rank, max_iters=300, seed=seed)
            if result is not None:
                # Verify
                valid, err = check_decomposition(result["U"], result["V"], result["W"], tol=1e-6)
                if valid:
                    best_rank = target_rank
                    best_result = result
                    found = True
                    break
        
        if found:
            break

    return best_result
