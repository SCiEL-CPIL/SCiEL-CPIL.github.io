import time
import numpy as np
import random

DIMENSIONS = [2, 2, 16]

def construct():
    """
    Attempt to find a rank < 56 exact decomposition for 2x2 * 2x16 matrix multiplication.
    Uses random restarts with gradient-free optimization (Nelder-Mead style) on the
    constraint that the tensor decomposition must be exact.
    
    Fallback: rank-56 Strassen-based decomposition.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    num_A = n * m  # 4
    num_B = m * p  # 32
    num_C = p * n  # 32
    
    # The matrix multiplication tensor T has entries T[i, j, k] = 1 if k = i // n * n + j // m ... 
    # Actually, for C = A * B where A is (n x m) and B is (m x p):
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # Tensor form: T[i, j, k] = 1 if the product contributes to C[i,k]
    # In tensor decomposition: T = sum_r U[:,r] outer V[:,r] outer W[:,r]
    # where T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    
    # Build the target tensor T of shape (num_A, num_B, num_C)
    T = np.zeros((num_A, num_B, num_C))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1  # A[i,j] * B[j,k] contributes to C[i,k]
                # Wait, C[i,k] = sum_j A[i,j]*B[j,k]
                # So T[i,j,k] = 1 for all i,j,k? No.
                # T[i,j,k] should be such that sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
                # and the result C[i,k] = sum_j A[i,j]*B[j,k] = sum_j sum_r U[i,r]*V[j,r]*W[k,r]
                # = sum_r U[i,r] * (sum_j V[j,r]*B[j,r]) * W[k,r] ... no that's not right either.
                
    # Let me reconsider the tensor convention.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We want: C[i,k] = sum_r (sum_j U[i,r] * A[i,j]... no
    # Standard tensor decomposition: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where T[i,j,k] is the matrix multiplication tensor.
    # T[i,j,k] = 1 if and only if the (i,j) element of A and (j,k) element of B 
    # multiply to contribute to C[i,k]. Since C[i,k] = sum_j A[i,j]*B[j,k],
    # T[i,j,k] = 1 for all valid i,j,k. But that gives a trivial rank.
    
    # Actually the correct interpretation: the multiplication tensor is defined as
    # T[i, j, k] = delta_{k, i*n + j} ... no.
    
    # Let me use the trace convention mentioned: W[k*n+i, r]
    # C has shape (n, p), so C[i, k] for i in [0,n), k in [0,p)
    # W index: k*n + i, so W[k*n+i, r]
    
    # The tensor T[i, j, k] where i indexes A (n*m=4), j indexes B (m*p=32), k indexes C (p*n=32)
    # T[i, j, k] = 1 if A's element (i//m, i%m) times B's element (j//p, j%p) 
    # contributes to C's element (k//n, k%n)
    # A[i_row, i_col] * B[j_row, j_col] -> C[i_row, j_col]
    # i = i_row * m + i_col, j = j_row * p + j_col, k = j_col * n + i_row
    # Contribution if i_col == j_row
    
    T = np.zeros((num_A, num_B, num_C))
    for i_row in range(n):
        for i_col in range(m):
            i = i_row * m + i_col
            for j_row in range(m):
                for j_col in range(p):
                    j = j_row * p + j_col
                    if i_col == j_row:
                        k = j_col * n + i_row
                        T[i, j, k] = 1
    
    # Verify T
    # num_A=4, num_B=32, num_C=32
    
    # Now try to find rank-r decomposition: T[i,j,k] = sum_{r=0}^{rank-1} U[i,r]*V[j,r]*W[k,r]
    
    def try_rank(rank, max_iter=2000, tol=1e-10):
        """Try to find rank-r decomposition using iterative optimization."""
        # Random initialization
        U = np.random.randn(num_A, rank) * 0.1
        V = np.random.randn(num_B, rank) * 0.1
        W = np.random.randn(num_C, rank) * 0.1
        
        for it in range(max_iter):
            # Compute current tensor estimate
            # T_est[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
            # Error = ||T - T_est||^2
            # Gradient w.r.t. U[i,r]: sum_{j,k} (T[i,j,k] - T_est[i,j,k]) * V[j,r] * W[k,r]
            
            T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = T - T_est
            err_norm = np.linalg.norm(err)
            
            if err_norm < tol:
                return True, U, V, W, err_norm
            
            # Gradient-based update (simple gradient descent with adaptive step)
            # dU[i,r] = sum_{j,k} err[i,j,k] * V[j,r] * W[k,r]
            dU = np.einsum('ijk,jr,kr->ir', err, V, W)
            dV = np.einsum('ijk,ir,kr->jr', err, U, W)
            dW = np.einsum('ijk,ir,jr->kr', err, U, V)
            
            # Adaptive learning rate
            step = 0.01
            U -= step * dU
            V -= step * dV
            W -= step * dW
            
            # Check deadline
            if time.time() > deadline - 10:
                return False, None, None, None, err_norm
        
        return False, None, None, None, np.linalg.norm(T - np.einsum('ir,jr,kr->ijk', U, V, W))
    
    # Try various ranks below 56
    best_rank = 56
    best_decomp = None
    
    for rank in [48, 50, 52, 54]:
        if time.time() > deadline - 30:
            break
        
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
            
            success, U, V, W, err_norm = try_rank(rank, max_iter=500)
            if success:
                best_rank = rank
                best_decomp = (U, V, W)
                break
        if best_rank < 56:
            break
    
    # If we found a better decomposition, return it
    if best_decomp is not None:
        U, V, W = best_decomp
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank-56 Strassen-based decomposition
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    rank = 56
    U = np.zeros((num_A, rank), dtype=float)
    V = np.zeros((num_B, rank), dtype=float)
    W = np.zeros((num_C, rank), dtype=float)
    
    S_U = np.array(strassen_U).T
    S_V = np.array(strassen_V).T
    S_W = np.array(strassen_W).T
    
    for l in range(8):
        col_start = 7 * l
        col_end = 7 * l + 7
        U[:, col_start:col_end] = S_U
        V[2*l:2*l+2, col_start:col_end] = S_V[:2, :]
        V[16+2*l:16+2*l+2, col_start:col_end] = S_V[2:, :]
        W[4*l, col_start:col_end] = S_W[0, :]
        W[4*l+2, col_start:col_end] = S_W[1, :]
        W[4*l+1, col_start:col_end] = S_W[2, :]
        W[4*l+3, col_start:col_end] = S_W[3, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
