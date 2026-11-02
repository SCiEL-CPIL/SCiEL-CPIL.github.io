import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # A: n x m (2x2), B: m x p (2x6), C: p x n (6x2) -> Wait, task says W[k*n+i, r]
    # Output C is 2x6.
    # Tensor T[i, j, k] where i in A (4), j in B (12), k in C (12).
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), ..., 5:(0,5), 6:(1,0), ..., 11:(1,5)
    # C indices: 0:(0,0), 1:(0,1), ..., 5:(0,5), 6:(1,0), ..., 11:(1,5)
    
    # Build the multiplication tensor
    # T[i, j, k] = 1 if C_k = A_i * B_j
    # C[row, col] = sum B[row, mid] * A[mid, col] ? No.
    # Standard: C = A * B.
    # A is 2x2. B is 2x6.
    # C[i, k] = sum_m A[i, m] B[m, k].
    # i in 0..1 (row of A/C)
    # m in 0..1 (col of A / row of B)
    # k in 0..5 (col of B/C)
    
    # Flatten indices:
    # A_idx = i * 2 + m
    # B_idx = m * 6 + k
    # C_idx = i * 6 + k
    
    T = np.zeros((4, 12, 12))
    for i in range(2):
        for m in range(2):
            for k in range(6):
                a_idx = i * 2 + m
                b_idx = m * 6 + k
                c_idx = i * 6 + k
                T[a_idx, b_idx, c_idx] = 1.0

    # Target Rank
    # Try to find rank 19
    target_rank = 19
    
    best_result = None
    best_err = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # Random Initialization
    # Use small random values
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        U = np.random.randn(4, target_rank) * 0.5
        V = np.random.randn(12, target_rank) * 0.5
        W = np.random.randn(12, target_rank) * 0.5
        
        # Simple Gradient Descent
        # L = || T - sum U_r V_r^T W_r^T ||^2
        # This is hard to optimize directly due to non-convexity.
        # Use ALS (Alternating Least Squares)
        
        for it in range(100):
            if time.time() > deadline:
                break
            
            # Fix V, W, optimize U
            # T_i.. = sum_r U_ir V_r.. W_r..
            # For each i, T_i (12x12) = sum_r U_ir (V_r (12x1) outer W_r (12x1)^T)
            # Flatten T_i to 144. Flatten (V_r outer W_r) to 144.
            # T_i = U_i @ M where M is 144 x R
            M = np.zeros((144, target_rank))
            for r in range(target_rank):
                M[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            
            for i in range(4):
                T_flat = T[i].flatten()
                # Solve U[i] = M^+ T_flat
                # Use least squares
                try:
                    U[i] = np.linalg.lstsq(M, T_flat, rcond=None)[0]
                except:
                    pass
            
            # Fix U, W, optimize V
            # T_.j. = sum_r V_jr U_.r W_.r
            # For each j, T_.j. (4x12) = sum_r V_jr (U_.r (4x1) outer W_.r (12x1)^T)
            M = np.zeros((48, target_rank))
            for r in range(target_rank):
                M[:, r] = np.outer(U[:, r], W[:, r]).flatten()
            
            for j in range(12):
                T_flat = T[:, j, :].flatten()
                try:
                    V[j] = np.linalg.lstsq(M, T_flat, rcond=None)[0]
                except:
                    pass
            
            # Fix U, V, optimize W
            # T_..k = sum_r W_kr U_.r V_.r
            # For each k, T_..k (4x12) = sum_r W_kr (U_.r (4x1) outer V_.r (12x1)^T)
            M = np.zeros((48, target_rank))
            for r in range(target_rank):
                M[:, r] = np.outer(U[:, r], V[:, r]).flatten()
            
            for k in range(12):
                T_flat = T[:, :, k].flatten()
                try:
                    W[k] = np.linalg.lstsq(M, T_flat, rcond=None)[0]
                except:
                    pass
            
            # Calculate error
            T_approx = np.zeros((4, 12, 12))
            for r in range(target_rank):
                T_approx += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            err = np.linalg.norm(T - T_approx)
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-5:
                break

    if best_result is not None and best_err < 1e-4:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to Rank 21 (Current)
    n, m, p = DIMENSIONS
    U_local = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_local = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_local = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    rank = 21
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    num_blocks = p // 2
    for k in range(num_blocks):
        base = k * 7
        for i in range(4):
            for r in range(7):
                U[i, base + r] = U_local[i][r]
        for r in range(7):
            V[2*k, base + r] = V_local[0][r]
            V[2*k+1, base + r] = V_local[1][r]
            V[6+2*k, base + r] = V_local[2][r]
            V[7+2*k, base + r] = V_local[3][r]
        for r in range(7):
            W[4*k, base + r] = W_local[0][r]
            W[4*k+2, base + r] = W_local[1][r]
            W[4*k+1, base + r] = W_local[2][r]
            W[4*k+3, base + r] = W_local[3][r]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
