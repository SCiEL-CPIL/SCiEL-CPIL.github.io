import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Build the rank-84 Strassen solution
    rank = 84
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    C_A = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ]
    
    C_B = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]
    
    Targets = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    for t in range(7):
        for i_loc in range(2):
            for j_loc in range(2):
                for k_loc in range(3):
                    for i_blk in range(2):
                        for j_blk in range(2):
                            coeff_a = C_A[t][i_blk][j_blk]
                            if coeff_a != 0:
                                I = i_blk * 2 + i_loc
                                J = j_blk * 2 + j_loc
                                U[I * m + J, col] = coeff_a
                    for j_blk in range(2):
                        for k_blk in range(2):
                            coeff_b = C_B[t][j_blk][k_blk]
                            if coeff_b != 0:
                                J = j_blk * 2 + j_loc
                                K = k_blk * 3 + k_loc
                                V[J * p + K, col] = coeff_b
                    for i_t, k_t, coeff_c in Targets[t]:
                        I = i_t * 2 + i_loc
                        K = k_t * 3 + k_loc
                        W[K * n + I, col] = coeff_c
                    col += 1
    
    current_rank = rank
    
    # Vectorized reduction
    # U: (nm, R), V: (mp, R), W: (pn, R)
    # Reshape for einsum:
    # U: (n, m, R)
    # V: (m, p, R)
    # W: (p, n, R)
    
    while current_rank > 0:
        if time.time() > deadline:
            break
            
        U_curr = U[:, :current_rank]
        V_curr = V[:, :current_rank]
        W_curr = W[:, :current_rank]
        
        U_reshaped = U_curr.reshape(n, m, current_rank)
        V_reshaped = V_curr.reshape(m, p, current_rank)
        W_reshaped = W_curr.reshape(p, n, current_rank)
        
        # T[i, j, k, r] = U[i, j, r] * V[j, k, r] * W[k, i, r]
        try:
            T = np.einsum('ijr,jkr,kir->ijk r', U_reshaped, V_reshaped, W_reshaped)
        except Exception:
            break
            
        # Check which columns r have T[:, :, :, r] == 0
        # T shape: (n, m, p, current_rank)
        # Sum of squares for each r
        norms = np.sum(T**2, axis=(0, 1, 2))
        
        # Find indices where norm is 0
        zero_indices = np.where(norms < 1e-9)[0]
        
        if len(zero_indices) == 0:
            break
            
        # Remove all zero columns at once
        # Create mask of columns to keep
        keep_mask = np.ones(current_rank, dtype=bool)
        keep_mask[zero_indices] = False
        
        U = U[:, keep_mask]
        V = V[:, keep_mask]
        W = W[:, keep_mask]
        current_rank = np.sum(keep_mask)
        
        if current_rank == 0:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
