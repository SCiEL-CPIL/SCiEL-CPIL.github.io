import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build the rank-28 Strassen decomposition
    rank28 = _build_strassen_rank28()
    
    # Try to find a rank-27 decomposition using algebraic reduction
    elapsed = time.time() - start_time
    if elapsed < 110:
        rank27 = _try_reduce_rank(rank28, n, m, p, time.time() + 140)
        if rank27 is not None:
            return rank27
    
    return rank28

def _build_strassen_rank28():
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(block_id, A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        base_col = block_id * 7
        
        col = base_col + 0
        U[A_rows[0] * m + A_cols[0]][col] += 1
        U[A_rows[1] * m + A_cols[1]][col] += 1
        V[B_rows[0] * p + B_cols[0]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] += 1
        
        col = base_col + 1
        U[A_rows[1] * m + A_cols[0]][col] += 1
        U[A_rows[1] * m + A_cols[1]][col] += 1
        V[B_rows[0] * p + B_cols[0]][col] += 1
        
        col = base_col + 2
        U[A_rows[0] * m + A_cols[0]][col] += 1
        V[B_rows[0] * p + B_cols[1]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] -= 1
        
        col = base_col + 3
        U[A_rows[1] * m + A_cols[1]][col] += 1
        V[B_rows[1] * p + B_cols[0]][col] += 1
        V[B_rows[0] * p + B_cols[0]][col] -= 1
        
        col = base_col + 4
        U[A_rows[0] * m + A_cols[0]][col] += 1
        U[A_rows[0] * m + A_cols[1]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] += 1
        
        col = base_col + 5
        U[A_rows[1] * m + A_cols[0]][col] += 1
        U[A_rows[0] * m + A_cols[0]][col] -= 1
        V[B_rows[0] * p + B_cols[0]][col] += 1
        V[B_rows[0] * p + B_cols[1]][col] += 1
        
        col = base_col + 6
        U[A_rows[0] * m + A_cols[1]][col] += 1
        U[A_rows[1] * m + A_cols[1]][col] -= 1
        V[B_rows[1] * p + B_cols[0]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] += 1
        
        w_idx_00 = C_cols[0] * n + C_rows[0]
        W[w_idx_00][base_col + 0] += 1
        W[w_idx_00][base_col + 3] += 1
        W[w_idx_00][base_col + 4] -= 1
        W[w_idx_00][base_col + 6] += 1
        
        w_idx_01 = C_cols[1] * n + C_rows[0]
        W[w_idx_01][base_col + 2] += 1
        W[w_idx_01][base_col + 4] += 1
        
        w_idx_10 = C_cols[0] * n + C_rows[1]
        W[w_idx_10][base_col + 1] += 1
        W[w_idx_10][base_col + 3] += 1
        
        w_idx_11 = C_cols[1] * n + C_rows[1]
        W[w_idx_11][base_col + 0] += 1
        W[w_idx_11][base_col + 1] -= 1
        W[w_idx_11][base_col + 2] += 1
        W[w_idx_11][base_col + 5] += 1
    
    add_strassen_block(0, [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    add_strassen_block(1, [0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    add_strassen_block(2, [0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    add_strassen_block(3, [0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(rank28, n, m, p, deadline):
    """Try to find a rank-27 decomposition by checking if one term is linearly dependent."""
    U28 = np.array(rank28["U"], dtype=float)
    V28 = np.array(rank28["V"], dtype=float)
    W28 = np.array(rank28["W"], dtype=float)
    
    rank = 28
    num_u = n * m  # 8
    num_v = m * p  # 16
    num_w = p * n  # 8
    
    # For each candidate rank term to remove, check if the tensor it contributes
    # can be expressed as a combination of the remaining terms.
    for remove_idx in range(rank):
        if time.time() > deadline - 10:
            return None
        
        U_rem = np.delete(U28, remove_idx, axis=1)
        V_rem = np.delete(V28, remove_idx, axis=1)
        W_rem = np.delete(W28, remove_idx, axis=1)
        
        # The removed term contributes: W28[:, remove_idx] * outer(U28[:, remove_idx], V28[:, remove_idx])
        # We need: for all k, W28[k, remove_idx] * U28[:, remove_idx] ⊗ V28[:, remove_idx]
        #          = sum_{r=0}^{26} c_{k,r} * U_rem[:,r] ⊗ V_rem[:,r]
        
        # Build the linear system for each k
        A = np.zeros(((num_u * num_v), rank - 1))
        for r in range(rank - 1):
            A[:, r] = np.outer(U_rem[:, r], V_rem[:, r]).flatten()
        
        # Check if A has full column rank (needed for unique solution)
        rank_A = np.linalg.matrix_rank(A)
        if rank_A < (rank - 1):
            # Underdetermined or singular - skip this candidate
            continue
        
        all_valid = True
        C_coeffs = np.zeros((num_w, rank - 1))
        
        for k in range(num_w):
            if time.time() > deadline - 5:
                return None
            
            lhs = W28[k, remove_idx] * np.outer(U28[:, remove_idx], V28[:, remove_idx]).flatten()
            
            # Solve A @ coeffs = lhs
            try:
                coeffs, residuals, rank_sol, sv = np.linalg.lstsq(A, lhs, rcond=1e-10)
                residual = np.linalg.norm(A @ coeffs - lhs)
                if residual > 1e-8:
                    all_valid = False
                    break
                C_coeffs[k, :] = coeffs
            except:
                all_valid = False
                break
        
        if all_valid:
            return {
                "U": U_rem.tolist(),
                "V": V_rem.tolist(),
                "W": C_coeffs.tolist()
            }
    
    return None
