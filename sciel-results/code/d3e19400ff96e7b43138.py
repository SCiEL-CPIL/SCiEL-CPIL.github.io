"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build the target tensor T of shape (n*m, m*p, p*n) = (8, 24, 12)
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def verify_decomp(U, V, W):
        """Verify that the decomposition exactly reconstructs T."""
        R = U.shape[1]
        prod = np.zeros_like(T)
        for r in range(R):
            prod += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return np.max(np.abs(prod - T))
    
    # Strassen's 7 products for 2×2 matrices
    # A_forms[s]: list of (row, col, coeff) for the s-th Strassen product's A part
    # B_forms[s]: list of (row, col, coeff) for the s-th Strassen product's B part
    # C_coeffs[s]: list of (row, col, coeff) for how the s-th product contributes to C
    
    A_forms = [
        [(0,0,1), (1,1,1)],       # M1: A00 + A11
        [(1,0,1), (1,1,1)],       # M2: A10 + A11
        [(0,0,1)],                 # M3: A00
        [(1,1,1)],                 # M4: A11
        [(0,0,1), (0,1,1)],       # M5: A00 + A01
        [(1,0,1), (0,0,-1)],      # M6: A10 - A00
        [(0,1,1), (1,1,-1)],      # M7: A01 - A11
    ]
    
    B_forms = [
        [(0,0,1), (1,1,1)],       # M1: B00 + B11
        [(0,0,1)],                 # M2: B00
        [(0,1,1), (1,1,-1)],      # M3: B01 - B11
        [(1,0,1), (0,0,-1)],      # M4: B10 - B00
        [(1,1,1)],                 # M5: B11
        [(0,0,1), (0,1,1)],       # M6: B00 + B01
        [(1,0,1), (1,1,1)],       # M7: B10 + B11
    ]
    
    C_coeffs = [
        [(0,0,1), (1,1,1)],       # M1: C00 += 1, C11 += 1
        [(1,0,1), (1,1,-1)],      # M2: C10 += 1, C11 -= 1
        [(0,1,1), (1,1,1)],       # M3: C01 += 1, C11 += 1
        [(0,0,1), (1,0,1)],       # M4: C00 += 1, C10 += 1
        [(0,0,-1), (0,1,1)],      # M5: C00 -= 1, C01 += 1
        [(1,1,1)],                 # M6: C11 += 1
        [(0,0,1)],                 # M7: C00 += 1
    ]
    
    # Construct rank-42 block-Strassen decomposition
    rank_42 = 42
    U = np.zeros((n * m, rank_42), dtype=np.float64)
    V = np.zeros((m * p, rank_42), dtype=np.float64)
    W = np.zeros((p * n, rank_42), dtype=np.float64)
    
    col = 0
    for a_block in range(2):  # A's column blocks: 0 (cols 0-1), 1 (cols 2-3)
        for b_block in range(6):  # B's 2x2 blocks
            # b_block 0: rows 0-1, cols 0-1
            # b_block 1: rows 0-1, cols 2-3
            # b_block 2: rows 0-1, cols 4-5
            # b_block 3: rows 2-3, cols 0-1
            # b_block 4: rows 2-3, cols 2-3
            # b_block 5: rows 2-3, cols 4-5
            b_row_offset = (b_block // 3) * 2
            b_col_offset = (b_block % 3) * 2
            
            for s in range(7):
                # A part
                for (i_loc, j_loc, coeff) in A_forms[s]:
                    i = i_loc
                    j = a_block * 2 + j_loc
                    U[i * m + j, col] += coeff
                
                # B part
                for (j_loc, k_loc, coeff) in B_forms[s]:
                    j = b_row_offset + j_loc
                    k = b_col_offset + k_loc
                    V[j * p + k, col] += coeff
                
                # C part: C[i, k] where i is from A's row, k is the output column
                # Output column k = b_col_offset + k_loc
                for (i_loc, k_loc, coeff) in C_coeffs[s]:
                    i = i_loc
                    k = b_col_offset + k_loc
                    c_idx = k * n + i
                    W[c_idx, col] += coeff
                
                col += 1
    
    err_42 = verify_decomp(U, V, W)
    print(f"Rank-42 error: {err_42}")
    
    best_rank = rank_42
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower rank via optimization
    np.random.seed(42)
    
    def try_find_decomposition(R, max_attempts=3, max_steps=150):
        """Try to find a rank-R decomposition."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            
            scale = 0.5
            U_try = np.random.randn(n * m, R) * scale
            V_try = np.random.randn(m * p, R) * scale
            W_try = np.random.randn(p * n, R) * scale
            
            lr = 0.01
            best_obj = float('inf')
            stall_count = 0
            
            for step in range(max_steps):
                if time.time() > deadline - 8:
                    return None
                
                prod = np.zeros_like(T)
                for r in range(R):
                    prod += np.einsum('i,j,k->ijk', U_try[:, r], V_try[:, r], W_try[:, r])
                
                diff = prod - T
                obj = np.dot(diff.flatten(), diff.flatten())
                
                if obj < 1e-12:
                    return U_try.copy(), V_try.copy(), W_try.copy()
                
                if obj < best_obj:
                    best_obj = obj
                    stall_count = 0
                else:
                    stall_count += 1
                
                if stall_count > 30:
                    lr *= 0.5
                    if lr < 1e-8:
                        break
                
                grad_U = np.zeros_like(U_try)
                grad_V = np.zeros_like(V_try)
                grad_W = np.zeros_like(W_try)
                
                for r in range(R):
                    grad_U[:, r] = np.einsum('ijk,j,k->i', diff, V_try[:, r], W_try[:, r])
                    grad_V[:, r] = np.einsum('ijk,i,k->j', diff, U_try[:, r], W_try[:, r])
                    grad_W[:, r] = np.einsum('ijk,i,j->k', diff, U_try[:, r], V_try[:, r])
                
                U_try -= lr * grad_U
                V_try -= lr * grad_V
                W_try -= lr * grad_W
            
            prod = np.zeros_like(T)
            for r in range(R):
                prod += np.einsum('i,j,k->ijk', U_try[:, r], V_try[:, r], W_try[:, r])
            if np.max(np.abs(prod - T)) < 1e-8:
                return U_try.copy(), V_try.copy(), W_try.copy()
        
        return None
    
    for R in range(12, 42):
        if time.time() > deadline - 15:
            break
        
        result = try_find_decomposition(R)
        if result is not None:
            U_found, V_found, W_found = result
            err = verify_decomp(U_found, V_found, W_found)
            if err < 1e-8:
                best_rank = R
                best_U, best_V, best_W = U_found, V_found, W_found
                break
    
    final_err = verify_decomp(best_U, best_V, best_W)
    if final_err > 1e-6:
        best_rank = rank_42
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

