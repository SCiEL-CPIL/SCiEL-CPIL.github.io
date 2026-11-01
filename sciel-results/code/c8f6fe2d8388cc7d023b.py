import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    size_a = n * m
    size_b = m * p
    size_c = p * n
    
    # Build the true tensor T[a, b, c]
    T_true = np.zeros((size_a, size_b, size_c), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1
    
    # Try Strassen-based decomposition
    # Rank = 7 * (p // 2) = 49
    rank = 7 * (p // 2)
    
    U = np.zeros((size_a, rank), dtype=int)
    V = np.zeros((size_b, rank), dtype=int)
    W = np.zeros((size_c, rank), dtype=int)
    
    # Strassen factors for 2x2x2
    # Indices for A (2x2): 0,1,2,3
    # Indices for B (2x2): 0,1,2,3 (local)
    # Indices for C (2x2): 0,1,2,3 (local)
    
    # Verified Strassen factors (U, V, W)
    # U[i, r], V[j, r], W[k, r]
    U_strassen = [
        [1, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0, 0, 0]
    ]
    V_strassen = [
        [1, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, -1, 0, 1],
        [1, 0, 0, 1, 0, -1, -1]
    ]
    W_strassen = [
        [1, -1, 1, 0, 0, 0, 0],
        [0, 1, 0, -1, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1]
    ]
    
    # Verify Strassen on 2x2x2
    U_s = np.array(U_strassen)
    V_s = np.array(V_strassen)
    W_s = np.array(W_strassen)
    T_s_recon = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
    T_s_true = np.zeros((4, 4, 4), dtype=int)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T_s_true[i*2+j, j*2+k, k*2+i] = 1
    if not np.array_equal(T_s_recon, T_s_true):
        # If Strassen factors are wrong, fall back to schoolbook
        rank = n * m * p
        U = np.zeros((size_a, rank), dtype=int)
        V = np.zeros((size_b, rank), dtype=int)
        W = np.zeros((size_c, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Construct full decomposition
    for b in range(p // 2):
        # Block b corresponds to columns 2b and 2b+1 of B and C
        # Local B indices:
        # B_{0, 2b} -> global 0*p + 2b = 2b
        # B_{0, 2b+1} -> global 0*p + 2b+1 = 2b+1
        # B_{1, 2b} -> global 1*p + 2b = p + 2b
        # B_{1, 2b+1} -> global 1*p + 2b+1 = p + 2b + 1
        
        # Local C indices:
        # C_{0, 2b} -> global 2b*n + 0 = 4b
        # C_{0, 2b+1} -> global (2b+1)*n + 0 = 4b+2
        # C_{1, 2b} -> global 2b*n + 1 = 4b+1
        # C_{1, 2b+1} -> global (2b+1)*n + 1 = 4b+3
        
        # Map local indices to global indices
        # A indices are always 0, 1, 2, 3
        # Local B indices 0,1,2,3 map to:
        b_idx_map = [2*b, 2*b+1, p+2*b, p+2*b+1]
        # Local C indices 0,1,2,3 map to:
        c_idx_map = [4*b, 4*b+2, 4*b+1, 4*b+3]
        
        for r in range(7):
            global_col = b * 7 + r
            for i_local in range(4):
                U[i_local][global_col] = U_strassen[i_local][r]
            for j_local in range(4):
                V[b_idx_map[j_local]][global_col] = V_strassen[j_local][r]
            for k_local in range(4):
                W[c_idx_map[k_local]][global_col] = W_strassen[k_local][r]
    
    # Verify full decomposition
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
    
    if not np.array_equal(T_recon, T_true):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((size_a, rank), dtype=int)
        V = np.zeros((size_b, rank), dtype=int)
        W = np.zeros((size_c, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
