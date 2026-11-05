"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 48 decomposition
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a Strassen block for 2x2x2
    def add_strassen_block(j_off, k_off):
        nonlocal col_idx
        # Local indices for A: i in {0,1}, j in {0,1}
        # Global U index: i * m + (j_off + j)
        # Local indices for B: j in {0,1}, k in {0,1}
        # Global V index: (j_off + j) * p + (k_off + k)
        # Local indices for C: i in {0,1}, k in {0,1}
        # Global W index: (k_off + k) * n + i
        
        # Define the 7 Strassen terms
        # Each term is (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
        # We use local relative indices for clarity, then map to global
        
        # M1: (a00 + a11)(b00 + b11) -> c00 + c11
        # U: a00 (i=0,j=0), a11 (i=1,j=1)
        # V: b00 (j=0,k=0), b11 (j=1,k=1)
        # W: c00 (k=0,i=0), c11 (k=1,i=1)
        
        terms = [
            # M1
            (
                [(0, 0), (1, 1)], [1, 1],
                [(0, 0), (1, 1)], [1, 1],
                [(0, 0), (1, 1)], [1, 1]
            ),
            # M2: (a10 + a11)b00 -> c10 - c11
            (
                [(1, 0), (1, 1)], [1, 1],
                [(0, 0)], [1],
                [(1, 0), (1, 1)], [1, -1]
            ),
            # M3: a00(b01 - b11) -> c01 + c11
            (
                [(0, 0)], [1],
                [(0, 1), (1, 1)], [1, -1],
                [(0, 1), (1, 1)], [1, 1]
            ),
            # M4: a11(b10 - b00) -> c00 + c10
            (
                [(1, 1)], [1],
                [(1, 0), (0, 0)], [1, -1],
                [(0, 0), (1, 0)], [1, 1]
            ),
            # M5: (a00 + a01)b11 -> -c00 + c01
            (
                [(0, 0), (0, 1)], [1, 1],
                [(1, 1)], [1],
                [(0, 0), (0, 1)], [-1, 1]
            ),
            # M6: (a10 - a00)(b00 + b01) -> c11
            (
                [(1, 0), (0, 0)], [1, -1],
                [(0, 0), (0, 1)], [1, 1],
                [(1, 1)], [1]
            ),
            # M7: (a01 - a11)(b10 + b11) -> c00
            (
                [(0, 1), (1, 1)], [1, -1],
                [(1, 0), (1, 1)], [1, 1],
                [(0, 0)], [1]
            ),
        ]
        
        for u_idx, u_val, v_idx, v_val, w_idx, w_val in terms:
            for (li, lj), val in zip(u_idx, u_val):
                global_i = li
                global_j = j_off + lj
                U[global_i * m + global_j][col_idx] += val
            
            for (lj, lk), val in zip(v_idx, v_val):
                global_j = j_off + lj
                global_k = k_off + lk
                V[global_j * p + global_k][col_idx] += val
            
            for (lk, li), val in zip(w_idx, w_val):
                global_k = k_off + lk
                global_i = li
                W[global_k * n + global_i][col_idx] += val
            
            col_idx += 1

    # Apply Strassen blocks
    # Part 1: j in {0, 1}
    add_strassen_block(0, 0)
    add_strassen_block(0, 2)
    add_strassen_block(0, 4)
    
    # Part 2: j in {2, 3}
    add_strassen_block(2, 0)
    add_strassen_block(2, 2)
    add_strassen_block(2, 4)
    
    # Part 3: j = 4
    # Rank 6 for 2x1x6
    # C_ik = A_i4 * B_4k
    for k in range(6):
        # U: A_i4 -> i in {0,1}, j=4
        U[0 * m + 4][col_idx] = 1
        U[1 * m + 4][col_idx] = 1
        
        # V: B_4k -> j=4, k
        V[4 * p + k][col_idx] = 1
        
        # W: C_ik -> k, i in {0,1}
        W[k * n + 0][col_idx] = 1
        W[k * n + 1][col_idx] = 1
        
        col_idx += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
