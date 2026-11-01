import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_standard(i_o, j_o, k_o, n_l, m_l, p_l):
        nonlocal col
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    U[(i_o + i) * m + (j_o + j)][col] = 1
                    V[(j_o + j) * p + (k_o + k)][col] = 1
                    W[(k_o + k) * n + (i_o + i)][col] = 1
                    col += 1

    def add_strassen(i_o, j_o, k_o):
        nonlocal col
        # Local indices 0..1 for i, j, k
        # Global indices:
        # U: (i_o+i)*5 + (j_o+j)
        # V: (j_o+j)*5 + (k_o+k)
        # W: (k_o+k)*2 + (i_o+i)
        
        # M1
        # u: 00, 11
        # v: 00, 11
        # w: 00, 11
        U[i_o * m + j_o][col] = 1
        U[(i_o + 1) * m + (j_o + 1)][col] = 1
        V[j_o * p + k_o][col] = 1
        V[(j_o + 1) * p + (k_o + 1)][col] = 1
        W[k_o * n + i_o][col] = 1
        W[(k_o + 1) * n + (i_o + 1)][col] = 1
        col += 1
        
        # M2
        # u: 10, 11
        # v: 00
        # w: 10 (1), 11 (-1)
        U[(i_o + 1) * m + j_o][col] = 1
        U[(i_o + 1) * m + (j_o + 1)][col] = 1
        V[j_o * p + k_o][col] = 1
        W[(k_o + 1) * n + i_o][col] = 1
        W[(k_o + 1) * n + (i_o + 1)][col] = -1
        col += 1
        
        # M3
        # u: 00
        # v: 01 (-1 at 11? No, 01 - 11) -> 01: 1, 11: -1
        # w: 01 (1), 11 (1)
        U[i_o * m + j_o][col] = 1
        V[j_o * p + (k_o + 1)][col] = 1
        V[(j_o + 1) * p + (k_o + 1)][col] = -1
        W[(k_o + 1) * n + i_o][col] = 1
        W[(k_o + 1) * n + (i_o + 1)][col] = 1
        col += 1
        
        # M4
        # u: 11
        # v: 10 (-1 at 00? No, 10 - 00) -> 10: 1, 00: -1
        # w: 00 (1), 10 (1)
        U[(i_o + 1) * m + (j_o + 1)][col] = 1
        V[(j_o + 1) * p + k_o][col] = 1
        V[j_o * p + k_o][col] = -1
        W[k_o * n + i_o][col] = 1
        W[(k_o + 1) * n + i_o][col] = 1
        col += 1
        
        # M5
        # u: 00, 10
        # v: 11
        # w: 00 (-1), 01 (1)
        U[i_o * m + j_o][col] = 1
        U[(i_o + 1) * m + j_o][col] = 1
        V[(j_o + 1) * p + (k_o + 1)][col] = 1
        W[k_o * n + i_o][col] = -1
        W[(k_o + 1) * n + i_o][col] = 1
        col += 1
        
        # M6
        # u: 11 (-1 at 00? No, 11 - 00) -> 11: 1, 00: -1
        # v: 00, 01
        # w: 11 (1)
        U[(i_o + 1) * m + (j_o + 1)][col] = 1
        U[i_o * m + j_o][col] = -1
        V[j_o * p + k_o][col] = 1
        V[j_o * p + (k_o + 1)][col] = 1
        W[(k_o + 1) * n + (i_o + 1)][col] = 1
        col += 1
        
        # M7
        # u: 01 (-1 at 11? No, 01 - 11) -> 01: 1, 11: -1
        # v: 10, 11
        # w: 00 (1)
        U[i_o * m + (j_o + 1)][col] = 1
        U[(i_o + 1) * m + (j_o + 1)][col] = -1
        V[(j_o + 1) * p + k_o][col] = 1
        V[(j_o + 1) * p + (k_o + 1)][col] = 1
        W[k_o * n + i_o][col] = 1
        col += 1

    # Part 1: C[0:2, 0:2]
    # 1. A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen(0, 0, 0)
    
    # 2. A[0:2, 2:3] * B[2:3, 0:2]
    add_standard(0, 2, 0, 2, 1, 2)
    
    # 3. A[0:2, 3:5] * B[3:5, 0:2]
    add_strassen(0, 3, 0)
    
    # Part 2: C[0:2, 2:5]
    # 4. A[0:2, 0:2] * B[0:2, 2:3]
    add_standard(0, 0, 2, 2, 2, 1)
    
    # 5. A[0:2, 0:2] * B[0:2, 3:5]
    add_strassen(0, 0, 3)
    
    # 6. A[0:2, 2:3] * B[2:3, 2:5]
    add_standard(0, 2, 2, 2, 1, 3)
    
    # 7. A[0:2, 3:5] * B[3:5, 2:3]
    add_standard(0, 3, 2, 2, 2, 1)
    
    # 8. A[0:2, 3:5] * B[3:5, 3:5]
    add_strassen(0, 3, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
