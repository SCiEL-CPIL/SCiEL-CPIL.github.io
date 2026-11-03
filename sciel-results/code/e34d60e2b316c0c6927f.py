import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    # U shape: (n*m, rank) = (6, 44)
    # V shape: (m*p, rank) = (24, 44)
    # W shape: (p*n, rank) = (16, 44)
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get indices
    # A indices: i in 0..1, j in 0..2 -> u = i*3 + j
    # B indices: j in 0..2, k in 0..7 -> v = j*8 + k
    # C indices: k in 0..7, i in 0..1 -> w = k*2 + i
    
    term = 0
    for j_block in range(4):
        # Columns 2*j_block and 2*j_block+1
        k0 = 2 * j_block
        k1 = 2 * j_block + 1
        
        # W indices for the 2x2 block
        w_00 = k0 * 2 + 0
        w_01 = k1 * 2 + 0
        w_10 = k0 * 2 + 1
        w_11 = k1 * 2 + 1
        
        # V indices for B_1 (rows 0,1) and B_2 (row 2)
        v_00 = 0 * 8 + k0
        v_01 = 0 * 8 + k1
        v_10 = 1 * 8 + k0
        v_11 = 1 * 8 + k1
        v_20 = 2 * 8 + k0
        v_21 = 2 * 8 + k1
        
        # U indices for A_1 (cols 0,1) and A_2 (col 2)
        u_00 = 0 * 3 + 0
        u_01 = 0 * 3 + 1
        u_10 = 1 * 3 + 0
        u_11 = 1 * 3 + 1
        u_02 = 0 * 3 + 2
        u_12 = 1 * 3 + 2
        
        # 1. P1 = (A00 + A11)(B0k0 + B1k1)
        U[u_00][term] = 1
        U[u_11][term] = 1
        V[v_00][term] = 1
        V[v_11][term] = 1
        W[w_00][term] = 1
        W[w_01][term] = 1
        W[w_10][term] = -1
        W[w_11][term] = 1
        term += 1
        
        # 2. P2 = (A10 + A11)B0k0
        U[u_10][term] = 1
        U[u_11][term] = 1
        V[v_00][term] = 1
        W[w_01][term] = 1
        term += 1
        
        # 3. P3 = A00(B0k1 - B1k1)
        U[u_00][term] = 1
        V[v_01][term] = 1
        V[v_11][term] = -1
        W[w_10][term] = 1
        W[w_11][term] = -1
        term += 1
        
        # 4. P4 = A11(B1k0 - B0k0)
        U[u_11][term] = 1
        V[v_10][term] = 1
        V[v_00][term] = -1
        W[w_00][term] = -1
        W[w_01][term] = 1
        term += 1
        
        # 5. P5 = (A00 + A01)B1k1
        U[u_00][term] = 1
        U[u_01][term] = 1
        V[v_11][term] = 1
        W[w_10][term] = 1
        W[w_11][term] = -1
        term += 1
        
        # 6. P6 = (A10 - A00)(B0k0 + B0k1)
        U[u_10][term] = 1
        U[u_00][term] = -1
        V[v_00][term] = 1
        V[v_01][term] = 1
        W[w_00][term] = 1
        term += 1
        
        # 7. P7 = (A01 - A11)(B1k0 + B1k1)
        U[u_01][term] = 1
        U[u_11][term] = -1
        V[v_10][term] = 1
        V[v_11][term] = 1
        W[w_00][term] = 1
        W[w_11][term] = 1
        term += 1
        
        # 8. Q1 = A02 B2k0
        U[u_02][term] = 1
        V[v_20][term] = 1
        W[w_00][term] = 1
        term += 1
        
        # 9. Q2 = A02 B2k1
        U[u_02][term] = 1
        V[v_21][term] = 1
        W[w_01][term] = 1
        term += 1
        
        # 10. Q3 = A12 B2k0
        U[u_12][term] = 1
        V[v_20][term] = 1
        W[w_10][term] = 1
        term += 1
        
        # 11. Q4 = A12 B2k1
        U[u_12][term] = 1
        V[v_21][term] = 1
        W[w_11][term] = 1
        term += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
