"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)
    
    # We use a rank 30 decomposition:
    # - 4 blocks of 2x2x2 Strassen (7 each) = 28
    # - 2 blocks of 2x2x1 (rank 1 each) = 2
    # Total: 30
    
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1: (y00+y11)(z00+z11) -> X00, X11
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M2: (y10+y11)z00 -> X10, -X11
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
        col += 1
        
        # M3: y00(z01-z11) -> X01, X11
        U[u_idx(y00[0], y00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M4: y11(z10-z00) -> X00, X10
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        col += 1
        
        # M5: (y00+y01)z11 -> -X00, X01
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y01[0], y01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        col += 1
        
        # M6: (y10-y00)(z00+z01) -> X11
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y00[0], y00[1])][col] = -1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M7: (y01-y11)(z10+z11) -> X00
        U[u_idx(y01[0], y01[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = -1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        col += 1

    # Block 1: A_L (cols 0,1) * B_11 (rows 0,1; cols 0,1)
    # Result: C rows 0,1; cols 0,1 -> W indices (k=0,i=0), (k=1,i=0), (k=0,i=1), (k=1,i=1)
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y1, z1, w1)
    
    # Block 2: A_R (cols 2,3) * B_21 (rows 2,3; cols 0,1)
    # Result: C rows 0,1; cols 0,1
    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y2, z2, w2)
    
    # Block 3: A_L (cols 0,1) * B_12 (rows 0,1; cols 2,3)
    # Result: C rows 0,1; cols 2,3 -> W indices (k=2,i=0), (k=3,i=0), (k=2,i=1), (k=3,i=1)
    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    w3 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y3, z3, w3)
    
    # Block 4: A_R (cols 2,3) * B_22 (rows 2,3; cols 2,3)
    # Result: C rows 0,1; cols 2,3
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y4, z4, w4)
    
    # Block 5: A_L (cols 0,1) * B_15 (rows 0,1; col 4)
    # 2x2x1: rank 4
    # C_0 = A_00 B_0 + A_01 B_1
    # C_1 = A_10 B_0 + A_11 B_1
    
    # Term 1: A_00 * B_0 -> C_0
    U[u_idx(0,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,0)][col] = 1
    col += 1
    
    # Term 2: A_01 * B_1 -> C_0
    U[u_idx(0,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,0)][col] = 1
    col += 1
    
    # Term 3: A_10 * B_0 -> C_1
    U[u_idx(1,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
    # Term 4: A_11 * B_1 -> C_1
    U[u_idx(1,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
    # Block 6: A_R (cols 2,3) * B_25 (rows 2,3; col 4)
    # 2x2x1: rank 4
    
    # Term 1: A_02 * B_24 -> C_0
    U[u_idx(0,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,0)][col] = 1
    col += 1
    
    # Term 2: A_03 * B_34 -> C_0
    U[u_idx(0,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,0)][col] = 1
    col += 1
    
    # Term 3: A_12 * B_24 -> C_1
    U[u_idx(1,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
    # Term 4: A_13 * B_34 -> C_1
    U[u_idx(1,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
