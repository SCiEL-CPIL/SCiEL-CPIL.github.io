"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)
    
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
    
    def add_term(u_indices, v_indices, w_indices, coeffs):
        nonlocal col
        if col >= rank:
            return
        for u, c in zip(u_indices, coeffs):
            U[u][col] += c
        for v, c in zip(v_indices, coeffs):
            V[v][col] += c
        for w, c in zip(w_indices, coeffs):
            W[w][col] += c
        col += 1

    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        w00, w01, w10, w11 = w_offsets
        
        # M1
        add_term([u_idx(y00[0], y00[1]), u_idx(y11[0], y11[1])],
                 [v_idx(z00[0], z00[1]), v_idx(z11[0], z11[1])],
                 [w_idx(w00[0], w00[1]), w_idx(w11[0], w11[1])],
                 [1, 1, 1, 1])
                 
        # M2
        add_term([u_idx(y10[0], y10[1]), u_idx(y11[0], y11[1])],
                 [v_idx(z00[0], z00[1])],
                 [w_idx(w10[0], w10[1]), w_idx(w11[0], w11[1])],
                 [1, 1, 1, -1])
                 
        # M3
        add_term([u_idx(y00[0], y00[1])],
                 [v_idx(z01[0], z01[1]), v_idx(z11[0], z11[1])],
                 [w_idx(w01[0], w01[1]), w_idx(w11[0], w11[1])],
                 [1, 1, -1, 1])
                 
        # M4
        add_term([u_idx(y11[0], y11[1])],
                 [v_idx(z10[0], z10[1]), v_idx(z00[0], z00[1])],
                 [w_idx(w00[0], w00[1]), w_idx(w10[0], w10[1])],
                 [1, 1, -1, 1])
                 
        # M5
        add_term([u_idx(y00[0], y00[1]), u_idx(y01[0], y01[1])],
                 [v_idx(z11[0], z11[1])],
                 [w_idx(w00[0], w00[1]), w_idx(w01[0], w01[1])],
                 [1, 1, 1, -1, 1])
                 
        # M6
        add_term([u_idx(y10[0], y10[1]), u_idx(y00[0], y00[1])],
                 [v_idx(z00[0], z00[1]), v_idx(z01[0], z01[1])],
                 [w_idx(w11[0], w11[1])],
                 [1, -1, 1, 1, 1])
                 
        # M7
        add_term([u_idx(y01[0], y01[1]), u_idx(y11[0], y11[1])],
                 [v_idx(z10[0], z10[1]), v_idx(z11[0], z11[1])],
                 [w_idx(w00[0], w00[1])],
                 [1, -1, 1, 1, 1])

    # Block 1: A_L B_11 -> C_11
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y1, z1, w1)
    
    # Block 2: A_R B_21 -> C_11
    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y2, z2, w2)
    
    # Block 3: A_L B_12 -> C_12
    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    w3 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y3, z3, w3)
    
    # Block 4: A_R B_22 -> C_12
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y4, z4, w4)
    
    # Block 5: A_L B_15 -> C_15 (2x2 * 2x1)
    # M1: A00*B04 -> C04
    add_term([u_idx(0,0)], [v_idx(0,4)], [w_idx(4,0)], [1, 1, 1])
    # M2: A01*B14 -> C04
    add_term([u_idx(0,1)], [v_idx(1,4)], [w_idx(4,0)], [1, 1, 1])
    # M3: A10*B04 -> C14
    add_term([u_idx(1,0)], [v_idx(0,4)], [w_idx(4,1)], [1, 1, 1])
    # M4: A11*B14 -> C14
    add_term([u_idx(1,1)], [v_idx(1,4)], [w_idx(4,1)], [1, 1, 1])
    
    # Block 6: A_R B_25 -> C_15 (2x2 * 2x1)
    # M1: A02*B24 -> C04
    add_term([u_idx(0,2)], [v_idx(2,4)], [w_idx(4,0)], [1, 1, 1])
    # M2: A03*B34 -> C04
    add_term([u_idx(0,3)], [v_idx(3,4)], [w_idx(4,0)], [1, 1, 1])
    # M3: A12*B24 -> C14
    add_term([u_idx(1,2)], [v_idx(2,4)], [w_idx(4,1)], [1, 1, 1])
    # M4: A13*B34 -> C14
    add_term([u_idx(1,3)], [v_idx(3,4)], [w_idx(4,1)], [1, 1, 1])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
