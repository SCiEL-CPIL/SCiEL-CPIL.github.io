"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U indices: 0,1,2,3 corresponding to A11, A12, A21, A22
    # V indices: 0,1,2,3 corresponding to B11, B12, B21, B22
    # W indices: 0,1,2,3 corresponding to C11, C12, C21, C22
    strassen_blocks = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1])
    ]
    
    def add_block(u_offset, v_offset, w_offset):
        # u_offset: list of 4 global U indices for A11, A12, A21, A22
        # v_offset: list of 4 global V indices for B11, B12, B21, B22
        # w_offset: list of 4 global W indices for C11, C12, C21, C22
        for i in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_blocks[i]
            for local_idx in range(4):
                if u_coeffs[local_idx] != 0:
                    U[u_offset[local_idx]][rank_offset] += u_coeffs[local_idx]
                if v_coeffs[local_idx] != 0:
                    V[v_offset[local_idx]][rank_offset] += v_coeffs[local_idx]
                if w_coeffs[local_idx] != 0:
                    W[w_offset[local_idx]][rank_offset] += w_coeffs[local_idx]
            rank_offset += 1

    # Block 1: A_L B_LL -> C_LL
    # A_L: rows 0,1; cols 0,1. U indices: i*4+j
    # A11: 0,0 -> 0; A12: 0,1 -> 1; A21: 1,0 -> 4; A22: 1,1 -> 5
    u1 = [0, 1, 4, 5]
    # B_LL: rows 0,1; cols 0,1. V indices: j*4+k
    # B11: 0,0 -> 0; B12: 0,1 -> 1; B21: 1,0 -> 4; B22: 1,1 -> 5
    v1 = [0, 1, 4, 5]
    # C_LL: rows 0,1; cols 0,1. W indices: k*2+i
    # C11: i=0,k=0 -> 0; C12: i=0,k=1 -> 2; C21: i=1,k=0 -> 1; C22: i=1,k=1 -> 3
    # Wait, W index is k*n+i.
    # C11 (i=0, k=0) -> 0*2+0 = 0
    # C12 (i=0, k=1) -> 1*2+0 = 2
    # C21 (i=1, k=0) -> 0*2+1 = 1
    # C22 (i=1, k=1) -> 1*2+1 = 3
    w1 = [0, 2, 1, 3]
    
    global rank_offset
    rank_offset = 0
    add_block(u1, v1, w1)
    
    # Block 2: A_R B_RL -> C_RL (adds to C_L)
    # A_R: rows 0,1; cols 2,3.
    # A11: 0,2 -> 2; A12: 0,3 -> 3; A21: 1,2 -> 6; A22: 1,3 -> 7
    u2 = [2, 3, 6, 7]
    # B_RL: rows 2,3; cols 0,1.
    # B11: 2,0 -> 8; B12: 2,1 -> 9; B21: 3,0 -> 12; B22: 3,1 -> 13
    v2 = [8, 9, 12, 13]
    # C_RL: same output as C_LL
    w2 = [0, 2, 1, 3]
    add_block(u2, v2, w2)
    
    # Block 3: A_L B_LR -> C_LR (part of C_R)
    # A_L: same as u1
    u3 = u1
    # B_LR: rows 0,1; cols 2,3.
    # B11: 0,2 -> 2; B12: 0,3 -> 3; B21: 1,2 -> 6; B22: 1,3 -> 7
    v3 = [2, 3, 6, 7]
    # C_LR: rows 0,1; cols 2,3.
    # C11 (i=0, k=2) -> 2*2+0 = 4
    # C12 (i=0, k=3) -> 3*2+0 = 6
    # C21 (i=1, k=2) -> 2*2+1 = 5
    # C22 (i=1, k=3) -> 3*2+1 = 7
    w3 = [4, 6, 5, 7]
    add_block(u3, v3, w3)
    
    # Block 4: A_R B_RR -> C_RR (part of C_R)
    # A_R: same as u2
    u4 = u2
    # B_RR: rows 2,3; cols 2,3.
    # B11: 2,2 -> 10; B12: 2,3 -> 11; B21: 3,2 -> 14; B22: 3,3 -> 15
    v4 = [10, 11, 14, 15]
    # C_RR: same output as C_LR
    w4 = [4, 6, 5, 7]
    add_block(u4, v4, w4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
