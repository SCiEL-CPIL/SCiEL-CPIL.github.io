"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 28 decomposition using Strassen's algorithm on 2x2 blocks
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen 2x2x2 block
    # u_idx: list of 4 U row indices [u00, u01, u10, u11]
    # v_idx: list of 4 V row indices [v00, v01, v10, v11]
    # w_idx: list of 4 W row indices [w00, w01, w10, w11]
    # col_offset: starting column in U, V, W
    def add_strassen_block(u_idx, v_idx, w_idx, col_offset):
        # M1 = (A00 + A11)(B00 + B11)
        U[u_idx[0]][col_offset + 0] += 1
        U[u_idx[3]][col_offset + 0] += 1
        V[v_idx[0]][col_offset + 0] += 1
        V[v_idx[3]][col_offset + 0] += 1
        W[w_idx[0]][col_offset + 0] += 1
        W[w_idx[3]][col_offset + 0] += 1
        
        # M2 = (A10 + A11)B00
        U[u_idx[2]][col_offset + 1] += 1
        U[u_idx[3]][col_offset + 1] += 1
        V[v_idx[0]][col_offset + 1] += 1
        W[w_idx[2]][col_offset + 1] += 1
        W[w_idx[3]][col_offset + 1] -= 1
        
        # M3 = A00(B01 - B11)
        U[u_idx[0]][col_offset + 2] += 1
        V[v_idx[1]][col_offset + 2] += 1
        V[v_idx[3]][col_offset + 2] -= 1
        W[w_idx[1]][col_offset + 2] += 1
        W[w_idx[3]][col_offset + 2] += 1
        
        # M4 = A11(B10 - B00)
        U[u_idx[3]][col_offset + 3] += 1
        V[v_idx[2]][col_offset + 3] += 1
        V[v_idx[0]][col_offset + 3] -= 1
        W[w_idx[0]][col_offset + 3] += 1
        W[w_idx[2]][col_offset + 3] += 1
        
        # M5 = (A00 + A01)B11
        U[u_idx[0]][col_offset + 4] += 1
        U[u_idx[1]][col_offset + 4] += 1
        V[v_idx[3]][col_offset + 4] += 1
        W[w_idx[0]][col_offset + 4] -= 1
        W[w_idx[1]][col_offset + 4] += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[u_idx[2]][col_offset + 5] += 1
        U[u_idx[0]][col_offset + 5] -= 1
        V[v_idx[0]][col_offset + 5] += 1
        V[v_idx[1]][col_offset + 5] += 1
        W[w_idx[3]][col_offset + 5] += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[u_idx[1]][col_offset + 6] += 1
        U[u_idx[3]][col_offset + 6] -= 1
        V[v_idx[2]][col_offset + 6] += 1
        V[v_idx[3]][col_offset + 6] += 1
        W[w_idx[0]][col_offset + 6] += 1

    # Block 1: A(0,1) x B(0,1) -> C(0,1) (Top-Left)
    # A rows 0,1; Cols 0,1 -> U indices 0,1,4,5
    # B rows 0,1; Cols 0,1 -> V indices 0,1,4,5
    # C rows 0,1; Cols 0,1 -> W indices 0,2,1,3 (w00, w01, w10, w11)
    # W mapping: k*n+i. 
    # C00 (k=0,i=0) -> 0. C01 (k=1,i=0) -> 2. C10 (k=0,i=1) -> 1. C11 (k=1,i=1) -> 3.
    u1 = [0, 1, 4, 5]
    v1 = [0, 1, 4, 5]
    w1 = [0, 2, 1, 3]
    add_strassen_block(u1, v1, w1, 0)

    # Block 2: A(2,3) x B(2,3) -> C(0,1) (Bottom-Left part of sum)
    # A rows 0,1; Cols 2,3 -> U indices 2,3,6,7
    # B rows 2,3; Cols 0,1 -> V indices 8,9,12,13
    # C same as Block 1
    u2 = [2, 3, 6, 7]
    v2 = [8, 9, 12, 13]
    w2 = [0, 2, 1, 3]
    add_strassen_block(u2, v2, w2, 7)

    # Block 3: A(0,1) x B(0,1) -> C(2,3) (Top-Right)
    # A same as Block 1
    # B rows 0,1; Cols 2,3 -> V indices 2,3,6,7
    # C rows 0,1; Cols 2,3 -> W indices 4,6,5,7
    # C02 (k=2,i=0) -> 4. C03 (k=3,i=0) -> 6. C12 (k=2,i=1) -> 5. C13 (k=3,i=1) -> 7.
    u3 = [0, 1, 4, 5]
    v3 = [2, 3, 6, 7]
    w3 = [4, 6, 5, 7]
    add_strassen_block(u3, v3, w3, 14)

    # Block 4: A(2,3) x B(2,3) -> C(2,3) (Bottom-Right)
    # A same as Block 2
    # B rows 2,3; Cols 2,3 -> V indices 10,11,14,15
    # C same as Block 3
    u4 = [2, 3, 6, 7]
    v4 = [10, 11, 14, 15]
    w4 = [4, 6, 5, 7]
    add_strassen_block(u4, v4, w4, 21)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
