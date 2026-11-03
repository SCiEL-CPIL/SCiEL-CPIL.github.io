import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS
    
    # Target rank estimate: 85
    # We will build up the rank dynamically or pre-allocate max possible.
    # Max schoolbook is 105.
    rank = 105
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        
        u00 = i0 * m + j0
        u01 = i0 * m + j1
        u10 = i1 * m + j0
        u11 = i1 * m + j1
        
        v00 = j0 * p + k0
        v01 = j0 * p + k1
        v10 = j1 * p + k0
        v11 = j1 * p + k1
        
        w00 = k0 * n + i0
        w01 = k1 * n + i0
        w10 = k0 * n + i1
        w11 = k1 * n + i1
        
        # M1
        U[u00][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        W[w11][col] = 1
        col += 1
        # M2
        U[u10][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        W[w10][col] = 1
        W[w11][col] = -1
        col += 1
        # M3
        U[u00][col] = 1
        V[v01][col] = 1
        V[v11][col] = -1
        W[w01][col] = 1
        W[w11][col] = 1
        col += 1
        # M4
        U[u11][col] = 1
        V[v10][col] = 1
        V[v00][col] = -1
        W[w00][col] = 1
        W[w10][col] = 1
        col += 1
        # M5
        U[u00][col] = 1
        U[u01][col] = 1
        V[v11][col] = 1
        W[w00][col] = -1
        W[w01][col] = 1
        col += 1
        # M6
        U[u10][col] = 1
        U[u00][col] = -1
        V[v00][col] = 1
        V[v01][col] = 1
        W[w11][col] = 1
        col += 1
        # M7
        U[u01][col] = 1
        U[u11][col] = -1
        V[v10][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        col += 1

    def add_3x2x2_block(i0, j0, k0):
        # i in {i0, i0+1, i0+2}, j in {j0, j0+1}, k in {k0, k0+1}
        # Use Strassen for i in {i0, i0+1}
        add_strassen_block(i0, j0, k0)
        # Use Schoolbook for i in {i0+2}
        add_schoolbook_block(i0+2, i0+3, j0, j0+2, k0, k0+2)

    def add_3x3x2_block(i0, j0, k0):
        # i in {i0..i0+2}, j in {j0..j0+2}, k in {k0, k0+1}
        # Split j into {j0, j0+1} and {j0+2}
        # 3x2x2 for j in {j0, j0+1}
        add_3x2x2_block(i0, j0, k0)
        # 3x1x2 for j in {j0+2}
        # Rank 1: U[i, j0+2] = 1, V[j0+2, k] = 1, W[k, i] = 1
        for i in range(i0, i0+3):
            U[i * m + (j0+2)][col] = 1
            for k in range(k0, k0+2):
                V[(j0+2) * p + k][col] = 1
                W[k * n + i][col] = 1
            col += 1 # Wait, this logic is wrong. 
            # The rank-1 term is a single tensor.
            # T[i, j, k] = U[i,j] V[j,k] W[k,i]
            # If U[i, j0+2] = 1, V[j0+2, k] = 1, W[k, i] = 1, then sum is 1.
            # This is ONE rank-1 term for the whole 3x1x2 block.
        
        # Correction:
        # The above loop added 3 terms.
        # We need to add 1 term.
        # U[i * m + (j0+2)][col] = 1 for all i? No, U is a matrix.
        # U[i, j] is a scalar.
        # So we set U[i * m + (j0+2)][col] = 1 for i in 0..2.
        # V[(j0+2) * p + k][col] = 1 for k in 0..1.
        # W[k * n + i][col] = 1 for all k, i.
        
        # Let's re-implement add_3x1x2 properly.
        
    # Re-doing add_3x3x2_block with correct rank-1 addition
    def add_3x1x2_block(i0, j0, k0):
        nonlocal col
        for i in range(i0, i0+3):
            U[i * m + j0][col] = 1
        for k in range(k0, k0+2):
            V[j0 * p + k][col] = 1
            for i in range(i0, i0+3):
                W[k * n + i][col] = 1
        col += 1

    def add_3x3x2_block_correct(i0, j0, k0):
        # i in {i0..i0+2}, j in {j0..j0+2}, k in {k0, k0+1}
        add_3x2x2_block(i0, j0, k0)
        add_3x1x2_block(i0, j0+2, k0)

    def add_3x2x3_block(i0, j0, k0):
        # i in {i0..i0+2}, j in {j0, j0+1}, k in {k0..k0+2}
        # Split k into {k0, k0+1} and {k0+2}
        # 3x2x2 for k in {k0, k0+1}
        add_3x2x2_block(i0, j0, k0)
        # 3x2x1 for k in {k0+2}
        # Rank 1
        for i in range(i0, i0+3):
            for j in range(j0, j0+2):
                U[i * m + j][col] = 1
                V[j * p + (k0+2)][col] = 1
                W[(k0+2) * n + i][col] = 1
        col += 1

    # Execute blocks
    # Block 1: 3x3x3 at (0,0,0)
    if time.time() < deadline:
        add_schoolbook_block(0, 3, 0, 3, 0, 3)
        
    # Block 2: 3x3x2 at (0,0,3)
    if time.time() < deadline:
        add_3x3x2_block_correct(0, 0, 3)
        
    # Block 3: 3x3x2 at (0,0,5)
    if time.time() < deadline:
        add_3x3x2_block_correct(0, 0, 5)
        
    # Block 4: 3x2x3 at (0,3,0)
    if time.time() < deadline:
        add_3x2x3_block(0, 3, 0)
        
    # Block 5: 3x2x2 at (0,3,3)
    if time.time() < deadline:
        add_3x2x2_block(0, 3, 3)
        
    # Block 6: 3x2x2 at (0,3,5)
    if time.time() < deadline:
        add_3x2x2_block(0, 3, 5)
        
    # Truncate to actual rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
