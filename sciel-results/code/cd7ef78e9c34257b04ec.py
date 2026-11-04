"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    # Rank 42 decomposition
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Each entry is (u_coef, v_coef, w_coef) for the rank-1 term
    # We need to map local indices to global indices.
    
    # Strassen terms:
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)B00 -> C10, C11(-1)
    # M3: A00(B01-B11) -> C01, C11
    # M4: A11(B10-B00) -> C00, C10
    # M5: (A00+A01)B11 -> C00(-1), C01
    # M6: (A10-A00)(B00+B01) -> C11
    # M7: (A01-A11)(B10+B11) -> C00
    
    # Define the 7 rank-1 components of Strassen
    # Format: (a_idx, a_val, b_idx, b_val, c_idx, c_val)
    # a_idx: (i, j) local 2x2
    # b_idx: (j, k) local 2x2
    # c_idx: (k, i) local 2x2
    
    strassen_components = [
        # M1
        [((0,0), 1), ((1,1), 1)],
        [((0,0), 1), ((1,1), 1)],
        [((0,0), 1), ((1,1), 1)],
        [((0,0), 1), ((1,1), 1)],
        # M2
        [((1,0), 1), ((1,1), 1)],
        [((0,0), 1)],
        [((0,1), 1), ((1,1), -1)],
        # M3
        [((0,0), 1)],
        [((0,1), 1), ((1,1), -1)],
        [((1,1), 1), ((1,0), 1)], # Wait, C indices for M3 are C01 and C11?
        # Let's re-verify standard Strassen mapping.
    ]
    
    # Let's stick to the hardcoded logic but refactor it into a helper function
    # to change the code structure.
    
    def add_strassen_block(i_start, j_start, k_start, offset):
        # Local indices a,b in {0,1}
        # Global A index: (i_start+a)*m + (j_start+b)
        # Global B index: (j_start+a)*p + (k_start+b)
        # Global C index: (k_start+b)*n + (i_start+a)
        
        # M1: (A00+A11)(B00+B11) -> C00+1, C11+1
        col = offset
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        # M2: (A10+A11)B00 -> C10+1, C11-1
        col = offset + 1
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] -= 1
        
        # M3: A00(B01-B11) -> C01+1, C11+1
        col = offset + 2
        U[(i_start+0)*m + (j_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        # M4: A11(B10-B00) -> C00+1, C10+1
        col = offset + 3
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] -= 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        
        # M5: (A00+A01)B11 -> C00-1, C01+1
        col = offset + 4
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        
        # M6: (A10-A00)(B00+B01) -> C11+1
        col = offset + 5
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+0)][col] -= 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        # M7: (A01-A11)(B10+B11) -> C00+1
        col = offset + 6
        U[(i_start+0)*m + (j_start+1)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] -= 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1

    # Apply Strassen to the 6 blocks
    # A is 2x4, B is 4x6.
    # Split A columns into 2 groups of 2: [0,1] and [2,3].
    # Split B rows into 2 groups of 2: [0,1] and [2,3].
    # Split B cols into 3 groups of 2: [0,1], [2,3], [4,5].
    
    # Block (0,0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0)
    # Block (0,1): A[0:2, 0:2] * B[0:2, 2:4] -> C[2:4, 0:2]
    add_strassen_block(0, 0, 2, 7)
    # Block (0,2): A[0:2, 0:2] * B[0:2, 4:6] -> C[4:6, 0:2]
    add_strassen_block(0, 0, 4, 14)
    
    # Block (1,0): A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 2, 0, 21)
    # Block (1,1): A[0:2, 2:4] * B[2:4, 2:4] -> C[2:4, 0:2]
    add_strassen_block(0, 2, 2, 28)
    # Block (1,2): A[0:2, 2:4] * B[2:4, 4:6] -> C[4:6, 0:2]
    add_strassen_block(0, 2, 4, 35)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
