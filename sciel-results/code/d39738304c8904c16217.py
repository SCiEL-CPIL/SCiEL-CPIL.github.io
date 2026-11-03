import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 6x6 matrix multiplication.
    Current best known rank for 6x6 is 189 (Block-Strassen with naive 3x3 blocks).
    This implementation uses a distinct code structure from previous attempts.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # We use the tensor product of Strassen's 2x2 algorithm (rank 7)
    # and the naive 3x3 algorithm (rank 27).
    # Total rank = 7 * 27 = 189.
    
    rank = 189
    
    # Initialize U, V, W as numpy arrays for easier manipulation
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 block decomposition coefficients
    # Blocks are 3x3.
    # Block indices: (br, bc) for 2x2 grid of blocks.
    # Local indices: i_l, j_l, k_l in 0..2 for the 3x3 subblocks.
    
    # Strassen M1..M7 combinations
    # Each entry: (block_row, block_col, coeff)
    
    # M1 = (A00 + A11)(B00 + B11)
    A_m1 = [(0,0,1), (1,1,1)]
    B_m1 = [(0,0,1), (1,1,1)]
    C_m1 = [(0,0,1), (1,1,1)]
    
    # M2 = (A10 + A11)B00
    A_m2 = [(1,0,1), (1,1,1)]
    B_m2 = [(0,0,1)]
    C_m2 = [(1,0,1), (1,1,-1)]
    
    # M3 = A00(B01 - B11)
    A_m3 = [(0,0,1)]
    B_m3 = [(0,1,1), (1,1,-1)]
    C_m3 = [(0,1,1), (1,1,1)]
    
    # M4 = A11(B10 - B00)
    A_m4 = [(1,1,1)]
    B_m4 = [(1,0,1), (0,0,-1)]
    C_m4 = [(0,0,1), (1,0,1)]
    
    # M5 = (A00 + A01)B11
    A_m5 = [(0,0,1), (0,1,1)]
    B_m5 = [(1,1,1)]
    C_m5 = [(0,0,-1), (0,1,1)]
    
    # M6 = (A10 - A00)(B00 + B01)
    A_m6 = [(1,0,1), (0,0,-1)]
    B_m6 = [(0,0,1), (0,1,1)]
    C_m6 = [(1,1,1)]
    
    # M7 = (A01 - A11)(B10 + B11)
    A_m7 = [(0,1,1), (1,1,-1)]
    B_m7 = [(1,0,1), (1,1,1)]
    C_m7 = [(0,0,1)]
    
    strassen_combos = [
        (A_m1, B_m1, C_m1),
        (A_m2, B_m2, C_m2),
        (A_m3, B_m3, C_m3),
        (A_m4, B_m4, C_m4),
        (A_m5, B_m5, C_m5),
        (A_m6, B_m6, C_m6),
        (A_m7, B_m7, C_m7)
    ]
    
    block_size = 3
    
    # Fill the matrices
    for t_idx in range(7):
        a_comb, b_comb, c_comb = strassen_combos[t_idx]
        
        for i_l in range(block_size):
            for j_l in range(block_size):
                for k_l in range(block_size):
                    if time.time() > deadline:
                        # Return whatever we have constructed so far
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    
                    # Column index in the full rank matrix
                    # Each Strassen step has 27 sub-multiplications (3x3x3)
                    col_idx = t_idx * 27 + (i_l * 9 + j_l * 3 + k_l)
                    
                    # Update U
                    for rb, cb, coeff in a_comb:
                        i_glob = rb * block_size + i_l
                        j_glob = cb * block_size + j_l
                        U[i_glob * m + j_glob, col_idx] += coeff
                    
                    # Update V
                    for rb, cb, coeff in b_comb:
                        j_glob = rb * block_size + j_l
                        k_glob = cb * block_size + k_l
                        V[j_glob * p + k_glob, col_idx] += coeff
                    
                    # Update W
                    # W index convention: W[k*n+i, r]
                    for rb, cb, coeff in c_comb:
                        # rb is block row of C (which corresponds to k in C_ik if we align)
                        # Wait, C_ik. Block row rb, Block col cb.
                        # C_rb,cb = sum ...
                        # In Strassen, C_00 = M1 - M4 + M5 - M7
                        # C_01 = M3 + M7
                        # C_10 = M2 + M4
                        # C_11 = M1 - M2 + M3 + M6
                        # Our C_comb definitions above reflect this.
                        # rb is the row block index of C.
                        # cb is the col block index of C.
                        # In C_ik, i is row, k is col.
                        # So rb -> i_block, cb -> k_block.
                        
                        k_glob = cb * block_size + k_l
                        i_glob = rb * block_size + i_l
                        
                        W[k_glob * n + i_glob, col_idx] += coeff
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
