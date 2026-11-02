import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 189 from Block-Strassen (7 * 27)
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Local 3x3 schoolbook parameters
    # i_l, j_l, k_l in 0..2
    # U_3 index: i_l * 3 + j_l
    # V_3 index: j_l * 3 + k_l
    # W_3 index: k_l * 3 + i_l
    
    # Strassen Combinations
    # Each entry is a list of (block_row, block_col, coeff)
    # Blocks are 2x2, so indices 0,1
    
    # M1 = (A00 + A11)(B00 + B11)
    A_comb_0 = [(0,0,1), (1,1,1)]
    B_comb_0 = [(0,0,1), (1,1,1)]
    C_comb_0 = [(0,0,1), (1,1,1)]
    
    # M2 = (A10 + A11)B00
    A_comb_1 = [(1,0,1), (1,1,1)]
    B_comb_1 = [(0,0,1)]
    C_comb_1 = [(1,0,1), (1,1,-1)]
    
    # M3 = A00(B01 - B11)
    A_comb_2 = [(0,0,1)]
    B_comb_2 = [(0,1,1), (1,1,-1)]
    C_comb_2 = [(0,1,1), (1,1,1)]
    
    # M4 = A11(B10 - B00)
    A_comb_3 = [(1,1,1)]
    B_comb_3 = [(1,0,1), (0,0,-1)]
    C_comb_3 = [(0,0,1), (1,0,1)]
    
    # M5 = (A00 + A01)B11
    A_comb_4 = [(0,0,1), (0,1,1)]
    B_comb_4 = [(1,1,1)]
    C_comb_4 = [(0,0,-1), (0,1,1)]
    
    # M6 = (A10 - A00)(B00 + B01)
    A_comb_5 = [(1,0,1), (0,0,-1)]
    B_comb_5 = [(0,0,1), (0,1,1)]
    C_comb_5 = [(1,1,1)]
    
    # M7 = (A01 - A11)(B10 + B11)
    A_comb_6 = [(0,1,1), (1,1,-1)]
    B_comb_6 = [(1,0,1), (1,1,1)]
    C_comb_6 = [(0,0,1)]
    
    A_combs = [A_comb_0, A_comb_1, A_comb_2, A_comb_3, A_comb_4, A_comb_5, A_comb_6]
    B_combs = [B_comb_0, B_comb_1, B_comb_2, B_comb_3, B_comb_4, B_comb_5, B_comb_6]
    C_combs = [C_comb_0, C_comb_1, C_comb_2, C_comb_3, C_comb_4, C_comb_5, C_comb_6]
    
    block_size = 3
    
    for t in range(7):
        for i_l in range(block_size):
            for j_l in range(block_size):
                for k_l in range(block_size):
                    col = t * 27 + (i_l * 9 + j_l * 3 + k_l)
                    
                    # Update U
                    for rb, cb, coeff in A_combs[t]:
                        i_glob = rb * block_size + i_l
                        j_glob = cb * block_size + j_l
                        U[i_glob * m + j_glob][col] += coeff
                        
                    # Update V
                    for rb, cb, coeff in B_combs[t]:
                        j_glob = rb * block_size + j_l
                        k_glob = cb * block_size + k_l
                        V[j_glob * p + k_glob][col] += coeff
                        
                    # Update W
                    for rb, cb, coeff in C_combs[t]:
                        k_glob = cb * block_size + k_l
                        i_glob = rb * block_size + i_l
                        W[k_glob * n + i_glob][col] += coeff
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
