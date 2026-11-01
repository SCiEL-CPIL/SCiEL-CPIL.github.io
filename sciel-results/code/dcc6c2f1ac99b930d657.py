"""Rewritten program for matrix multiplication 2,2,9 tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 9
    # Rank calculation:
    # 4 blocks of 2x2 (Strassen rank 7 each) + 1 block of 2x1 (rank 4)
    # Total rank = 4 * 7 + 4 = 32
    rank = 32
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a Strassen block
    def add_strassen_block(block_idx, v_e, v_f, v_g, v_h, w_00, w_01, w_10, w_11):
        # block_idx is the starting column index in U,V,W
        # v_e, v_f, v_g, v_h are indices in V for the 2x2 block
        # w_00, w_01, w_10, w_11 are indices in W for the 2x2 output block
        
        # Strassen multiplications:
        # M1 = (a00 + a11) * (e + h)
        # M2 = (a10 + a11) * e
        # M3 = a00 * (f - h)
        # M4 = a11 * (g - e)
        # M5 = (a00 + a01) * h
        # M6 = (a10 - a11) * (f + h)
        # M7 = (a01 - a11) * (e + g)
        
        # U indices: 0:a00, 1:a01, 2:a10, 3:a11
        
        # Col 0: M1
        U[0, block_idx] += 1.0
        U[3, block_idx] += 1.0
        V[v_e, block_idx] += 1.0
        V[v_h, block_idx] += 1.0
        W[w_00, block_idx] += 1.0
        W[w_11, block_idx] += 1.0
        
        # Col 1: M2
        U[2, block_idx+1] += 1.0
        U[3, block_idx+1] += 1.0
        V[v_e, block_idx+1] += 1.0
        W[w_10, block_idx+1] += 1.0
        W[w_11, block_idx+1] += -1.0
        
        # Col 2: M3
        U[0, block_idx+2] += 1.0
        V[v_f, block_idx+2] += 1.0
        V[v_h, block_idx+2] += -1.0
        W[w_01, block_idx+2] += 1.0
        W[w_11, block_idx+2] += 1.0
        
        # Col 3: M4
        U[3, block_idx+3] += 1.0
        V[v_g, block_idx+3] += 1.0
        V[v_e, block_idx+3] += -1.0
        W[w_00, block_idx+3] += 1.0
        W[w_10, block_idx+3] += 1.0
        
        # Col 4: M5
        U[0, block_idx+4] += 1.0
        U[1, block_idx+4] += 1.0
        V[v_h, block_idx+4] += 1.0
        W[w_00, block_idx+4] += -1.0
        W[w_01, block_idx+4] += 1.0
        
        # Col 5: M6
        U[2, block_idx+5] += 1.0
        U[3, block_idx+5] += -1.0
        V[v_f, block_idx+5] += 1.0
        V[v_h, block_idx+5] += 1.0
        W[w_11, block_idx+5] += 1.0
        
        # Col 6: M7
        U[1, block_idx+6] += 1.0
        U[3, block_idx+6] += -1.0
        V[v_e, block_idx+6] += 1.0
        V[v_g, block_idx+6] += 1.0
        W[w_00, block_idx+6] += 1.0
        W[w_10, block_idx+6] += -1.0

    # Helper to add a 2x1 block
    def add_2x1_block(block_idx, v_e, v_g, w_0, w_1):
        # M1 = a00 * e
        # M2 = a01 * g
        # M3 = a10 * e
        # M4 = a11 * g
        
        # Col 0: M1
        U[0, block_idx] += 1.0
        V[v_e, block_idx] += 1.0
        W[w_0, block_idx] += 1.0
        
        # Col 1: M2
        U[1, block_idx+1] += 1.0
        V[v_g, block_idx+1] += 1.0
        W[w_0, block_idx+1] += 1.0
        
        # Col 2: M3
        U[2, block_idx+2] += 1.0
        V[v_e, block_idx+2] += 1.0
        W[w_1, block_idx+2] += 1.0
        
        # Col 3: M4
        U[3, block_idx+3] += 1.0
        V[v_g, block_idx+3] += 1.0
        W[w_1, block_idx+3] += 1.0

    # Block 1: cols 0,1
    # V indices: e=0, f=1, g=9, h=10
    # W indices: w_00=0, w_01=2, w_10=1, w_11=3
    add_strassen_block(0, 0, 1, 9, 10, 0, 2, 1, 3)
    
    # Block 2: cols 2,3
    # V indices: e=2, f=3, g=11, h=12
    # W indices: w_00=4, w_01=6, w_10=5, w_11=7
    add_strassen_block(7, 2, 3, 11, 12, 4, 6, 5, 7)
    
    # Block 3: cols 4,5
    # V indices: e=4, f=5, g=13, h=14
    # W indices: w_00=8, w_01=10, w_10=9, w_11=11
    add_strassen_block(14, 4, 5, 13, 14, 8, 10, 9, 11)
    
    # Block 4: cols 6,7
    # V indices: e=6, f=7, g=15, h=16
    # W indices: w_00=12, w_01=14, w_10=13, w_11=15
    add_strassen_block(21, 6, 7, 15, 16, 12, 14, 13, 15)
    
    # Block 5: col 8
    # V indices: e=8, g=17
    # W indices: w_0=16, w_1=17
    add_2x1_block(28, 8, 17, 16, 17)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
