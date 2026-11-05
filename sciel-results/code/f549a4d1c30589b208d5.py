"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
DEADLINE = START_TIME + 120  # 120s wall-clock limit

def _time_remaining():
    return DEADLINE - time.time()

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    
    # We pad the inner dimension m=5 to m_pad=6 to allow full Strassen decomposition.
    m_pad = 6
    rank = 42  # 3 blocks of (2x2x2) Strassen * 2 sub-blocks of output = 3 * 2 * 7 = 42
    
    # Initialize padded factor matrices
    U_pad = np.zeros((n * m_pad, rank), dtype=int)
    V_pad = np.zeros((m_pad * p, rank), dtype=int)
    W_pad = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # i_base: 0 (n=2, so only one pair of rows)
        # j_base: 0, 2, 4 (inner dimension pairs)
        # k_base: 0, 2, 4 (output dimension pairs)
        
        # M0: (A00+A11)(B00+B11)
        U_pad[i_base * m_pad + j_base][col] += 1
        U_pad[(i_base+1) * m_pad + (j_base+1)][col] += 1
        V_pad[j_base * p + k_base][col] += 1
        V_pad[(j_base+1) * p + (k_base+1)][col] += 1
        W_pad[k_base * n + i_base][col] += 1
        W_pad[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M1: (A10+A11)B00
        U_pad[(i_base+1) * m_pad + j_base][col] += 1
        U_pad[(i_base+1) * m_pad + (j_base+1)][col] += 1
        V_pad[j_base * p + k_base][col] += 1
        W_pad[k_base * n + (i_base+1)][col] += 1
        W_pad[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        
        # M2: A00(B01-B11)
        U_pad[i_base * m_pad + j_base][col] += 1
        V_pad[j_base * p + (k_base+1)][col] += 1
        V_pad[(j_base+1) * p + (k_base+1)][col] -= 1
        W_pad[(k_base+1) * n + i_base][col] += 1
        W_pad[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M3: A11(B10-B00)
        U_pad[(i_base+1) * m_pad + (j_base+1)][col] += 1
        V_pad[(j_base+1) * p + k_base][col] += 1
        V_pad[j_base * p + k_base][col] -= 1
        W_pad[k_base * n + i_base][col] += 1
        W_pad[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # M4: (A00+A01)B11
        U_pad[i_base * m_pad + j_base][col] += 1
        U_pad[i_base * m_pad + (j_base+1)][col] += 1
        V_pad[(j_base+1) * p + (k_base+1)][col] += 1
        W_pad[k_base * n + i_base][col] -= 1
        W_pad[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # M5: (A10-A00)(B00+B01)
        U_pad[(i_base+1) * m_pad + j_base][col] += 1
        U_pad[i_base * m_pad + j_base][col] -= 1
        V_pad[j_base * p + k_base][col] += 1
        V_pad[j_base * p + (k_base+1)][col] += 1
        W_pad[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M6: (A01-A11)(B10+B11)
        U_pad[i_base * m_pad + (j_base+1)][col] += 1
        U_pad[(i_base+1) * m_pad + (j_base+1)][col] -= 1
        V_pad[(j_base+1) * p + k_base][col] += 1
        V_pad[(j_base+1) * p + (k_base+1)][col] += 1
        W_pad[k_base * n + i_base][col] += 1
        col += 1

    # Apply Strassen blocks for all 3 inner pairs and 3 output pairs
    for j_base in range(0, m_pad, 2):
        for k_base in range(0, p, 2):
            add_strassen_block(0, j_base, k_base)
            
    if col != rank:
        # Should be exactly 42
        pass

    # Slice to original dimensions
    # U: (n * m, rank) -> (2 * 5, 42)
    U = U_pad[:n * m, :]
    # V: (m * p, rank) -> (5 * 6, 42)
    V = V_pad[:m * p, :]
    # W: (p * n, rank) -> (6 * 2, 42)
    W = W_pad
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
