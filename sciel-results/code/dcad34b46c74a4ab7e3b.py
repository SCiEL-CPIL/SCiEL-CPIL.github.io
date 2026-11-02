"""
Optimized Matrix Multiplication Tensor Decomposition for 2x4 * 4x4.
Uses Strassen's algorithm on 2x2 blocks to achieve rank 28.
The tensor rank for 2x4 * 4x4 is known to be 28 (Landsberg).
"""

import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    # Initialize factor matrices as NumPy arrays of zeros
    # U: (n*m, rank) -> (8, 28)
    # V: (m*p, rank) -> (16, 28)
    # W: (p*n, rank) -> (8, 28)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def idx_a(i, j):
        return i * m + j
        
    def idx_b(i, j):
        return i * p + j
        
    def idx_c(i, j):
        # Trace convention: W[k*n + i, r] where output C[i,j] corresponds to row i, col j.
        # Based on the task description "W[k*n+i,r]":
        # C has dimensions n x p (rows i, cols k).
        # The index is k*n + i.
        return j * n + i

    def add_strassen_block(offset, a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        """
        Adds a Strassen decomposition for a 2x2 block multiplication.
        A sub-block: A[a_r0:a_r0+2, a_c0:a_c0+2]
        B sub-block: B[b_r0:b_r0+2, b_c0:b_c0+2]
        C sub-block: C[c_r0:c_r0+2, c_c0:c_c0+2]
        
        Standard Strassen formulas for P = A*B:
        M1 = (A00 + A11) * (B00 + B11)
        M2 = (A10 + A11) * B00
        M3 = A00 * (B01 - B11)
        M4 = A11 * (B10 - B00)
        M5 = (A00 + A01) * B11
        M6 = (A10 - A00) * (B00 + B01)
        M7 = (A01 - A11) * (B10 + B11)
        
        P00 = M1 - M5 + M7
        P01 = M3 + M5
        P10 = M2 + M4
        P11 = M1 - M2 + M3 + M6
        """
        # U indices for A
        u_00 = idx_a(a_r0, a_c0)
        u_01 = idx_a(a_r0, a_c0 + 1)
        u_10 = idx_a(a_r0 + 1, a_c0)
        u_11 = idx_a(a_r0 + 1, a_c0 + 1)
        
        # V indices for B
        v_00 = idx_b(b_r0, b_c0)
        v_01 = idx_b(b_r0, b_c0 + 1)
        v_10 = idx_b(b_r0 + 1, b_c0)
        v_11 = idx_b(b_r0 + 1, b_c0 + 1)
        
        # W indices for C
        w_00 = idx_c(c_r0, c_c0)
        w_01 = idx_c(c_r0, c_c0 + 1)
        w_10 = idx_c(c_r0 + 1, c_c0)
        w_11 = idx_c(c_r0 + 1, c_c0 + 1)
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[u_00, offset + 0] += 1
        U[u_11, offset + 0] += 1
        V[v_00, offset + 0] += 1
        V[v_11, offset + 0] += 1
        # P00 += M1, P11 += M1
        W[w_00, offset + 0] += 1
        W[w_11, offset + 0] += 1
        
        # M2 = (A10 + A11) * B00
        U[u_10, offset + 1] += 1
        U[u_11, offset + 1] += 1
        V[v_00, offset + 1] += 1
        # P10 += M2, P11 -= M2
        W[w_10, offset + 1] += 1
        W[w_11, offset + 1] -= 1
        
        # M3 = A00 * (B01 - B11)
        U[u_00, offset + 2] += 1
        V[v_01, offset + 2] += 1
        V[v_11, offset + 2] -= 1
        # P01 += M3, P11 += M3
        W[w_01, offset + 2] += 1
        W[w_11, offset + 2] += 1
        
        # M4 = A11 * (B10 - B00)
        U[u_11, offset + 3] += 1
        V[v_10, offset + 3] += 1
        V[v_00, offset + 3] -= 1
        # P00 += M4, P10 += M4
        W[w_00, offset + 3] += 1
        W[w_10, offset + 3] += 1
        
        # M5 = (A00 + A01) * B11
        U[u_00, offset + 4] += 1
        U[u_01, offset + 4] += 1
        V[v_11, offset + 4] += 1
        # P00 -= M5, P01 += M5
        W[w_00, offset + 4] -= 1
        W[w_01, offset + 4] += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[u_10, offset + 5] += 1
        U[u_00, offset + 5] -= 1
        V[v_00, offset + 5] += 1
        V[v_01, offset + 5] += 1
        # P11 += M6
        W[w_11, offset + 5] += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[u_01, offset + 6] += 1
        U[u_11, offset + 6] -= 1
        V[v_10, offset + 6] += 1
        V[v_11, offset + 6] += 1
        # P00 += M7
        W[w_00, offset + 6] += 1

    # Decompose the 2x4 * 4x4 into four 2x2 * 2x2 blocks
    # A is 2x4. Split into A_left (2x2) and A_right (2x2).
    # B is 4x4. Split into 2x2 blocks: Top-Left, Top-Right, Bottom-Left, Bottom-Right.
    # C is 2x4. Split into C_left (2x2) and C_right (2x2).
    
    # Block 1: A_left * B_top_left -> C_left
    # A rows 0-1, cols 0-1. B rows 0-1, cols 0-1. C rows 0-1, cols 0-1.
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A_right * B_bottom_left -> C_left
    # A rows 0-1, cols 2-3. B rows 2-3, cols 0-1. C rows 0-1, cols 0-1.
    add_strassen_block(7, 0, 2, 2, 0, 0, 0)
    
    # Block 3: A_left * B_top_right -> C_right
    # A rows 0-1, cols 0-1. B rows 0-1, cols 2-3. C rows 0-1, cols 2-3.
    add_strassen_block(14, 0, 0, 0, 2, 0, 2)
    
    # Block 4: A_right * B_bottom_right -> C_right
    # A rows 0-1, cols 2-3. B rows 2-3, cols 2-3. C rows 0-1, cols 2-3.
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)
    
    # Convert to lists of lists to match expected interface format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
