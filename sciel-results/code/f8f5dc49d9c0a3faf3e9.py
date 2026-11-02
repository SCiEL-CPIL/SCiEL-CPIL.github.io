"""Optimized Rank-22 decomposition for 2x3x4 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 22
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a Strassen term
    def add_strassen_term(col_offset, b_col_offset):
        # b_col_offset is 0 for first 2 cols, 2 for next 2 cols
        
        # Local A indices in U:
        # a00: 0, a01: 1, a10: 3, a11: 4
        
        # Local B indices in V:
        # b00: b_col_offset + 0
        # b01: b_col_offset + 1
        # b10: b_col_offset + 4
        # b11: b_col_offset + 5
        
        # Local C indices in W:
        # c00: b_col_offset * 2 + 0
        # c01: b_col_offset * 2 + 1
        # c10: b_col_offset * 2 + 4
        # c11: b_col_offset * 2 + 5
        
        # M1: (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
        U[0, col_offset + 0] += 1
        U[4, col_offset + 0] += 1
        V[b_col_offset, col_offset + 0] += 1
        V[b_col_offset + 4, col_offset + 0] += 1
        W[b_col_offset * 2, col_offset + 0] += 1
        W[b_col_offset * 2 + 5, col_offset + 0] += 1
        
        # M2: (a10+a11)b00 -> c10 += 1, c11 -= 1
        U[3, col_offset + 1] += 1
        U[4, col_offset + 1] += 1
        V[b_col_offset, col_offset + 1] += 1
        W[b_col_offset * 2 + 4, col_offset + 1] += 1
        W[b_col_offset * 2 + 5, col_offset + 1] -= 1
        
        # M3: a00(b01-b11) -> c01 += 1, c11 += 1
        U[0, col_offset + 2] += 1
        V[b_col_offset + 1, col_offset + 2] += 1
        V[b_col_offset + 5, col_offset + 2] -= 1
        W[b_col_offset * 2 + 1, col_offset + 2] += 1
        W[b_col_offset * 2 + 5, col_offset + 2] += 1
        
        # M4: a11(b10-b00) -> c00 += 1, c10 += 1
        U[4, col_offset + 3] += 1
        V[b_col_offset + 4, col_offset + 3] += 1
        V[b_col_offset, col_offset + 3] -= 1
        W[b_col_offset * 2, col_offset + 3] += 1
        W[b_col_offset * 2 + 4, col_offset + 3] += 1
        
        # M5: (a00+a01)b11 -> c00 -= 1, c01 += 1
        U[0, col_offset + 4] += 1
        U[1, col_offset + 4] += 1
        V[b_col_offset + 5, col_offset + 4] += 1
        W[b_col_offset * 2, col_offset + 4] -= 1
        W[b_col_offset * 2 + 1, col_offset + 4] += 1
        
        # M6: (a10-a00)(b00+b01) -> c11 += 1
        U[3, col_offset + 5] += 1
        U[0, col_offset + 5] -= 1
        V[b_col_offset, col_offset + 5] += 1
        V[b_col_offset + 1, col_offset + 5] += 1
        W[b_col_offset * 2 + 5, col_offset + 5] += 1
        
        # M7: (a01-a11)(b10+b11) -> c00 += 1
        U[1, col_offset + 6] += 1
        U[4, col_offset + 6] -= 1
        V[b_col_offset + 4, col_offset + 6] += 1
        V[b_col_offset + 5, col_offset + 6] += 1
        W[b_col_offset * 2, col_offset + 6] += 1
        
    # Add Strassen terms for B cols 0-1
    add_strassen_term(0, 0)
    
    # Add Strassen terms for B cols 2-3
    add_strassen_term(7, 2)
    
    # Add outer product terms for a (A col 2) and b (B row 2)
    # a is U index 2 (i=0, j=2) and 5 (i=1, j=2)
    # b is V index 8 (j=2, k=0), 9 (j=2, k=1), 10 (j=2, k=2), 11 (j=2, k=3)
    
    # a0 * b0 -> C[0,0]
    U[2, 14] += 1
    V[8, 14] += 1
    W[0, 14] += 1
    
    # a0 * b1 -> C[0,1]
    U[2, 15] += 1
    V[9, 15] += 1
    W[1, 15] += 1
    
    # a0 * b2 -> C[0,2]
    U[2, 16] += 1
    V[10, 16] += 1
    W[2, 16] += 1
    
    # a0 * b3 -> C[0,3]
    U[2, 17] += 1
    V[11, 17] += 1
    W[3, 17] += 1
    
    # a1 * b0 -> C[1,0]
    U[5, 18] += 1
    V[8, 18] += 1
    W[4, 18] += 1
    
    # a1 * b1 -> C[1,1]
    U[5, 19] += 1
    V[9, 19] += 1
    W[5, 19] += 1
    
    # a1 * b2 -> C[1,2]
    U[5, 20] += 1
    V[10, 20] += 1
    W[6, 20] += 1
    
    # a1 * b3 -> C[1,3]
    U[5, 21] += 1
    V[11, 21] += 1
    W[7, 21] += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
