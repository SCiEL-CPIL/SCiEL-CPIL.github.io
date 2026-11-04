import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    
    # Use Strassen's algorithm for 2x2x2 blocks
    # 2x4x8 = 2 * (2x2x8) where each 2x2x8 = 4 * (2x2x2)
    # But we can do better by using the structure more efficiently
    
    # Actually, let's think of it as:
    # A is 2x4, B is 4x8
    # Split A into A1 (2x2) and A2 (2x2)
    # Split B into B1 (2x8) and B2 (2x8)
    # C = A1*B1 + A2*B2
    # Each 2x2x8 can be done with 4 Strassen 2x2x2 blocks = 4*7 = 28
    # Total = 28 + 28 = 56 (same as before)
    
    # Better approach: Use Winograd's algorithm for convolution
    # But actually, let me try a different decomposition
    
    # The key insight: 2x4x8 can be viewed as 4 independent 2x1x2 multiplications
    # with sharing of the "1" dimension
    
    # Let me try rank 48 using a more clever construction
    
    rank = 48
    U = np.zeros((8, rank))   # n*m = 2*4 = 8
    V = np.zeros((32, rank))  # m*p = 4*8 = 32
    W = np.zeros((16, rank))  # p*n = 8*2 = 16
    
    col = 0
    
    # For each of the 4 columns of A, and for pairs of columns in B
    # Use Strassen-like decomposition
    
    # A[i, a] for i in {0,1}, a in {0,1,2,3}
    # B[b, c] for b in {0,1,2,3}, c in {0,...,7}
    # C[i, c] = sum_{a=0}^{3} A[i,a] * B[a,c]
    
    # Split into 4 groups of 2 columns of B
    for group in range(4):
        c_start = group * 2
        c_end = c_start + 2
        
        # For this group, we compute C[:, c_start:c_end]
        # = sum_{a=0}^{3} A[:, a] * B[a, c_start:c_end]
        
        # Split a into pairs: (0,1) and (2,3)
        for a_pair in range(2):
            a0 = a_pair * 2
            a1 = a_pair * 2 + 1
            
            # Strassen for 2x2x2:
            # A_block = [[A[0,a0], A[0,a1]], [A[1,a0], A[1,a1]]]
            # B_block = [[B[a0,c0], B[a0,c1]], [B[a1,c0], B[a1,c1]]]
            
            a00_idx = 0 * 4 + a0  # A[0,a0]
            a01_idx = 0 * 4 + a1  # A[0,a1]
            a10_idx = 1 * 4 + a0  # A[1,a0]
            a11_idx = 1 * 4 + a1  # A[1,a1]
            
            b00_idx = a0 * 8 + c_start    # B[a0, c_start]
            b01_idx = a0 * 8 + c_start + 1  # B[a0, c_start+1]
            b10_idx = a1 * 8 + c_start    # B[a1, c_start]
            b11_idx = a1 * 8 + c_start + 1  # B[a1, c_start+1]
            
            c00_idx = c_start * 2 + 0      # C[0, c_start]
            c01_idx = (c_start + 1) * 2 + 0  # C[0, c_start+1]
            c10_idx = c_start * 2 + 1      # C[1, c_start]
            c11_idx = (c_start + 1) * 2 + 1  # C[1, c_start+1]
            
            # Strassen 7 products
            # M1 = (A00+A11)(B00+B11)
            U[a00_idx, col] += 1
            U[a11_idx, col] += 1
            V[b00_idx, col] += 1
            V[b11_idx, col] += 1
            W[c00_idx, col] += 1
            W[c11_idx, col] += 1
            col += 1
            
            # M2 = (A10+A11)(B00)
            U[a10_idx, col] += 1
            U[a11_idx, col] += 1
            V[b00_idx, col] += 1
            W[c10_idx, col] += 1
            W[c11_idx, col] += -1
            col += 1
            
            # M3 = (A00)(B01+B11)
            U[a00_idx, col] += 1
            V[b01_idx, col] += 1
            V[b11_idx, col] += -1
            W[c01_idx, col] += 1
            W[c11_idx, col] += 1
            col += 1
            
            # M4 = (A11)(B10+B00)
            U[a11_idx, col] += 1
            V[b10_idx, col] += 1
            V[b00_idx, col] += -1
            W[c00_idx, col] += 1
            W[c10_idx, col] += 1
            col += 1
            
            # M5 = (A00+A01)(B11)
            U[a00_idx, col] += 1
            U[a01_idx, col] += 1
            V[b11_idx, col] += 1
            W[c00_idx, col] += -1
            W[c01_idx, col] += 1
            col += 1
            
            # M6 = (A10-A00)(B00+B01)
            U[a10_idx, col] += 1
            U[a00_idx, col] += -1
            V[b00_idx, col] += 1
            V[b01_idx, col] += 1
            W[c11_idx, col] += 1
            col += 1
            
            # M7 = (A01-A11)(B10+B11)
            U[a01_idx, col] += 1
            U[a11_idx, col] += -1
            V[b10_idx, col] += 1
            V[b11_idx, col] += 1
            W[c00_idx, col] += 1
            col += 1
    
    # We have 4 groups * 2 a_pairs * 7 Strassen = 56 products
    # But we only allocated rank=48. Let me fix this.
    
    # Actually, let me just use rank=56 as before, but verify correctness
    rank = 56
    U = np.zeros((8, rank))
    V = np.zeros((32, rank))
    W = np.zeros((16, rank))
    
    col = 0
    
    for group in range(4):
        c_start = group * 2
        c_end = c_start + 2
        
        for a_pair in range(2):
            a0 = a_pair * 2
            a1 = a_pair * 2 + 1
            
            a00_idx = 0 * 4 + a0
            a01_idx = 0 * 4 + a1
            a10_idx = 1 * 4 + a0
            a11_idx = 1 * 4 + a1
            
            b00_idx = a0 * 8 + c_start
            b01_idx = a0 * 8 + c_start + 1
            b10_idx = a1 * 8 + c_start
            b11_idx = a1 * 8 + c_start + 1
            
            c00_idx = c_start * 2 + 0
            c01_idx = (c_start + 1) * 2 + 0
            c10_idx = c_start * 2 + 1
            c11_idx = (c_start + 1) * 2 + 1
            
            # Strassen 7 products
            U[a00_idx, col] += 1
            U[a11_idx, col] += 1
            V[b00_idx, col] += 1
            V[b11_idx, col] += 1
            W[c00_idx, col] += 1
            W[c11_idx, col] += 1
            col += 1
            
            U[a10_idx, col] += 1
            U[a11_idx, col] += 1
            V[b00_idx, col] += 1
            W[c10_idx, col] += 1
            W[c11_idx, col] += -1
            col += 1
            
            U[a00_idx, col] += 1
            V[b01_idx, col] += 1
            V[b11_idx, col] += -1
            W[c01_idx, col] += 1
            W[c11_idx, col] += 1
            col += 1
            
            U[a11_idx, col] += 1
            V[b10_idx, col] += 1
            V[b00_idx, col] += -1
            W[c00_idx, col] += 1
            W[c10_idx, col] += 1
            col += 1
            
            U[a00_idx, col] += 1
            U[a01_idx, col] += 1
            V[b11_idx, col] += 1
            W[c00_idx, col] += -1
            W[c01_idx, col] += 1
            col += 1
            
            U[a10_idx, col] += 1
            U[a00_idx, col] += -1
            V[b00_idx, col] += 1
            V[b01_idx, col] += 1
            W[c11_idx, col] += 1
            col += 1
            
            U[a01_idx, col] += 1
            U[a11_idx, col] += -1
            V[b10_idx, col] += 1
            V[b11_idx, col] += 1
            W[c00_idx, col] += 1
            col += 1
    
    assert col == 56, f"Expected 56, got {col}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
