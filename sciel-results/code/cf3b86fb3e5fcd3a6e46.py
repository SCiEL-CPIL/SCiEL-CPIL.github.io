"""
Tensor decomposition for 3x4 * 4x8 matrix multiplication.
Uses Strassen's algorithm for 2x2 blocks where possible.
"""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    _start_time = time.time()
    _deadline = _start_time + 120
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    
    Strategy:
    - Partition A (3x4) into A1 (2x4) and A2 (1x4)
    - Partition B (4x8) into B1 (4x4) and B2 (4x4)
    - For A1*B1: use Strassen on 2x2 blocks (4 blocks of 2x2*2x2, each rank 7 = 28 total)
    - For A1*B2: use Strassen on 2x2 blocks (4 blocks of 2x2*2x2, each rank 7 = 28 total)
    - For A2*B1: schoolbook 1x4*4x4, rank 16
    - For A2*B2: schoolbook 1x4*4x4, rank 16
    Total rank: 28 + 28 + 16 + 16 = 88
    """
    n, m, p = DIMENSIONS
    
    # U has shape (n*m, rank) = (12, rank)
    # V has shape (m*p, rank) = (32, rank)
    # W has shape (p*n, rank) = (24, rank)
    
    terms = []  # List of (u_vector, v_vector, w_vector)
    
    def time_check():
        return time.time() > _deadline
    
    def add_strassen_block(i0, j0, k0):
        """
        Add Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        """
        if time_check():
            return
        
        nm = n * m
        mp = m * p
        pn = p * n
        
        a00_u = i0 * m + j0
        a01_u = i0 * m + (j0 + 1)
        a10_u = (i0 + 1) * m + j0
        a11_u = (i0 + 1) * m + (j0 + 1)
        
        b00_v = j0 * p + k0
        b01_v = j0 * p + (k0 + 1)
        b10_v = (j0 + 1) * p + k0
        b11_v = (j0 + 1) * p + (k0 + 1)
        
        c00_w = k0 * n + i0
        c01_w = (k0 + 1) * n + i0
        c10_w = k0 * n + (i0 + 1)
        c11_w = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a00 + a11) * (b00 + b11)
        u1 = [0.0] * nm; u1[a00_u] = 1.0; u1[a11_u] = 1.0
        v1 = [0.0] * mp; v1[b00_v] = 1.0; v1[b11_v] = 1.0
        w1 = [0.0] * pn; w1[c00_w] = 1.0; w1[c11_w] = 1.0
        terms.append((u1, v1, w1))
        
        # M2 = (a10 + a11) * b00
        u2 = [0.0] * nm; u2[a10_u] = 1.0; u2[a11_u] = 1.0
        v2 = [0.0] * mp; v2[b00_v] = 1.0
        w2 = [0.0] * pn; w2[c10_w] = 1.0; w2[c11_w] = -1.0
        terms.append((u2, v2, w2))
        
        # M3 = a00 * (b01 - b11)
        u3 = [0.0] * nm; u3[a00_u] = 1.0
        v3 = [0.0] * mp; v3[b01_v] = 1.0; v3[b11_v] = -1.0
        w3 = [0.0] * pn; w3[c01_w] = 1.0; w3[c11_w] = 1.0
        terms.append((u3, v3, w3))
        
        # M4 = a11 * (b10 - b00)
        u4 = [0.0] * nm; u4[a11_u] = 1.0
        v4 = [0.0] * mp; v4[b10_v] = 1.0; v4[b00_v] = -1.0
        w4 = [0.0] * pn; w4[c00_w] = 1.0; w4[c10_w] = 1.0
        terms.append((u4, v4, w4))
        
        # M5 = (a00 + a01) * b11
        u5 = [0.0] * nm; u5[a00_u] = 1.0; u5[a01_u] = 1.0
        v5 = [0.0] * mp; v5[b11_v] = 1.0
        w5 = [0.0] * pn; w5[c00_w] = -1.0; w5[c01_w] = 1.0
        terms.append((u5, v5, w5))
        
        # M6 = (a10 - a00) * (b00 + b01)
        u6 = [0.0] * nm; u6[a10_u] = 1.0; u6[a00_u] = -1.0
        v6 = [0.0] * mp; v6[b00_v] = 1.0; v6[b01_v] = 1.0
        w6 = [0.0] * pn; w6[c11_w] = 1.0
        terms.append((u6, v6, w6))
        
        # M7 = (a01 - a11) * (b10 + b11)
        u7 = [0.0] * nm; u7[a01_u] = 1.0; u7[a11_u] = -1.0
        v7 = [0.0] * mp; v7[b10_v] = 1.0; v7[b11_v] = 1.0
        w7 = [0.0] * pn; w7[c00_w] = 1.0
        terms.append((u7, v7, w7))
        
    def add_schoolbook_1x4_4x4(i_row, j0, k0):
        """
        Add schoolbook decomposition for A[i_row, j0:j0+4] * B[j0:j0+4, k0:k0+4]
        This is a 1x4 * 4x4 multiplication, rank 16.
        """
        if time_check():
            return
        nm = n * m
        mp = m * p
        pn = p * n
        for j in range(4):
            for k in range(4):
                u_idx = i_row * m + (j0 + j)
                v_idx = (j0 + j) * p + (k0 + k)
                w_idx = (k0 + k) * n + i_row
                u_vec = [0.0] * nm; u_vec[u_idx] = 1.0
                v_vec = [0.0] * mp; v_vec[v_idx] = 1.0
                w_vec = [0.0] * pn; w_vec[w_idx] = 1.0
                terms.append((u_vec, v_vec, w_vec))
                
    # A1*B1: A1 is rows 0,1 of A (2x4). B1 is cols 0-3 of B (4x4).
    # Split A1 into A[0:2, 0:2] and A[0:2, 2:4]
    # Split B1 into B[0:2, 0:4] and B[2:4, 0:4]
    # Then split B[0:2, 0:4] into B[0:2, 0:2] and B[0:2, 2:4]
    # And B[2:4, 0:4] into B[2:4, 0:2] and B[2:4, 2:4]
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_block(i0=0, j0=0, k0=0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen_block(i0=0, j0=0, k0=2)
    
    # Block 3: A[0:2, 2:4] * B[2:4, 0:2]
    add_strassen_block(i0=0, j0=2, k0=0)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4]
    add_strassen_block(i0=0, j0=2, k0=2)
    
    # A1*B2: A1 is rows 0,1 of A (2x4). B2 is cols 4-7 of B (4x4).
    # Block 5: A[0:2, 0:2] * B[0:2, 4:6]
    add_strassen_block(i0=0, j0=0, k0=4)
    
    # Block 6: A[0:2, 0:2] * B[0:2, 6:8]
    add_strassen_block(i0=0, j0=0, k0=6)
    
    # Block 7: A[0:2, 2:4] * B[2:4, 4:6]
    add_strassen_block(i0=0, j0=2, k0=4)
    
    # Block 8: A[0:2, 2:4] * B[2:4, 6:8]
    add_strassen_block(i0=0, j0=2, k0=6)
    
    # A2*B1: A2 is row 2 of A (1x4). B1 is cols 0-3 of B (4x4).
    add_schoolbook_1x4_4x4(i_row=2, j0=0, k0=0)
    
    # A2*B2: A2 is row 2 of A (1x4). B2 is cols 4-7 of B (4x4).
    add_schoolbook_1x4_4x4(i_row=2, j0=0, k0=4)
    
    rank = len(terms)
    
    # Build final matrices using numpy for speed
    nm = n * m
    mp = m * p
    pn = p * n
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for r in range(rank):
        u_vec, v_vec, w_vec = terms[r]
        U[:, r] = u_vec
        V[:, r] = v_vec
        W[:, r] = w_vec
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
