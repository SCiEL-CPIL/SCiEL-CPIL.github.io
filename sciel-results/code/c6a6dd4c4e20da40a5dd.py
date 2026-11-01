import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 110.0

def _time_left():
    return _DEADLINE - time.time()

def construct():
    n, m, p = 2, 4, 8
    A_dim = n * m   # 8
    B_dim = m * p   # 32
    C_dim = p * n   # 16
    
    # We'll use Strassen decomposition for each 2x2 x 2x2 block.
    # The 2x4 x 4x8 multiplication is decomposed as:
    # - Split B's 8 columns into 4 groups of 2: B_g (4x2) for g=0,1,2,3
    # - C_g (2x2) = A (2x4) * B_g (4x2)
    # - Split A into A_0 (2x2, cols 0-1) and A_1 (2x2, cols 2-3)
    # - C_g = A_0 * B_g[0:2,:] + A_1 * B_g[2:4,:]
    # - Each 2x2 x 2x2 uses Strassen (rank 7)
    # - Total: 4 groups * 2 (A_0, A_1) * 7 (Strassen) = 56
    
    rank = 56
    U = np.zeros((A_dim, rank), dtype=float)
    V = np.zeros((B_dim, rank), dtype=float)
    W = np.zeros((C_dim, rank), dtype=float)
    
    # Strassen products for 2x2 x 2x2:
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    #
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    
    # For each group g (0-3), and each A-block (A_0 for cols 0-1, A_1 for cols 2-3):
    # We compute A_block * B_g_slice where B_g_slice is 2x2
    
    # A_block is 2x2:
    # For A_0: rows of A are [A[0,0], A[0,1], A[0,2], A[0,3]] and [A[1,0], A[1,1], A[1,2], A[1,3]]
    # A_0 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # In flat index: A[0] = A[0,0], A[1] = A[0,1], A[4] = A[1,0], A[5] = A[1,1]
    
    # For A_1: A_1 = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    # In flat index: A[2] = A[0,2], A[3] = A[0,3], A[6] = A[1,2], A[7] = A[1,3]
    
    # B_g is 4x2: B_g[j,k] = B[j*8 + g*2 + k]
    # For group g, B_g[0:2,:] = rows 0,1 of B_g (2x2)
    # B_g[0,0] = B[g*2+0] = B[g*2], B_g[0,1] = B[g*2+1]
    # B_g[1,0] = B[8+g*2], B_g[1,1] = B[8+g*2+1]
    # B_g[2:4,:] = rows 2,3 of B_g (2x2)
    # B_g[2,0] = B[16+g*2], B_g[2,1] = B[16+g*2+1]
    # B_g[3,0] = B[24+g*2], B_g[3,1] = B[24+g*2+1]
    
    # C_g is 2x2: C_g[i,k] = C[g*2+k]*2 + i ... wait
    # C is 8x2, indexed as C[k*n + i] = C[k*2 + i]
    # C_g[i,k] corresponds to C[k*2 + i]
    # So C_g[0,0] = C[0*2+0] = C[0], C_g[0,1] = C[1*2+0] = C[2]
    # C_g[1,0] = C[0*2+1] = C[1], C_g[1,1] = C[1*2+1] = C[3]
    
    col = 0
    for g in range(4):  # 4 groups of 2 columns in B
        for a_block in range(2):  # A_0 (cols 0-1) or A_1 (cols 2-3)
            # A_block indices:
            if a_block == 0:
                # A_0 = [[A[0], A[1]], [A[4], A[5]]]
                x00_idx, x01_idx = 0, 1
                x10_idx, x11_idx = 4, 5
            else:
                # A_1 = [[A[2], A[3]], [A[6], A[7]]]
                x00_idx, x01_idx = 2, 3
                x10_idx, x11_idx = 6, 7
            
            # B_g slice indices:
            if a_block == 0:
                # B_g[0:2,:] -> rows 0,1 of B_g
                y00_idx, y01_idx = g*2, g*2+1
                y10_idx, y11_idx = 8+g*2, 8+g*2+1
            else:
                # B_g[2:4,:] -> rows 2,3 of B_g
                y00_idx, y01_idx = 16+g*2, 16+g*2+1
                y10_idx, y11_idx = 24+g*2, 24+g*2+1
            
            # C_g indices: C_g[i,k] = C[g*2+k*1 + i]... 
            # C is 8x2, C[k*n+i] = C[k*2+i]
            # C_g[i,k] = C[k*2+i] where k is column in C_g (0 or 1), i is row (0 or 1)
            # c00 = C[0*2+0] = C[0], c01 = C[1*2+0] = C[2]
            # c10 = C[0*2+1] = C[1], c11 = C[1*2+1] = C[3]
            c00_idx = g*2 + 0  # C[0] of group g
            c01_idx = g*2 + 1  # C[2] of group g... wait
            # Actually C is indexed as C[k*n + i] = C[k*2 + i]
            # For group g, the 2 columns are at positions 2g and 2g+1 in the 8-column C
            # C_g[i, 0] = C[(2g)*2 + i] = C[4g + i]... no
            # C is 8x2: row k (0-7), col i (0-1). Flat: C[k*2+i]
            # Group g corresponds to columns 2g and 2g+1 of B, which produce rows 2g and 2g+1 of C
            # So C_g[i, 0] = C[2g * 2 + i] = C[4g + i]
            # C_g[i, 1] = C[(2g+1) * 2 + i] = C[4g + 2 + i]
            c00_idx = 4*g + 0  # C_g[0,0]
            c01_idx = 4*g + 2  # C_g[0,1]
            c10_idx = 4*g + 1  # C_g[1,0]
            c11_idx = 4*g + 3  # C_g[1,1]
            
            # Strassen products:
            # M1 = (x00+x11)(y00+y11)
            U[x00_idx, col] = 1.0
            U[x11_idx, col] = 1.0
            V[y00_idx, col] = 1.0
            V[y11_idx, col] = 1.0
            # Z00 += M1, Z11 += M1
            W[c00_idx, col] = 1.0
            W[c11_idx, col] = 1.0
            col += 1
            
            # M2 = (x10+x11)y00
            U[x10_idx, col] = 1.0
            U[x11_idx, col] = 1.0
            V[y00_idx, col] = 1.0
            # Z10 += M2, Z11 -= M2
            W[c10_idx, col] = 1.0
            W[c11_idx, col] = -1.0
            col += 1
            
            # M3 = x00(y01-y11)
            U[x00_idx, col] = 1.0
            V[y01_idx, col] = 1.0
            V[y11_idx, col] = -1.0
            # Z01 += M3, Z11 += M3
            W[c01_idx, col] = 1.0
            W[c11_idx, col] = 1.0
            col += 1
            
            # M4 = x11(y10-y00)
            U[x11_idx, col] = 1.0
            V[y10_idx, col] = 1.0
            V[y00_idx, col] = -1.0
            # Z00 += M4, Z10 += M4
            W[c00_idx, col] = 1.0
            W[c10_idx, col] = 1.0
            col += 1
            
            # M5 = (x00+x01)y11
            U[x00_idx, col] = 1.0
            U[x01_idx, col] = 1.0
            V[y11_idx, col] = 1.0
            # Z00 -= M5, Z01 += M5
            W[c00_idx, col] = -1.0
            W[c01_idx, col] = 1.0
            col += 1
            
            # M6 = (x10-x00)(y00+y01)
            U[x10_idx, col] = 1.0
            U[x00_idx, col] = -1.0
            V[y00_idx, col] = 1.0
            V[y01_idx, col] = 1.0
            # Z11 += M6
            W[c11_idx, col] = 1.0
            col += 1
            
            # M7 = (x01-x11)(y10+y11)
            U[x01_idx, col] = 1.0
            U[x11_idx, col] = -1.0
            V[y10_idx, col] = 1.0
            V[y11_idx, col] = 1.0
            # Z00 += M7
            W[c00_idx, col] = 1.0
            col += 1
    
    # Verify col == rank
    assert col == rank, f"col={col}, rank={rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
