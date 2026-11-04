import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # The matrix multiplication tensor T_{i,j,k} = delta_{j,k} for C[i,k] = sum_j A[i,j]*B[j,k]
    # We need: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Current approach: 8 Strassen blocks, each with 7 multiplications = 56 rank
    # This is the known best for 2x4 * 4x8 using block Strassen
    
    rank = 56
    U = np.zeros((8, rank))
    V = np.zeros((32, rank))
    W = np.zeros((16, rank))
    
    col = 0
    # 8 blocks: (a_col_off, b_row_off, b_col_off, c_col_off)
    # A submatrix: A[0:2, a_col_off:a_col_off+2]
    # B submatrix: B[b_row_off:b_row_off+2, b_col_off:b_col_off+2]
    # C submatrix: C[0:2, c_col_off:c_col_off+2]
    blocks = [
        (0, 0, 0, 0),  # A1 * B[0:2, 0:2] -> C[:, 0:2]
        (0, 0, 2, 2),  # A1 * B[0:2, 2:4] -> C[:, 2:4]
        (0, 0, 4, 4),  # A1 * B[0:2, 4:6] -> C[:, 4:6]
        (0, 0, 6, 6),  # A1 * B[0:2, 6:8] -> C[:, 6:8]
        (2, 2, 0, 0),  # A2 * B[2:4, 0:2] -> C[:, 0:2]
        (2, 2, 2, 2),  # A2 * B[2:4, 2:4] -> C[:, 2:4]
        (2, 2, 4, 4),  # A2 * B[2:4, 4:6] -> C[:, 4:6]
        (2, 2, 6, 6),  # A2 * B[2:4, 6:8] -> C[:, 6:8]
    ]
    
    for (a_col_off, b_row_off, b_col_off, c_col_off) in blocks:
        # A submatrix indices: A[i,j] -> i*4+j
        a00 = 0*4 + a_col_off      # A[0, a_col_off]
        a01 = 0*4 + a_col_off + 1  # A[0, a_col_off+1]
        a10 = 1*4 + a_col_off      # A[1, a_col_off]
        a11 = 1*4 + a_col_off + 1  # A[1, a_col_off+1]
        
        # B submatrix indices: B[j,k] -> j*8+k
        b00 = b_row_off*8 + b_col_off
        b01 = b_row_off*8 + b_col_off + 1
        b10 = (b_row_off+1)*8 + b_col_off
        b11 = (b_row_off+1)*8 + b_col_off + 1
        
        # C submatrix indices: C[i,k] -> k*2+i
        c00 = c_col_off*2 + 0      # C[0, c_col_off]
        c01 = (c_col_off+1)*2 + 0  # C[0, c_col_off+1]
        c10 = c_col_off*2 + 1      # C[1, c_col_off]
        c11 = (c_col_off+1)*2 + 1  # C[1, c_col_off+1]
        
        # P1 = (a00+a11)(b00+b11)
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1   # C00 += P1
        W[c11, col] += 1   # C11 += P1
        col += 1
        
        # P2 = (a10+a11)*b00
        U[a10, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1   # C10 += P2
        W[c11, col] += -1  # C11 -= P2
        col += 1
        
        # P3 = a00*(b01-b11)
        U[a00, col] += 1
        V[b01, col] += 1
        V[b11, col] += -1
        W[c01, col] += 1   # C01 += P3
        W[c11, col] += 1   # C11 += P3
        col += 1
        
        # P4 = a11*(b10-b00)
        U[a11, col] += 1
        V[b10, col] += 1
        V[b00, col] += -1
        W[c00, col] += 1   # C00 += P4
        W[c10, col] += 1   # C10 += P4
        col += 1
        
        # P5 = (a00+a01)*b11
        U[a00, col] += 1
        U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1  # C00 -= P5
        W[c01, col] += 1   # C01 += P5
        col += 1
        
        # P6 = (a10-a00)*(b00+b01)
        U[a10, col] += 1
        U[a00, col] += -1
        V[b00, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1   # C11 += P6
        col += 1
        
        # P7 = (a01-a11)*(b10+b11)
        U[a01, col] += 1
        U[a11, col] += -1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1   # C00 += P7
        col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
