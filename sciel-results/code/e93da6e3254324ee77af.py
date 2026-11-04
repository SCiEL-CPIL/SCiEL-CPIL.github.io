"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7) indexed by [i*2+j, c]
    V: (4, 7) indexed by [j*2+k, c]
    W: (4, 7) indexed by [k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # c=0: P1 = (A11+A22)(B11+B22)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # c=1: P2 = (A21+A22)B11
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1
    
    # c=2: P3 = A11(B21-B22)
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[2,2] = 1
    
    # c=3: P4 = A22(B12-B11)
    U[3,3] = 1
    V[1,3] = 1; V[0,3] = -1
    W[0,3] = -1; W[1,3] = 1
    
    # c=4: P5 = (A11+A12)B22
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = 1; W[2,4] = 1
    
    # c=5: P6 = (A21-A11)(B11+B12)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # c=6: P7 = (A12-A22)(B21+B22)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = -1; W[3,6] = 1
    
    return U, V, W

def _strassen_4x4_4x4():
    """Return U, V, W for 4x4x4 matrix multiplication using 2x2 Strassen blocks.
    U: (16, 49) indexed by [i*4+j, c]
    V: (16, 49) indexed by [j*4+k, c]
    W: (16, 49) indexed by [k*4+i, c]
    """
    sU, sV, sW = _strassen_2x2()
    
    # Block-level Strassen coefficients
    # A_coeff[x][bi][bj] = coefficient of A block (bi,bj) in block product x
    A_coeff = [
        [[1, 0], [0, 1]],   # M1: (A00+A11)(B00+B11)
        [[0, 0], [1, 1]],   # M2: (A10+A11)B00
        [[1, 0], [0, 0]],   # M3: A00(B10-B11)
        [[0, 0], [0, 1]],   # M4: A11(B01-B00)
        [[1, 1], [0, 0]],   # M5: (A00+A01)B11
        [[-1, 0], [1, 0]],  # M6: (A10-A00)(B00+B01)
        [[0, 1], [0, -1]],  # M7: (A01-A11)(B10+B11)
    ]
    
    B_coeff = [
        [[1, 0], [0, 1]],   # M1: (A00+A11)(B00+B11)
        [[1, 0], [0, 0]],   # M2: (A10+A11)B00
        [[0, 1], [0, -1]],  # M3: A00(B10-B11)
        [[-1, 1], [0, 0]],  # M4: A11(B01-B00)
        [[0, 0], [0, 1]],   # M5: (A00+A01)B11
        [[1, 1], [0, 0]],   # M6: (A10-A00)(B00+B01)
        [[0, 0], [1, 1]],   # M7: (A01-A11)(B10+B11)
    ]
    
    C_coeff = [
        [[1, 0], [0, 1]],   # M1: C00, C11
        [[0, 0], [1, 0]],   # M2: C10
        [[0, 1], [0, 1]],   # M3: C01, C11
        [[-1, 1], [0, 0]],  # M4: C00, C01
        [[0, 1], [0, 1]],   # M5: C01, C11
        [[0, 0], [0, 1]],   # M6: C11
        [[-1, 0], [0, 1]],  # M7: C00, C11
    ]
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for x in range(7):
        for y in range(7):
            c = x * 7 + y
            for bi in range(2):
                for bj in range(2):
                    a_coeff = A_coeff[x][bi][bj]
                    if a_coeff == 0:
                        continue
                    for ii in range(2):
                        for ij in range(2):
                            i = bi * 2 + ii
                            j = bj * 2 + ij
                            U4[i*4+j, c] = a_coeff * sU[ii*2+ij, y]
            
            for bj in range(2):
                for bk in range(2):
                    b_coeff = B_coeff[x][bj][bk]
                    if b_coeff == 0:
                        continue
                    for ij in range(2):
                        for ik in range(2):
                            j = bj * 2 + ij
                            k = bk * 2 + ik
                            V4[j*4+k, c] = b_coeff * sV[ij*2+ik, y]
            
            for bi in range(2):
                for bk in range(2):
                    c_coeff = C_coeff[x][bi][bk]
                    if c_coeff == 0:
                        continue
                    for ii in range(2):
                        for ik in range(2):
                            i = bi * 2 + ii
                            k = bk * 2 + ik
                            W4[k*4+i, c] = c_coeff * sW[ik*2+ii, y]
    
    return U4, V4, W4

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    U4, V4, W4 = _strassen_4x4_4x4()
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Half 1: columns 0-3 of B
    U[:, 0:49] = U4
    for j in range(4):
        for r in range(4):
            V[j*8+r, 0:49] = V4[j*4+r, :]
    for r in range(4):
        for i in range(4):
            W[r*4+i, 0:49] = W4[r*4+i, :]
    
    # Half 2: columns 4-7 of B
    U[:, 49:98] = U4
    for j in range(4):
        for r in range(4, 8):
            V[j*8+r, 49:98] = V4[j*4+(r-4), :]
    for r in range(4, 8):
        for i in range(4):
            W[r*4+i, 49:98] = W4[(r-4)*4+i, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

