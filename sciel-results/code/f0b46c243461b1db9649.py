import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    rank = 28
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)

    # Strassen's 7 multiplications for 2x2 x 2x2 -> 2x2
    # X indices: [0,1,2,3] correspond to A[0,0], A[0,1], A[1,0], A[1,1]
    # Y indices: [0,1,2,3] correspond to B[0,0], B[0,1], B[1,0], B[1,1]
    # Z indices: [0,1,2,3] correspond to C[0,0], C[0,1], C[1,0], C[1,1]
    # But we use the trace convention W[k*n+i, r] for the full 2x4 x 4x4 -> 2x4
    # For a 2x2 block at rows r0..r1 of A and cols c0..c1 of B, contributing to C rows r0..r1, cols c0..c1:
    # A block indices in U: [r0, r0+1, r0+4, r0+5] (since A is 2x4, row r has indices r and r+4)
    # B block indices in V: [c0, c0+1, c0+4, c0+5] (since B is 4x4, col c has indices c, c+1, c+4, c+5... wait)
    # B is 4x4. V has 16 entries. B[i,j] is at V[i*4+j, r].
    # For B block (rows rB0..rB1, cols cB0..cB1):
    #   B[rB0, cB0] -> V[rB0*4+cB0]
    #   B[rB0, cB1] -> V[rB0*4+cB1]
    #   B[rB1, cB0] -> V[rB1*4+cB0]
    #   B[rB1, cB1] -> V[rB1*4+cB1]
    # C block (rows rC0..rC1, cols cC0..cC1):
    #   C[rC0, cC0] -> W[rC0*2+cC0]
    #   C[rC0, cC1] -> W[rC0*2+cC1]
    #   C[rC1, cC0] -> W[rC1*2+cC0]
    #   C[rC1, cC1] -> W[rC1*2+cC1]

    def add_strassen_block(A_idx, B_idx, C_idx, base):
        """
        A_idx: list of 4 indices into U for the 2x2 A block [a00, a01, a10, a11]
        B_idx: list of 4 indices into V for the 2x2 B block [b00, b01, b10, b11]
        C_idx: list of 4 indices into W for the 2x2 C block [c00, c01, c10, c11]
        base: offset in the rank dimension
        """
        a00, a01, a10, a11 = A_idx
        b00, b01, b10, b11 = B_idx
        c00, c01, c10, c11 = C_idx

        # M1 = (a00+a11)(b00+b11)
        U[a00, base+0] += 1
        U[a11, base+0] += 1
        V[b00, base+0] += 1
        V[b11, base+0] += 1
        W[c00, base+0] += 1
        W[c11, base+0] += 1

        # M2 = a11*(b10-b00)
        U[a11, base+1] += 1
        V[b10, base+1] += 1
        V[b00, base+1] -= 1
        W[c10, base+1] += 1
        W[c11, base+1] -= 1

        # M3 = (a00-a10)*b10
        U[a00, base+2] += 1
        U[a10, base+2] -= 1
        V[b10, base+2] += 1
        W[c01, base+2] += 1
        W[c11, base+2] += 1

        # M4 = (a01+a11)*b11
        U[a01, base+3] += 1
        U[a11, base+3] += 1
        V[b11, base+3] += 1
        W[c00, base+3] += 1
        W[c10, base+3] += 1

        # M5 = (a00+a01)*(b01-b11)
        U[a00, base+4] += 1
        U[a01, base+4] += 1
        V[b01, base+4] += 1
        V[b11, base+4] -= 1
        W[c00, base+4] -= 1
        W[c01, base+4] += 1

        # M6 = (a10-a00)*(b00+b01)
        U[a10, base+5] += 1
        U[a00, base+5] -= 1
        V[b00, base+5] += 1
        V[b01, base+5] += 1
        W[c11, base+5] += 1

        # M7 = (a11-a10)*(b10+b11)
        U[a11, base+6] += 1
        U[a10, base+6] -= 1
        V[b10, base+6] += 1
        V[b11, base+6] += 1
        W[c00, base+6] += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A block: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[4], A[1,1]=U[5]
    # B block: B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[4], B[1,1]=V[5]
    # C block: C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
    add_strassen_block([0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 2, 3], 0)

    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # A block: A[0,2]=U[2], A[0,3]=U[3], A[1,2]=U[6], A[1,3]=U[7]
    # B block: B[2,0]=V[8], B[2,1]=V[9], B[3,0]=V[12], B[3,1]=V[13]
    # C block: C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
    add_strassen_block([2, 3, 6, 7], [8, 9, 12, 13], [0, 1, 2, 3], 7)

    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # A block: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[4], A[1,1]=U[5]
    # B block: B[0,2]=V[2], B[0,3]=V[3], B[1,2]=V[6], B[1,3]=V[7]
    # C block: C[0,2]=W[4], C[0,3]=W[5], C[1,2]=W[6], C[1,3]=W[7]
    add_strassen_block([0, 1, 4, 5], [2, 3, 6, 7], [4, 5, 6, 7], 14)

    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # A block: A[0,2]=U[2], A[0,3]=U[3], A[1,2]=U[6], A[1,3]=U[7]
    # B block: B[2,2]=V[10], B[2,3]=V[11], B[3,2]=V[14], B[3,3]=V[15]
    # C block: C[0,2]=W[4], C[0,3]=W[5], C[1,2]=W[6], C[1,3]=W[7]
    add_strassen_block([2, 3, 6, 7], [10, 11, 14, 15], [4, 5, 6, 7], 21)

    # Convert to lists for the expected interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
