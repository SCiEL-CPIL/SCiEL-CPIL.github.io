import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 150.0  # wall-clock limit

def _elapsed():
    return time.time() - _start_time

def _deadline_exceeded(margin=5.0):
    return _elapsed() > _DEADLINE - margin

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Decomposition:
    # A (3x4) = [A1 (3x3), A2 (3x1)]
    # B (4x5) = [B1 (3x5); B2 (1x5)]
    # C = A1*B1 + A2*B2
    
    # A2*B2: 3x1x5 -> Rank 15
    # A1*B1: 3x3x5 -> Split B1 into B11 (3x3) and B12 (3x2)
    # A1*B11: 3x3x3 -> Rank 26 (using Strassen 2x2 for 2x2 subblock)
    # A1*B12: 3x3x2 -> Rank 18 (Schoolbook)
    # Total Rank: 15 + 26 + 18 = 59
    
    rank = 59
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A2*B2 (3x1x5)
    # A2[i,0] = A[i,3], B2[0,k] = B[3,k]
    # C[i,k] += A[i,3] * B[3,k]
    # U index: i*4+3, V index: 3*5+k, W index: k*3+i
    for i in range(n):
        for k in range(p):
            U[i * m + 3, col] = 1.0
            V[3 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    # col = 15
    
    # Part 2: A1*B12 (3x3x2) Schoolbook
    # A1[i,j] (i=0..2, j=0..2), B12[j,k] (j=0..2, k=3..4)
    # C[i,k] += A1[i,j] * B12[j,k]
    for i in range(n):
        for j in range(3):
            for k in range(3, p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    # col = 15 + 18 = 33
    
    # Part 3: A1*B11 (3x3x3) Rank 26
    # X = A1 (3x3), Y = B11 (3x3)
    # X = [X11(2x2) X12(2x1); X21(1x2) X22(1x1)]
    # Y = [Y11(2x2) Y12(2x1); Y21(1x2) Y22(1x1)]
    
    # Indices:
    # X11: X[0,0]=U[0], X[0,1]=U[1], X[1,0]=U[4], X[1,1]=U[5]
    # X12: X[0,2]=U[2], X[1,2]=U[6]
    # X21: X[2,0]=U[8], X[2,1]=U[9]
    # X22: X[2,2]=U[10]
    
    # Y11: Y[0,0]=V[0], Y[0,1]=V[1], Y[1,0]=V[5], Y[1,1]=V[6]
    # Y12: Y[0,2]=V[2], Y[1,2]=V[6] -- Wait, Y index is j*5+k.
    # Y[0,0]=V[0], Y[0,1]=V[1], Y[0,2]=V[2]
    # Y[1,0]=V[5], Y[1,1]=V[6], Y[1,2]=V[7]
    # Y[2,0]=V[10], Y[2,1]=V[11], Y[2,2]=V[12]
    
    # W indices: W[k*3+i]
    # Z[0,0]=W[0], Z[0,1]=W[3], Z[0,2]=W[6]
    # Z[1,0]=W[1], Z[1,1]=W[4], Z[1,2]=W[7]
    # Z[2,0]=W[2], Z[2,1]=W[5], Z[2,2]=W[8]
    
    # 3.1: X11*Y11 (2x2x2) Strassen Rank 7
    # a=X[0,0]=U[0], b=X[0,1]=U[1], c=X[1,0]=U[4], d=X[1,1]=U[5]
    # e=Y[0,0]=V[0], f=Y[0,1]=V[1], g=Y[1,0]=V[5], h=Y[1,1]=V[6]
    
    # M1 = (a+d)(e+h)
    U[0, col] += 1.0; U[5, col] += 1.0
    V[0, col] += 1.0; V[6, col] += 1.0
    W[0, col] += 1.0  # Z00
    W[8, col] += 1.0  # Z22 (Wait, Z22 is part of X21*Y12 etc? No, Z11 block is 2x2. Z[0,0] and Z[1,1] are in Z11. Z[0,1] and Z[1,0] are in Z11.
    # Z11 corresponds to rows 0,1 and cols 0,1.
    # W[0*3+0]=W[0] is Z[0,0]
    # W[0*3+1]=W[1] is Z[0,1]
    # W[1*3+0]=W[3] is Z[1,0]
    # W[1*3+1]=W[4] is Z[1,1]
    
    # Correction: W index is k*n+i.
    # Z[0,0] -> k=0, i=0 -> W[0]
    # Z[0,1] -> k=1, i=0 -> W[3]
    # Z[1,0] -> k=0, i=1 -> W[1]
    # Z[1,1] -> k=1, i=1 -> W[4]
    
    # Strassen Reconstruction:
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # M1 = (a+d)(e+h)
    U[0, col] += 1.0; U[5, col] += 1.0
    V[0, col] += 1.0; V[6, col] += 1.0
    W[0, col] += 1.0  # Z00
    W[4, col] += 1.0  # Z11
    col += 1
    
    # M2 = (c+d)e
    U[4, col] += 1.0; U[5, col] += 1.0
    V[0, col] += 1.0
    W[1, col] += 1.0  # Z10
    W[4, col] += -1.0 # Z11
    col += 1
    
    # M3 = a(f-h)
    U[0, col] += 1.0
    V[1, col] += 1.0; V[6, col] += -1.0
    W[3, col] += 1.0  # Z01
    W[4, col] += 1.0  # Z11
    col += 1
    
    # M4 = d(g-e)
    U[5, col] += 1.0
    V[5, col] += 1.0; V[0, col] += -1.0
    W[0, col] += 1.0  # Z00
    W[1, col] += 1.0  # Z10
    col += 1
    
    # M5 = (a+b)h
    U[0, col] += 1.0; U[1, col] += 1.0
    V[6, col] += 1.0
    W[0, col] += -1.0 # Z00
    W[3, col] += 1.0  # Z01
    col += 1
    
    # M6 = (c-a)(e+f)
    U[4, col] += 1.0; U[0, col] += -1.0
    V[0, col] += 1.0; V[1, col] += 1.0
    W[4, col] += 1.0  # Z11
    col += 1
    
    # M7 = (b-d)(g+h)
    U[1, col] += 1.0; U[5, col] += -1.0
    V[5, col] += 1.0; V[6, col] += 1.0
    W[0, col] += 1.0  # Z00
    col += 1
    
    # 3.2: X12*Y21 (2x1x2) Schoolbook Rank 4
    # X12: X[0,2]=U[2], X[1,2]=U[6]
    # Y21: Y[2,0]=V[10], Y[2,1]=V[11]
    # Z00 += X[0,2]*Y[2,0]
    # Z01 += X[0,2]*Y[2,1]
    # Z10 += X[1,2]*Y[2,0]
    # Z11 += X[1,2]*Y[2,1]
    
    U[2, col] += 1.0; V[10, col] += 1.0; W[0, col] += 1.0; col += 1
    U[2, col] += 1.0; V[11, col] += 1.0; W[3, col] += 1.0; col += 1
    U[6, col] += 1.0; V[10, col] += 1.0; W[1, col] += 1.0; col += 1
    U[6, col] += 1.0; V[11, col] += 1.0; W[4, col] += 1.0; col += 1
    
    # 3.3: X11*Y12 (2x2x1) Schoolbook Rank 4
    # X11: U[0], U[1], U[4], U[5]
    # Y12: Y[0,2]=V[2], Y[1,2]=V[7]
    # Z02 += X[0,0]*Y[0,2] + X[0,1]*Y[1,2]
    # Z12 += X[1,0]*Y[0,2] + X[1,1]*Y[1,2]
    # W indices: Z[0,2]->W[6], Z[1,2]->W[7]
    
    U[0, col] += 1.0; V[2, col] += 1.0; W[6, col] += 1.0; col += 1
    U[1, col] += 1.0; V[7, col] += 1.0; W[6, col] += 1.0; col += 1
    U[4, col] += 1.0; V[2, col] += 1.0; W[7, col] += 1.0; col += 1
    U[5, col] += 1.0; V[7, col] += 1.0; W[7, col] += 1.0; col += 1
    
    # 3.4: X12*Y22 (2x1x1) Schoolbook Rank 2
    # X12: U[2], U[6]
    # Y22: Y[2,2]=V[12]
    # Z02 += X[0,2]*Y[2,2]
    # Z12 += X[1,2]*Y[2,2]
    
    U[2, col] += 1.0; V[12, col] += 1.0; W[6, col] += 1.0; col += 1
    U[6, col] += 1.0; V[12, col] += 1.0; W[7, col] += 1.0; col += 1
    
    # 3.5: X21*Y11 (1x2x2) Schoolbook Rank 4
    # X21: X[2,0]=U[8], X[2,1]=U[9]
    # Y11: V[0], V[1], V[5], V[6]
    # Z20 += X[2,0]*Y[0,0] + X[2,1]*Y[1,0]
    # Z21 += X[2,0]*Y[0,1] + X[2,1]*Y[1,1]
    # W indices: Z[2,0]->W[2], Z[2,1]->W[5]
    
    U[8, col] += 1.0; V[0, col] += 1.0; W[2, col] += 1.0; col += 1
    U[9, col] += 1.0; V[5, col] += 1.0; W[2, col] += 1.0; col += 1
    U[8, col] += 1.0; V[1, col] += 1.0; W[5, col] += 1.0; col += 1
    U[9, col] += 1.0; V[6, col] += 1.0; W[5, col] += 1.0; col += 1
    
    # 3.6: X22*Y21 (1x1x2) Schoolbook Rank 2
    # X22: X[2,2]=U[10]
    # Y21: V[10], V[11]
    # Z20 += X[2,2]*Y[2,0]
    # Z21 += X[2,2]*Y[2,1]
    
    U[10, col] += 1.0; V[10, col] += 1.0; W[2, col] += 1.0; col += 1
    U[10, col] += 1.0; V[11, col] += 1.0; W[5, col] += 1.0; col += 1
    
    # 3.7: X21*Y12 (1x2x1) Schoolbook Rank 2
    # X21: U[8], U[9]
    # Y12: V[2], V[7]
    # Z22 += X[2,0]*Y[0,2] + X[2,1]*Y[1,2]
    # W index: Z[2,2]->W[8]
    
    U[8, col] += 1.0; V[2, col] += 1.0; W[8, col] += 1.0; col += 1
    U[9, col] += 1.0; V[7, col] += 1.0; W[8, col] += 1.0; col += 1
    
    # 3.8: X22*Y22 (1x1x1) Rank 1
    # X22: U[10]
    # Y22: V[12]
    # Z22 += X[2,2]*Y[2,2]
    
    U[10, col] += 1.0; V[12, col] += 1.0; W[8, col] += 1.0; col += 1
    
    # Verify col == 59
    if col != 59:
        # Fallback to schoolbook if logic error
        rank = 60
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = (i * m + j) * p + k
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    col = c + 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
