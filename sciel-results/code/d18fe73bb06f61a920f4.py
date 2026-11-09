"""Hybrid Strassen(4x4x4) + Schoolbook(4x4x3) decomposition for 4x4 * 4x7."""
import time

DIMENSIONS = [4, 4, 7]

def construct():
    """Construct rank-97 tensor decomposition for 4x4 * 4x7 matrix multiplication."""
    n, m, p = DIMENSIONS  # 4, 4, 7
    rank = 97  # 49 (Strassen 4x4x4) + 48 (Schoolbook 4x4x3)
    
    U = [[0] * rank for _ in range(n * m)]  # (16, 97)
    V = [[0] * rank for _ in range(m * p)]  # (28, 97)
    W = [[0] * rank for _ in range(p * n)]  # (28, 97)
    
    # Strassen 2x2x2 base case (rank 7)
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    # T0: (A00+A11)(B00+B11)
    U2[0][0]=1; U2[3][0]=1; V2[0][0]=1; V2[3][0]=1; W2[0][0]=1; W2[3][0]=1
    # T1: (A10+A11)B00
    U2[2][1]=1; U2[3][1]=1; V2[0][1]=1; W2[2][1]=1
    # T2: A00(B10-B11)
    U2[0][2]=1; V2[2][2]=1; V2[3][2]=-1; W2[1][2]=1
    # T3: A11(B01-B00)
    U2[3][3]=1; V2[1][3]=1; V2[0][3]=-1; W2[0][3]=1; W2[2][3]=1
    # T4: (A00+A01)B11
    U2[0][4]=1; U2[1][4]=1; V2[3][4]=1; W2[0][4]=-1; W2[1][4]=1
    # T5: (A10-A00)(B00+B01)
    U2[2][5]=1; U2[0][5]=-1; V2[0][5]=1; V2[1][5]=1; W2[3][5]=1
    # T6: (A01-A11)(B10+B11)
    U2[1][6]=1; U2[3][6]=-1; V2[2][6]=1; V2[3][6]=1; W2[0][6]=1
    
    # Outer Strassen coefficients for 4x4x4 (block-level 2x2x2)
    oa = [[[1,0],[0,1]], [[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[0,1]],
           [[1,1],[0,0]], [[-1,0],[1,0]], [[0,1],[0,-1]]]
    ob = [[[1,0],[0,1]], [[1,0],[0,0]], [[0,1],[0,-1]], [[1,-1],[0,0]],
           [[0,0],[0,1]], [[1,1],[0,0]], [[0,1],[0,1]]]
    oc = [[[1,0],[0,1]], [[0,0],[1,0]], [[0,1],[0,0]], [[1,0],[1,0]],
           [[-1,1],[0,0]], [[0,0],[0,1]], [[1,0],[0,0]]]
    
    # Build 4x4x4 Strassen decomposition (49 terms)
    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            # U factor: U[i*4+j, col]
            for ih in range(2):
                for il in range(2):
                    for jh in range(2):
                        for jl in range(2):
                            i = 2*ih + il
                            j = 2*jh + jl
                            v = oa[t][ih][jh] * U2[il*2+jl][s]
                            if v != 0:
                                U[i*4+j][col] += v
            # V factor: V[j*4+k, col]
            for jh in range(2):
                for jl in range(2):
                    for kh in range(2):
                        for kl in range(2):
                            j = 2*jh + jl
                            k = 2*kh + kl
                            v = ob[t][jh][kh] * V2[jl*2+kl][s]
                            if v != 0:
                                V[j*4+k][col] += v
            # W factor: W[k*4+i, col]
            for ih in range(2):
                for il in range(2):
                    for kh in range(2):
                        for kl in range(2):
                            i = 2*ih + il
                            k = 2*kh + kl
                            v = oc[t][ih][kh] * W2[kl*2+il][s]
                            if v != 0:
                                W[k*4+i][col] += v
    
    # Schoolbook for remaining 4x4x3 (48 terms)
    for j in range(4):
        for k in range(4, 7):
            col = 49 + j*3 + (k-4)
            for i in range(4):
                U[i*4+j][col] = 1
                V[j*4+k][col] = 1
                W[k*4+i][col] = 1
    
    return {"U": U, "V": V, "W": W}
