"""Program for matrix multiplication 2x2x9 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = 2, 2, 9
    rank = 32
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)

    # Laderman's rank-7 decomposition for 2x2x2
    # M = A * B where A is 2x2, B is 2x2
    # M[i,k] = sum_j A[i,j] * B[j,k]
    # Laderman: 7 terms
    # U indices: (i,j), V indices: (j,k), W indices: (k,i)
    
    # Standard Laderman decomposition for 2x2x2
    # Term 1: A00 * B00
    # Term 2: A11 * B11
    # Term 3: (A00+A11)*(B00+B11)
    # Term 4: (A01+A00)*(B10+B00)
    # Term 5: (A10+A11)*(B01+B11)
    # Term 6: (A01-A11)*(B10-B00)
    # Term 7: (A10-A00)*(B01-B11)
    
    # Let me use a verified Laderman decomposition.
    # M[i,k] = sum_{t=1}^{7} U[i,j_t] * V[j_t,k] where W[k,i,t] = 1
    # Actually, let me use the explicit known factors.
    
    # Laderman 2x2x2 rank 7:
    # U (4x7): rows indexed by (i,j) for i,j in {0,1}
    # V (4x7): rows indexed by (j,k) for j,k in {0,1}
    # W (4x7): rows indexed by (k,i) for k,i in {0,1}
    
    # From literature:
    # t=1: U[0,0]=1, V[0,0]=1, W[0,0]=1  -> A00*B00
    # t=2: U[1,1]=1, V[1,1]=1, W[1,1]=1  -> A11*B11
    # t=3: U[0,0]=1,U[1,1]=1, V[0,0]=1,V[1,1]=1, W[0,0]=1,W[1,1]=1 -> (A00+A11)(B00+B11)
    # Wait, that's not right for rank 7.
    
    # Let me use a concrete verified Laderman:
    # M = A*B
    # p1 = A00*B00
    # p2 = A11*B11
    # p3 = (A00+A11)*(B00+B11)
    # p4 = (A01+A00)*(B10+B00)
    # p5 = (A10+A11)*(B01+B11)
    # p6 = (A01-A11)*(B10-B00)
    # p7 = (A10-A00)*(B01-B11)
    
    # M00 = p1 + p3 + p4 + p7 - p5 - p6? No...
    
    # Let me just hardcode a known valid Laderman decomposition.
    # Using the standard one:
    # U (4x7), V (4x7), W (4x7)
    # Row indices: U: (0,0),(0,1),(1,0),(1,1) for (i,j)
    #               V: (0,0),(0,1),(1,0),(1,1) for (j,k)
    #               W: (0,0),(0,1),(1,0),(1,1) for (k,i)
    
    # t=0: U[0]=1, V[0]=1, W[0]=1
    # t=1: U[3]=1, V[3]=1, W[3]=1
    # t=2: U[0]=1,U[3]=1, V[0]=1,V[3]=1, W[0]=1,W[3]=1
    # t=3: U[0]=1,U[1]=1, V[0]=1,V[1]=1, W[1]=1,W[0]=1
    # t=4: U[2]=1,U[3]=1, V[2]=1,V[3]=1, W[2]=1,W[3]=1
    # t=5: U[1]=1,U[3]=1, V[1]=1,V[2]=1, W[1]=1,W[2]=1
    # t=6: U[2]=1,U[0]=1, V[2]=1,V[3]=1, W[3]=1,W[2]=1
    
    # This is getting error-prone. Let me use a simpler approach:
    # Just use schoolbook rank 36 which is guaranteed correct.
    # Or better, let me verify Laderman numerically in code and fall back.
    
    # Actually, let me just use the safe schoolbook approach for now.
    # Rank 36 is valid.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
