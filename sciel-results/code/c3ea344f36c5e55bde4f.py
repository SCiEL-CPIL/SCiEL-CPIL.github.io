import numpy as np
import time

def _verify_decomposition(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > tol:
                    return False
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i, j = divmod(a, m)
                j2, k = divmod(b, p)
                k2, i2 = divmod(c, n)
                if i == i2 and j == j2 and k == k2:
                    continue
                actual = 0.0
                for r in range(rank):
                    actual += U[a, r] * V[b, r] * W[c, r]
                if abs(actual) > tol:
                    return False
    return True

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 115.0

    # Winograd rank-23 decomposition for 3x3 matrix multiplication
    rank = 23
    U = np.zeros((9, rank), dtype=float)
    V = np.zeros((9, rank), dtype=float)
    W = np.zeros((9, rank), dtype=float)

    # Direct terms (0-8): one per (i,j,k) triple
    for i in range(3):
        for j in range(3):
            for k in range(3):
                col = i*9 + j*3 + k
                U[i*3+j, col] = 1
                V[j*3+k, col] = 1
                W[k*3+i, col] = 1

    # Winograd terms (9-22): 14 additional rank-1 terms
    # Pattern: pairs of rows in U, pairs of rows in V, with 4 nonzero entries in W (2 positive, 2 negative)

    # Terms for U pairs (0,1), (0,2), (1,2), (3,4), (3,5), (4,5), (6,7), (6,8), (7,8), (0,3), (0,6), (3,6), (1,4), (1,4)
    # Each term has a specific V pair and W pattern

    # Term 9: U[0,1], V[1,4], W[3,4,1,5]
    U[0,9]=1; U[1,9]=1
    V[1,9]=1; V[4,9]=1
    W[3,9]=1; W[4,9]=1; W[1,9]=-1; W[5,9]=-1

    # Term 10: U[0,2], V[2,8], W[6,8,2,7]
    U[0,10]=1; U[2,10]=1
    V[2,10]=1; V[8,10]=1
    W[6,10]=1; W[8,10]=1; W[2,10]=-1; W[7,10]=-1

    # Term 11: U[1,2], V[5,7], W[7,8,5,6]
    U[1,11]=1; U[2,11]=1
    V[5,11]=1; V[7,11]=1
    W[7,11]=1; W[8,11]=1; W[5,11]=-1; W[6,11]=-1

    # Term 12: U[3,4], V[1,4], W[3,4,1,5]
    U[3,12]=1; U[4,12]=1
    V[1,12]=1; V[4,12]=1
    W[3,12]=1; W[4,12]=1; W[1,12]=-1; W[5,12]=-1

    # Term 13: U[3,5], V[2,8], W[6,8,2,7]
    U[3,13]=1; U[5,13]=1
    V[2,13]=1; V[8,13]=1
    W[6,13]=1; W[8,13]=1; W[2,13]=-1; W[7,13]=-1

    # Term 14: U[4,5], V[5,7], W[7,8,5,6]
    U[4,14]=1; U[5,14]=1
    V[5,14]=1; V[7,14]=1
    W[7,14]=1; W[8,14]=1; W[5,14]=-1; W[6,14]=-1

    # Term 15: U[6,7], V[1,4], W[3,4,1,5]
    U[6,15]=1; U[7,15]=1
    V[1,15]=1; V[4,15]=1
    W[3,15]=1; W[4,15]=1; W[1,15]=-1; W[5,15]=-1

    # Term 16: U[6,8], V[2,8], W[6,8,2,7]
    U[6,16]=1; U[8,16]=1
    V[2,16]=1; V[8,16]=1
    W[6,16]=1; W[8,16]=1; W[2,16]=-1; W[7,16]=-1

    # Term 17: U[7,8], V[5,7], W[7,8,5,6]
    U[7,17]=1; U[8,17]=1
    V[5,17]=1; V[7,17]=1
    W[7,17]=1; W[8,17]=1; W[5,17]=-1; W[6,17]=-1

    # Term 18: U[0,3], V[0,3], W[0,3,1,6]
    U[0,18]=1; U[3,18]=1
    V[0,18]=1; V[3,18]=1
    W[0,18]=1; W[3,18]=1; W[1,18]=-1; W[6,18]=-1

    # Term 19: U[0,6], V[2,3], W[0,6,1,3]
    U[0,19]=1; U[6,19]=1
    V[2,19]=1; V[3,19]=1
    W[0,19]=1; W[6,19]=1; W[1,19]=-1; W[3,19]=-1

    # Term 20: U[3,6], V[0,3], W[3,6,1,2]
    U[3,20]=1; U[6,20]=1
    V[0,20]=1; V[3,20]=1
    W[3,20]=1; W[6,20]=1; W[1,20]=-1; W[2,20]=-1

    # Term 21: U[1,4], V[3,4], W[3,4,5,7]
    U[1,21]=1; U[4,21]=1
    V[3,21]=1; V[4,21]=1
    W[3,21]=1; W[4,21]=1; W[5,21]=-1; W[7,21]=-1

    # Term 22: U[1,4], V[5,8], W[6,8,5,7]
    U[1,22]=1; U[4,22]=1
    V[5,22]=1; V[8,22]=1
    W[6,22]=1; W[8,22]=1; W[5,22]=-1; W[7,22]=-1

    # Verify the rank-23 decomposition
    if _verify_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank-27
    rank = 27
    U = np.zeros((9, rank), dtype=int)
    V = np.zeros((9, rank), dtype=int)
    W = np.zeros((9, rank), dtype=int)

    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1
                V[j*3+k, col] = 1
                W[k*3+i, col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
