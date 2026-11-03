import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 7]

def _build_tensor():
    """Build the matrix multiplication tensor T[n*m*p] as a flat list."""
    n, m, p = DIMENSIONS
    size = n * m * p
    T = [0.0] * size
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m * p + j * p + k] = 1.0
    return T, n, m, p

def _verify(U, V, W, n, m, p, T):
    """Verify that the decomposition is exact using NumPy."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
        
    U_np = np.array(U, dtype=float).reshape(n*m, rank)
    V_np = np.array(V, dtype=float).reshape(m*p, rank)
    W_np = np.array(W, dtype=float).reshape(p*n, rank)
    
    U_t = U_np.reshape(n, m, rank)
    V_t = V_np.reshape(m, p, rank)
    W_t = W_np.reshape(p, n, rank)
    
    try:
        result = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        T_np = np.array(T, dtype=float).reshape(n, m, p)
        if np.allclose(result, T_np, atol=1e-9):
            return True
    except Exception:
        pass
        
    return False

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10 + A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B10 - B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01 - B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00 + A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W, 7

def _schoolbook_2x2x1():
    """
    Return schoolbook decomposition for 2x2x1 matrix multiplication.
    Rank 4.
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    
    # A is 2x2, B is 2x1. C is 2x1.
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    
    # Term 1: A[0,0]B[0,0]
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # Term 2: A[0,1]B[1,0]
    U[1][1] = 1
    V[1][1] = 1
    W[0][1] = 1
    
    # Term 3: A[1,0]B[0,0]
    U[2][2] = 1
    V[0][2] = 1
    W[1][2] = 1
    
    # Term 4: A[1,1]B[1,0]
    U[3][3] = 1
    V[1][3] = 1
    W[1][3] = 1
    
    return U, V, W, 4

def _decomp_2x2x3():
    """
    Return decomposition for 2x2x3 matrix multiplication.
    Rank 11 (7 for 2x2x2 + 4 for 2x2x1).
    """
    U = [[0]*11 for _ in range(4)]
    V = [[0]*11 for _ in range(6)]
    W = [[0]*11 for _ in range(6)]
    
    sU, sV, sW, srank = _strassen_2x2x2()
    bU, bV, bW, brank = _schoolbook_2x2x1()
    
    # Copy Strassen part (k=0,1)
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i*2+j][r] += sU[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V[j*3+k][r] += sV[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[k*2+i][r] += sW[k*2+i][r]
                
    # Copy Schoolbook part (k=2)
    # B is 2x3. Here B is 2x1 at index k=2.
    # V indices: j*3 + k. k=2.
    # W indices: k*2 + i. k=2.
    for r in range(brank):
        for i in range(2):
            for j in range(2):
                U[i*2+j][srank+r] += bU[i*2+j][r]
        for j in range(2):
            V[j*3+2][srank+r] += bV[j][r]
        for i in range(2):
            W[2*2+i][srank+r] += bW[i][r]
            
    return U, V, W, 11

def _decomp_2x1x7():
    """
    Return decomposition for 2x1x7 matrix multiplication.
    Rank 1.
    A is 2x1, B is 1x7. C is 2x7.
    C[i,k] = A[i,0] B[0,k].
    """
    U = [[0]*1 for _ in range(2)]
    V = [[0]*1 for _ in range(7)]
    W = [[0]*1 for _ in range(14)]
    
    # U[i,0] = A[i,0]
    # V[0,k] = B[0,k]
    # W[k,i] = 1
    
    for i in range(2):
        U[i][0] = 1
    for k in range(7):
        V[k][0] = 1
    for k in range(7):
        for i in range(2):
            W[k*2+i][0] = 1
            
    return U, V, W, 1

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    T, n, m, p = _build_tensor()
    
    total_rank = 26
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j in {0,1}, k in {0,1,2}. 2x2x3. Rank 11.
    U1, V1, W1, r1 = _decomp_2x2x3()
    for r in range(r1):
        for i in range(2):
            for j in range(2):
                U[i * m + j][col] += U1[i * 2 + j][r]
        for j in range(2):
            for k in range(3):
                V[j * p + k][col] += V1[j * 3 + k][r]
        for k in range(3):
            for i in range(2):
                W[k * n + i][col] += W1[k * 2 + i][r]
    col += r1
    
    # Part 2: j in {0,1}, k in {3,4}. 2x2x2. Rank 7.
    U2, V2, W2, r2 = _strassen_2x2x2()
    for r in range(r2):
        for i in range(2):
            for j in range(2):
                U[i * m + j][col] += U2[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[j * p + (3 + k)][col] += V2[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[(3 + k) * n + i][col] += W2[k * 2 + i][r]
    col += r2
    
    # Part 3: j in {0,1}, k in {5,6}. 2x2x2. Rank 7.
    U3, V3, W3, r3 = _strassen_2x2x2()
    for r in range(r3):
        for i in range(2):
            for j in range(2):
                U[i * m + j][col] += U3[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[j * p + (5 + k)][col] += V3[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[(5 + k) * n + i][col] += W3[k * 2 + i][r]
    col += r3
    
    # Part 4: j=2, k in {0..6}. 2x1x7. Rank 1.
    U4, V4, W4, r4 = _decomp_2x1x7()
    for r in range(r4):
        for i in range(2):
            U[i * m + 2][col] += U4[i][r]
        for k in range(7):
            V[2 * p + k][col] += V4[k][r]
        for k in range(7):
            for i in range(2):
                W[k * n + i][col] += W4[k * 2 + i][r]
    col += r4
    
    if col != total_rank:
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    if not _verify(U, V, W, n, m, p, T):
        # Fallback
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
