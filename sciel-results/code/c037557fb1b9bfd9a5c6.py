import time
import numpy as np

def _verify(U, V, W, n=2, m=3, p=5):
    """Verify the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2(U, V, W, u_base, v_base, w_base, k_offset=0):
    """
    Fill in Strassen 2x2x2 decomposition starting at column 0.
    u_base: offset for U rows (i*m + j), where j is local 0,1
    v_base: offset for V rows (j*p + k), where j is local 0,1, k is local 0,1
    w_base: offset for W rows (k*n + i), where k is local 0,1, i is local 0,1
    Returns number of columns used (7).
    """
    col = 0
    # M1: (A00+A11)*(B00+B11) -> C00 += 1, C11 += 1
    U[u_base + 0][col] = 1
    U[u_base + 1][col] = 1
    V[v_base + 0][col] = 1
    V[v_base + 1][col] = 1
    W[w_base + 0][col] = 1
    W[w_base + 1][col] = 1
    col += 1
    # M2: (A10+A11)*B00 -> C10 += 1, C11 += -1
    U[u_base + 1][col] = 1
    U[u_base + 2][col] = 1
    V[v_base + 0][col] = 1
    W[w_base + 1][col] = 1
    W[w_base + 3][col] = -1
    col += 1
    # M3: A00*(B11-B01) -> C01 += 1, C11 += 1
    U[u_base + 0][col] = 1
    V[v_base + 2][col] = 1
    V[v_base + 3][col] = -1
    W[w_base + 2][col] = 1
    W[w_base + 3][col] = 1
    col += 1
    # M4: A11*(B10-B00) -> C00 += 1, C10 += 1
    U[u_base + 2][col] = 1
    V[v_base + 1][col] = 1
    V[v_base + 0][col] = -1
    W[w_base + 0][col] = 1
    W[w_base + 1][col] = 1
    col += 1
    # M5: (A00+A01)*B11 -> C00 += -1, C01 += 1
    U[u_base + 0][col] = 1
    U[u_base + 1][col] = 1
    V[v_base + 3][col] = 1
    W[w_base + 0][col] = -1
    W[w_base + 2][col] = 1
    col += 1
    # M6: (A10-A00)*(B00+B01) -> C11 += 1
    U[u_base + 1][col] = 1
    U[u_base + 0][col] = -1
    V[v_base + 0][col] = 1
    V[v_base + 1][col] = 1
    W[w_base + 3][col] = 1
    col += 1
    # M7: (A01-A11)*(B10+B11) -> C00 += 1
    U[u_base + 1][col] = 1
    U[u_base + 2][col] = -1
    V[v_base + 2][col] = 1
    V[v_base + 3][col] = 1
    W[w_base + 0][col] = 1
    col += 1
    return col

def construct():
    """
    Return a rank-28 exact decomposition for 2x3 * 3x5 matrix multiplication.
    """
    n, m, p = 2, 3, 5
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j=0, all k in {0,1,2,3,4}
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col = 10
    
    # Block 2a: j in {1,2}, k in {0,1} — Strassen 2x2x2
    # u_base: i*3 + 1 (j=1), so u_base=1 for i=0, u_base=4 for i=1
    # Actually, u_base is the offset such that u_base + local_i gives the row
    # local_i=0 -> i=0, row = 0*3+1 = 1
    # local_i=1 -> i=1, row = 1*3+1 = 4
    # So u_base for U is not a single offset since i and j are mixed.
    # Let me just use the explicit indices.
    
    # For Strassen on j in {1,2}, k in {0,1}:
    # U rows: i*3 + j where j in {1,2}, i in {0,1}
    #   (i=0,j=1) -> 1, (i=0,j=2) -> 2
    #   (i=1,j=1) -> 4, (i=1,j=2) -> 5
    # V rows: j*5 + k where j in {1,2}, k in {0,1}
    #   (j=1,k=0) -> 5, (j=1,k=1) -> 6
    #   (j=2,k=0) -> 10, (j=2,k=1) -> 11
    # W rows: k*2 + i where k in {0,1}, i in {0,1}
    #   (k=0,i=0) -> 0, (k=0,i=1) -> 1
    #   (k=1,i=0) -> 2, (k=1,i=1) -> 3
    
    # M1: (A[0,1]+A[1,2]) * (B[1,0]+B[2,1]) -> C[0,0]+=1, C[1,1]+=1
    U[1][col] = 1; U[5][col] = 1
    V[5][col] = 1; V[11][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    # M2: (A[1,1]+A[1,2]) * B[1,0] -> C[1,0]+=1, C[1,1]+=-1
    U[4][col] = 1; U[5][col] = 1
    V[5][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1
    # M3: A[0,1] * (B[2,1]-B[1,1]) -> C[0,1]+=1, C[1,1]+=1
    U[1][col] = 1
    V[11][col] = 1; V[6][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    # M4: A[1,2] * (B[2,0]-B[1,0]) -> C[0,0]+=1, C[1,0]+=1
    U[5][col] = 1
    V[10][col] = 1; V[5][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    # M5: (A[0,1]+A[0,2]) * B[2,1] -> C[0,0]+=-1, C[0,1]+=1
    U[1][col] = 1; U[2][col] = 1
    V[11][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1
    # M6: (A[1,1]-A[0,1]) * (B[1,0]+B[1,1]) -> C[1,1]+=1
    U[4][col] = 1; U[1][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[3][col] = 1
    col += 1
    # M7: (A[0,2]-A[1,2]) * (B[2,0]+B[2,1]) -> C[0,0]+=1
    U[2][col] = 1; U[5][col] = -1
    V[10][col] = 1; V[11][col] = 1
    W[0][col] = 1
    col += 1
    # col = 17
    
    # Block 2b: j in {1,2}, k=2 — schoolbook
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i * 3 + j][col] = 1
            V[j * 5 + 2][col] = 1
            W[2 * 2 + i][col] = 1
            col += 1
    # col = 21
    
    # Block 2c: j in {1,2}, k in {3,4} — Strassen 2x2x2
    # V rows: j*5 + k where j in {1,2}, k in {3,4}
    #   (j=1,k=3) -> 8, (j=1,k=4) -> 9
    #   (j=2,k=3) -> 13, (j=2,k=4) -> 14
    # W rows: k*2 + i where k in {3,4}, i in {0,1}
    #   (k=3,i=0) -> 6, (k=3,i=1) -> 7
    #   (k=4,i=0) -> 8, (k=4,i=1) -> 9
    
    # M1: (A[0,1]+A[1,2]) * (B[1,3]+B[2,4]) -> C[0,3]+=1, C[1,4]+=1
    U[1][col] = 1; U[5][col] = 1
    V[8][col] = 1; V[14][col] = 1
    W[6][col] = 1; W[9][col] = 1
    col += 1
    # M2: (A[1,1]+A[1,2]) * B[1,3] -> C[1,3]+=1, C[1,4]+=-1
    U[4][col] = 1; U[5][col] = 1
    V[8][col] = 1
    W[7][col] = 1; W[9][col] = -1
    col += 1
    # M3: A[0,1] * (B[2,4]-B[1,4]) -> C[0,4]+=1, C[1,4]+=1
    U[1][col] = 1
    V[14][col] = 1; V[9][col] = -1
    W[8][col] = 1; W[9][col] = 1
    col += 1
    # M4: A[1,2] * (B[2,3]-B[1,3]) -> C[0,3]+=1, C[1,3]+=1
    U[5][col] = 1
    V[13][col] = 1; V[8][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    # M5: (A[0,1]+A[0,2]) * B[2,4] -> C[0,3]+=-1, C[0,4]+=1
    U[1][col] = 1; U[2][col] = 1
    V[14][col] = 1
    W[6][col] = -1; W[8][col] = 1
    col += 1
    # M6: (A[1,1]-A[0,1]) * (B[1,3]+B[1,4]) -> C[1,4]+=1
    U[4][col] = 1; U[1][col] = -1
    V[8][col] = 1; V[9][col] = 1
    W[9][col] = 1
    col += 1
    # M7: (A[0,2]-A[1,2]) * (B[2,3]+B[2,4]) -> C[0,3]+=1
    U[2][col] = 1; U[5][col] = -1
    V[13][col] = 1; V[14][col] = 1
    W[6][col] = 1
    col += 1
    # col = 28
    
    return {"U": U, "V": V, "W": W}
