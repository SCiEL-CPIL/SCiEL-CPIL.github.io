"""Matrix multiplication 4x4x7 - Strassen-based decomposition for reduced rank."""
import time
import sys

DIMENSIONS = [4, 4, 7]

def _strassen_2x2x2():
    """Return U, V, W matrices for 2x2x2 Strassen decomposition (rank 7).
    
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    where i indexes rows of A (0,1), j indexes cols of A / rows of B, k indexes cols of B.
    
    Strassen formulas:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n = 2
    rank = 7
    U = [[0]*rank for _ in range(n*n)]
    V = [[0]*rank for _ in range(n*n)]
    W = [[0]*rank for _ in range(n*n)]
    
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # For each Strassen product M_c = (sum_a u_{i,j,c} A_{i,j}) * (sum_b v_{j,k,c} B_{j,k})
    # Then C_{i,k} += w_{k,i,c} * M_c
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00 (i=0,j=0) +=1, A11 (i=1,j=1) +=1
    U[0*2+0][0] = 1  # A00
    U[1*2+1][0] = 1  # A11
    # V: B00 (j=0,k=0) +=1, B11 (j=1,k=1) +=1
    V[0*2+0][0] = 1  # B00
    V[1*2+1][0] = 1  # B11
    # W: C00 += M1, C11 += M1
    # C00: i=0,k=0 -> W[0*2+0][0] = W[0][0]
    W[0*2+0][0] = 1
    # C11: i=1,k=1 -> W[1*2+1][0] = W[3][0]
    W[1*2+1][0] = 1
    
    # M2 = (A10+A11)B00
    U[1*2+0][1] = 1  # A10
    U[1*2+1][1] = 1  # A11
    V[0*2+0][1] = 1  # B00
    # C10 = M2: i=1,k=0 -> W[0*2+1][1] = W[1][1]
    W[0*2+1][1] = 1
    
    # M3 = A00(B10-B11)
    U[0*2+0][2] = 1  # A00
    V[1*2+0][2] = 1  # B10
    V[1*2+1][2] = -1  # B11
    # C01 = M3: i=0,k=1 -> W[1*2+0][2] = W[2][2]
    W[1*2+0][2] = 1
    
    # M4 = A11(B01-B00)
    U[1*2+1][3] = 1  # A11
    V[0*2+1][3] = 1  # B01
    V[0*2+0][3] = -1  # B00
    # C00 += M4: i=0,k=0 -> W[0*2+0][3] = W[0][3]
    W[0*2+0][3] = 1
    # C10 += M4: i=1,k=0 -> W[0*2+1][3] = W[1][3]
    W[0*2+1][3] = 1
    
    # M5 = (A00+A01)B11
    U[0*2+0][4] = 1  # A00
    U[0*2+1][4] = 1  # A01
    V[1*2+1][4] = 1  # B11
    # C00 -= M5: i=0,k=0 -> W[0*2+0][4] = W[0][4] = -1
    W[0*2+0][4] = -1
    # C01 += M5: i=0,k=1 -> W[1*2+0][4] = W[2][4] = 1
    W[1*2+0][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[1*2+0][5] = 1  # A10
    U[0*2+0][5] = -1  # A00
    V[0*2+0][5] = 1  # B00
    V[0*2+1][5] = 1  # B01
    # C11 += M6: i=1,k=1 -> W[1*2+1][5] = W[3][5] = 1
    W[1*2+1][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[0*2+1][6] = 1  # A01
    U[1*2+1][6] = -1  # A11
    V[1*2+0][6] = 1  # B10
    V[1*2+1][6] = 1  # B11
    # C00 += M7: i=0,k=0 -> W[0*2+0][6] = W[0][6] = 1
    W[0*2+0][6] = 1
    # C11 += M7: i=1,k=1 -> W[1*2+1][6] = W[3][6] = 1
    W[1*2+1][6] = 1
    
    return U, V, W

def _build_strassen_4x4x4():
    """Build 4x4x4 decomposition using Kronecker product of 2x2 Strassen.
    
    We partition 4x4 matrices into 2x2 blocks:
    A = [[A00, A01], [A10, A11]] where each block is 2x2
    B = [[B00, B01], [B10, B11]]
    
    Then C = A*B is computed using Strassen on the 2x2 block level,
    with each block multiplication being a 2x2x2 Strassen decomposition.
    
    This gives rank 7*7 = 49.
    """
    n = 4
    m = 4
    p = 4
    U_small, V_small, W_small = _strassen_2x2x2()
    
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            # U[i*4+j, c] for i in 0..3, j in 0..3
            # Block (bi, bj) where bi = i//2, bj = j//2, local_i = i%2, local_j = j%2
            # U_small[bi*2+bj, c1] gives the coefficient for A_{bi,bj}
            # V_small[bj*2+bk, c2] gives the coefficient for B_{bk,bj}... wait
            # 
            # Let me think more carefully.
            # 
            # In the 4x4x4 case with 2x2 blocks:
            # U_4x4[i*4+j, c1*7+c2] = sum over block contributions
            # 
            # Actually, the Kronecker structure is:
            # U_4x4[i*4+j, c1*7+c2] = U_small[(i//2)*2+(j//2), c1] * U_small[(i%2)*2+(j%2), c2]
            # V_4x4[j*4+k, c1*7+c2] = V_small[(j//2)*2+(k//2), c1] * V_small[(j%2)*2+(k%2), c2]
            # W_4x4[k*4+i, c1*7+c2] = W_small[(k//2)*2+(i//2), c1] * W_small[(k%2)*2+(i%2), c2]
            
            for i in range(4):
                for j in range(4):
                    bi, bj = i // 2, j // 2
                    li, lj = i % 2, j % 2
                    val = U_small[bi*2+bj][c1] * U_small[li*2+lj][c2]
                    U[i*4+j][c] = val
            
            for j in range(4):
                for k in range(4):
                    bj, bk = j // 2, k // 2
                    lj, lk = j % 2, k % 2
                    val = V_small[bj*2+bk][c1] * V_small[lj*2+lk][c2]
                    V[j*4+k][c] = val
            
            for k in range(4):
                for i in range(4):
                    bk, bi = k // 2, i // 2
                    lk, li = k % 2, i % 2
                    val = W_small[bk*2+bi][c1] * W_small[lk*2+li][c2]
                    W[k*4+i][c] = val
    
    return U, V, W

def _build_schoolbook_4x4x3():
    """Build 4x4x3 schoolbook decomposition (rank 48).
    
    U[i*4+j, c], V[j*3+k, c], W[k*4+i, c]
    """
    n = 4
    m = 4
    p = 3
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return U, V, W

def _concatenate(U1, V1, W1, U2, V2, W2):
    """Concatenate two decompositions side by side (increase rank)."""
    rank1 = len(U1[0]) if U1 else 0
    rank2 = len(U2[0]) if U2 else 0
    rows = len(U1)
    new_rank = rank1 + rank2
    U = [[0]*new_rank for _ in range(rows)]
    # V and W have different numbers of rows, so handle separately
    rowsV = len(V1)
    V = [[0]*new_rank for _ in range(rowsV)]
    rowsW = len(W1)
    W = [[0]*new_rank for _ in range(rowsW)]
    
    for i in range(rows):
        for c in range(rank1):
            U[i][c] = U1[i][c]
        for c in range(rank2):
            U[i][rank1+c] = U2[i][c]
    
    for i in range(rowsV):
        for c in range(rank1):
            V[i][c] = V1[i][c]
        for c in range(rank2):
            V[i][rank1+c] = V2[i][c]
    
    for i in range(rowsW):
        for c in range(rank1):
            W[i][c] = W1[i][c]
        for c in range(rank2):
            W[i][rank1+c] = W2[i][c]
    
    return U, V, W

def _verify(U, V, W):
    """Verify that the decomposition is correct for 4x4x7.
    
    Check that for all i,j,k:
    sum_c U[i*4+j, c] * V[j*3+k, c] * W[k*4+i, c] == (1 if this is a valid matmul term)
    
    Actually, the tensor for matrix multiplication C = A*B where A is 4x4 and B is 4x7:
    T[i,j,k] = 1 if k is a valid output column (k < 7) and j is valid...
    
    Wait, let me re-read the convention.
    U shape: (n*m, rank) = (16, rank), indexed U[i*m+j, c]
    V shape: (m*p, rank) = (28, rank), indexed V[j*p+k, c]  
    W shape: (p*n, rank) = (28, rank), indexed W[k*n+i, c]
    
    The tensor T[i,j,k] for multiplying an (n x m) matrix by an (m x p) matrix to get (n x p):
    T[i,j,k] = 1 for all valid i,j,k (it's the identity tensor in the sense that
    C[i,k] = sum_j A[i,j] * B[j,k])
    
    So the check is: for each (i,j,k) with i in [0,4), j in [0,4), k in [0,7):
    sum_c U[i*4+j, c] * V[j*7+k, c] * W[k*4+i, c] should equal 1.
    """
    n, m, p = 4, 4, 7
    rank = len(U[0])
    
    # For each (i, j, k), check the contraction
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False, (i, j, k, s)
    return True, None

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Build the 4x4x4 Strassen decomposition (rank 49)
    U1, V1, W1 = _build_strassen_4x4x4()
    
    # Build the 4x4x3 schoolbook decomposition (rank 48)
    U2, V2, W2 = _build_schoolbook_4x4x3()
    
    # Concatenate
    U, V, W = _concatenate(U1, V1, W1, U2, V2, W2)
    
    # Verify correctness
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        valid, info = _verify(U, V, W)
        if not valid:
            # Fallback to schoolbook
            return _build_schoolbook_4x4x7()
    
    return {"U": U, "V": V, "W": W}

def _build_schoolbook_4x4x7():
    """Fallback: full schoolbook decomposition for 4x4x7 (rank 112)."""
    n, m, p = 4, 4, 7
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

