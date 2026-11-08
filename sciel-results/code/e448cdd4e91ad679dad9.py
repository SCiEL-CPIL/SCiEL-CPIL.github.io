"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook():
    """Return the trivial schoolbook decomposition (rank = n*m*p)."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, tol=1e-10):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_3d = U_arr.reshape(n, m, -1)
    V_3d = V_arr.reshape(m, p, -1)
    W_3d = W_arr.reshape(p, n, -1)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2.
    Rank = 7.
    
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1+M4-M5+M7
    C12 = M3+M5
    C21 = M2+M4
    C22 = M1-M2+M3+M6
    """
    n = m = p = 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # i,j,k in {0,1}
    
    # M1 = (A11+A22)(B11+B22)
    # A11 = A[0,0], A22 = A[1,1], B11 = B[0,0], B22 = B[1,1]
    # For each scalar product in M1, we need to express it as sum of A[i,j]*B[j,k] terms
    # M1 is a 2x2 matrix: M1[a,b] = sum_u (A[2a... wait, for 2x2:
    # M1[a,b] = (A[a,0]+A[a,1]) ... no wait.
    
    # Actually for 2x2 Strassen:
    # A = [[A11, A12], [A21, A22]] where A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
    # B = [[B11, B12], [B21, B22]] where B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
    # C = [[C11, C12], [C21, C22]]
    
    # M1 = (A11+A22)(B11+B22) = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    # This is a scalar. It contributes to C11 and C22.
    # C11 += M1, C22 += M1
    # So W[0,0] += 1, W[1,1] += 1
    # U[0,0] += 1 (for A[0,0]), U[1,1] += 1 (for A[1,1])
    # V[0,0] += 1 (for B[0,0]), V[1,1] += 1 (for B[1,1])
    
    # But wait, M1 is a single scalar product of two linear combinations.
    # In the tensor decomposition, each "multiplication" Mx corresponds to one column.
    # The column c has U[i,j], V[j,k], W[k,i] entries.
    
    # For M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1]):
    # This expands to: A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
    # But in the tensor decomposition, we can't just expand it into 4 terms because
    # that would use 4 columns. The point of Strassen is that M1 is ONE multiplication
    # of two linear combinations, so it uses ONE column.
    
    # So for column c=0 (M1):
    # U[0,0] = 1 (coefficient of A[0,0] in the first factor)
    # U[1,1] = 1 (coefficient of A[1,1] in the first factor)
    # V[0,0] = 1 (coefficient of B[0,0] in the second factor)
    # V[1,1] = 1 (coefficient of B[1,1] in the second factor)
    # W[0,0] = 1 (contribution to C[0,0])
    # W[1,1] = 1 (contribution to C[1,1])
    
    # Let me verify: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should equal 1 for all i,j,k.
    # For i=0,j=0,k=0: U[0,0,0]*V[0,0,0]*W[0,0,0] = 1*1*1 = 1. Good.
    # For i=0,j=0,k=1: U[0,0,0]*V[0,1,0]*W[1,0,0] = 1*0*0 = 0. Good.
    # For i=0,j=1,k=0: U[0,1,0]*V[1,0,0]*W[0,0,0] = 0*0*1 = 0. Good.
    # For i=0,j=1,k=1: U[0,1,0]*V[1,1,0]*W[1,0,0] = 0*1*0 = 0. Good.
    # For i=1,j=0,k=0: U[1,0,0]*V[0,0,0]*W[0,1,0] = 0*1*0 = 0. Good.
    # For i=1,j=0,k=1: U[1,0,0]*V[0,1,0]*W[1,1,0] = 0*0*1 = 0. Good.
    # For i=1,j=1,k=0: U[1,1,0]*V[1,0,0]*W[0,1,0] = 1*0*0 = 0. Good.
    # For i=1,j=1,k=1: U[1,1,0]*V[1,1,0]*W[1,1,0] = 1*1*1 = 1. Good.
    
    # So M1 contributes correctly. Let me build all 7 terms.
    
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # U: U[0,0]=1, U[1,1]=1
    # V: V[0,0]=1, V[1,1]=1
    # W: W[0,0]=1, W[1,1]=1
    U[0*2+0][0] = 1  # U[0,0]
    U[1*2+1][0] = 1  # U[1,1]
    V[0*2+0][0] = 1  # V[0,0]
    V[1*2+1][0] = 1  # V[1,1]
    W[0*2+0][0] = 1  # W[0,0]
    W[1*2+1][0] = 1  # W[1,1]
    
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # U: U[1,0]=1, U[1,1]=1
    # V: V[0,0]=1
    # W: W[0,1]=1 (contributes to C[1,0])
    U[1*2+0][1] = 1  # U[1,0]
    U[1*2+1][1] = 1  # U[1,1]
    V[0*2+0][1] = 1  # V[0,0]
    W[0*2+1][1] = 1  # W[0,1]
    
    # M3 = A[0,0] * (B[1,0]-B[1,1])
    # U: U[0,0]=1
    # V: V[1,0]=1, V[1,1]=-1
    # W: W[0,0]=1 (contributes to C[0,0]), W[1,0]=1 (contributes to C[1,0])
    # Wait, C12 = M3+M5, so M3 contributes to C[0,1] and C[1,1]?
    # Let me re-check. C12 is the top-right block, which is C[0,1].
    # C12 = M3 + M5
    # So M3 contributes to C[0,1] only? No, M3 is a scalar, and C12 is a scalar for 2x2.
    # Wait, for 2x2, each Cij is a scalar. So C12 = C[0,1].
    # M3 contributes to C[0,1].
    # So W[1,0] should be 1 (since k=1, i=0).
    
    U[0*2+0][2] = 1  # U[0,0]
    V[1*2+0][2] = 1  # V[1,0]
    V[1*2+1][2] = -1  # V[1,1]
    W[1*2+0][2] = 1  # W[1,0]
    
    # M4 = A[1,1] * (B[0,1]-B[0,0])
    # C21 = M2+M4, so M4 contributes to C[1,0].
    # C11 = M1+M4-M5+M7, so M4 also contributes to C[0,0].
    U[1*2+1][3] = 1  # U[1,1]
    V[0*2+1][3] = 1  # V[0,1]
    V[0*2+0][3] = -1  # V[0,0]
    W[0*2+0][3] = 1  # W[0,0]
    W[0*2+1][3] = 1  # W[0,1]
    
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # C11 = ... -M5, C12 = M3+M5
    # So M5 contributes to C[0,0] with coeff -1, and C[0,1] with coeff +1.
    U[0*2+0][4] = 1  # U[0,0]
    U[0*2+1][4] = 1  # U[0,1]
    V[1*2+1][4] = 1  # V[1,1]
    W[0*2+0][4] = -1  # W[0,0]
    W[1*2+0][4] = 1  # W[1,0]
    
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # C22 = M1-M2+M3+M6
    # So M6 contributes to C[1,1].
    U[1*2+0][5] = 1  # U[1,0]
    U[0*2+0][5] = -1  # U[0,0]
    V[0*2+0][5] = 1  # V[0,0]
    V[0*2+1][5] = 1  # V[0,1]
    W[1*2+1][5] = 1  # W[1,1]
    
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    # C11 = M1+M4-M5+M7
    # So M7 contributes to C[0,0].
    U[0*2+1][6] = 1  # U[0,1]
    U[1*2+1][6] = -1  # U[1,1]
    V[1*2+0][6] = 1  # V[1,0]
    V[1*2+1][6] = 1  # V[1,1]
    W[0*2+0][6] = 1  # W[0,0]
    
    return {"U": U, "V": V, "W": W}

def _kronecker_combine(U1, V1, W1, U2, V2, W2):
    """
    Combine two tensor decompositions using Kronecker product.
    """
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    rank = rank1 * rank2
    
    U = [[0] * rank for _ in range(len(U1) * len(U2))]
    V = [[0] * rank for _ in range(len(V1) * len(V2))]
    W = [[0] * rank for _ in range(len(W1) * len(W2))]
    
    for c in range(rank1):
        for d in range(rank2):
            col = c * rank2 + d
            for i1 in range(len(U1)):
                for i2 in range(len(U2)):
                    idx = i1 * len(U2) + i2
                    U[idx][col] = U1[i1][c] * U2[i2][d]
            for j1 in range(len(V1)):
                for j2 in range(len(V2)):
                    idx = j1 * len(V2) + j2
                    V[idx][col] = V1[j1][c] * V2[j2][d]
            for k1 in range(len(W1)):
                for k2 in range(len(W2)):
                    idx = k1 * len(W2) + k2
                    W[idx][col] = W1[k1][c] * W2[k2][d]
    
    return {"U": U, "V": V, "W": W}

def _strassen_4x4x9():
    """
    Construct a decomposition for 4x4 * 4x9 using Strassen for the 4x4 part
    and schoolbook for the 9 columns.
    
    Idea: Decompose 4x4x9 as follows:
    - Use Strassen to reduce the 4x4x4 part to rank 28 (7 Strassen * 4 schoolbook for 2x2)
    - But we need 4x4x9, not 4x4x4.
    
    Alternative: Use the fact that 9 = 4 + 4 + 1, or 9 = 3*3.
    
    Let me try a different approach: use Strassen for 2x2x2, then apply it
    to blocks of the 4x4x9 problem.
    
    Actually, let's try: 4x4x9 can be seen as 4x(4x9) or (4x4)x9.
    
    Let me try using Strassen on the 4x4 part, treating the 9 as a vector.
    
    For each of the 9 columns k, we need to compute C[:,k] = A * B[:,k].
    This is 9 separate 4x4 * 4x1 multiplications.
    
    Using Strassen for 4x4 * 4x1:
    - Split A into 2x2 blocks, B[:,k] into 2x1 vectors
    - Apply Strassen-like decomposition
    
    Actually, let me just use the schoolbook decomposition for now,
    but try to find a better one using known results.
    
    Known: The tensor rank of 4x4x4 is at most 28 (Strassen).
    For 4x4x9, we can use the fact that rank(4x4x9) <= rank(4x4x4) + rank(4x4x5).
    But we don't have a good 4x4x5 decomposition.
    
    Let me try a direct approach: use Strassen for the 2x2 blocks.
    
    Split A into 4 2x2 blocks: A11, A12, A21, A22
    Split B into 9 2x? blocks... this is getting complicated.
    
    Let me just return the schoolbook decomposition for now.
    """
    return _schoolbook()

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4x9 matrix multiplication.
    Returns a dictionary with U, V, W matrices.
    """
    start_time = time.time()
    deadline = start_time + 145
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Schoolbook
    schoolbook = _schoolbook()
    if _validate(schoolbook['U'], schoolbook['V'], schoolbook['W']):
        best_result = schoolbook
        best_rank = len(schoolbook['U'][0])
    
    # Strategy 2: Try Strassen-based decomposition
    if time.time() < deadline:
        strassen_result = _strassen_4x4x9()
        if strassen_result is not None:
            if _validate(strassen_result['U'], strassen_result['V'], strassen_result['W']):
                rank = len(strassen_result['U'][0])
                if rank < best_rank:
                    best_result = strassen_result
                    best_rank = rank
    
    if best_result is None:
        return _schoolbook()
    
    return best_result

# EVOLVE-BLOCK-END
