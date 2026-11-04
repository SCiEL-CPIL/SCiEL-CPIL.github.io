import time
import numpy as np

START = time.time()
DEADLINE = START + 130.0

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # M3 = A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # M4 = A11(B10-B00)
    U[3,3]=1; V[0,3]=-1; V[2,3]=1; W[0,3]=1; W[1,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6 = (A10-A00)(B00+B01)
    U[0,5]=-1; U[2,5]=1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_2x2x4():
    """Build 2x2x4 decomposition using shared Strassen U, rank 7."""
    U2, V2, W2 = _strassen_2x2x2()
    # U is 4x7 (same for both k-blocks)
    # V is 8x7: first 4 rows for k=0,1; next 4 rows for k=2,3
    # W is 8x7: first 4 rows for k=0,1; next 4 rows for k=2,3
    
    rank = 7
    U = U2.copy()  # 4x7
    V = np.zeros((8, rank), dtype=np.int16)
    W = np.zeros((8, rank), dtype=np.int16)
    
    # For k=0,1: use V2 and W2 directly
    V[0:4, :] = V2
    W[0:4, :] = W2
    
    # For k=2,3: same V2 and W2 (since B structure is the same)
    V[4:8, :] = V2
    W[4:8, :] = W2
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = 1 for matrix mult C[i,k] = sum_j A[i,j]*B[j,k]
    # Check: for each rank component r, U[:,r] * V[:,r] * W[:,r] should sum to T
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    T[i,j,k] += U[i,r] * V[j,r] * W[k,r]
    
    # Expected: T[i,j,k] = 1 if i corresponds to (i_row, i_col), 
    # j to (j_row, j_col), k to (k_row, k_col) where i_col == j_row and k_row == i_row
    # Indexing: i = i_row*n + i_col? No, i is in [0, n*m)
    # Standard: A is n×m, B is m×p, C is n×p
    # U index: i in [0, n*m) -> (i//m, i%m) = (row_A, col_A)
    # V index: j in [0, m*p) -> (j//p, j%p) = (row_B, col_B)  
    # W index: k in [0, p*n) -> (k//n, k%n) = (row_C, col_C)
    # T[i,j,k] = 1 iff col_A == row_B AND row_C == row_A
    
    valid = True
    for i in range(n*m):
        i_row, i_col = i // m, i % m
        for j in range(m*p):
            j_row, j_col = j // p, j % p
            for k in range(p*n):
                k_row, k_col = k // n, k % n
                expected = 1 if (i_col == j_row and k_row == i_row) else 0
                if T[i,j,k] != expected:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    return valid

def _build_4x4x8_kron():
    """Build 4x4x8 via Kronecker product of 2x2x4 and 2x2x2."""
    U1, V1, W1 = _build_2x2x4()  # 2x2x4, rank 7
    U2, V2, W2 = _strassen_2x2x2()  # 2x2x2, rank 7
    
    r1, r2 = 7, 7
    rank = r1 * r2  # 49
    
    n1, m1, p1 = 2, 2, 4
    n2, m2, p2 = 2, 2, 2
    
    U = np.zeros((n1*m1 * n2*m2, rank), dtype=np.int16)
    V = np.zeros((m1*p1 * m2*p2, rank), dtype=np.int16)
    W = np.zeros((p1*n1 * p2*n2, rank), dtype=np.int16)
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(n1*m1):
                for i2 in range(n2*m2):
                    U[i1 * (n2*m2) + i2, c] = U1[i1, c1] * U2[i2, c2]
            for j1 in range(m1*p1):
                for j2 in range(m2*p2):
                    V[j1 * (m2*p2) + j2, c] = V1[j1, c1] * V2[j2, c2]
            for k1 in range(p1*n1):
                for k2 in range(p2*n2):
                    W[k1 * (p2*n2) + k2, c] = W1[k1, c1] * W2[k2, c2]
    
    return U, V, W

def construct():
    """
    Construct rank-49 decomposition for 4x4 * 4x8 matrix multiplication.
    Uses Kronecker product of 2x2x4 (rank 7) and 2x2x2 (rank 7).
    """
    if time.time() > DEADLINE:
        # Fallback: return the known-good rank-98 decomposition
        return _construct_fallback()
    
    n, m, p = 4, 4, 8
    
    U, V, W = _build_4x4x8_kron()
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # If verification fails, fall back to rank 98
        return _construct_fallback()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_fallback():
    """Fallback rank-98 decomposition."""
    U2, V2, W2 = _strassen_2x2x2()
    rank49 = 49
    U4 = np.zeros((16, rank49), dtype=np.int16)
    V4 = np.zeros((16, rank49), dtype=np.int16)
    W4 = np.zeros((16, rank49), dtype=np.int16)
    for c0 in range(7):
        for c1 in range(7):
            c = c0*7 + c1
            for i in range(4):
                for j in range(4):
                    U4[i*4+j, c] = U2[(i//2)*2+(j//2), c0] * U2[(i%2)*2+(j%2), c1]
            for j in range(4):
                for k in range(4):
                    V4[j*4+k, c] = V2[(j//2)*2+(k//2), c0] * V2[(j%2)*2+(k%2), c1]
            for k in range(4):
                for i in range(4):
                    W4[k*4+i, c] = W2[(k//2)*2+(i//2), c0] * W2[(k%2)*2+(i%2), c1]
    
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V4[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W4[k*4 + i, :]
            
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4, 8):
            V[j*8 + k, 49:98] = V4[j*4 + (k-4), :]
    for k in range(4, 8):
        for i in range(4):
            W[k*4 + i, 49:98] = W4[(k-4)*4 + i, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
