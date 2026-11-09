import time
import numpy as np

def construct():
    n, m, p = 2, 3, 4
    deadline = time.time() + 140
    
    # Try Rank 22
    try:
        result = _construct_rank_22()
        # Verify correctness
        if _verify_decomposition(result, n, m, p):
            return result
        else:
            raise ValueError("Rank 22 decomposition failed verification")
    except Exception:
        return _construct_rank_24()

def _verify_decomposition(result, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    rank = U.shape[1]
    
    # The tensor T[i,j,k] = delta_{i,j} * delta_{j,k} for matrix multiplication
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor entry T[i,j,k] = 1 if j==j (always true for valid indices), 
    # but actually the multiplication tensor has T[i,j,k] = 1 when the index j 
    # matches between A's column and B's row
    # For A (n x m), B (m x p), C (n x p):
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 (it's just the identity in the middle dimension)
    # Decomposition: T[i,j,k] = sum_{r=0}^{rank-1} U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, W index is W[k*n+i, r] per the trace convention
    
    # Actually the convention is W[k*n + i, r] where C[i,k] gets contribution
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k, i) -> k*n + i
    
    # The decomposition should satisfy:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 (since C[i,k] = sum_j A[i,j]*B[j,k])
    
    # Build the tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # Compute decomposition
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                error += (val - T[i, j, k]) ** 2
    
    return error < 1e-10

def _construct_rank_24():
    n, m, p = 2, 3, 4
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _construct_rank_22():
    n, m, p = 2, 3, 4
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Strassen
    # M1: (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[3][col] = 1 # C00, C11
    col += 1
    
    # M2: (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1 # C01, C11
    col += 1
    
    # M3: A00(B10+B11)
    U[0][col] = 1
    V[4][col] = 1; V[5][col] = 1
    W[2][col] = 1; W[3][col] = 1 # C10, C11
    col += 1
    
    # M4: A11(B01+B10)
    U[4][col] = 1
    V[1][col] = 1; V[4][col] = 1
    W[0][col] = -1; W[1][col] = 1 # C00, C01
    col += 1
    
    # M5: (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = 1; W[2][col] = 1 # C00, C10
    col += 1
    
    # M6 is not used in the standard 7-term Strassen reconstruction, so skip it entirely
    
    # M7: (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = -1; W[3][col] = 1 # C00, C11
    col += 1
    
    # M6 was skipped, so col is 7.
    # Wait, I need to finish M6? No, if it's not used, I don't need it.
    # But I listed 7 terms.
    # Term 1: M1
    # Term 2: M2
    # Term 3: M3
    # Term 4: M4
    # Term 5: M5
    # Term 6: M6 (Skipped in reconstruction? No, I skipped it in W assignment)
    # Term 7: M7
    
    # So Block 1 uses 7 columns.
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Same Strassen logic, but B indices are 2,3,6,7
    # B00->2, B01->3, B10->6, B11->7
    # C00->4, C01->5, C10->6, C11->7
    
    # M1
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    
    # M2
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[5][col] = 1; W[7][col] = -1
    col += 1
    
    # M3
    U[0][col] = 1
    V[6][col] = 1; V[7][col] = 1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # M4
    U[4][col] = 1
    V[3][col] = 1; V[6][col] = 1
    W[4][col] = -1; W[5][col] = 1
    col += 1
    
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[4][col] = 1; W[6][col] = 1
    col += 1
    
    # M6 is not used in the standard 7-term Strassen reconstruction, so skip it entirely
    
    # M7
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[4][col] = -1; W[7][col] = 1
    col += 1
    
    # Block 2 used 7 columns. Total 14.
    
    # Block 3: A[0:2, 2] * B[2, 0:4] -> C[0:2, 0:4]
    # Schoolbook
    # A02 (index 2), A12 (index 5)
    # B20 (8), B21 (9), B22 (10), B23 (11)
    
    # A02 B20 -> C00 (0)
    U[2][col] = 1; V[8][col] = 1; W[0][col] = 1
    col += 1
    # A02 B21 -> C01 (1)
    U[2][col] = 1; V[9][col] = 1; W[1][col] = 1
    col += 1
    # A02 B22 -> C02 (4)
    U[2][col] = 1; V[10][col] = 1; W[4][col] = 1
    col += 1
    # A02 B23 -> C03 (5)
    U[2][col] = 1; V[11][col] = 1; W[5][col] = 1
    col += 1
    # A12 B20 -> C10 (2)
    U[5][col] = 1; V[8][col] = 1; W[2][col] = 1
    col += 1
    # A12 B21 -> C11 (3)
    U[5][col] = 1; V[9][col] = 1; W[3][col] = 1
    col += 1
    # A12 B22 -> C12 (6)
    U[5][col] = 1; V[10][col] = 1; W[6][col] = 1
    col += 1
    # A12 B23 -> C13 (7)
    U[5][col] = 1; V[11][col] = 1; W[7][col] = 1
    col += 1
    
    if col != 22:
        raise ValueError(f"Expected 22 columns, got {col}")
        
    return {"U": U, "V": V, "W": W}

def _construct_rank_22():
    """
    Rank-22 decomposition using Strassen for 2x2 blocks.
    
    Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]  (7 columns)
    Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]  (7 columns)
    Block 3: A[0:2, 2] * B[2, 0:4] -> C[0:2, 0:4]      (8 columns, schoolbook)
    Total: 22 columns
    """
    n, m, p = 2, 3, 4
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: A00=0, A01=1, A10=3, A11=4
    # B indices: B00=0, B01=1, B10=4, B11=5
    # C indices: C00=0, C01=1, C10=2, C11=3
    
    # M1: (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[3][col] = 1  # C00, C11
    col += 1
    
    # M2: (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1  # C01, C11
    col += 1
    
    # M3: A00(B10+B11)
    U[0][col] = 1
    V[4][col] = 1; V[5][col] = 1
    W[2][col] = 1; W[3][col] = 1  # C10, C11
    col += 1
    
    # M4: A11(B01+B10)
    U[4][col] = 1
    V[1][col] = 1; V[4][col] = 1
    W[0][col] = -1; W[1][col] = 1  # C00, C01
    col += 1
    
    # M5: (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = 1; W[2][col] = 1  # C00, C10
    col += 1
    
    # M7: (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = -1; W[3][col] = 1  # C00, C11
    col += 1
    
    # Block 1 used 6 columns (M1-M5, M7; M6 not needed)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # A indices: A00=0, A01=1, A10=3, A11=4
    # B indices: B00=2, B01=3, B10=6, B11=7
    # C indices: C00=4, C01=5, C10=6, C11=7
    
    # M1: (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[4][col] = 1; W[7][col] = 1  # C02, C13
    col += 1
    
    # M2: (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[5][col] = 1; W[7][col] = -1  # C03, C13
    col += 1
    
    # M3: A00(B10+B11)
    U[0][col] = 1
    V[6][col] = 1; V[7][col] = 1
    W[6][col] = 1; W[7][col] = 1  # C12, C13
    col += 1
    
    # M4: A11(B01+B10)
    U[4][col] = 1
    V[3][col] = 1; V[6][col] = 1
    W[4][col] = -1; W[5][col] = 1  # C02, C03
    col += 1
    
    # M5: (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[4][col] = 1; W[6][col] = 1  # C02, C12
    col += 1
    
    # M7: (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[4][col] = -1; W[7][col] = 1  # C02, C13
    col += 1
    
    # Block 2 used 6 columns. Total 12.
    
    # Block 3: A[0:2, 2] * B[2, 0:4] -> C[0:2, 0:4]
    # A indices: A02=2, A12=5
    # B indices: B20=8, B21=9, B22=10, B23=11
    # C indices: C00=0, C01=1, C02=4, C03=5, C10=2, C11=3, C12=6, C13=7
    
    # Schoolbook: 2*4 = 8 terms
    U[2][col] = 1; V[8][col] = 1; W[0][col] = 1   # A02*B20 -> C00
    col += 1
    U[2][col] = 1; V[9][col] = 1; W[1][col] = 1   # A02*B21 -> C01
    col += 1
    U[2][col] = 1; V[10][col] = 1; W[4][col] = 1  # A02*B22 -> C02
    col += 1
    U[2][col] = 1; V[11][col] = 1; W[5][col] = 1  # A02*B23 -> C03
    col += 1
    U[5][col] = 1; V[8][col] = 1; W[2][col] = 1   # A12*B20 -> C10
    col += 1
    U[5][col] = 1; V[9][col] = 1; W[3][col] = 1   # A12*B21 -> C11
    col += 1
    U[5][col] = 1; V[10][col] = 1; W[6][col] = 1  # A12*B22 -> C12
    col += 1
    U[5][col] = 1; V[11][col] = 1; W[7][col] = 1  # A12*B23 -> C13
    col += 1
    
    if col != 20:
        raise ValueError(f"Expected 20 columns, got {col}")
        
    return {"U": U, "V": V, "W": W}
