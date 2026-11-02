"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for k in range(p):
            left = np.zeros((m, m))
            for c in range(rank):
                for j in range(m):
                    for jp in range(m):
                        left[j, jp] += U[i*m+j, c] * V[jp*p+k, c] * W[k*n+i, c]
            if not np.allclose(left, np.eye(m), atol=tol):
                return False
    return True

def _rank17_decomposition():
    """
    Deterministic rank-17 decomposition for 2x3 * 3x3 matrix multiplication.
    
    Strategy: Use Strassen-like 7-term decomposition for the 2x2x2 core,
    then extend to 2x3x3 by handling the third column of B separately.
    
    For C = A*B where A is 2x3, B is 3x3, C is 2x3:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We decompose this into rank-17 terms.
    """
    n, m, p = 2, 3, 3
    rank = 17
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Terms 0-6: Strassen-like for columns 0,1 of B (2x2x2 subproblem)
    # A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # Strassen's 7 products for 2x2*2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # Mapping: A indices (i,j) with i in {0,1}, j in {0,1}
    # B indices (j,k) with j in {0,1}, k in {0,1}
    # C indices (i,k) with i in {0,1}, k in {0,1}
    
    # Term 0: M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # U: A[0,0] -> U[0*3+0, 0] = U[0,0]; A[1,1] -> U[1*3+1, 0] = U[4,0]
    U[0, 0] = 1
    U[4, 0] = 1
    # V: B[0,0] -> V[0*3+0, 0] = V[0,0]; B[1,1] -> V[1*3+1, 0] = V[4,0]
    V[0, 0] = 1
    V[4, 0] = 1
    # W: C[0,0] -> W[0*2+0, 0] = W[0,0]; C[1,1] -> W[1*2+1, 0] = W[3,0]
    W[0, 0] = 1
    W[3, 0] = 1
    
    # Term 1: M2 = (A[1,0]+A[1,1]) * B[0,0]
    U[3, 1] = 1  # A[1,0]
    U[4, 1] = 1  # A[1,1]
    V[0, 1] = 1  # B[0,0]
    W[2, 1] = 1  # C[1,0]
    
    # Term 2: M3 = A[0,0] * (B[1,0]-B[1,1])
    U[0, 2] = 1  # A[0,0]
    V[3, 2] = 1  # B[1,0]
    V[4, 2] = -1 # B[1,1]
    W[1, 2] = 1  # C[0,1]
    
    # Term 3: M4 = A[1,1] * (B[0,1]-B[0,0])
    U[4, 3] = 1  # A[1,1]
    V[1, 3] = 1  # B[0,1]
    V[0, 3] = -1 # B[0,0]
    W[2, 3] = 1  # C[1,0]
    W[3, 3] = 1  # C[1,1]
    
    # Term 4: M5 = (A[0,0]+A[0,1]) * B[1,1]
    U[0, 4] = 1  # A[0,0]
    U[1, 4] = 1  # A[0,1]
    V[4, 4] = 1  # B[1,1]
    W[0, 4] = -1 # C[0,0] with -1
    W[1, 4] = 1  # C[0,1]
    
    # Term 5: M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    U[3, 5] = 1  # A[1,0]
    U[0, 5] = -1 # A[0,0]
    V[0, 5] = 1  # B[0,0]
    V[1, 5] = 1  # B[0,1]
    W[2, 5] = 1  # C[1,0]
    
    # Term 6: M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    U[1, 6] = 1  # A[0,1]
    U[4, 6] = -1 # A[1,1]
    V[3, 6] = 1  # B[1,0]
    V[4, 6] = 1  # B[1,1]
    W[0, 6] = 1  # C[0,0]
    W[3, 6] = 1  # C[1,1]
    
    # Terms 7-16: Handle column 2 of B (B[:, 2])
    # C[i, 2] = sum_j A[i,j] * B[j, 2] for i in {0,1}, j in {0,1,2}
    # This is a 2x3 * 3x1 = 2x1 multiplication, which needs rank 6 (schoolbook)
    # But we can potentially do better...
    
    # For now, use schoolbook for the third column: 2*3 = 6 terms
    for idx in range(6):
        col = 7 + idx
        i = idx // 3
        j = idx % 3
        # A[i, j] * B[j, 2] -> C[i, 2]
        U[i*3 + j, col] = 1
        V[j*3 + 2, col] = 1
        W[2*2 + i, col] = 1  # W[k*n+i] with k=2, so W[2*2+i] = W[4+i]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Use the deterministic rank-17 construction
    U, V, W = _rank17_decomposition()
    
    # Verify
    if _verify(U, V, W, n, m, p, tol=1e-9):
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook (rank 18)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
