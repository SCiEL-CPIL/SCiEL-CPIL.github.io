"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7).
    Indexing: U[i*2+j, :], V[j*2+k, :], W[k*2+i, :]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_block_decomposition(n, m, p):
    """
    Construct a structured decomposition for n x m x p matrix multiplication
    using slicing and Strassen for 2x2x2 blocks.
    """
    # Dimensions are fixed to 3, 4, 5 for this task
    # n=3, m=4, p=5
    
    # Total rank calculation:
    # Block 1 (2x2x2): 7
    # Block 2 (2x2x3): 7 (for 2x2x2 part) + 4 (for 2x2x1 part) = 11
    # Block 3 (2x2x2): 7
    # Block 4 (2x2x3): 7 + 4 = 11
    # Block 5 (1x2x2): 4
    # Block 6 (1x2x3): 6
    # Block 7 (1x2x2): 4
    # Block 8 (1x2x3): 6
    # Total: 7 + 11 + 7 + 11 + 4 + 6 + 4 + 6 = 56
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    def add_strassen(i_start, j_start, k_start):
        """
        Add Strassen decomposition for A[i_start:i_start+2, j_start:j_start+2] @ B[j_start:j_start+2, k_start:k_start+2]
        """
        nonlocal col
        U_s, V_s, W_s = _strassen_2x2x2()
        for r in range(4):
            i_local = r // 2
            j_local = r % 2
            for c in range(7):
                if U_s[r, c] != 0:
                    U[(i_start + i_local) * m + j_start + j_local, col + c] = U_s[r, c]
        
        for r in range(4):
            j_local = r // 2
            k_local = r % 2
            for c in range(7):
                if V_s[r, c] != 0:
                    V[(j_start + j_local) * p + k_start + k_local, col + c] = V_s[r, c]
        
        for r in range(4):
            k_local = r // 2
            i_local = r % 2
            for c in range(7):
                if W_s[r, c] != 0:
                    W[(k_start + k_local) * n + i_start + i_local, col + c] = W_s[r, c]
        
        col += 7
    
    # Helper to add 2x2x1 block (A is 2x2, B is 2x1, C is 2x1) - trivial rank 4
    def add_2x2x1(i_start, j_start, k_idx):
        """
        Add trivial decomposition for A[i_start:i_start+2, j_start:j_start+2] @ B[j_start:j_start+2, k_idx]
        Result: C[i_start:i_start+2, k_idx]
        """
        nonlocal col
        for ii in range(2):
            for jj in range(2):
                U[(i_start + ii) * m + j_start + jj, col] = 1
                V[(j_start + jj) * p + k_idx, col] = 1
                W[k_idx * n + i_start + ii, col] = 1
                col += 1
                
    # Helper to add trivial 1xa x bxc block
    def add_trivial_1x(a, b, c, i_idx, j_start, k_start):
        """
        Add trivial decomposition for A[i_idx, j_start:j_start+a] @ B[j_start:j_start+a, k_start:k_start+c]
        A is 1xa, B is axc, C is 1xc.
        """
        nonlocal col
        for jj in range(a):
            for kk in range(c):
                U[i_idx * m + j_start + jj, col] = 1
                V[(j_start + jj) * p + k_start + kk, col] = 1
                W[(k_start + kk) * n + i_idx, col] = 1
                col += 1

    # Now assemble the 8 blocks:
    
    # Block 1: i=0,1; j=0,1; k=0,1 -> 2x2x2 Strassen
    add_strassen(0, 0, 0)
    
    # Block 2: i=0,1; j=0,1; k=2,3,4 -> 2x2x3 = 2x2x2 (k=2,3) + 2x2x1 (k=4)
    # 2x2x2 part: A[0:2, 0:2] @ B[0:2, 2:4]
    add_strassen(0, 0, 2)
    # 2x2x1 part: A[0:2, 0:2] @ B[0:2, 4]
    add_2x2x1(0, 0, 4)
    
    # Block 3: i=0,1; j=2,3; k=0,1 -> 2x2x2 Strassen
    add_strassen(0, 2, 0)
    
    # Block 4: i=0,1; j=2,3; k=2,3,4 -> 2x2x3 = 2x2x2 (k=2,3) + 2x2x1 (k=4)
    add_strassen(0, 2, 2)
    add_2x2x1(0, 2, 4)
    
    # Block 5: i=2; j=0,1; k=0,1 -> 1x2x2 trivial
    add_trivial_1x(2, 2, 2, 2, 0, 0)
    
    # Block 6: i=2; j=0,1; k=2,3,4 -> 1x2x3 trivial
    add_trivial_1x(2, 2, 3, 2, 0, 2)
    
    # Block 7: i=2; j=2,3; k=0,1 -> 1x2x2 trivial
    add_trivial_1x(2, 2, 2, 2, 2, 0)
    
    # Block 8: i=2; j=2,3; k=2,3,4 -> 1x2x3 trivial
    add_trivial_1x(2, 2, 3, 2, 2, 2)
    
    assert col == 56, f"Expected rank 56, got {col}"
    
    # Verify
    if not _verify(U, V, W, n, m, p, tol=1e-8):
        return None
    
    return U, V, W

def _construct_schoolbook(n, m, p):
    """Construct the standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try the structured decomposition first
    result = _construct_block_decomposition(n, m, p)
    
    if result is not None:
        U, V, W = result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook
    U, V, W = _construct_schoolbook(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
