import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _construct_2x3x9():
    """
    Construct a tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.
    
    Strategy: Use Strassen's algorithm for 2x2x2 blocks where possible.
    - For each pair of columns (k1,k2) in B, use Strassen (7 terms) for A[0:2, 0:2] * B[0:2, k1:k2]
    - For the last column (k=8), use direct multiplication (4 terms)
    - For the third row of A (row 2), use direct multiplication with all 9 columns of B (18 terms)
    
    Total rank: 4 pairs * 7 + 4 + 18 = 28 + 4 + 18 = 50
    """
    n, m, p = 2, 3, 9
    rank = 50
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen's 7 rank-1 terms for 2x2x2 matrix multiplication
    # Each term: (U_dict, V_dict, W_dict)
    # Keys are (i,j) in local 2x2 coordinates
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00(B10-B00)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11(B01-B10)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, k1:k2] for 4 column pairs
    k_pairs = [(0,1), (2,3), (4,5), (6,7)]
    
    for (k1, k2) in k_pairs:
        for (u_dict, v_dict, w_dict) in strassen_terms:
            # U part: A[li, lj] -> U[li*m + lj, col_idx]
            for (li, lj), coeff in u_dict.items():
                U[li * m + lj, col_idx] = coeff
            
            # V part: B[lj, lk] -> V[lj*p + actual_k, col_idx]
            for (lj, lk), coeff in v_dict.items():
                actual_k = k1 if lk == 0 else k2
                V[lj * p + actual_k, col_idx] = coeff
            
            # W part: C[li, lk] -> W[actual_k*n + li, col_idx]
            for (li, lk), coeff in w_dict.items():
                actual_k = k1 if lk == 0 else k2
                W[actual_k * n + li, col_idx] = coeff
            
            col_idx += 1
    
    # Part 2: Direct multiplication for A[0:2, 0:2] * B[0:2, 8]
    # C[i, 8] = A[i, 0]*B[0, 8] + A[i, 1]*B[1, 8]
    for i in range(2):
        for j in range(2):
            U[i * m + j, col_idx] = 1
            V[j * p + 8, col_idx] = 1
            W[8 * n + i, col_idx] = 1
            col_idx += 1
    
    # Part 3: Direct multiplication for A[0:2, 2] * B[2, 0:9]
    # C[i, k] += A[i, 2] * B[2, k]
    for i in range(2):
        for k in range(9):
            U[i * m + 2, col_idx] = 1
            V[2 * p + k, col_idx] = 1
            W[k * n + i, col_idx] = 1
            col_idx += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _verify_decomposition(U, V, W, n, m, p, atol=1e-10):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Reconstruct the tensor: T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T_computed = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    
    # Expected tensor: T[i*m+j, j*p+k, k*n+i] = 1 for all valid i, j, k
    T_expected = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1.0
    
    return np.allclose(T_computed, T_expected, atol=atol)

def _rank1_decomposition(n, m, p):
    """Trivial rank-1 decomposition with rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x9 matrix.
    
    Returns: {'U': U, 'V': V, 'W': W} where U has shape (n*m, rank),
    V has shape (m*p, rank), W has shape (p*n, rank).
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    try:
        U, V, W = _construct_2x3x9()
        
        # Verify the decomposition
        if time.time() < deadline:
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
            else:
                # Fallback to rank-1 if verification fails
                U, V, W = _rank1_decomposition(n, m, p)
                return {"U": U, "V": V, "W": W}
        else:
            # Return the constructed result (correct by construction)
            return {"U": U, "V": V, "W": W}
            
    except Exception:
        # Fallback to rank-1 decomposition
        U, V, W = _rank1_decomposition(n, m, p)
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
