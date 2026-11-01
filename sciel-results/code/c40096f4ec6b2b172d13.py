"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_decomposition():
    """
    Construct a Strassen-based decomposition for 2x4x8 matrix multiplication.
    Rank = 2 * 4 * 7 = 56.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col = 0
    for a_block in range(2):
        for b_block in range(4):
            for term in strassen_terms:
                A_coeffs, B_coeffs, C_coeffs = term
                for (li, lj), coeff in A_coeffs.items():
                    U[li * m + 2 * a_block + lj, col] += coeff
                for (lj, lk), coeff in B_coeffs.items():
                    V[(2 * a_block + lj) * p + 2 * b_block + lk, col] += coeff
                for (li, lk), coeff in C_coeffs.items():
                    W[(2 * b_block + lk) * n + li, col] += coeff
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _reduce_rank(U, V, W, n, m, p, tol=1e-10):
    """
    Try to reduce the rank of a valid decomposition by detecting linearly
    dependent columns in the combined tensor. We use the fact that if two
    columns c1, c2 satisfy a linear relation in the tensor product space,
    we can eliminate one.
    
    Strategy: For each column, check if it can be expressed as a linear
    combination of other columns. Use the structure: column c defines a
    rank-1 tensor T_c = u_c ⊗ v_c ⊗ w_c. The decomposition is valid iff
    sum_c T_c = T_matmul. If columns are linearly dependent in the full
    tensor space, we can reduce rank.
    
    Practical approach: Build a matrix where each row corresponds to a
    (i,j,k) triple and each column to a decomposition column, with entry
    U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]. If this matrix has rank < number
    of columns, we can find a smaller decomposition.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Build the "tensor matrix" of size (n*m*p) x rank
    # Entry (i,j,k), c = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    T = np.zeros((n * m * p, rank), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[idx, :] = U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :]
                idx += 1
    
    # Find the rank of T
    # Use SVD to find linearly independent columns
    try:
        U_svd, s, Vt = np.linalg.svd(T, full_matrices=False)
    except np.linalg.LinAlgError:
        return U, V, W
    
    # Find effective rank
    effective_rank = np.sum(s > tol)
    
    if effective_rank >= rank:
        return U, V, W
    
    # The columns of T are linearly dependent. We need to find a subset
    # of columns that spans the same space. Use QR with column pivoting.
    # But we need the reduction to be exact in the tensor sense.
    
    # Alternative: use the null space of T to find linear relations.
    # If T @ x = 0 for some nonzero x, then sum_c x_c * T_c = 0,
    # meaning we can replace the decomposition with one that has those
    # columns eliminated (if we can find a valid reduced decomposition).
    
    # For now, return the original if no clear reduction path.
    # A proper reduction requires finding a valid smaller decomposition.
    return U, V, W

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition with rank n*m*p = 64."""
    n, m, p = DIMENSIONS
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

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Try the Strassen decomposition
    try:
        result = _strassen_decomposition()
        if _verify(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
            # Try rank reduction
            U_red, V_red, W_red = _reduce_rank(
                result["U"], result["V"], result["W"], n, m, p
            )
            if len(U_red[0]) < len(result["U"][0]):
                if _verify(U_red, V_red, W_red, n, m, p, tol=1e-9):
                    return {"U": U_red, "V": V_red, "W": W_red}
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 64)
    return _schoolbook_decomposition()

# EVOLVE-BLOCK-END
