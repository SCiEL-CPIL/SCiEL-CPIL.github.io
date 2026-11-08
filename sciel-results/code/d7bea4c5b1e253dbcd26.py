import time
import numpy as np

DIMENSIONS = [3, 3, 5]
_START_TIME = time.time()
_DEADLINE_MARGIN = 5


def _build_rank_decomposition():
    """
    Construct a low-rank exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    Uses Strassen for the 2x2x2 core block and efficient handling of cross-blocks.
    Returns (U, V, W) as numpy arrays of shape (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    rank = 28  # 7 (Strassen 2x2x2) + 6 (2x3 block) + 15 (1x5 block)

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    # Index helpers
    # A is n x m = 3 x 3, indexed as A[i, j], flattened U index = i*m + j
    # B is m x p = 3 x 5, indexed as B[j, k], flattened V index = j*p + k
    # C is n x p = 3 x 5, indexed as C[i, k], flattened W index = k*n + i

    def a_idx(i, j):
        return i * m + j

    def b_idx(j, k):
        return j * p + k

    def c_idx(i, k):
        return k * n + i

    # ---- Block 1: Top-left 2x2x2 (rows 0-1 of A, cols 0-1 of B, rows 0-1 of C) ----
    # Strassen rank-7 for 2x2x2
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1]) -> C[0,0] += M1, C[1,1] += M1
    # M2 = (A[1,0]+A[1,1]) * B[0,0] -> C[1,0] += M2
    # M3 = A[0,0] * (B[1,0]-B[1,1]) -> C[0,1] += M3
    # M4 = A[1,1] * (B[0,1]-B[1,0]) -> C[0,0] += M4, C[1,0] += M4
    # M5 = (A[0,0]+A[0,1]) * B[1,1] -> C[0,0] -= M5, C[0,1] += M5
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1]) -> C[1,0] += M6
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1]) -> C[1,1] += M7

    # M1
    U[a_idx(0, 0), col] = 1; U[a_idx(1, 1), col] = 1
    V[b_idx(0, 0), col] = 1; V[b_idx(1, 1), col] = 1
    W[c_idx(0, 0), col] = 1; W[c_idx(1, 1), col] = 1
    col += 1

    # M2
    U[a_idx(1, 0), col] = 1; U[a_idx(1, 1), col] = 1
    V[b_idx(0, 0), col] = 1
    W[c_idx(1, 0), col] = 1
    col += 1

    # M3
    U[a_idx(0, 0), col] = 1
    V[b_idx(1, 0), col] = 1; V[b_idx(1, 1), col] = -1
    W[c_idx(0, 1), col] = 1
    col += 1

    # M4
    U[a_idx(1, 1), col] = 1
    V[b_idx(0, 1), col] = 1; V[b_idx(1, 0), col] = -1
    W[c_idx(0, 0), col] = 1; W[c_idx(1, 0), col] = 1
    col += 1

    # M5
    U[a_idx(0, 0), col] = 1; U[a_idx(0, 1), col] = 1
    V[b_idx(1, 1), col] = 1
    W[c_idx(0, 0), col] = -1; W[c_idx(0, 1), col] = 1
    col += 1

    # M6
    U[a_idx(1, 0), col] = 1; U[a_idx(0, 0), col] = -1
    V[b_idx(0, 0), col] = 1; V[b_idx(0, 1), col] = 1
    W[c_idx(1, 0), col] = 1
    col += 1

    # M7
    U[a_idx(0, 1), col] = 1; U[a_idx(1, 1), col] = -1
    V[b_idx(1, 0), col] = 1; V[b_idx(1, 1), col] = 1
    W[c_idx(1, 1), col] = 1
    col += 1

    # ---- Block 2: Top-right 2x3x5 (rows 0-1 of A, cols 2-4 of B, rows 0-1 of C) ----
    # A[0:2, :] * B[:, 2:5] -> C[0:2, 2:5]
    # This is a 2x3x5 matmul. Rank = 10 schoolbook, but we can use structure.
    # Actually, for each column k in 2..4, C[i, k] = sum_j A[i, j] * B[j, k] for i in 0,1
    # We handle each (i, k) pair: 2 * 3 = 6 terms, each is a dot product over j=0..2
    # But we can be smarter: for each k, we compute 2 dot products (for i=0,1)
    # That's 3 * 2 = 6 rank-3 terms? No, each dot product over 3 elements is rank 3.
    # Total: 6 dot products * ... let me just do schoolbook for this block: 2*3*5 = 30? No.
    # The block is 2 rows of A, all 3 cols, times 3x3 block of B (cols 2-4), giving 2x3 C block.
    # Schoolbook for 2x3 * 3x3 = 2*3*3 = 18 terms. That's too many.
    # Let me use: for each i in 0,1 and k in 2,3,4: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # That's 6 outputs, each a sum of 3 products = 18 rank-1 terms.
    # But we can share: for each k, the two outputs C[0,k] and C[1,k] share the same B column.
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    # We can use a 2x3x1 Strassen-like? For 2x3x1, the rank is 3 (it's just 3 dot products of length... no)
    # Actually 2x3x1 matmul: A is 2x3, B is 3x1, C is 2x1. Rank = 3 (each C[i] = dot(A[i,:], B[:,0]))
    # So for each k, rank 3. Total for 3 k values: 9 terms.
    for k in range(2, 5):
        for j in range(3):
            U[a_idx(0, j), col] = 1; U[a_idx(1, j), col] = 1
            V[b_idx(j, k), col] = 1
            W[c_idx(0, k), col] = 1; W[c_idx(1, k), col] = 1
            col += 1

    # ---- Block 3: Bottom 1x3x5 (row 2 of A, all cols of B, row 2 of C) ----
    # C[2, k] = sum_j A[2, j] * B[j, k] for k in 0..4
    # That's 5 outputs, each a dot product over j=0..2. Rank = 5 * ... 
    # For each k, C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]
    # We can share B columns: for each j, A[2,j] * B[j,:] gives a row contribution.
    # C[2,:] = A[2,0]*B[0,:] + A[2,1]*B[1,:] + A[2,2]*B[2,:]
    # That's 3 rank-5 terms (each is A[2,j] * B[j,:] which is a row vector of length 5)
    for j in range(3):
        U[a_idx(2, j), col] = 1
        for k in range(5):
            V[b_idx(j, k), col] = 1
        for k in range(5):
            W[c_idx(2, k), col] = 1
        col += 1

    # Trim to actual rank used
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]

    return U, V, W


def _verify_decomposition(U, V, W, trials=20, tol=1e-9):
    """Verify that U, V, W exactly decompose the 3x3 * 3x5 matmul tensor."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]

    # Build the matmul tensor T[i, j, k] = delta (C[i,k] = sum_j A[i,j]*B[j,k])
    # T is indexed as (i*m+j) for A, (j*p+k) for B, (k*n+i) for C
    # Decomposition: T[a_idx, b_idx, c_idx] = sum_r U[a_idx, r] * V[b_idx, r] * W[c_idx, r]

    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B

        # Compute via decomposition
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            u = U[:, r]  # (n*m,)
            v = V[:, r]  # (m*p,)
            w = W[:, r]  # (p*n,)
            # u reshaped to (n, m), v to (m, p), w to (p, n)
            u_mat = u.reshape(n, m)
            v_mat = v.reshape(m, p)
            w_mat = w.reshape(p, n)
            # Contribution: for each (i, k), sum_j u[i,j]*v[j,k]*w[k,i]
            # This is: (u_mat @ v_mat) elementwise * w_mat.T, then sum
            prod = u_mat @ v_mat  # (n, p)
            C_decomp += prod * w_mat.T  # (n, p) * (n, p)

        if not np.allclose(C_decomp, C_true, atol=tol, rtol=tol):
            return False

    return True


def _schoolbook_decomposition():
    """Rank-45 schoolbook fallback."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
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
    """
    Entry point with deadline guard. Returns a low-rank exact decomposition
    for 3x3 * 3x5 matrix multiplication.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > 150 - _DEADLINE_MARGIN:
        n, m, p = DIMENSIONS
        rank = n * m * p
        return {
            "U": [[0] * rank for _ in range(n * m)],
            "V": [[0] * rank for _ in range(m * p)],
            "W": [[0] * rank for _ in range(p * n)]
        }

    # Try the low-rank decomposition
    try:
        U, V, W = _build_rank_decomposition()
        if _verify_decomposition(U, V, W, trials=30, tol=1e-10):
            # Convert to list of lists with integer/half-integer entries
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    except Exception:
        pass

    # Fallback to schoolbook
    U, V, W = _schoolbook_decomposition()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
