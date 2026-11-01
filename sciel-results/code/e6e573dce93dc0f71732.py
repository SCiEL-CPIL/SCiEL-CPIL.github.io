"""Combined program: structural rank-56 Strassen construction with fast verification."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    W indexed by k*n + i (output factor follows W[k*n+i, r])
    """
    try:
        U_arr = np.asarray(U, dtype=np.float64)
        V_arr = np.asarray(V, dtype=np.float64)
        W_arr = np.asarray(W, dtype=np.float64)

        if U_arr.ndim != 2 or V_arr.ndim != 2 or W_arr.ndim != 2:
            return False

        rank = U_arr.shape[1]
        if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
            return False
        if U_arr.shape[0] != n * m or V_arr.shape[0] != m * p or W_arr.shape[0] != p * n:
            return False

        # Build a dense tensor view: T[a, b, c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # For efficiency, use einsum for valid triples and spot-check invalid.
        # Valid triples: (i*m+j, j*p+k, k*n+i) must equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_arr[a, :], np.dot(V_arr[b, :], W_arr[c, :]))
                    if np.abs(val - 1.0) > tol:
                        return False

        # Check a sample of invalid triples (all would be too slow for large p)
        # But for exactness, we need all. Use vectorized approach.
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We can compute this as: for each r, outer(U[:,r], outer(V[:,r], W[:,r]))
        # But that's O(rank * n*m * m*p * p*n) which is 56 * 4 * 32 * 32 = 229376 ops -- fine.
        # Actually let's just do the full check with einsum for speed.
        # T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        # This creates a (4, 32, 32) tensor = 4096 entries. Very manageable.
        T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)

        # Check valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if np.abs(T[a, b, c] - 1.0) > tol:
                        return False

        # Check invalid entries are zero
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_set.add((i * m + j, j * p + k, k * n + i))

        # Iterate over all and check non-valid ones
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in valid_set:
                        if np.abs(T[a, b, c]) > tol:
                            return False

        return True
    except Exception:
        return False


def _construct_rank56(n=2, m=2, p=16):
    """
    Structurally distinct rank-56 decomposition.
    
    Treats the 2x2x16 multiplication as the tensor product of:
      - A rank-7 2x2x2 Strassen core (handling the n,m dimensions)
      - An identity-like structure across the 8 column-blocks of B (p dimension)
    
    The key structural insight: each of the 8 blocks of B (columns 2t, 2t+1)
    is processed by the same Strassen pattern, but the U factor is shared
    across all blocks while V and W are block-diagonal in the p-index.
    
    This gives rank = 7 * 8 = 56 with the following structure:
      U[:, 7*t : 7*t+7] = Strassen_U  (same for all t)
      V[2t:2t+2, 7*t:7*t+7] = Strassen_V[:2, :]
      V[16+2t:16+2t+2, 7*t:7*t+7] = Strassen_V[2:, :]
      W[4t:4t+4, 7*t:7*t+7] = Strassen_W  (with proper index mapping)
    """
    # Strassen rank-7 decomposition for 2x2x2
    # Indices: A: [A00, A01, A10, A11] -> U rows 0,1,2,3
    #          B: [B00, B01, B10, B11] -> V rows 0,1,2,3
    #          C: [C00, C01, C10, C11] -> W rows 0,1,2,3
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    S_U = np.zeros((4, 7))
    S_V = np.zeros((4, 7))
    S_W = np.zeros((4, 7))

    # M1: (A00+A11)(B00+B11)
    S_U[0, 0] = 1; S_U[3, 0] = 1
    S_V[0, 0] = 1; S_V[3, 0] = 1
    S_W[0, 0] = 1; S_W[3, 0] = 1

    # M2: (A10+A11)B00
    S_U[2, 1] = 1; S_U[3, 1] = 1
    S_V[0, 1] = 1
    S_W[2, 1] = 1; S_W[3, 1] = -1

    # M3: A00(B01-B11)
    S_U[0, 2] = 1
    S_V[1, 2] = 1; S_V[3, 2] = -1
    S_W[1, 2] = 1; S_W[3, 2] = 1

    # M4: A11(B10-B00)
    S_U[3, 3] = 1
    S_V[2, 3] = 1; S_V[0, 3] = -1
    S_W[0, 3] = 1; S_W[2, 3] = 1

    # M5: (A00+A01)B11
    S_U[0, 4] = 1; S_U[1, 4] = 1
    S_V[3, 4] = 1
    S_W[0, 4] = -1; S_W[1, 4] = 1

    # M6: (A10-A00)(B00+B01)
    S_U[2, 5] = 1; S_U[0, 5] = -1
    S_V[0, 5] = 1; S_V[1, 5] = 1
    S_W[3, 5] = 1

    # M7: (A01-A11)(B10+B11)
    S_U[1, 6] = 1; S_U[3, 6] = -1
    S_V[2, 6] = 1; S_V[3, 6] = 1
    S_W[0, 6] = 1

    num_blocks = p // 2  # 8
    rank = 7 * num_blocks

    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    for t in range(num_blocks):
        col_start = 7 * t
        col_end = col_start + 7

        # U: same Strassen pattern for every block
        U[:, col_start:col_end] = S_U

        # V: B[j, k] where j in {0,1}, k in {2t, 2t+1}
        # V row index = j * p + k
        # Local B00 = B[0, 2t] -> V[0*16 + 2t] = V[2t]
        # Local B01 = B[0, 2t+1] -> V[0*16 + 2t+1] = V[2t+1]
        # Local B10 = B[1, 2t] -> V[1*16 + 2t] = V[16+2t]
        # Local B11 = B[1, 2t+1] -> V[1*16 + 2t+1] = V[16+2t+1]
        V[2 * t, col_start:col_end] = S_V[0, :]
        V[2 * t + 1, col_start:col_end] = S_V[1, :]
        V[16 + 2 * t, col_start:col_end] = S_V[2, :]
        V[16 + 2 * t + 1, col_start:col_end] = S_V[3, :]

        # W: C[i, k] where i in {0,1}, k in {2t, 2t+1}
        # W row index = k * n + i
        # Local C00 = C[0, 2t] -> W[2t*2 + 0] = W[4t]
        # Local C01 = C[0, 2t+1] -> W[(2t+1)*2 + 0] = W[4t+2]
        # Local C10 = C[1, 2t] -> W[2t*2 + 1] = W[4t+1]
        # Local C11 = C[1, 2t+1] -> W[(2t+1)*2 + 1] = W[4t+3]
        # S_W rows: 0=C00, 1=C01, 2=C10, 3=C11
        W[4 * t, col_start:col_end] = S_W[0, :]       # C00
        W[4 * t + 2, col_start:col_end] = S_W[1, :]   # C01
        W[4 * t + 1, col_start:col_end] = S_W[2, :]   # C10
        W[4 * t + 3, col_start:col_end] = S_W[3, :]   # C11

    return U, V, W


def construct():
    """
    Returns a rank-56 exact tensor decomposition for 2x2x16 matrix multiplication.
    Uses the structural Strassen-block construction.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS

    try:
        U, V, W = _construct_rank56(n, m, p)

        # Quick verification
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback to schoolbook if verification fails (shouldn't happen)
            rank = n * m * p
            U_fb = np.zeros((n * m, rank))
            V_fb = np.zeros((m * p, rank))
            W_fb = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, col] = 1
                        V_fb[j * p + k, col] = 1
                        W_fb[k * n + i, col] = 1
                        col += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    except Exception:
        # Last-resort fallback
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
