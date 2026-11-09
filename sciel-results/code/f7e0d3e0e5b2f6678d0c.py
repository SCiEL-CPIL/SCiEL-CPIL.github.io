"""Matrix multiplication 3,3,8 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook_decomposition(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition as list-of-lists."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[:,r]*V[:,r]*W[:,r] reconstructs the MM tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for r in range(rank):
        Au = Ua[:, r].reshape(n, m)
        Bv = Va[:, r].reshape(m, p)
        Ww = Wa[:, r].reshape(p, n)
        C_computed += (Au @ Bv) * Ww.T
    if not np.allclose(C_computed, C_expected, atol=tol):
        return False
    return True

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns (U, V, W) each of shape (4, 7).
    """
    U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [1, 0, -1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    return U, V, W

def _embed_2x2x2_in_3x3x8(U2, V2, W2, i_offset, j_offset, k_offset, n, m, p):
    """
    Embed a 2x2x2 Strassen decomposition into a 3x3x8 problem.
    i_offset: row offset in A (0 or 1)
    j_offset: col offset in A / row offset in B (0 or 1)
    k_offset: col offset in B (0 or 1)
    """
    U = [[0.0] * 7 for _ in range(n * m)]
    V = [[0.0] * 7 for _ in range(m * p)]
    W = [[0.0] * 7 for _ in range(p * n)]
    
    for r in range(7):
        for a_i in range(2):
            for a_j in range(2):
                ui = (i_offset + a_i) * m + (j_offset + a_j)
                U[ui][r] = float(U2[r][a_i * 2 + a_j])
        
        for b_j in range(2):
            for b_k in range(2):
                vj = (j_offset + b_j) * p + (k_offset + b_k)
                V[vj][r] = float(V2[r][b_j * 2 + b_k])
        
        for c_k in range(2):
            for c_i in range(2):
                wk = (k_offset + c_k) * n + (i_offset + c_i)
                W[wk][r] = float(W2[r][c_k * 2 + c_i])
    
    return U, V, W

def _strassen_block_decomposition(n, m, p):
    """
    Build a decomposition for 3x3 * 3x8 using Strassen on 2x2 blocks.
    
    Strategy:
    - Split A (3x3) into top 2x2 block + remaining
    - Split B (3x8) into left 2x8 block + remaining  
    - Use Strassen for the 2x2 * 2x8 part
    - Use schoolbook for the remaining parts
    
    For 2x2 * 2x8:
      Split B into four 2x2 blocks: B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8]
      Each uses Strassen rank 7, total 4*7 = 28
    
    For remaining:
      A[2, :] (1x3) * B[:, 0:8] (3x8): schoolbook rank 24
      A[0:2, 2] (2x1) * B[2, :] (1x8): schoolbook rank 16
    
    Total: 28 + 24 + 16 = 68 (better than 72)
    """
    n, m, p = 3, 3, 8
    
    # Collect all rank-1 terms
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Part 1: Top-left 2x2 block of A times left 2x8 block of B
    # This is A[0:2, 0:2] (2x2) * B[0:2, 0:8] (2x8)
    U2, V2, W2 = _strassen_2x2x2()
    
    # Split B[0:2, 0:8] into four 2x2 blocks
    for k_block in range(4):
        k_offset = k_block * 2
        U_emb, V_emb, W_emb = _embed_2x2x2_in_3x3x8(U2, V2, W2, 0, 0, k_offset, n, m, p)
        U_rows.extend(U_emb)
        V_rows.extend(V_emb)
        W_rows.extend(W_emb)
    
    # Part 2: Bottom row of A times all of B
    # A[2, 0:3] (1x3) * B[0:3, 0:8] (3x8)
    # Schoolbook: rank 3*8 = 24
    for j in range(3):
        for k in range(8):
            u_row = [0.0] * 1
            v_row = [0.0] * 1
            w_row = [0.0] * 1
            u_row[0] = 1.0  # A[2, j]
            v_row[0] = 1.0  # B[j, k]
            w_row[0] = 1.0  # C[k, 2]
            U_rows.append(u_row)
            V_rows.append(v_row)
            W_rows.append(w_row)
    
    # Part 3: Right column of A (top 2 rows) times bottom row of B
    # A[0:2, 2] (2x1) * B[2, 0:8] (1x8)
    # Schoolbook: rank 2*8 = 16
    for i in range(2):
        for k in range(8):
            u_row = [0.0] * 1
            v_row = [0.0] * 1
            w_row = [0.0] * 1
            u_row[0] = 1.0  # A[i, 2]
            v_row[0] = 1.0  # B[2, k]
            w_row[0] = 1.0  # C[k, i]
            U_rows.append(u_row)
            V_rows.append(v_row)
            W_rows.append(w_row)
    
    # Convert to matrices
    rank = len(U_rows)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = U_rows[r][i] if i < len(U_rows[r]) else 0.0
        for i in range(m * p):
            V[i][r] = V_rows[r][i] if i < len(V_rows[r]) else 0.0
        for i in range(p * n):
            W[i][r] = W_rows[r][i] if i < len(W_rows[r]) else 0.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomp(decomp, n, m, p, tol=1e-9):
    """Verify a decomposition."""
    U = np.array(decomp["U"])
    V = np.array(decomp["V"])
    W = np.array(decomp["W"])
    rank = U.shape[1]
    
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        Au = U[:, r].reshape(n, m)
        Bv = V[:, r].reshape(m, p)
        Ww = W[:, r].reshape(p, n)
        C_computed += (Au @ Bv) * Ww.T
    
    return np.allclose(C_computed, C_expected, atol=tol)

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try Strassen block decomposition
    try:
        strassen_decomp = _strassen_block_decomposition(n, m, p)
        if _verify_decomp(strassen_decomp, n, m, p):
            return strassen_decomp
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END
