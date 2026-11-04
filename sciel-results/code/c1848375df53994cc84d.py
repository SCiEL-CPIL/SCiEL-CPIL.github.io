import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _build_2x2x2_strassen():
    """
    Build the rank-7 Strassen decomposition for 2x2x2.
    """
    terms = [
        ([1, 0, 0, -1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([0, 0, 1, -1], [0, 1, 0, 1], [0, 0, 0, 1]),
        ([1, 0, -1, 0], [1, 0, 1, 0], [1, 0, 0, 0]),
    ]
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build a rank-49 decomposition for 4x4x4 using Strassen.
    """
    U2, V2, W2 = _build_2x2x2_strassen()
    
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    # Map 2x2 Strassen to 4x4x4
    # The 4x4 matrix is divided into four 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # The rank-49 decomposition uses 7x7 = 49 multiplications.
    # Each multiplication corresponds to a pair of Strassen operations.
    
    # For each of the 7 Strassen operations on the left (A) and 7 on the right (B),
    # we have a product.
    
    # The coefficients for the 4x4x4 Strassen decomposition are derived from the 2x2x2 ones.
    
    block_U = [
        [1, 0, 0, -1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [1, 0, -1, 0],
    ]
    block_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ]
    c_combos = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    
    for op_idx in range(7):
        lc = block_U[op_idx]
        rc = block_V[op_idx]
        cc = c_combos[op_idx]
        for t in range(7):
            col = op_idx * 7 + t
            for a_blk in range(4):
                if lc[a_blk] == 0:
                    continue
                a_row = a_blk // 2
                a_col = a_blk % 2
                for li in range(2):
                    for lj in range(2):
                        u_idx = (a_row * 2 + li) * 4 + (a_col * 2 + lj)
                        U[u_idx, col] += lc[a_blk] * U2[2 * li + lj, t]
            for b_blk in range(4):
                if rc[b_blk] == 0:
                    continue
                b_row = b_blk // 2
                b_col = b_blk % 2
                for lj in range(2):
                    for lk in range(2):
                        v_idx = (b_row * 2 + lj) * 4 + (b_col * 2 + lk)
                        V[v_idx, col] += rc[b_blk] * V2[2 * lj + lk, t]
            for c_blk in range(4):
                c_coeff = cc[c_blk]
                if c_coeff == 0:
                    continue
                c_row = c_blk // 2
                c_col = c_blk % 2
                for lk in range(2):
                    for li in range(2):
                        w_idx = (c_row * 2 + lk) * 4 + (c_col * 2 + li)
                        W[w_idx, col] += c_coeff * W2[2 * lk + li, t]
    
    return U, V, W

def _build_4x4x8_rank98():
    """
    Rank-98 decomposition: Strassen for both halves.
    """
    U44, V44, W44 = _build_4x4x4_strassen()
    
    rank = 98
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((32, rank), dtype=int)
    W = np.zeros((32, rank), dtype=int)
    
    # C1 = A*B1: columns 0..48
    U[:, 0:49] = U44
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j * 8 + k, c] = V44[j * 4 + k, c]
    W[0:16, 0:49] = W44
    
    # C2 = A*B2: columns 49..97
    U[:, 49:98] = U44
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j * 8 + (k + 4), 49 + c] = V44[j * 4 + k, c]
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[(k + 4) * 4 + i, 49 + c] = W44[k * 4 + i, c]
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            u_vec = U[i * m + j]
            for k in range(p):
                v_vec = V[j * p + k]
                w_vec = W[k * n + i]
                total = np.dot(u_vec, v_vec * w_vec)
                if abs(total - 1.0) > tol:
                    return False
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x8 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Try rank-98 first (two rank-49 halves)
    try:
        U, V, W = _build_4x4x8_rank98()
        if _verify_decomposition(U, V, W, n, m, p):
            if time.time() < deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition (rank 128)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
