"""Matrix multiplication 4x4x8 with structural decomposition search."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _build_2x2x2_strassen():
    """Build the rank-7 Strassen decomposition for 2x2x2."""
    terms = [
        ([1, 0, 0, -1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([0, 0, 1, -1], [0, 1, 0, 1], [0, 0, 0, 1]),
        ([1, 0, -1, 0], [1, 0, 1, 0], [1, 0, 0, 0]),
    ]
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    return U, V, W


def _build_4x4x4_strassen2():
    """Build a rank-49 decomposition for 4x4x4 using Strassen applied to 2x2 blocks."""
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    U2, V2, W2 = _build_2x2x2_strassen()
    
    # Block-level Strassen coefficients
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    # C blocks: C00, C01, C10, C11 (each 2x2)
    #
    # M1 = (A00-A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A11)(B01+B11)
    # M7 = (A00-A10)(B00+B10)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For each M term, the A-factor is a linear combo of A blocks,
    # B-factor is a linear combo of B blocks.
    # Each block product (2x2 x 2x2) uses rank-7 Strassen.
    
    # A-block coefficients for each M: [A00, A01, A10, A11]
    a_coeffs = [
        [1, 0, 0, -1],  # M1: A00 - A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [0, 0, 1, -1],  # M6: A10 - A11
        [1, 0, -1, 0],  # M7: A00 - A10
    ]
    
    # B-block coefficients for each M: [B00, B01, B10, B11]
    b_coeffs = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01 - B11
        [0, 0, 1, -1],  # M4: B10 - B00
        [0, 0, 0, 1],   # M5: B11
        [0, 1, 0, 1],   # M6: B01 + B11
        [1, 0, 1, 0],   # M7: B00 + B10
    ]
    
    # C-block coefficients for each M: [C00, C01, C10, C11]
    c_coeffs = [
        [1, 0, 0, 1],   # M1: C00 + C11
        [0, 0, 1, -1],  # M2: C10 - C11
        [0, 1, 0, 1],   # M3: C01 + C11
        [1, 0, 1, 0],   # M4: C00 + C10
        [-1, 1, 0, 0],  # M5: -C00 + C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0],   # M7: C00
    ]
    
    for op_idx in range(7):
        for t in range(7):
            col = op_idx * 7 + t
            # U: A-factor
            for blk in range(4):
                coeff = a_coeffs[op_idx][blk]
                if coeff == 0:
                    continue
                br = blk // 2
                bc = blk % 2
                for li in range(2):
                    for lj in range(2):
                        u_idx = (br * 2 + li) * 4 + (bc * 2 + lj)
                        U[u_idx, col] += coeff * U2[2 * li + lj, t]
            # V: B-factor
            for blk in range(4):
                coeff = b_coeffs[op_idx][blk]
                if coeff == 0:
                    continue
                br = blk // 2
                bc = blk % 2
                for lj in range(2):
                    for lk in range(2):
                        v_idx = (br * 2 + lj) * 4 + (bc * 2 + lk)
                        V[v_idx, col] += coeff * V2[2 * lj + lk, t]
            # W: C-factor
            for blk in range(4):
                coeff = c_coeffs[op_idx][blk]
                if coeff == 0:
                    continue
                br = blk // 2
                bc = blk % 2
                for lk in range(2):
                    for li in range(2):
                        w_idx = (br * 2 + lk) * 4 + (bc * 2 + li)
                        W[w_idx, col] += coeff * W2[2 * lk + li, t]
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the decomposition."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(total - 1.0) > tol:
                    return False
    return True


def _build_4x4x8_rank98():
    """Build the rank-98 decomposition for 4x4x8 by splitting B into two 4x4 halves."""
    U44, V44, W44 = _build_4x4x4_strassen2()
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    U[:, 0:49] = U44
    V[0:16, 0:49] = V44
    W[0:16, 0:49] = W44
    U[:, 49:98] = U44
    V[16:32, 49:98] = V44
    W[16:32, 49:98] = W44
    return U, V, W


def _schoolbook_decomposition(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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


def _try_reduce_rank(U0, V0, W0, n, m, p, target_rank, time_limit=5.0):
    """
    Try to reduce the rank of a decomposition using alternating least squares.
    Start from a known decomposition and try to find a lower-rank one.
    """
    start = time.time()
    nU, nV, nW = n * m, m * p, p * n
    
    # Initialize with random low-rank
    U = np.random.randn(nU, target_rank)
    V = np.random.randn(nV, target_rank)
    W = np.random.randn(nW, target_rank)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = float('inf')
    
    for iteration in range(200):
        if time.time() - start > time_limit:
            break
        
        # Update U: for each row i*m+j, solve least squares
        for idx in range(nU):
            i = idx // m
            j = idx % m
            # Target: for all k, sum_c U[idx,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is: sum_c U[idx,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
            # So U[idx,:] should satisfy: U[idx,:] @ diag(V[j*p+k,:] * W[k*n+i,:]) ... 
            # Actually: for each k, we need sum_c U[idx,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # Let A_k = V[j*p+k,:] * W[k*n+i,:] (element-wise product)
            # Then we need: U[idx,:] @ A_k^T = 1 for all k
            # This is a linear system: M @ U[idx,:] = 1 where M[k,c] = V[j*p+k,c] * W[k*n+i,c]
            M = np.outer(V[j * p:], W[:, i])  # shape (p, target_rank)
            # Actually: M[k, c] = V[j*p+k, c] * W[k*n+i, c]
            M = V[j * p:(j + 1) * p, :] * W[0:p, i]  # wrong indexing
            # W index: k*n+i, so for k in range(p), W[k*n+i, c]
            M = np.zeros((p, target_rank))
            for k in range(p):
                M[k, :] = V[j * p + k, :] * W[k * n + i, :]
            try:
                sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                U[idx, :] = sol
            except:
                pass
        
        # Update V: for each row j*p+k
        for idx in range(nV):
            j = idx // p
            k = idx % p
            M = np.zeros((m, target_rank))
            for i in range(m):
                M[i, :] = U[i * m + j, :] * W[k * n + i, :]
            try:
                sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                V[idx, :] = sol
            except:
                pass
        
        # Update W: for each row k*n+i
        for idx in range(nW):
            k = idx // n
            i = idx % n
            M = np.zeros((m, target_rank))
            for j in range(m):
                M[j, :] = U[i * m + j, :] * V[j * p + k, :]
            try:
                sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                W[idx, :] = sol
            except:
                pass
        
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    err += (total - 1.0) ** 2
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Check if we found a valid decomposition
    if _verify_decomposition(best_U, best_V, best_W, n, m, p):
        return best_U, best_V, best_W
    return None


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x8 matrix.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin
    
    # First, build the rank-98 Strassen decomposition as baseline
    U98, V98, W98 = _build_4x4x8_rank98()
    if _verify_decomposition(U98, V98, W98, n, m, p):
        best_rank = 98
        best_result = {"U": U98.tolist(), "V": V98.tolist(), "W": W98.tolist()}
    else:
        U98, V98, W98 = _schoolbook_decomposition(n, m, p)
        best_rank = 128
        best_result = {"U": U98.tolist(), "V": V98.tolist(), "W": W98.tolist()}
    
    # Try to find lower-rank decompositions
    for target_rank in range(best_rank - 1, 31, -1):
        if time.time() > deadline - 10:
            break
        result = _try_reduce_rank(U98, V98, W98, n, m, p, target_rank, 
                                  time_limit=min(5.0, deadline - time.time() - 5))
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = target_rank
            best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            break  # Found the highest rank that works, stop
    
    return best_result

# EVOLVE-BLOCK-END
