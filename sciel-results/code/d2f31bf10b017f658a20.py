"""Matrix multiplication 6x6x6 via Block-Strassen rank decomposition.

This program constructs a rank-189 exact tensor decomposition for 6x6 matrix
multiplication by partitioning the matrices into 3x3 blocks of 2x2 submatrices
and applying Strassen's rank-7 algorithm to each block product.

The previous implementation failed because the verification step incorrectly
simulated the Strassen products. This version verifies the 2x2 Strassen
decomposition by explicitly constructing its U, V, W factors and checking
against standard matrix multiplication. If verification passes, it constructs
the full 6x6x6 decomposition; otherwise, it falls back to schoolbook (rank 216).
"""
import time
import random

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _strassen22_tables():
    """
    Returns (M, W_local) for 2x2 Strassen decomposition.
    M: List of 7 tuples (u_vec, v_vec) where u_vec, v_vec are length-4 lists.
    W_local: 4x7 matrix where W_local[out_idx][m_idx] is the coefficient.
    """
    M = [
        ([1, 0, 0, 1], [1, 0, 0, 1]),  # M0 = (A00+A11)*(B00+B11)
        ([0, 0, 1, 1], [1, 0, 0, 0]),  # M1 = (A10+A11)*B00
        ([1, 0, 0, 0], [0, 1, 0, -1]), # M2 = A00*(B01-B11)
        ([0, 0, 0, 1], [0, 0, 1, -1]), # M3 = A11*(B10-B00)
        ([1, 1, 0, 0], [0, 0, 0, 1]),  # M4 = (A00+A01)*B11
        ([1, 0, -1, 0], [1, 1, 0, 0]), # M5 = (A10-A00)*(B00+B01)
        ([0, 1, 0, -1], [0, 1, 1, 0]), # M6 = (A01-A11)*(B10+B11)
    ]
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    W_local = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    return M, W_local

def _verify_strassen22(M, W_local, trials=20):
    """
    Verify the 2x2 Strassen decomposition.
    """
    for _ in range(trials):
        A = [[random.randint(-10, 10) for _ in range(2)] for _ in range(2)]
        B = [[random.randint(-10, 10) for _ in range(2)] for _ in range(2)]
        
        C_expected = [[0, 0], [0, 0]]
        for i in range(2):
            for k in range(2):
                for j in range(2):
                    C_expected[i][j] += A[i][k] * B[k][j]
                    
        a_flat = [A[0][0], A[0][1], A[1][0], A[1][1]]
        b_flat = [B[0][0], B[0][1], B[1][0], B[1][1]]
        
        m_vals = [0] * 7
        for r in range(7):
            sum_u = sum(M[r][0][i] * a_flat[i] for i in range(4))
            sum_v = sum(M[r][1][i] * b_flat[i] for i in range(4))
            m_vals[r] = sum_u * sum_v
            
        C_calc = [[0, 0], [0, 0]]
        for out_idx in range(4):
            c_val = 0
            for r in range(7):
                c_val += W_local[out_idx][r] * m_vals[r]
            C_calc[out_idx // 2][out_idx % 2] = c_val
                
        if C_expected != C_calc:
            return False
            
    return True

def _build_block_strassen():
    """
    Construct U, V, W for 6x6x6 matrix multiplication using Block-Strassen.
    Rank = 9 (block positions) * 3 (sum indices) * 7 (Strassen terms) = 189.
    """
    n, m, p = DIMENSIONS
    M, W_local = _strassen22_tables()
    
    if not _verify_strassen22(M, W_local):
        return None
        
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    Wm = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i_block in range(3):
        for k_block in range(3):
            for j_block in range(3):
                for s in range(7):
                    u_vec, v_vec = M[s]
                    
                    # Update U: A block (i_block, j_block)
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr * 2 + lc
                            global_row = i_block * 2 + lr
                            global_col = j_block * 2 + lc
                            U[global_row * m + global_col][col] += u_vec[local_idx]
                            
                    # Update V: B block (j_block, k_block)
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr * 2 + lc
                            global_row = j_block * 2 + lr
                            global_col = k_block * 2 + lc
                            V[global_row * p + global_col][col] += v_vec[local_idx]
                            
                    # Update W: C block (i_block, k_block)
                    for out_idx in range(4):
                        sign = W_local[out_idx][s]
                        if sign != 0:
                            lr = out_idx // 2
                            lc = out_idx % 2
                            global_i = i_block * 2 + lr
                            global_k = k_block * 2 + lc
                            Wm[global_k * n + global_i][col] += sign
                            
                    col += 1
                    
    if col != rank:
        return None
        
    return {"U": U, "V": V, "W": Wm}

def _build_schoolbook():
    """Fallback schoolbook decomposition with rank 216."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    Wm = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                # W index: k*n + i
                Wm[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": Wm}

def _verify_full_decomposition(result, trials=10):
    """Verify the full 6x6x6 decomposition using numpy."""
    import numpy as np
    n, m, p = DIMENSIONS
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    for _ in range(trials):
        A = np.random.randint(-5, 5, (n, m)).astype(float)
        B = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A @ B
        
        a_flat = A.flatten()
        b_flat = B.flatten()
        u_proj = U.T @ a_flat
        v_proj = V.T @ b_flat
        m_vals = u_proj * v_proj
        c_flat = W.T @ m_vals
        C_calc = c_flat.reshape((p, n)).T
        
        if not np.allclose(C_calc, C_expected, atol=1e-9):
            return False
    return True

def construct():
    deadline = time.time() + 110
    # Try Block-Strassen first
    try:
        result = _build_block_strassen()
        if result is not None and time.time() < deadline:
            if _verify_full_decomposition(result):
                return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _build_schoolbook()

# EVOLVE-BLOCK-END
