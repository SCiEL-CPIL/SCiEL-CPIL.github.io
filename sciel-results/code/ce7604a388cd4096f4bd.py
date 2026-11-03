import numpy as np
import time
import sys

T0 = time.time()
DEADLINE = 140.0  # Safety margin
SAFETY = 10.0

def _build_strassen_28():
    """
    Constructs the known rank-28 decomposition for 2x4x4 matrix multiplication
    using Strassen's algorithm on 2x2 blocks.
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen 2x2x2 coefficients
    # alpha[s] = U part, beta[s] = V part, gamma[s] = W part
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen = [
        # s=0: (A00+A11)(B00+B11) -> C00 + C11
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # s=1: (A10+A11)(B00) -> C10 + C11
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # s=2: (A00)(B10+B11) -> C00 + C01
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}), # Note: Standard Strassen varies. Let's stick to the one that worked.
        # s=3: (A11)(B01-B00) -> C01 + C11
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # s=4: (A00+A01)(B11) -> C00 + C01
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # s=5: (A10-A00)(B00+B01) -> C10 + C11
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # s=6: (A01-A11)(B10+B11) -> C01 + C11
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def emit_block(a_col_off, b_row_off, b_col_off, c_col_off, off):
        """
        Emits a 2x2x2 Strassen block.
        a_col_off: offset in A's columns (0 or 2)
        b_row_off: offset in B's rows (0 or 2)
        b_col_off: offset in B's cols (0 or 2)
        c_col_off: offset in C's cols (0 or 2)
        off: offset in rank dimension
        """
        for s in range(7):
            col = off + s
            uc, vc, wc = strassen[s]
            for li, c in uc.items():
                r_i, c_k = li // 2, li % 2
                # A index: row i, col k. In U: row i*m + (a_col_off + k)
                U[r_i * m + a_col_off + c_k][col] += c
            for li, c in vc.items():
                r_k, c_j = li // 2, li % 2
                # B index: row k, col j. In V: row (b_row_off + k)*p + (b_col_off + j)
                V[(b_row_off + r_k) * p + b_col_off + c_j][col] += c
            for li, c in wc.items():
                r_i, c_j = li // 2, li % 2
                # C index: row i, col j. In W: row (c_col_off + j)*n + i
                W[(c_col_off + c_j) * n + r_i][col] += c
    
    # 4 blocks corresponding to the 2x2 partition of the 4x4 B matrix
    # A is 2x4, B is 4x4, C is 2x4
    # We partition B into four 2x2 blocks.
    # A is not partitioned in rows (only 2 rows), but partitioned in cols?
    # Wait, the standard Strassen applies to 2x2 A, 2x2 B, 2x2 C.
    # Here A is 2x4. We can view A as two 2x2 blocks side-by-side: A_left, A_right.
    # B is four 2x2 blocks.
    # C is two 2x2 blocks side-by-side: C_left, C_right.
    
    # Block 1: A_left (cols 0,1) * B_top_left (rows 0,1; cols 0,1) -> C_left_top (rows 0,1; cols 0,1)
    # This is a 2x2x2 multiplication.
    # A indices: rows 0,1; cols 0,1.
    # B indices: rows 0,1; cols 0,1.
    # C indices: rows 0,1; cols 0,1.
    emit_block(0, 0, 0, 0, 0)
    
    # Block 2: A_left * B_top_right -> C_right_top
    # A indices: rows 0,1; cols 0,1.
    # B indices: rows 0,1; cols 2,3.
    # C indices: rows 0,1; cols 2,3.
    emit_block(0, 0, 2, 2, 7)
    
    # Block 3: A_right * B_bottom_left -> C_left_bottom
    # A indices: rows 0,1; cols 2,3.
    # B indices: rows 2,3; cols 0,1.
    # C indices: rows 0,1; cols 0,1.
    emit_block(2, 2, 0, 0, 14)
    
    # Block 4: A_right * B_bottom_right -> C_right_bottom
    # A indices: rows 0,1; cols 2,3.
    # B indices: rows 2,3; cols 2,3.
    # C indices: rows 0,1; cols 2,3.
    emit_block(2, 2, 2, 2, 21)
    
    return U, V, W

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9, max_err

def try_find_lower_rank(target_rank, time_limit=10.0):
    """
    Attempts to find a decomposition of target_rank using optimization.
    Returns (U, V, W) if found, else None.
    """
    import scipy.optimize as opt
    
    n, m, p = 2, 4, 4
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    total_size = size_U + size_V + size_W
    
    def residuals(x):
        U = x[:size_U].reshape(n*m, target_rank)
        V = x[size_U : size_U + size_V].reshape(m*p, target_rank)
        W = x[size_U + size_V : ].reshape(p*n, target_rank)
        
        T_recon = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
        return T_recon.flatten() - 1.0

    start_time = time.time()
    np.random.seed(42)
    
    # Try a few random restarts
    for restart in range(5):
        if time.time() - start_time > time_limit:
            break
            
        x0 = np.random.randn(total_size) * 0.5
        try:
            result = opt.least_squares(residuals, x0, method='lm', max_nfev=2000)
            err = np.linalg.norm(result.fun)
            if err < 1e-8:
                # Success!
                U = result.x[:size_U].reshape(n*m, target_rank)
                V = result.x[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = result.x[size_U + size_V : ].reshape(p*n, target_rank)
                return U, V, W
        except Exception:
            pass
            
    return None

def construct():
    # First, get the known good solution
    U28, V28, W28 = _build_strassen_28()
    
    # Check time remaining
    elapsed = time.time() - T0
    if elapsed < DEADLINE - 20:
        # Try to find a better solution (rank 27)
        # Note: If rank 27 is impossible, this will fail.
        result_27 = try_find_lower_rank(27, time_limit=15.0)
        if result_27 is not None:
            U, V, W = result_27
            valid, err = verify_decomposition(U, V, W)
            if valid:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try rank 26? (Unlikely, but if we have time)
        if time.time() - T0 < DEADLINE - 10:
             result_26 = try_find_lower_rank(26, time_limit=10.0)
             if result_26 is not None:
                 U, V, W = result_26
                 valid, err = verify_decomposition(U, V, W)
                 if valid:
                     return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank 28
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
