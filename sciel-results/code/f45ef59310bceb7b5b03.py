"""
Matrix multiplication 2x4 by 4x4, exact tensor decomposition.
Uses Strassen's algorithm on four 2x2 sub-blocks to achieve rank 28 (vs schoolbook 32).
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]
TIME_LIMIT = 150.0
START_TIME = time.time()


def check_time_and_return(best_result):
    if best_result is not None and (time.time() - START_TIME) > TIME_LIMIT - 10.0:
        return best_result
    return None


def construct():
    best_result = None
    # Try rank 20 first
    try:
        best_result = _construct_rank20()
        if best_result is not None:
            if _verify_decomposition(best_result):
                return best_result
    except Exception:
        pass
    # Try rank 28
    if best_result is None:
        try:
            best_result = _construct_strassen_28()
            if best_result is not None:
                if _verify_decomposition(best_result):
                    return best_result
        except Exception:
            pass
    # Fallback to schoolbook
    if best_result is None:
        best_result = _construct_schoolbook_32()
    return best_result


def _verify_decomposition(result, trials=5):
    """Verify that U @ diag(V) @ W^T reconstructs the matmul tensor."""
    import random
    n, m, p = DIMENSIONS
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    rank = U.shape[1]
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        # Reconstruct C from decomposition
        # C[i,k] = sum_r U[i*m+j... wait, need to be careful
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_l V[j*p+l, r] * B[j,l]) ... no
        # Actually: C[i,k] = sum_r U_col_r(A) * V_col_r(B) * W_col_r[C_index]
        # where U_col_r(A) = sum_{i,j} U[i*m+j, r] * A[i,j]
        #       V_col_r(B) = sum_{j,k} V[j*p+k, r] * B[j,k]
        #       and W contributes: C[i,k] = sum_r W[k*n+i, r] * U_col_r(A) * V_col_r(B)
        # Wait, that's not right either. The tensor decomposition is:
        # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # So C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum_{i,j} U[i*m+j,r]*A[i,j]) * ... 
        # No. The correct reconstruction:
        # C[i,k] = sum_{j} A[i,j] * B[j,k]
        # = sum_r (sum_{i',j} U[i'*m+j, r] * A[i',j]) * (sum_{j',k'} V[j'*p+k', r] * B[j',k']) * W[k*n+i, r]
        # That's not right. Let me think again.
        # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # The matmul: C[i,k] = sum_j A[i,j] * B[j,k]
        # We need: sum_j A[i,j]*B[j,k] = sum_r (sum_{i',j'} U[i'*m+j',r]*A[i',j']) * ... 
        # Actually no. The way it works:
        # For each r, compute:
        #   u_r = sum_{i,j} U[i*m+j, r] * A[i,j]  (scalar)
        #   v_r = sum_{j,k} V[j*p+k, r] * B[j,k]  (scalar)
        #   Then C[i,k] += W[k*n+i, r] * u_r * v_r
        # Let's verify: C[i,k] = sum_r W[k*n+i,r] * (sum_{i',j} U[i'*m+j,r]*A[i',j]) * (sum_{j',k'} V[j'*p+k',r]*B[j',k'])
        # This should equal sum_j A[i,j]*B[j,k] if the decomposition is correct.
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_r = 0.0
            for i in range(n):
                for j in range(m):
                    u_r += U[i * m + j, r] * A[i, j]
            v_r = 0.0
            for j in range(m):
                for k in range(p):
                    v_r += V[j * p + k, r] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += W[k * n + i, r] * u_r * v_r
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True


def _construct_schoolbook_32():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 32
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


def _get_242_rank10():
    """
    Find a rank-10 exact decomposition for [2,4,2] matmul.
    A: 2x4, B: 4x2, C: 2x2.
    Returns (U_sub, V_sub, W_sub) with shapes (8,10), (8,10), (4,10).
    Uses numerical solve then rounds to half-integers.
    """
    import random
    # Tensor: T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,1}
    # U: 8x10, V: 8x10, W: 4x10
    # T[i,j,k] = sum_r U[i*4+j, r] * V[j*2+k, r] * W[k*2+i, r]
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for attempt in range(200):
        if (time.time() - START_TIME) > TIME_LIMIT - 20:
            break
        # Initialize with random values
        np.random.seed(attempt)
        U = np.random.randn(8, 10)
        V = np.random.randn(8, 10)
        W = np.random.randn(4, 10)
        
        # Use alternating least squares / gradient descent
        # For fixed V, W: solve for U (linear)
        # For fixed U, W: solve for V (linear)
        # For fixed U, V: solve for W (linear)
        
        for it in range(200):
            # Solve for U: T[i,j,k] = sum_r U[i*4+j,r]*V[j*2+k,r]*W[k*2+i,r]
            # For each (i,j): T[i,j,k] for all k -> linear system in U[i*4+j, :]
            for i in range(2):
                for j in range(4):
                    # Build system: for k in {0,1}: T[i,j,k] = sum_r U[i*4+j,r] * V[j*2+k,r] * W[k*2+i,r]
                    # This is: [V[j*2+0,0]*W[0*2+i,0], ..., V[j*2+0,9]*W[0*2+i,9], V[j*2+1,0]*W[1*2+i,0], ..., V[j*2+1,9]*W[1*2+i,9]] dot U[i*4+j,:] = [T[i,j,0], T[i,j,1]]
                    # Wait, that's not right. Let me re-index.
                    # T[i,j,k] = sum_r U[i*4+j, r] * V[j*2+k, r] * W[k*2+i, r]
                    # For fixed i,j: we have 2 equations (k=0,1) and 10 unknowns (U[i*4+j,0..9])
                    # This is underdetermined. We need to use all equations together.
                    pass
            
            # Better: set up the full linear system for U
            # 16 equations (2*4*2), 80 unknowns (8*10). Underdetermined.
            # Use least squares with regularization.
            
            # Build A_U matrix: 16x80
            # For each (i,j,k): equation T[i,j,k] = sum_r U[i*4+j,r] * V[j*2+k,r] * W[k*2+i,r]
            # So A_U[(i,j,k), (i*4+j)*10 + r] = V[j*2+k,r] * W[k*2+i,r]
            A_U = np.zeros((16, 80))
            for i in range(2):
                for j in range(4):
                    for k in range(2):
                        eq_idx = i*4*2 + j*2 + k
                        for r in range(10):
                            A_U[eq_idx, (i*4+j)*10 + r] = V[j*2+k, r] * W[k*2+i, r]
            b_U = np.ones(16)
            U_flat, *_ = np.linalg.lstsq(A_U, b_U, rcond=None)
            U = U_flat.reshape(8, 10)
            
            # Solve for V
            A_V = np.zeros((16, 80))
            for i in range(2):
                for j in range(4):
                    for k in range(2):
                        eq_idx = i*4*2 + j*2 + k
                        for r in range(10):
                            A_V[eq_idx, (j*2+k)*10 + r] = U[i*4+j, r] * W[k*2+i, r]
            b_V = np.ones(16)
            V_flat, *_ = np.linalg.lstsq(A_V, b_V, rcond=None)
            V = V_flat.reshape(8, 10)
            
            # Solve for W
            A_W = np.zeros((16, 40))
            for i in range(2):
                for j in range(4):
                    for k in range(2):
                        eq_idx = i*4*2 + j*2 + k
                        for r in range(10):
                            A_W[eq_idx, (k*2+i)*10 + r] = U[i*4+j, r] * V[j*2+k, r]
            b_W = np.ones(16)
            W_flat, *_ = np.linalg.lstsq(A_W, b_W, rcond=None)
            W = W_flat.reshape(4, 10)
            
            # Check error
            err = 0.0
            for i in range(2):
                for j in range(4):
                    for k in range(2):
                        val = sum(U[i*4+j, r] * V[j*2+k, r] * W[k*2+i, r] for r in range(10))
                        err += (val - 1.0)**2
            if err < 1e-10:
                break
        
        # Round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check if rounded version works
        err_r = 0.0
        for i in range(2):
            for j in range(4):
                for k in range(2):
                    val = sum(U_r[i*4+j, r] * V_r[j*2+k, r] * W_r[k*2+i, r] for r in range(10))
                    err_r += (val - 1.0)**2
        
        if err_r < 1e-10 and err_r < best_err:
            best_err = err_r
            best_U = U_r
            best_V = V_r
            best_W = W_r
    
    if best_U is not None:
        return (best_U.tolist(), best_V.tolist(), best_W.tolist())
    
    # Fallback: try integer rounding
    for attempt in range(100):
        if (time.time() - START_TIME) > TIME_LIMIT - 20:
            break
        np.random.seed(1000 + attempt)
        U = np.random.randn(8, 10)
        V = np.random.randn(8, 10)
        W = np.random.randn(4, 10)
        
        for it in range(300):
            A_U = np.zeros((16, 80))
            for i in range(2):
                for j in range(4):
                    for k in range(2):
                        eq_idx = i*4*2 + j*2 + k
                        for r in range(10):
                            A_U[eq_idx, (i*4+j)*10 + r] = V[j*2+k, r] * W[k*2+i, r]
            b_U = np.ones(16)
            U_flat, *_ = np.linalg.lstsq(A_U, b_U, rcond=None)
            U = U_flat.reshape(8, 10)
            
            A_V = np.zeros((16, 80))
            for i in range(2):
                for j in range(4):
                    for k in range(2):
                        eq_idx = i*4*2 + j*2 + k
                        for r in range(10):
                            A_V[eq_idx, (j*2+k)*10 + r] = U[i*4+j, r] * W[k*2+i, r]
            b_V = np.ones(16)
            V_flat, *_ = np.linalg.lstsq(A_V, b_V, rcond=None)
            V = V_flat.reshape(8, 10)
            
            A_W = np.zeros((16, 40))
            for i in range(2):
                for j in range(4):
                    for k in range(2):
                        eq_idx = i*4*2 + j*2 + k
                        for r in range(10):
                            A_W[eq_idx, (k*2+i)*10 + r] = U[i*4+j, r] * V[j*2+k, r]
            b_W = np.ones(16)
            W_flat, *_ = np.linalg.lstsq(A_W, b_W, rcond=None)
            W = W_flat.reshape(4, 10)
            
            err = 0.0
            for i in range(2):
                for j in range(4):
                    for k in range(2):
                        val = sum(U[i*4+j, r] * V[j*2+k, r] * W[k*2+i, r] for r in range(10))
                        err += (val - 1.0)**2
            if err < 1e-12:
                break
        
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        err_r = 0.0
        for i in range(2):
            for j in range(4):
                for k in range(2):
                    val = sum(U_r[i*4+j, r] * V_r[j*2+k, r] * W_r[k*2+i, r] for r in range(10))
                    err_r += (val - 1.0)**2
        
        if err_r < 1e-10 and err_r < best_err:
            best_err = err_r
            best_U = U_r
            best_V = V_r
            best_W = W_r
    
    if best_U is not None:
        return (best_U.tolist(), best_V.tolist(), best_W.tolist())
    
    return None


def _construct_rank20():
    """
    2x4 * 4x4 using two [2,4,2] subproblems of rank 10 each.
    """
    n, m, p = DIMENSIONS
    rank = 20
    U = [[0]*rank for _ in range(n*m)]  # 8x20
    V = [[0]*rank for _ in range(m*p)]  # 16x20
    W = [[0]*rank for _ in range(p*n)]  # 8x20
    
    result_242 = _get_242_rank10()
    if result_242 is None:
        return None
    U_sub, V_sub, W_sub = result_242
    
    # C_L = A * B_L (B_L = B[:, 0:2])
    for r in range(10):
        for i in range(n*m):
            U[i][r] += U_sub[i][r]
        for j in range(4):
            for k in range(2):
                V[j*4+k][r] += V_sub[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[k*2+i][r] += W_sub[k*2+i][r]
    
    # C_R = A * B_R (B_R = B[:, 2:4])
    for r in range(10):
        for i in range(n*m):
            U[i][r+10] += U_sub[i][r]
        for j in range(4):
            for k in range(2):
                V[j*4+(k+2)][r+10] += V_sub[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[(k+2)*2+i][r+10] += W_sub[k*2+i][r]
    
    return {"U": U, "V": V, "W": W}


def _construct_strassen_28():
    """
    2x4 * 4x4 using four 2x2 Strassen blocks.
    """
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    sub_products = [
        {'a_map': lambda r, c: r * 4 + c, 'b_map': lambda r, c: r * 4 + c, 'c_map': lambda r, c: c * 2 + r},
        {'a_map': lambda r, c: r * 4 + (c + 2), 'b_map': lambda r, c: (r + 2) * 4 + c, 'c_map': lambda r, c: c * 2 + r},
        {'a_map': lambda r, c: r * 4 + c, 'b_map': lambda r, c: r * 4 + (c + 2), 'c_map': lambda r, c: (c + 2) * 2 + r},
        {'a_map': lambda r, c: r * 4 + (c + 2), 'b_map': lambda r, c: (r + 2) * 4 + (c + 2), 'c_map': lambda r, c: (c + 2) * 2 + r},
    ]
    
    col = 0
    for sp in sub_products:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for loc_u, coeff in u_coeffs.items():
                r, c = loc_u // 2, loc_u % 2
                global_u = sp['a_map'](r, c)
                U[global_u][col] += coeff
            for loc_v, coeff in v_coeffs.items():
                r, c = loc_v // 2, loc_v % 2
                global_v = sp['b_map'](r, c)
                V[global_v][col] += coeff
            for loc_w, coeff in w_coeffs.items():
                r, c = loc_w // 2, loc_w % 2
                global_w = sp['c_map'](r, c)
                W[global_w][col] += coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}
