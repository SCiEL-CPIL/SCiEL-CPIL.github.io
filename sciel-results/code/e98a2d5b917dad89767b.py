"""Matrix multiplication 2x4 by 4x4, exact tensor decomposition.
Uses Strassen's algorithm on four 2x2 sub-blocks (rank 28), then attempts
linear-dependency reduction to lower the rank further."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]
TIME_LIMIT = 150.0
START_TIME = time.time()


def _time_remaining():
    return TIME_LIMIT - 10.0 - (time.time() - START_TIME)


def _verify_decomposition(result, num_tests=20):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    rank = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False
    for _ in range(num_tests):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        # Compute C from decomposition: C[i,k] = sum_r U[i*m+j... wait
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # The decomposition: for each column r, U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Actually the standard form: C[i,k] = sum_r U[i, r] * V[r, k] * W[k, i]... 
        # No. Let me use the tensor form:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # where T[i,j,k] should equal (A*B)[i,k] summed over j... 
        # Actually for matrix mult: (AB)[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor is T[i,j,k] = delta for the multiplication tensor.
        # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Let's verify by contracting:
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                C_computed[i, k] = s
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True


def _construct_strassen_28():
    """
    2x4 * 4x4 using four 2x2 Strassen blocks, rank 28.
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
        {'a_map': lambda r, c: r * 4 + c,
         'b_map': lambda r, c: r * 4 + c,
         'c_map': lambda r, c: c * 2 + r},
        {'a_map': lambda r, c: r * 4 + (c + 2),
         'b_map': lambda r, c: (r + 2) * 4 + c,
         'c_map': lambda r, c: c * 2 + r},
        {'a_map': lambda r, c: r * 4 + c,
         'b_map': lambda r, c: r * 4 + (c + 2),
         'c_map': lambda r, c: (c + 2) * 2 + r},
        {'a_map': lambda r, c: r * 4 + (c + 2),
         'b_map': lambda r, c: (r + 2) * 4 + (c + 2),
         'c_map': lambda r, c: (c + 2) * 2 + r},
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


def _try_reduce_rank(result, max_iterations=5000):
    """
    Attempt to reduce rank by finding linear dependencies among the rank-1 terms.
    We look for a linear combination of columns that is zero in one of U, V, W,
    allowing us to eliminate that column.
    
    Strategy: For each pair of columns (a, b), check if there exists a scalar λ
    such that column a - λ*column b is zero in U (or V, or W). If so, we can
    replace column a with column a - λ*column b in the other two matrices and
    eliminate column a.
    """
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    rank = U.shape[1]
    
    reduced = True
    while reduced and _time_remaining() > 20:
        reduced = False
        for a in range(rank - 1):
            if _time_remaining() < 20:
                break
            for b in range(a + 1, rank):
                # Check if U[:,a] is proportional to U[:,b]
                ua = U[:, a]
                ub = U[:, b]
                if np.allclose(ua, 0, atol=1e-12) and not np.allclose(ub, 0, atol=1e-12):
                    # Column a is zero in U; check if it's also zero in V or W
                    if np.allclose(V[:, a], 0, atol=1e-12) or np.allclose(W[:, a], 0, atol=1e-12):
                        # This column is entirely zero; remove it
                        U = np.delete(U, a, axis=1)
                        V = np.delete(V, a, axis=1)
                        W = np.delete(W, a, axis=1)
                        rank -= 1
                        reduced = True
                        break
                    continue
                
                # Check proportionality in U
                if not np.allclose(ua, 0, atol=1e-12):
                    # Find λ such that ua ≈ λ * ub
                    non_zero = np.abs(ub) > 1e-12
                    if np.any(non_zero):
                        lam = ua[non_zero] / ub[non_zero]
                        if np.std(lam) < 1e-10 and np.allclose(ua, lam * ub, atol=1e-10):
                            # U[:,a] = λ * U[:,b]
                            # Replace col a with col a - λ*col b in V and W
                            V[:, a] = V[:, a] - lam * V[:, b]
                            W[:, a] = W[:, a] - lam * W[:, b]
                            # Now col a is zero in U
                            U[:, a] = 0.0
                            # Check if we can eliminate col a (zero in U)
                            # It contributes nothing to the tensor since U col is zero
                            U = np.delete(U, a, axis=1)
                            V = np.delete(V, a, axis=1)
                            W = np.delete(W, a, axis=1)
                            rank -= 1
                            reduced = True
                            break
                        # Also try the reverse: ub = λ * ua
                        non_zero2 = np.abs(ua) > 1e-12
                        if np.any(non_zero2):
                            lam2 = ub[non_zero2] / ua[non_zero2]
                            if np.std(lam2) < 1e-10 and np.allclose(ub, lam2 * ua, atol=1e-10):
                                V[:, b] = V[:, b] - lam2 * V[:, a]
                                W[:, b] = W[:, b] - lam2 * W[:, a]
                                U[:, b] = 0.0
                                U = np.delete(U, b, axis=1)
                                V = np.delete(V, b, axis=1)
                                W = np.delete(W, b, axis=1)
                                rank -= 1
                                reduced = True
                                break
            if reduced:
                continue
            
            # Check proportionality in V
            va = V[:, a]
            vb = V[:, b]
            if not np.allclose(va, 0, atol=1e-12) and not np.allclose(vb, 0, atol=1e-12):
                non_zero = np.abs(vb) > 1e-12
                if np.any(non_zero):
                    lam = va[non_zero] / vb[non_zero]
                    if np.std(lam) < 1e-10 and np.allclose(va, lam * vb, atol=1e-10):
                        U[:, a] = U[:, a] - lam * U[:, b]
                        W[:, a] = W[:, a] - lam * W[:, b]
                        V[:, a] = 0.0
                        V = np.delete(V, a, axis=1)
                        U = np.delete(U, a, axis=1)
                        W = np.delete(W, a, axis=1)
                        rank -= 1
                        reduced = True
                        break
                    non_zero2 = np.abs(va) > 1e-12
                    if np.any(non_zero2):
                        lam2 = vb[non_zero2] / va[non_zero2]
                        if np.std(lam2) < 1e-10 and np.allclose(vb, lam2 * va, atol=1e-10):
                            U[:, b] = U[:, b] - lam2 * U[:, a]
                            W[:, b] = W[:, b] - lam2 * W[:, a]
                            V[:, b] = 0.0
                            V = np.delete(V, b, axis=1)
                            U = np.delete(U, b, axis=1)
                            W = np.delete(W, b, axis=1)
                            rank -= 1
                            reduced = True
                            break
            
            # Check proportionality in W
            wa = W[:, a]
            wb = W[:, b]
            if not np.allclose(wa, 0, atol=1e-12) and not np.allclose(wb, 0, atol=1e-12):
                non_zero = np.abs(wb) > 1e-12
                if np.any(non_zero):
                    lam = wa[non_zero] / wb[non_zero]
                    if np.std(lam) < 1e-10 and np.allclose(wa, lam * wb, atol=1e-10):
                        U[:, a] = U[:, a] - lam * U[:, b]
                        V[:, a] = V[:, a] - lam * V[:, b]
                        W[:, a] = 0.0
                        W = np.delete(W, a, axis=1)
                        U = np.delete(U, a, axis=1)
                        V = np.delete(V, a, axis=1)
                        rank -= 1
                        reduced = True
                        break
                    non_zero2 = np.abs(wa) > 1e-12
                    if np.any(non_zero2):
                        lam2 = wb[non_zero2] / wa[non_zero2]
                        if np.std(lam2) < 1e-10 and np.allclose(wb, lam2 * wa, atol=1e-10):
                            U[:, b] = U[:, b] - lam2 * U[:, a]
                            V[:, b] = V[:, b] - lam2 * V[:, a]
                            W[:, b] = 0.0
                            W = np.delete(W, b, axis=1)
                            U = np.delete(U, b, axis=1)
                            V = np.delete(V, b, axis=1)
                            rank -= 1
                            reduced = True
                            break
        if rank <= 1:
            break
    
    if rank <= 0:
        return None
    
    # Convert back to list format
    U_list = [[float(U[i, j]) for j in range(rank)] for i in range(U.shape[0])]
    V_list = [[float(V[i, j]) for j in range(rank)] for i in range(V.shape[0])]
    W_list = [[float(W[i, j]) for j in range(rank)] for i in range(W.shape[0])]
    
    # Clean up near-zero values
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx, val in enumerate(row):
                if abs(val) < 1e-10:
                    row[idx] = 0.0
                elif abs(val - round(val)) < 1e-10:
                    row[idx] = float(round(val))
                elif abs(val - round(val * 2) / 2) < 1e-10:
                    row[idx] = float(round(val * 2) / 2)
    
    return {"U": U_list, "V": V_list, "W": W_list}


def _construct_schoolbook_32():
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
    best_result = None
    best_rank = float('inf')
    
    try:
        # Start with Strassen rank-28
        result_28 = _construct_strassen_28()
        if _verify_decomposition(result_28):
            best_result = result_28
            best_rank = 28
        
        # Try to reduce rank via linear dependency elimination
        if best_result is not None and _time_remaining() > 30:
            reduced = _try_reduce_rank(best_result)
            if reduced is not None:
                new_rank = reduced['U'][0].__len__() if reduced['U'] else 0
                if new_rank < best_rank and _verify_decomposition(reduced):
                    best_result = reduced
                    best_rank = new_rank
        
    except Exception:
        pass
    
    # Fallback to schoolbook if nothing else works
    if best_result is None:
        best_result = _construct_schoolbook_32()
    
    return best_result
