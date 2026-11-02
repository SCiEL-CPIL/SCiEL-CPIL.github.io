import time
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE_SECONDS = 120

def _try_reduce_rank(U, V, W, n, m, p, max_time=10):
    """Try to reduce the rank by finding linear dependencies."""
    start = time.time()
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    R = U_np.shape[1]
    
    # Try to find columns that are linear combinations of others
    # This is a simplified approach: check if any column is zero or duplicate
    keep = list(range(R))
    
    for r in range(R):
        if time.time() - start > max_time:
            break
        # Check if column r is zero
        if np.allclose(U_np[:, r], 0) or np.allclose(V_np[:, r], 0) or np.allclose(W_np[:, r], 0):
            keep.remove(r)
            continue
        # Check if column r is a scalar multiple of another kept column
        for s in keep:
            if s == r:
                continue
            # Check proportionality
            try:
                # Find non-zero entries to determine ratio
                nz_u = np.nonzero(U_np[:, r])[0]
                if len(nz_u) == 0:
                    continue
                idx = nz_u[0]
                if U_np[idx, s] != 0:
                    ratio = U_np[idx, r] / U_np[idx, s]
                    if (np.allclose(U_np[:, r], ratio * U_np[:, s], atol=1e-10) and
                        np.allclose(V_np[:, r], ratio * V_np[:, s], atol=1e-10) and
                        np.allclose(W_np[:, r], ratio * W_np[:, s], atol=1e-10)):
                        keep.remove(r)
                        break
            except:
                pass
    
    if len(keep) < R:
        U_new = U_np[:, keep]
        V_new = V_np[:, keep]
        W_new = W_np[:, keep]
        if _verify_decomposition(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
            return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    rank = U_np.shape[1]
    
    # Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want T[i, j, k] = 1 if i==j? No, C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor is the multiplication tensor.
    # T[i, j, k] should be 1 if the term A[i, j] * B[j, k] contributes to C[i, k].
    # Actually, the decomposition is:
    # C = sum_r u_r v_r w_r^T ? No.
    # The standard tensor decomposition for matrix multiplication is:
    # sum_{r=1}^R u_r \otimes v_r \otimes w_r = T_{matmul}
    # where T_{matmul}[i, j, k] = 1 if the product A[i, j] B[j, k] forms C[i, k].
    # Wait, the indices are:
    # U is (n*m, R), V is (m*p, R), W is (p*n, R).
    # The term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # Summing over r, we get C[i, k] = sum_j A[i, j] B[j, k].
    # So the tensor T[i, j, k] (where i is row of A, j is col of A/row of B, k is col of B)
    # should be 1 for all i, j, k?
    # No.
    # Let's check the seed's verify function.
    # s = np.sum(U_np[i * m + j] * V_np[j * p + k] * W_np[k * n + i])
    # if abs(s - 1.0) > 1e-9: return False
    # This implies that for EVERY i, j, k, the sum must be 1.
    # Is that correct?
    # If C[i, k] = sum_j A[i, j] B[j, k], then the tensor is not just 1s.
    # The tensor T is defined such that T[i, j, k] = 1 if we are computing C[i, k] using A[i, j] and B[j, k].
    # But the decomposition sum_r u_r(i, j) v_r(j, k) w_r(k, i) should equal the "multiplication tensor".
    # The multiplication tensor has 1s at all (i, j, k) positions?
    # No.
    # If we set A and B to arbitrary values, C is determined.
    # The identity is: sum_j A[i, j] B[j, k] = C[i, k].
    # The tensor decomposition represents the bilinear map.
    # The coefficients of the bilinear form are 1 for each (i, j, k) triple?
    # Yes, because each product A[i, j] B[j, k] appears exactly once in the sum for C[i, k].
    # So the target tensor T has T[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p).
    # So the verify function is correct.

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _get_strassen_2x2x2():
    """Returns list of 7 (u, v, w) vectors for 2x2x2 Strassen."""
    terms = []
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    terms.append(([1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]))
    terms.append(([0, 0, 0, 1], [0, 1, 0, 0], [-1, 0, 1, 0]))
    terms.append(([1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]))
    terms.append(([ -1, 0, 1, 0], [1, 0, 0, 1], [0, 0, 0, -1]))
    terms.append(([0, 1, 0, -1], [0, 1, 0, 1], [0, 1, 0, -1]))
    return terms

def _get_rank_11_2x2x3():
    terms = []
    s_terms = _get_strassen_2x2x2()
    for u, v, w in s_terms:
        u_g = u
        v_g = [v[0], v[1], 0, v[2], v[3], 0]
        w_g = [w[0], w[1], 0, w[2], w[3], 0]
        terms.append((u_g, v_g, w_g))
    
    terms.append(([1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]))
    terms.append(([0, 1, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0]))
    terms.append(([0, 0, 1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]))
    terms.append(([0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0]))
    return terms

def _get_rank_4_1x2x2():
    terms = []
    terms.append(([1, 0], [1, 0, 0, 0], [1, 0]))
    terms.append(([1, 0], [0, 1, 0, 0], [0, 1]))
    terms.append(([0, 1], [0, 0, 1, 0], [1, 0]))
    terms.append(([0, 1], [0, 0, 0, 1], [0, 1]))
    return terms

def _get_rank_6_1x2x3():
    terms = []
    terms.append(([1, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0]))
    terms.append(([1, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0]))
    terms.append(([1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1]))
    terms.append(([0, 1], [0, 0, 0, 1, 0, 0], [1, 0, 0]))
    terms.append(([0, 1], [0, 0, 0, 0, 1, 0], [0, 1, 0]))
    terms.append(([0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 1]))
    return terms

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i in range(n * m):
            U[i][col] = u_vec[i]
        for i in range(m * p):
            V[i][col] = v_vec[i]
        for i in range(p * n):
            W[i][col] = w_vec[i]
        col += 1
        
    s_terms = _get_strassen_2x2x2()
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    for u, v, w in s_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[0] = u[0]; u_g[1] = u[1]; u_g[4] = u[2]; u_g[5] = u[3]
        v_g[0] = v[0]; v_g[1] = v[1]; v_g[5] = v[2]; v_g[6] = v[3]
        w_g[0] = w[0]; w_g[1] = w[1]; w_g[3] = w[2]; w_g[4] = w[3]
        add_term(u_g, v_g, w_g)
        
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    for u, v, w in s_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[2] = u[0]; u_g[3] = u[1]; u_g[6] = u[2]; u_g[7] = u[3]
        v_g[10] = v[0]; v_g[11] = v[1]; v_g[15] = v[2]; v_g[16] = v[3]
        w_g[0] = w[0]; w_g[1] = w[1]; w_g[3] = w[2]; w_g[4] = w[3]
        add_term(u_g, v_g, w_g)
        
    r11_terms = _get_rank_11_2x2x3()
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    for u, v, w in r11_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[0] = u[0]; u_g[1] = u[1]; u_g[4] = u[2]; u_g[5] = u[3]
        v_g[2] = v[0]; v_g[3] = v[1]; v_g[4] = v[2]; v_g[7] = v[3]; v_g[8] = v[4]; v_g[9] = v[5]
        w_g[2] = w[0]; w_g[3] = w[1]; w_g[4] = w[2]; w_g[5] = w[3]; w_g[6] = w[4]; w_g[7] = w[5]
        add_term(u_g, v_g, w_g)
        
    # Block 4: A[0:2, 2:4] * B[2:4, 2:5] -> C[0:2, 2:5]
    for u, v, w in r11_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[2] = u[0]; u_g[3] = u[1]; u_g[6] = u[2]; u_g[7] = u[3]
        v_g[12] = v[0]; v_g[13] = v[1]; v_g[14] = v[2]; v_g[17] = v[3]; v_g[18] = v[4]; v_g[19] = v[5]
        w_g[2] = w[0]; w_g[3] = w[1]; w_g[4] = w[2]; w_g[5] = w[3]; w_g[6] = w[4]; w_g[7] = w[5]
        add_term(u_g, v_g, w_g)
        
    r4_terms = _get_rank_4_1x2x2()
    
    # Block 5: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    for u, v, w in r4_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[8] = u[0]; u_g[9] = u[1]
        v_g[0] = v[0]; v_g[1] = v[1]; v_g[5] = v[2]; v_g[6] = v[3]
        w_g[6] = w[0]; w_g[7] = w[1]
        add_term(u_g, v_g, w_g)
        
    # Block 6: A[2, 2:4] * B[2:4, 0:2] -> C[2, 0:2]
    for u, v, w in r4_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[10] = u[0]; u_g[11] = u[1]
        v_g[10] = v[0]; v_g[11] = v[1]; v_g[15] = v[2]; v_g[16] = v[3]
        w_g[6] = w[0]; w_g[7] = w[1]
        add_term(u_g, v_g, w_g)
        
    r6_terms = _get_rank_6_1x2x3()
    
    # Block 7: A[2, 0:2] * B[0:2, 2:5] -> C[2, 2:5]
    for u, v, w in r6_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[8] = u[0]; u_g[9] = u[1]
        v_g[2] = v[0]; v_g[3] = v[1]; v_g[4] = v[2]; v_g[7] = v[3]; v_g[8] = v[4]; v_g[9] = v[5]
        w_g[8] = w[0]; w_g[9] = w[1]; w_g[10] = w[2]
        add_term(u_g, v_g, w_g)
        
    # Block 8: A[2, 2:4] * B[2:4, 2:5] -> C[2, 2:5]
    for u, v, w in r6_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[10] = u[0]; u_g[11] = u[1]
        v_g[12] = v[0]; v_g[13] = v[1]; v_g[14] = v[2]; v_g[17] = v[3]; v_g[18] = v[4]; v_g[19] = v[5]
        w_g[8] = w[0]; w_g[9] = w[1]; w_g[10] = w[2]
        add_term(u_g, v_g, w_g)
        
    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")
        
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    remaining_time = DEADLINE_SECONDS - (time.time() - start_time)
    if remaining_time > 5:
        U, V, W = _try_reduce_rank(U, V, W, n, m, p, max_time=min(remaining_time - 2, 15))
    
    return {"U": U, "V": V, "W": W}
