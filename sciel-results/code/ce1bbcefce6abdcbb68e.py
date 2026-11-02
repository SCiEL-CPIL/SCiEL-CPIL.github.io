import time
import numpy as np

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 135.0  # 15s safety margin

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()

    def verify_fast(U, V, W, rank):
        """Vectorized verification using einsum."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Reshape and use einsum
        T = np.einsum('ir,jr,kr->ijk',
                       Ua.reshape(n, m, rank),
                       Va.reshape(m, p, rank),
                       Wa.reshape(p, n, rank))
        return bool(np.allclose(T, 1.0, atol=1e-8))

    def verify_slow(U, V, W, rank):
        """Fallback verification."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = np.dot(Ua[i*m+j, :rank], Va[j*p+k, :rank] * Wa[k*n+i, :rank])
        return bool(np.allclose(T, 1.0, atol=1e-8))

    # ---- Build the rank-64 structured decomposition ----
    target_rank = 64
    U_new = np.zeros((n * m, target_rank))
    V_new = np.zeros((m * p, target_rank))
    W_new = np.zeros((p * n, target_rank))

    col = 0

    def add_strassen_2x2x2(offset_i, offset_j, offset_k, col_start):
        nonlocal col
        col = col_start
        # M1: (A00+A11)(B00+B11) -> C00:+1, C11:+1
        U_new[(offset_i+0)*m+(offset_j+0), col] += 1
        U_new[(offset_i+1)*m+(offset_j+1), col] += 1
        V_new[(offset_j+0)*p+(offset_k+0), col] += 1
        V_new[(offset_j+1)*p+(offset_k+1), col] += 1
        W_new[(offset_k+0)*n+(offset_i+0), col] += 1
        W_new[(offset_k+1)*n+(offset_i+1), col] += 1
        col += 1
        # M2: (A10+A11)B00 -> C10:+1, C11:-1
        U_new[(offset_i+1)*m+(offset_j+0), col] += 1
        U_new[(offset_i+1)*m+(offset_j+1), col] += 1
        V_new[(offset_j+0)*p+(offset_k+0), col] += 1
        W_new[(offset_k+0)*n+(offset_i+1), col] += 1
        W_new[(offset_k+1)*n+(offset_i+1), col] += -1
        col += 1
        # M3: A00(B10-B11) -> C01:+1, C11:+1
        U_new[(offset_i+0)*m+(offset_j+0), col] += 1
        V_new[(offset_j+1)*p+(offset_k+0), col] += 1
        V_new[(offset_j+1)*p+(offset_k+1), col] += -1
        W_new[(offset_k+0)*n+(offset_i+0), col] += 1
        W_new[(offset_k+1)*n+(offset_i+1), col] += 1
        col += 1
        # M4: A11(B01-B00) -> C00:+1, C10:+1
        U_new[(offset_i+1)*m+(offset_j+1), col] += 1
        V_new[(offset_j+0)*p+(offset_k+1), col] += 1
        V_new[(offset_j+0)*p+(offset_k+0), col] += -1
        W_new[(offset_k+0)*n+(offset_i+0), col] += 1
        W_new[(offset_k+1)*n+(offset_i+1), col] += 1
        col += 1
        # M5: (A00+A01)B11 -> C00:-1, C01:+1
        U_new[(offset_i+0)*m+(offset_j+0), col] += 1
        U_new[(offset_i+0)*m+(offset_j+1), col] += 1
        V_new[(offset_j+1)*p+(offset_k+1), col] += 1
        W_new[(offset_k+0)*n+(offset_i+0), col] += -1
        W_new[(offset_k+1)*n+(offset_i+1), col] += 1
        col += 1
        # M7: (A01-A10)(B10+B11) -> C00:-1, C11:+1
        U_new[(offset_i+0)*m+(offset_j+1), col] += 1
        U_new[(offset_i+1)*m+(offset_j+0), col] += -1
        V_new[(offset_j+1)*p+(offset_k+0), col] += 1
        V_new[(offset_j+1)*p+(offset_k+1), col] += 1
        W_new[(offset_k+0)*n+(offset_i+0), col] += -1
        W_new[(offset_k+1)*n+(offset_i+1), col] += 1
        col += 1
        return col

    # 8 Strassen blocks
    for j_block in [0, 1]:
        for k_block in [0, 1]:
            for k_sub in [0, 1]:
                offset_i = 0
                offset_j = j_block * 2
                offset_k = k_block * 4 + k_sub * 2
                col = add_strassen_2x2x2(offset_i, offset_j, offset_k, col)

    # 4 rank-2 blocks for i=2
    for j_block in [0, 1]:
        for k_block in [0, 1]:
            offset_i = 2
            offset_j = j_block * 2
            offset_k = k_block * 4
            # term a: j = offset_j
            U_new[offset_i*m + offset_j, col] = 1
            for kk in range(4):
                V_new[offset_j*p + (offset_k+kk), col] = 1
                W_new[(offset_k+kk)*n + offset_i, col] = 1
            col += 1
            # term b: j = offset_j+1
            U_new[offset_i*m + (offset_j+1), col] = 1
            for kk in range(4):
                V_new[(offset_j+1)*p + (offset_k+kk), col] = 1
                W_new[(offset_k+kk)*n + offset_i, col] = 1
            col += 1

    # Convert to list format
    U_list = U_new.tolist()
    V_list = V_new.tolist()
    W_list = W_new.tolist()

    best_result = {"U": U_list, "V": V_list, "W": W_list}
    best_rank = target_rank

    if not verify_fast(U_list, V_list, W_list, target_rank):
        # Fallback to naive
        naive_rank = n * m * p
        U_naive = np.zeros((n * m, naive_rank))
        V_naive = np.zeros((m * p, naive_rank))
        W_naive = np.zeros((p * n, naive_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i * m + j, col] = 1
                    V_naive[j * p + k, col] = 1
                    W_naive[k * n + i, col] = 1
                    col += 1
        best_result = {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
        best_rank = naive_rank

    if time.time() > DEADLINE - 20:
        return best_result

    # ---- Efficient linear-dependency elimination ----
    def eliminate_columns(U, V, W, R):
        """Remove linearly dependent columns from the decomposition."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # Build the full tensor coefficient matrix
        # M[i,j,k,r] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        M = np.zeros((n*m*p, R))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = i*m*p + j*p + k
                    for r in range(R):
                        M[row, r] = Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
        
        # Find redundant columns using rank analysis
        kept = list(range(R))
        improved = True
        while improved and len(kept) > 1:
            improved = False
            # Try to remove each column
            for idx in range(len(kept)):
                r = kept[idx]
                others = [k for k in kept if k != r]
                if not others:
                    continue
                sub = M[:, others]
                # Check if column r is in span of others
                try:
                    rank_sub = np.linalg.matrix_rank(sub, tol=1e-10)
                    combined = np.column_stack([sub, M[:, r:r+1]])
                    rank_comb = np.linalg.matrix_rank(combined, tol=1e-10)
                except Exception:
                    continue
                if rank_comb == rank_sub:
                    kept.remove(r)
                    improved = True
                    break
        
        return kept

    kept, R_reduced = eliminate_columns(best_result["U"], best_result["V"], best_result["W"], best_rank)
    if R_reduced < best_rank:
        Ua = np.array(best_result["U"], dtype=float)
        Va = np.array(best_result["V"], dtype=float)
        Wa = np.array(best_result["W"], dtype=float)
        U_red = Ua[:, kept].tolist()
        V_red = Va[:, kept].tolist()
        W_red = Wa[:, kept].tolist()
        if verify_fast(U_red, V_red, W_red, R_reduced):
            best_result = {"U": U_red, "V": V_red, "W": W_red}
            best_rank = R_reduced

    if time.time() > DEADLINE - 10:
        return best_result

    # ---- Local search: add random columns then re-eliminate ----
    rng = np.random.RandomState(42)
    current = best_result
    cur_rank = best_rank
    
    for trial in range(200):
        if time.time() > DEADLINE - 5:
            break
        
        Ua = np.array(current["U"], dtype=float)
        Va = np.array(current["V"], dtype=float)
        Wa = np.array(current["W"], dtype=float)
        R_cur = cur_rank
        
        # Add 2 random columns at a time for more aggressive search
        for extra in range(2):
            new_U = np.column_stack([Ua, rng.randn(n*m)])
            new_V = np.column_stack([Va, rng.randn(m*p)])
            new_W = np.column_stack([Wa, rng.randn(p*n)])
            R_try = R_cur + 1
            
            kept2 = eliminate_columns(new_U.tolist(), new_V.tolist(), new_W.tolist(), R_try)
            R2 = len(kept2)
            
            if R2 < cur_rank:
                U2 = new_U[:, kept2].tolist()
                V2 = new_V[:, kept2].tolist()
                W2 = new_W[:, kept2].tolist()
                if verify_fast(U2, V2, W2, R2):
                    current = {"U": U2, "V": V2, "W": W2}
                    cur_rank = R2
                    best_result = current
                    best_rank = R2
                    break
            elif time.time() > DEADLINE - 5:
                break
    
    return best_result
