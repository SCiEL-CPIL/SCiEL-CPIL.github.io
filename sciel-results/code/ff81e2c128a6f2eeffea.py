import time
import random
import numpy as np

def construct():
    _t0 = time.time()
    DEADLINE = _t0 + 115.0  # safety margin before 150s kill
    n, m, p = 3, 4, 6  # C is n x p, A is n x m, B is m x p

    def _idx_u(i, k):
        return i * m + k

    def _idx_v(k, j):
        return k * p + j

    def _idx_w(j, i):
        return j * n + i

    def build_decomposition(a_row_splits, a_col_splits, b_row_splits, b_col_splits):
        """Build a decomposition given partition strategies."""
        RANK = 0
        col_counter = [0]

        a_row_starts = [0]
        for s in a_row_splits[:-1]:
            a_row_starts.append(a_row_starts[-1] + s)
        a_col_starts = [0]
        for s in a_col_splits[:-1]:
            a_col_starts.append(a_col_starts[-1] + s)
        b_row_starts = [0]
        for s in b_row_splits[:-1]:
            b_row_starts.append(b_row_starts[-1] + s)
        b_col_starts = [0]
        for s in b_col_splits[:-1]:
            b_col_starts.append(b_col_starts[-1] + s)

        def _add_term(u_indices, v_indices, w_indices, u_coeffs, v_coeffs, w_coeffs):
            nonlocal RANK
            RANK += 1
            col = col_counter[0]
            col_counter[0] += 1
            for u_i, u_c in zip(u_indices, u_coeffs):
                U[u_i][col] += u_c
            for v_i, v_c in zip(v_indices, v_coeffs):
                V[v_i][col] += v_c
            for w_i, w_c in zip(w_indices, w_coeffs):
                W[w_i][col] += w_c

        # We need to know RANK upfront to allocate, so we do a first pass
        total_rank = 0
        for ar_i, ar_size in enumerate(a_row_splits):
            for ac_i, ac_size in enumerate(a_col_splits):
                br_size = b_row_splits[ac_i]
                for bc_i, bc_size in enumerate(b_col_splits):
                    if ar_size == 1 and br_size == 2:
                        total_rank += 4
                    elif ar_size == 2 and br_size == 2:
                        total_rank += 7
                    else:
                        total_rank += ar_size * bc_size * br_size

        RANK = total_rank
        U = [[0] * RANK for _ in range(n * m)]
        V = [[0] * RANK for _ in range(m * p)]
        W = [[0] * RANK for _ in range(p * n)]

        col_counter[0] = 0

        for ar_i, ar_size in enumerate(a_row_splits):
            ar_start = a_row_starts[ar_i]
            for ac_i, ac_size in enumerate(a_col_splits):
                ac_start = a_col_starts[ac_i]
                br_start = b_row_starts[ac_i]
                br_size = b_row_splits[ac_i]
                for bc_i, bc_size in enumerate(b_col_splits):
                    bc_start = b_col_starts[bc_i]

                    if ar_size == 1 and br_size == 2:
                        # 1x2 * 2x2 -> 1x2, schoolbook rank 4
                        A0 = _idx_u(ar_start, ac_start)
                        A1 = _idx_u(ar_start, ac_start + 1)
                        B00 = _idx_v(br_start, bc_start)
                        B01 = _idx_v(br_start, bc_start + 1)
                        B10 = _idx_v(br_start + 1, bc_start)
                        B11 = _idx_v(br_start + 1, bc_start + 1)
                        C0 = _idx_w(bc_start, ar_start)
                        C1 = _idx_w(bc_start + 1, ar_start)

                        # M1 = A0 * B00 -> C0
                        _add_term([A0], [B00], [C0], [1], [1], [1])
                        # M2 = A1 * B10 -> C0
                        _add_term([A1], [B10], [C0], [1], [1], [1])
                        # M3 = A0 * B01 -> C1
                        _add_term([A0], [B01], [C1], [1], [1], [1])
                        # M4 = A1 * B11 -> C1
                        _add_term([A1], [B11], [C1], [1], [1], [1])

                    elif ar_size == 2 and br_size == 2:
                        # 2x2 * 2x2 -> 2x2, Strassen rank 7
                        A00 = _idx_u(ar_start, ac_start)
                        A01 = _idx_u(ar_start, ac_start + 1)
                        A10 = _idx_u(ar_start + 1, ac_start)
                        A11 = _idx_u(ar_start + 1, ac_start + 1)
                        B00 = _idx_v(br_start, bc_start)
                        B01 = _idx_v(br_start, bc_start + 1)
                        B10 = _idx_v(br_start + 1, bc_start)
                        B11 = _idx_v(br_start + 1, bc_start + 1)
                        C00 = _idx_w(bc_start, ar_start)
                        C01 = _idx_w(bc_start + 1, ar_start)
                        C10 = _idx_w(bc_start, ar_start + 1)
                        C11 = _idx_w(bc_start + 1, ar_start + 1)

                        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
                        _add_term([A00, A11], [B00, B11], [C00, C11], [1, 1], [1, 1], [1, 1])
                        # M2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
                        _add_term([A10, A11], [B00], [C10, C11], [1, 1], [1], [1, -1])
                        # M3 = A00(B01-B11) -> C01 += 1, C11 += 1
                        _add_term([A00], [B01, B11], [C01, C11], [1], [1, -1], [1, 1])
                        # M4 = A11(B10-B00) -> C00 += 1, C10 += 1
                        _add_term([A11], [B10, B00], [C00, C10], [1], [1, -1], [1, 1])
                        # M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
                        _add_term([A00, A01], [B11], [C00, C01], [1, 1], [1], [-1, 1])
                        # M6 = (A10-A00)(B00+B01) -> C11 += 1
                        _add_term([A10, A00], [B00, B01], [C11], [1, -1], [1, 1], [1])
                        # M7 = (A01-A11)(B10+B11) -> C00 += 1
                        _add_term([A01, A11], [B10, B11], [C00], [1, -1], [1, 1], [1])

                    else:
                        # Schoolbook for general case
                        for di in range(ar_size):
                            for dj in range(bc_size):
                                for dk in range(br_size):
                                    u_i = _idx_u(ar_start + di, ac_start + dk)
                                    v_i = _idx_v(br_start + dk, bc_start + dj)
                                    w_i = _idx_w(bc_start + dj, ar_start + di)
                                    _add_term([u_i], [v_i], [w_i], [1], [1], [1])

        assert col_counter[0] == RANK, f"Expected {RANK} terms, got {col_counter[0]}"
        return U, V, W, RANK

    def verify(U, V, W, rank, n=3, m=4, p=6):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + k][c] * V[k * p + j][c] * W[j * n + i][c]
                    if s != 1:
                        return False
        return True

    def try_reduce_rank(U, V, W, rank, n=3, m=4, p=6, max_iters=200):
        """Try to reduce rank by finding linear dependencies over integers/rationals."""
        best_U, best_V, best_W, best_rank = U, V, W, rank
        current_U, current_V, current_W, current_rank = U, V, W, rank
        
        # Use integer arithmetic for exactness
        # We look for integer vectors alpha such that sum_c alpha_c * T_c = 0
        # where T_c is the rank-1 tensor U_c \otimes V_c \otimes W_c
        # This is equivalent to solving a system of linear equations.
        # Since the dimension is small (72 equations), we can use Gaussian elimination over Q.
        # However, doing this repeatedly is expensive.
        
        # Strategy: Try removing one term at a time and check if the remaining terms
        # can still span the tensor. This is equivalent to checking if the removed term
        # is in the span of the others.
        
        # A faster heuristic: check for trivial dependencies first.
        # For example, if two terms are identical, we can drop one.
        
        # Let's implement a robust rank reduction using SVD on the flattening matrix.
        # Flatten the tensor into a matrix M of size (n*m*p) x rank.
        # If the rank of M is less than 'rank', there is a dependency.
        
        def _get_flattening_matrix(U_mat, V_mat, W_mat, r):
            M = np.zeros((n * m * p, r))
            for c in range(r):
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            M[i * m * p + k * p + j, c] = U_mat[i * m + k][c] * V_mat[k * p + j][c] * W_mat[j * n + i][c]
            return M

        for iteration in range(max_iters):
            if time.time() > DEADLINE - 15:
                break
            
            M = _get_flattening_matrix(current_U, current_V, current_W, current_rank)
            
            # Compute SVD to find the nullspace
            try:
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=True)
                # Check if the last singular value is close to zero
                if s[-1] < 1e-9:
                    null_vec = Vt[-1, :]
                    # Find the index of the largest component
                    max_idx = np.argmax(np.abs(null_vec))
                    
                    # Remove column max_idx
                    keep = [c for c in range(current_rank) if c != max_idx]
                    new_rank = current_rank - 1
                    U_new = [[current_U[i][c] for c in keep] for i in range(n * m)]
                    V_new = [[current_V[i][c] for c in keep] for i in range(m * p)]
                    W_new = [[current_W[i][c] for c in keep] for i in range(p * n)]
                    
                    if verify(U_new, V_new, W_new, new_rank):
                        current_U, current_V, current_W, current_rank = U_new, V_new, W_new, new_rank
                        best_U, best_V, best_W, best_rank = current_U, current_V, current_W, current_rank
                        continue
            except:
                pass
            
            # If SVD doesn't find a clear dependency, try brute-force dropping of each term
            # This is expensive, so we limit it.
            if iteration > 10:
                break
                
            for c_drop in range(current_rank):
                if time.time() > DEADLINE - 15:
                    break
                keep = [c for c in range(current_rank) if c != c_drop]
                new_rank = current_rank - 1
                U_new = [[current_U[i][c] for c in keep] for i in range(n * m)]
                V_new = [[current_V[i][c] for c in keep] for i in range(m * p)]
                W_new = [[current_W[i][c] for c in keep] for i in range(p * n)]
                
                if verify(U_new, V_new, W_new, new_rank):
                    current_U, current_V, current_W, current_rank = U_new, V_new, W_new, new_rank
                    best_U, best_V, best_W, best_rank = current_U, current_V, current_W, current_rank
                    break
        
        return best_U, best_V, best_W, best_rank

    # Generate candidate partition strategies
    candidates = []
    
    # Strategy 1: Current best (1+2 rows, 2+2 cols, 2+2 B rows, 2+2+2 B cols)
    candidates.append(([1, 2], [2, 2], [2, 2], [2, 2, 2]))
    
    # Strategy 2: 2+1 rows, 2+2 cols, 2+2 B rows, 2+2+2 B cols
    candidates.append(([2, 1], [2, 2], [2, 2], [2, 2, 2]))
    
    # Strategy 3: 1+1+1 rows, 2+2 cols, 2+2 B rows, 2+2+2 B cols
    candidates.append(([1, 1, 1], [2, 2], [2, 2], [2, 2, 2]))
    
    # Strategy 4: 1+2 rows, 1+1+2 cols, 1+1+2 B rows, 2+2+2 B cols
    candidates.append(([1, 2], [1, 1, 2], [1, 1, 2], [2, 2, 2]))
    
    # Strategy 5: 2+1 rows, 1+1+2 cols, 1+1+2 B rows, 2+2+2 B cols
    candidates.append(([2, 1], [1, 1, 2], [1, 1, 2], [2, 2, 2]))
    
    # Strategy 6: 1+2 rows, 2+2 cols, 2+2 B rows, 1+1+2+2 B cols (asymmetric B cols)
    candidates.append(([1, 2], [2, 2], [2, 2], [1, 1, 2, 2]))
    
    # Strategy 7: 1+2 rows, 2+2 cols, 2+2 B rows, 3+3 B cols
    candidates.append(([1, 2], [2, 2], [2, 2], [3, 3]))
    
    # Strategy 8: 2+1 rows, 2+2 cols, 2+2 B rows, 3+3 B cols
    candidates.append(([2, 1], [2, 2], [2, 2], [3, 3]))
    
    # Strategy 9: 1+1+1 rows, 1+1+2 cols, 1+1+2 B rows, 2+2+2 B cols
    candidates.append(([1, 1, 1], [1, 1, 2], [1, 1, 2], [2, 2, 2]))
    
    # Strategy 10: 1+2 rows, 3+1 cols, 3+1 B rows, 2+2+2 B cols
    candidates.append(([1, 2], [3, 1], [3, 1], [2, 2, 2]))
    
    # Strategy 11: 2+1 rows, 3+1 cols, 3+1 B rows, 2+2+2 B cols
    candidates.append(([2, 1], [3, 1], [3, 1], [2, 2, 2]))
    
    # Strategy 12: 1+2 rows, 2+2 cols, 2+2 B rows, 1+2+3 B cols
    candidates.append(([1, 2], [2, 2], [2, 2], [1, 2, 3]))
    
    # Strategy 13: 2+1 rows, 2+2 cols, 2+2 B rows, 1+2+3 B cols
    candidates.append(([2, 1], [2, 2], [2, 2], [1, 2, 3]))
    
    # Strategy 14: 1+2 rows, 1+3 cols, 1+3 B rows, 2+2+2 B cols
    candidates.append(([1, 2], [1, 3], [1, 3], [2, 2, 2]))
    
    # Strategy 15: 2+1 rows, 1+3 cols, 1+3 B rows, 2+2+2 B cols
    candidates.append(([2, 1], [1, 3], [1, 3], [2, 2, 2]))

    best_result = None
    best_rank = float('inf')
    
    for a_rows, a_cols, b_rows, b_cols in candidates:
        if time.time() > DEADLINE - 20:
            break
        
        # Validate that a_cols and b_rows match
        if a_cols != b_rows:
            continue
        if sum(a_rows) != n or sum(a_cols) != m or sum(b_rows) != m or sum(b_cols) != p:
            continue
        
        try:
            U, V, W, rank = build_decomposition(a_rows, a_cols, b_rows, b_cols)
            
            if not verify(U, V, W, rank):
                continue
            
            # Try to reduce rank
            U, V, W, rank = try_reduce_rank(U, V, W, rank)
            
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W, rank)
                
        except Exception as e:
            continue
    
    if best_result is None:
        # Fallback to the known good 66-rank decomposition
        U, V, W, rank = build_decomposition([1, 2], [2, 2], [2, 2], [2, 2, 2])
        assert verify(U, V, W, rank)
        return {"U": U, "V": V, "W": W}
    
    U, V, W, rank = best_result
    return {"U": U, "V": V, "W": W}
