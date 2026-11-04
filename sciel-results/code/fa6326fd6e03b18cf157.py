import time
import numpy as np

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT - 5.0

    def to_clean_list(arr):
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                half = round(val * 2) / 2
                if abs(val - half) < 1e-8:
                    clean_row.append(half if half != int(half) else int(half))
                else:
                    clean_row.append(val)
            result.append(clean_row)
        return result

    def build_decomposition(j_split, k_split):
        """Build decomposition given splits of middle and output dims."""
        U = [[0] * 200 for _ in range(n * m)]
        V = [[0] * 200 for _ in range(m * p)]
        W = [[0] * 200 for _ in range(p * n)]
        col = 0

        def add_strassen_block(i_base, j_base, k_base):
            nonlocal col
            if col >= 200:
                return
            U[i_base * m + j_base][col] += 1
            U[(i_base+1) * m + (j_base+1)][col] += 1
            V[j_base * p + k_base][col] += 1
            V[(j_base+1) * p + (k_base+1)][col] += 1
            W[k_base * n + i_base][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] += 1
            col += 1
            
            U[(i_base+1) * m + j_base][col] += 1
            U[(i_base+1) * m + (j_base+1)][col] += 1
            V[j_base * p + k_base][col] += 1
            W[k_base * n + (i_base+1)][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] -= 1
            col += 1
            
            U[i_base * m + j_base][col] += 1
            V[j_base * p + (k_base+1)][col] += 1
            V[(j_base+1) * p + (k_base+1)][col] -= 1
            W[(k_base+1) * n + i_base][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] += 1
            col += 1
            
            U[(i_base+1) * m + (j_base+1)][col] += 1
            V[(j_base+1) * p + k_base][col] += 1
            V[j_base * p + k_base][col] -= 1
            W[k_base * n + i_base][col] += 1
            W[k_base * n + (i_base+1)][col] += 1
            col += 1
            
            U[i_base * m + j_base][col] += 1
            U[i_base * m + (j_base+1)][col] += 1
            V[(j_base+1) * p + (k_base+1)][col] += 1
            W[k_base * n + i_base][col] -= 1
            W[(k_base+1) * n + i_base][col] += 1
            col += 1
            
            U[(i_base+1) * m + j_base][col] += 1
            U[i_base * m + j_base][col] -= 1
            V[j_base * p + k_base][col] += 1
            V[j_base * p + (k_base+1)][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] += 1
            col += 1
            
            U[i_base * m + (j_base+1)][col] += 1
            U[(i_base+1) * m + (j_base+1)][col] -= 1
            V[(j_base+1) * p + k_base][col] += 1
            V[(j_base+1) * p + (k_base+1)][col] += 1
            W[k_base * n + i_base][col] += 1
            col += 1

        def add_naive_block(i_base, j_base, k_base, n_rows=1, n_cols=1, n_out=1):
            nonlocal col
            for ii in range(n_rows):
                for kk in range(n_out):
                    if col >= 200:
                        return
                    U[(i_base+ii) * m + (j_base)][col] += 1
                    V[(j_base) * p + (k_base+kk)][col] += 1
                    W[(k_base+kk) * n + (i_base+ii)][col] += 1
                    col += 1

        # Process blocks based on splits
        j_positions = []
        start = 0
        for size in j_split:
            j_positions.append((start, size))
            start += size

        k_positions = []
        start = 0
        for size in k_split:
            k_positions.append((start, size))
            start += size

        for (j_start, j_size) in j_positions:
            for (k_start, k_size) in k_positions:
                if j_size == 2 and k_size == 2:
                    add_strassen_block(0, j_start, k_start)
                elif j_size == 1 and k_size == 2:
                    # Naive: 1 row of A (both i=0,1), 1 col of B, 2 cols of output
                    for ii in range(2):
                        for kk in range(2):
                            if col >= 200:
                                return None, 0
                            U[(ii) * m + j_start][col] += 1
                            V[j_start * p + (k_start+kk)][col] += 1
                            W[(k_start+kk) * n + ii][col] += 1
                            col += 1
                elif j_size == 2 and k_size == 1:
                    for ii in range(2):
                        for jj in range(2):
                            if col >= 200:
                                return None, 0
                            U[ii * m + (j_start+jj)][col] += 1
                            V[(j_start+jj) * p + k_start][col] += 1
                            W[k_start * n + ii][col] += 1
                            col += 1
                elif j_size == 1 and k_size == 1:
                    for ii in range(2):
                        if col >= 200:
                            return None, 0
                        U[ii * m + j_start][col] += 1
                        V[j_start * p + k_start][col] += 1
                        W[k_start * n + ii][col] += 1
                        col += 1
                elif j_size == 3 and k_size == 2:
                    # Naive 2x3 * 3x2
                    for ii in range(2):
                        for jj in range(3):
                            for kk in range(2):
                                if col >= 200:
                                    return None, 0
                                U[ii * m + (j_start+jj)][col] += 1
                                V[(j_start+jj) * p + (k_start+kk)][col] += 1
                                W[(k_start+kk) * n + ii][col] += 1
                                col += 1
                elif j_size == 2 and k_size == 3:
                    for ii in range(2):
                        for jj in range(2):
                            for kk in range(3):
                                if col >= 200:
                                    return None, 0
                                U[ii * m + (j_start+jj)][col] += 1
                                V[(j_start+jj) * p + (k_start+kk)][col] += 1
                                W[(k_start+kk) * n + ii][col] += 1
                                col += 1
                elif j_size == 3 and k_size == 3:
                    for ii in range(2):
                        for jj in range(3):
                            for kk in range(3):
                                if col >= 200:
                                    return None, 0
                                U[ii * m + (j_start+jj)][col] += 1
                                V[(j_start+jj) * p + (k_start+kk)][col] += 1
                                W[(k_start+kk) * n + ii][col] += 1
                                col += 1
                else:
                    # Fallback naive
                    for ii in range(2):
                        for jj in range(j_size):
                            for kk in range(k_size):
                                if col >= 200:
                                    return None, 0
                                U[ii * m + (j_start+jj)][col] += 1
                                V[(j_start+jj) * p + (k_start+kk)][col] += 1
                                W[(k_start+kk) * n + ii][col] += 1
                                col += 1

        if col == 0:
            return None, 0

        U_arr = np.array(U[:200], dtype=float)[:, :col]
        V_arr = np.array(V[:200], dtype=float)[:, :col]
        W_arr = np.array(W[:200], dtype=float)[:, :col]
        return (U_arr, V_arr, W_arr), col

    def reduce_rank(U_arr, V_arr, W_arr, col):
        """Reduce rank using SVD-based column selection and greedy elimination."""
        T_size = n * m * p
        M = np.zeros((T_size, col))
        for r in range(col):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        M[idx, r] = U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]

        rank_M = np.linalg.matrix_rank(M, tol=1e-10)

        if rank_M >= col:
            return U_arr, V_arr, W_arr, col

        # SVD-based: find a maximal linearly independent set of columns
        # Use QR with column pivoting
        try:
            from scipy.linalg import svd
            U_svd, s_vals, Vt_svd = svd(M, full_matrices=False)
            # Effective rank
            eff_rank = np.sum(s_vals > 1e-10)
            if eff_rank < rank_M:
                rank_M = eff_rank

            # Select columns with largest singular values contribution
            # Use QR with column pivoting for a numerically stable basis
            from scipy.linalg import qr
            Q, R, P = qr(M, pivoting=True)
            # The first rank_M columns (after permutation) form a basis
            keep_indices = P[:rank_M].tolist()
        except:
            # Fallback to greedy
            keep = list(range(col))
            for idx in list(keep):
                if time.time() > deadline:
                    break
                test_keep = [k for k in keep if k != idx]
                if len(test_keep) == 0:
                    break
                M_test = M[:, test_keep]
                if np.linalg.matrix_rank(M_test, tol=1e-10) == rank_M:
                    keep = test_keep

            keep_indices = keep

        if len(keep_indices) >= col:
            return U_arr, V_arr, W_arr, col

        # Further greedy elimination on the reduced set
        keep = list(keep_indices)
        for _pass in range(10):
            if time.time() > deadline:
                break
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for idx in list(keep):
                    test_keep = [k for k in keep if k != idx]
                    if len(test_keep) == 0:
                        break
                    M_test = M[:, test_keep]
                    if np.linalg.matrix_rank(M_test, tol=1e-10) == rank_M:
                        keep = test_keep
                        changed = True
                        break

        if len(keep) >= col:
            return U_arr, V_arr, W_arr, col

        new_rank = len(keep)
        U_new = U_arr[:, keep]
        V_new = V_arr[:, keep]
        W_new = W_arr[:, keep]

        # Verify
        verified = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_new[i*m+j, r] * V_new[j*p+k, r] * W_new[k*n+i, r] for r in range(new_rank))
                    if abs(val - 1.0) > 1e-9:
                        verified = False
                        break
                if not verified:
                    break
            if not verified:
                break

        if verified:
            return U_new, V_new, W_new, new_rank
        else:
            return U_arr, V_arr, W_arr, col

    # Try multiple block decompositions
    best_result = None
    best_rank = 999

    # Decomposition strategies: (j_split, k_split)
    strategies = [
        ([2, 2, 1], [2, 2, 2]),  # Current: 54
        ([2, 1, 2], [2, 2, 2]),  # Alternative ordering
        ([1, 2, 2], [2, 2, 2]),  # Another ordering
        ([2, 2, 1], [1, 2, 2, 1]),  # Finer k split
        ([2, 2, 1], [2, 1, 2, 1]),
        ([2, 1, 2], [2, 2, 1, 1]),
        ([3, 2], [3, 3]),  # 3+2 middle, 3+3 output
        ([2, 3], [3, 3]),  # 2+3 middle, 3+3 output
        ([2, 2, 1], [3, 3]),  # 2+2+1 middle, 3+3 output
        ([2, 1, 2], [3, 3]),
        ([1, 2, 2], [3, 3]),
    ]

    for j_split, k_split in strategies:
        if time.time() > deadline:
            break
        result, col = build_decomposition(j_split, k_split)
        if result is None:
            continue
        U_arr, V_arr, W_arr = result
        if col >= best_rank:
            continue

        U_red, V_red, W_red, new_rank = reduce_rank(U_arr, V_arr, W_arr, col)

        if new_rank < best_rank:
            best_rank = new_rank
            best_result = (U_red, V_red, W_red)

        if best_rank <= 50:
            break

    if best_result is not None:
        U_out, V_out, W_out = best_result
        return {"U": to_clean_list(U_out), "V": to_clean_list(V_out), "W": to_clean_list(W_out)}

    # Fallback: original 54-rank decomposition
    U = [[0] * 54 for _ in range(n * m)]
    V = [[0] * 54 for _ in range(m * p)]
    W = [[0] * 54 for _ in range(p * n)]
    col = 0

    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_naive_2112_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    add_naive_2112_block(0, 4, 0)
    add_naive_2112_block(0, 4, 2)
    add_naive_2112_block(0, 4, 4)

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    return {"U": to_clean_list(U_arr), "V": to_clean_list(V_arr), "W": to_clean_list(W_arr)}
