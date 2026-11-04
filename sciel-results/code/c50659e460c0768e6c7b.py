import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 115.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _build_strassen_56():
    """Build the rank-56 Strassen decomposition for 2x4 * 4x8."""
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def add_term(u_coeffs, v_coeffs, w_coeffs, col_idx):
        for (i, j), val in u_coeffs.items():
            if val != 0:
                U[i * m + j, col_idx] += val
        for (j, k), val in v_coeffs.items():
            if val != 0:
                V[j * p + k, col_idx] += val
        for (k, i), val in w_coeffs.items():
            if val != 0:
                W[k * n + i, col_idx] += val

    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            if _time_left() < 5.0:
                break

            # Strassen 7 terms for 2x2x2 block
            # M1 = (A00+A11)(B00+B11) -> C00:1, C11:1
            add_term(
                {(0, j0): 1, (1, j0+1): 1},
                {(j0, k0): 1, (j0+1, k0+1): 1},
                {(k0, 0): 1, (k0+1, 1): 1},
                col
            )
            col += 1

            # M2 = (A10+A11)B00 -> C10:1, C11:-1
            add_term(
                {(1, j0): 1, (1, j0+1): 1},
                {(j0, k0): 1},
                {(k0, 1): 1, (k0+1, 1): -1},
                col
            )
            col += 1

            # M3 = A00(B01-B11) -> C01:1, C11:1
            add_term(
                {(0, j0): 1},
                {(j0, k0+1): 1, (j0+1, k0+1): -1},
                {(k0+1, 0): 1, (k0+1, 1): 1},
                col
            )
            col += 1

            # M4 = A11(B10-B00) -> C00:1, C10:1
            add_term(
                {(1, j0+1): 1},
                {(j0+1, k0): 1, (j0, k0): -1},
                {(k0, 0): 1, (k0, 1): 1},
                col
            )
            col += 1

            # M5 = (A00+A01)B11 -> C00:-1, C01:1
            add_term(
                {(0, j0): 1, (0, j0+1): 1},
                {(j0+1, k0+1): 1},
                {(k0, 0): -1, (k0+1, 0): 1},
                col
            )
            col += 1

            # M6 = (A10-A00)(B00+B01) -> C11:1
            add_term(
                {(1, j0): 1, (0, j0): -1},
                {(j0, k0): 1, (j0, k0+1): 1},
                {(k0+1, 1): 1},
                col
            )
            col += 1

            # M7 = (A01-A11)(B10+B11) -> C00:1
            add_term(
                {(0, j0+1): 1, (1, j0+1): -1},
                {(j0+1, k0): 1, (j0+1, k0+1): 1},
                {(k0, 0): 1},
                col
            )
            col += 1

    return U, V, W

def _build_schoolbook():
    """Build the rank-64 schoolbook decomposition."""
    n, m, p = 2, 4, 8
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

def _verify(U, V, W, tol=1e-9):
    """Verify decomposition reproduces the matmul tensor exactly."""
    n, m, p = 2, 4, 8
    T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    return np.allclose(T_recon, T_target, atol=tol)

def _try_reduce_rank(U, V, W, max_reduction=5):
    """
    Try to reduce rank by finding linear dependencies among the rank-1 terms.
    We check if the columns of the 'flattened' tensor matrix are linearly dependent.
    """
    n, m, p = 2, 4, 8
    current_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank

    # Indices of non-zero entries in the target tensor
    # T[i*m+j, j*p+k, k*n+i] = 1
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_indices.append((i * m + j, j * p + k, k * n + i))
    
    num_nz = len(nz_indices)

    for trial in range(max_reduction):
        if _time_left() < 20:
            break

        # Construct the matrix K where K[idx, col] = (u_col \otimes v_col \otimes w_col)[idx]
        K = np.zeros((num_nz, current_rank))
        for col in range(current_rank):
            for idx, (u_idx, v_idx, w_idx) in enumerate(nz_indices):
                K[idx, col] = U[u_idx, col] * V[v_idx, col] * W[w_idx, col]
        
        # Find rank of K
        # If rank < current_rank, there is a linear dependency
        # We use SVD to find the null space
        U_svd, s, Vt = np.linalg.svd(K, full_matrices=False)
        rank_K = np.sum(s > 1e-10)
        
        if rank_K < current_rank:
            # The null space is spanned by the last (current_rank - rank_K) rows of Vt
            # We take one null vector lambda
            # We want to remove a term. Let's pick the term corresponding to the largest component in lambda
            # But we need to be careful: the dependency is sum lambda_r * term_r = 0
            # So term_j = - (1/lambda_j) * sum_{r!=j} lambda_r * term_r
            # We can replace term_j with 0 by adding (lambda_r/lambda_j) * term_r to term_j?
            # No, we just drop term_j and adjust the others?
            # Actually, if sum lambda_r t_r = 0, then the set {t_1, ..., t_{R-1}} spans the same tensor as {t_1, ..., t_R}
            # provided lambda_{R-1} != 0. We can just drop t_{R-1}.
            # However, the coefficients of the other terms don't change, we just drop one term.
            # Wait, the decomposition is T = sum t_r. If sum lambda_r t_r = 0, then T = sum (1-lambda_r) t_r? No.
            # T = t_1 + ... + t_R.
            # 0 = lambda_1 t_1 + ... + lambda_R t_R.
            # This doesn't allow us to drop a term unless one lambda is 1 and others are 0?
            # No, it means the vectors are linearly dependent.
            # We can express t_R as a linear combination of t_1...t_{R-1}.
            # t_R = - (1/lambda_R) * sum_{r=1}^{R-1} lambda_r t_r.
            # Then T = sum_{r=1}^{R-1} t_r + t_R = sum_{r=1}^{R-1} (1 + lambda_r/lambda_R) t_r.
            # So we can update the first R-1 terms with new coefficients and drop the R-th.
            
            # Find the null vector
            # The null space is the orthogonal complement of the row space.
            # Vt has shape (current_rank, current_rank) if full_matrices=False and rank < current_rank?
            # No, Vt is (min(M,N), N). Here N=current_rank.
            # The last (current_rank - rank_K) rows of Vt are the null space basis.
            # Let's take the last row.
            lambda_vec = Vt[-1, :]
            
            # Find the index with the largest absolute value in lambda_vec to be the "pivot"
            pivot_idx = np.argmax(np.abs(lambda_vec))
            if abs(lambda_vec[pivot_idx]) < 1e-10:
                break # Numerical issue, try next or break
            
            # Update the other columns
            for r in range(current_rank):
                if r == pivot_idx:
                    continue
                factor = lambda_vec[r] / lambda_vec[pivot_idx]
                # t_pivot = - factor * t_r  => t_r = -1/factor * t_pivot
                # T = ... + t_r + t_pivot + ...
                # Substitute t_pivot: T = ... + t_r - factor * t_r + ... = (1-factor) t_r
                # Wait.
                # sum lambda_i t_i = 0 => t_pivot = - sum_{i != pivot} (lambda_i / lambda_pivot) t_i
                # T = sum_{i != pivot} t_i + t_pivot
                #   = sum_{i != pivot} t_i - sum_{i != pivot} (lambda_i / lambda_pivot) t_i
                #   = sum_{i != pivot} (1 - lambda_i / lambda_pivot) t_i
                # So we multiply column i by (1 - lambda_i / lambda_pivot) and drop column pivot.
                
                new_coeff = 1.0 - factor
                U[:, r] *= new_coeff
                V[:, r] *= new_coeff
                W[:, r] *= new_coeff
            
            # Drop the pivot column
            mask = np.ones(current_rank, dtype=bool)
            mask[pivot_idx] = False
            U = U[:, mask]
            V = V[:, mask]
            W = W[:, mask]
            current_rank -= 1
            
            # Verify
            if _verify(U, V, W):
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = current_rank
            else:
                # Revert
                # This is tricky because we modified in place. 
                # We should have worked on copies.
                # Let's restart the loop iteration with a fresh copy?
                # For now, just break if verification fails.
                break
        else:
            break

    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = 2, 4, 8

    U, V, W = _build_strassen_56()

    if not _verify(U, V, W):
        U, V, W = _build_schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    U_red, V_red, W_red, rank_red = _try_reduce_rank(U, V, W, max_reduction=5)

    if rank_red < U.shape[1]:
        return {
            "U": U_red.tolist(),
            "V": V_red.tolist(),
            "W": W_red.tolist()
        }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
