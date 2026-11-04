import numpy as np
import time

def _verify_decomposition(U, V, W, n=4, m=4, p=4):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = int(np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]))
                if s != 1:
                    return False
    return True

def _try_reduce_rank(U, V, W, n=4, m=4, p=4, deadline=None):
    """Try to reduce rank by greedily removing linearly dependent terms."""
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    removed = []
    
    # Sort columns by norm to try removing small-norm terms first
    col_norms = np.linalg.norm(current_U * current_V * current_W, axis=0)
    col_order = np.argsort(col_norms)
    
    for attempt in range(current_rank):
        if deadline is not None and time.time() > deadline:
            break
        found_removal = False
        for col in col_order:
            if col in removed or col >= current_rank:
                continue
            # Try removing this column
            mask = np.ones(current_rank, dtype=bool)
            mask[col] = False
            test_U = current_U[:, mask]
            test_V = current_V[:, mask]
            test_W = current_W[:, mask]
            
            if _verify_decomposition(test_U, test_V, test_W, n, m, p):
                current_U = test_U
                current_V = test_V
                current_W = test_W
                current_rank -= 1
                removed.append(col)
                found_removal = True
                break
        if not found_removal:
            break
    
    return current_U, current_V, current_W

def construct():
    """
    Matrix multiplication 4x4x4 via Kronecker product of Strassen's 2x2x2 algorithm.
    
    Uses the rank-7 Strassen decomposition for 2x2x2, then takes the Kronecker
    product to get a rank-49 decomposition for 4x4x4. Row permutations are applied
    to match the linear indexing conventions required by the task.
    
    Then attempts greedy rank reduction to find a lower-rank decomposition.
    
    Interface: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    The decomposition satisfies: for all i,j,k,
        sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    """
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120s limit
    
    # Strassen's 2x2x2 algorithm
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Verify 2x2 Strassen
    if not _verify_decomposition(U2, V2, W2, 2, 2, 2):
        # Fallback to schoolbook
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii * m + jj][col] = 1
                    V[jj * p + kk][col] = 1
                    W[kk * n + ii][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Kronecker products give 16x49 matrices
    U4_raw = np.kron(U2, U2)
    V4_raw = np.kron(V2, V2)
    W4_raw = np.kron(W2, W2)
    
    # Compute row permutations to match task's linear indexing
    perm_U = np.zeros(16, dtype=int)
    perm_V = np.zeros(16, dtype=int)
    perm_W = np.zeros(16, dtype=int)
    
    for S in range(16):
        i = S // 4
        j = S % 4
        i_out, i_in = i // 2, i % 2
        j_out, j_in = j // 2, j % 2
        perm_U[S] = (i_out * 2 + j_out) * 4 + (i_in * 2 + j_in)
        
        j = S // 4
        k = S % 4
        j_out, j_in = j // 2, j % 2
        k_out, k_in = k // 2, k % 2
        perm_V[S] = (j_out * 2 + k_out) * 4 + (j_in * 2 + k_in)
        
        k = S // 4
        i = S % 4
        k_out, k_in = k // 2, k % 2
        i_out, i_in = i // 2, i % 2
        perm_W[S] = (k_out * 2 + i_out) * 4 + (k_in * 2 + i_in)
    
    U4 = U4_raw[perm_U, :]
    V4 = V4_raw[perm_V, :]
    W4 = W4_raw[perm_W, :]
    
    # Verify 4x4 correctness
    if not _verify_decomposition(U4, V4, W4, 4, 4, 4):
        # Fallback to schoolbook
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii * m + jj][col] = 1
                    V[jj * p + kk][col] = 1
                    W[kk * n + ii][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Attempt greedy rank reduction
    reduced_U, reduced_V, reduced_W = _try_reduce_rank(U4, V4, W4, 4, 4, 4, deadline)
    
    # Verify reduced decomposition
    if _verify_decomposition(reduced_U, reduced_V, reduced_W, 4, 4, 4):
        return {
            "U": reduced_U.tolist(),
            "V": reduced_V.tolist(),
            "W": reduced_W.tolist()
        }
    else:
        # Fall back to rank-49
        return {
            "U": U4.tolist(),
            "V": V4.tolist(),
            "W": W4.tolist()
        }
