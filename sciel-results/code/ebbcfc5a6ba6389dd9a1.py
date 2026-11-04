import time
import random
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Build the known rank-28 Strassen solution
    U28, V28, W28 = _build_strassen_28(n, m, p)
    
    # Verify it's correct
    if not _verify(U28, V28, W28, n, m, p):
        # Fallback: schoolbook rank-32
        U32, V32, W32 = _build_schoolbook_32(n, m, p)
        return {"U": U32, "V": V32, "W": W32}
    
    best_rank = 28
    best_result = {"U": U28, "V": V28, "W": W28}
    
    # Try to find rank-27 or lower via sub-problem optimization
    if time.time() < deadline - 30:
        result = _search_lower_rank(n, m, p, deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                rank = len(U[0])
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
    
    return best_result


def _build_strassen_28(n, m, p):
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
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    blocks = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2)
    ]
    
    col_idx = 0
    for a_off, b_row_off, b_col_off, c_off in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for local_idx, val in u_coeffs.items():
                g_idx = (local_idx // 2) * m + a_off + (local_idx % 2)
                U[g_idx][col_idx] = val
            for local_idx, val in v_coeffs.items():
                g_idx = (b_row_off + local_idx // 2) * p + b_col_off + (local_idx % 2)
                V[g_idx][col_idx] = val
            for local_idx, val in w_coeffs.items():
                g_idx = (c_off + local_idx % 2) * n + (local_idx // 2)
                W[g_idx][col_idx] = val
            col_idx += 1
    
    return U, V, W


def _build_schoolbook_32(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _search_lower_rank(n, m, p, deadline):
    """
    Try to find a rank-27 or lower decomposition by optimizing sub-problems.
    
    Strategy: The 2x4x4 matmul can be decomposed into:
    - Two [2,4,2] sub-problems (left and right halves of B)
    - Combine results
    
    If we can find rank-9 for [2,4,2] instead of rank-10, we save 2 total.
    Similarly for [2,2,4] sub-problems.
    
    We use alternating least squares with rounding to half-integers.
    """
    # Try direct optimization for rank 27
    for target_rank in [27, 26, 25]:
        if time.time() > deadline - 15:
            break
        
        result = _optimize_rank(n, m, p, target_rank, deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return U, V, W
    
    # Try sub-problem approach: optimize [2,4,2] and [2,2,4] separately
    if time.time() < deadline - 20:
        result = _optimize_subproblems(n, m, p, deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return U, V, W
    
    return None


def _optimize_rank(n, m, p, target_rank, deadline):
    """
    Use alternating least squares to find a rank-target_rank decomposition.
    """
    np.random.seed(42)
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    num_restarts = 5
    max_iters = 200
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        np.random.seed(42 + restart)
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Solve for U
            U = _solve_factor(U, V, W, n, m, p, target_rank, 'U')
            # Solve for V
            V = _solve_factor(U, V, W, n, m, p, target_rank, 'V')
            # Solve for W
            W = _solve_factor(U, V, W, n, m, p, target_rank, 'W')
            
            # Check convergence
            err = _compute_error(U, V, W, n, m, p)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                break
        
        if best_err < 1e-10:
            # Try to round to half-integers
            rounded = _round_to_half_int(best_U, best_V, best_W, n, m, p)
            if rounded is not None:
                U_r, V_r, W_r = rounded
                if _verify(U_r, V_r, W_r, n, m, p):
                    return U_r, V_r, W_r
    
    return None


def _solve_factor(U, V, W, n, m, p, rank, factor):
    """
    Solve for one factor in the tensor decomposition using least squares.
    """
    if factor == 'U':
        # For each (i,j): T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Build linear system
        A = np.zeros((n * m * p, n * m * rank))
        b = np.ones(n * m * p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        col_idx = (i * m + j) * rank + r
                        A[eq_idx, col_idx] = V[j * p + k, r] * W[k * n + i, r]
                    eq_idx += 1
        
        U_flat, *_ = np.linalg.lstsq(A, b, rcond=None)
        return U_flat.reshape(n * m, rank)
    
    elif factor == 'V':
        A = np.zeros((n * m * p, m * p * rank))
        b = np.ones(n * m * p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        col_idx = (j * p + k) * rank + r
                        A[eq_idx, col_idx] = U[i * m + j, r] * W[k * n + i, r]
                    eq_idx += 1
        
        V_flat, *_ = np.linalg.lstsq(A, b, rcond=None)
        return V_flat.reshape(m * p, rank)
    
    elif factor == 'W':
        A = np.zeros((n * m * p, p * n * rank))
        b = np.ones(n * m * p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        col_idx = (k * n + i) * rank + r
                        A[eq_idx, col_idx] = U[i * m + j, r] * V[j * p + k, r]
                    eq_idx += 1
        
        W_flat, *_ = np.linalg.lstsq(A, b, rcond=None)
        return W_flat.reshape(p * n, rank)
    
    return None


def _compute_error(U, V, W, n, m, p):
    """Compute the maximum error in the tensor decomposition."""
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(U.shape[1]):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                err = abs(s - 1)
                if err > max_err:
                    max_err = err
    return max_err


def _round_to_half_int(U, V, W, n, m, p):
    """
    Try to round the factors to half-integer values while preserving the decomposition.
    """
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U_r]
    V_list = [[float(x) for x in row] for row in V_r]
    W_list = [[float(x) for x in row] for row in W_r]
    
    # Verify
    if _verify(U_list, V_list, W_list, n, m, p):
        return U_list, V_list, W_list
    
    # Try different rounding strategies
    for offset in [0, 0.25, 0.5, 0.75]:
        U_r = np.round((U + offset) * 2) / 2 - offset
        V_r = np.round((V + offset) * 2) / 2 - offset
        W_r = np.round((W + offset) * 2) / 2 - offset
        
        U_list = [[float(x) for x in row] for row in U_r]
        V_list = [[float(x) for x in row] for row in V_r]
        W_list = [[float(x) for x in row] for row in W_r]
        
        if _verify(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
    
    return None


def _optimize_subproblems(n, m, p, deadline):
    """
    Optimize sub-problems [2,4,2] and [2,2,4] separately, then combine.
    
    The 2x4x4 matmul can be viewed as:
    C[:, 0:2] = A @ B[:, 0:2]  -> [2,4,2] problem
    C[:, 2:4] = A @ B[:, 2:4]  -> [2,4,2] problem
    
    If we find rank-9 for each [2,4,2], total rank = 18 + some overhead.
    But this doesn't directly give us the full decomposition structure.
    
    Alternative: Use the block structure more carefully.
    """
    # For now, this is a placeholder for more sophisticated sub-problem optimization
    # that could potentially find rank-27 or lower
    return None
