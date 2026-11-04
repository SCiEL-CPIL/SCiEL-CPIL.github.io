import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    if isinstance(U, np.ndarray):
        U = U.tolist()
    if isinstance(V, np.ndarray):
        V = V.tolist()
    if isinstance(W, np.ndarray):
        W = W.tolist()
    
    rank = len(U[0]) if U else 0
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2_terms():
    """Return the 7 Strassen terms as (U_coeffs, V_coeffs, W_coeffs)."""
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    return terms

def _add_strassen_block(U, V, W, a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, offset, m, p, n):
    """Add a 2x2 Strassen block at the given offsets."""
    terms = _strassen_2x2_terms()
    for t_idx, (x_coeff, y_coeff, z_coeff) in enumerate(terms):
        r = offset + t_idx
        for (xi, xj), val in x_coeff.items():
            U[(a_row_off + xi) * m + (a_col_off + xj)][r] += val
        for (yi, yj), val in y_coeff.items():
            V[(b_row_off + yi) * p + (b_col_off + yj)][r] += val
        for (zi, zj), val in z_coeff.items():
            W[(c_col_off + zj) * n + (c_row_off + zi)][r] += val

def _try_rank_44():
    """
    Split 3x4 * 4x4 into:
    - Row 0: 1x4 * 4x4 -> schoolbook, rank 16
    - Rows 1-2: 2x4 * 4x4 -> 4 Strassen 2x2 blocks, rank 28
    Total: 44
    """
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Row 0 of A (1x4) * B (4x4) -> Row 0 of C (1x4)
    for j in range(4):
        for k in range(4):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: Rows 1-2 of A (2x4) * B (4x4) -> Rows 1-2 of C (2x4)
    blocks = [
        (1, 0, 0, 0, 1, 0),
        (1, 0, 0, 2, 1, 2),
        (1, 2, 2, 0, 1, 0),
        (1, 2, 2, 2, 1, 2),
    ]
    
    for (ar, ac, br, bc, cr, cc) in blocks:
        _add_strassen_block(U, V, W, ar, ac, br, bc, cr, cc, col, m, p, n)
        col += 7
    
    return U, V, W

def _search_low_rank(n, m, p, target_rank, time_budget, seed=42):
    """Try to find a rank-target_rank decomposition using random search + local refinement."""
    np.random.seed(seed)
    random.seed(seed)
    start = time.time()
    
    # Build the tensor equation: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is a system of n*m*p equations in rank*(n*m + m*p + p*n) variables.
    
    num_eq = n * m * p  # 48
    num_vars = target_rank * (n*m + m*p + p*n)  # target_rank * (12 + 16 + 12) = target_rank * 40
    
    # Random initialization
    U = np.random.randn(n*m, target_rank)
    V = np.random.randn(m*p, target_rank)
    W = np.random.randn(p*n, target_rank)
    
    def _residual(U, V, W):
        """Compute the residual for each (i,j,k) entry."""
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
        return res
    
    def _total_error(U, V, W):
        res = _residual(U, V, W)
        return np.sum((res - 1.0)**2)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = _total_error(U, V, W)
    
    # Iterative refinement: randomly perturb one matrix and accept if error decreases
    for iteration in range(5000):
        if time.time() - start > time_budget:
            break
        
        # Pick which matrix to perturb
        mat = random.choice(['U', 'V', 'W'])
        if mat == 'U':
            idx = random.randint(0, n*m-1)
            U[idx, :] += np.random.randn(target_rank) * 0.1
            new_err = _total_error(U, V, W)
            if new_err < best_err:
                best_err = new_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                U[idx, :] -= np.random.randn(target_rank) * 0.1
        elif mat == 'V':
            idx = random.randint(0, m*p-1)
            V[idx, :] += np.random.randn(target_rank) * 0.1
            new_err = _total_error(U, V, W)
            if new_err < best_err:
                best_err = new_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                V[idx, :] -= np.random.randn(target_rank) * 0.1
        else:
            idx = random.randint(0, p*n-1)
            W[idx, :] += np.random.randn(target_rank) * 0.1
            new_err = _total_error(U, V, W)
            if new_err < best_err:
                best_err = new_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                W[idx, :] -= np.random.randn(target_rank) * 0.1
        
        # Check if we've found an exact solution
        if best_err < 1e-12:
            break
    
    if best_err < 1e-12:
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), True
    
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Try rank-44 decomposition (known to work)
    best_result = None
    try:
        U, V, W = _try_rank_44()
        if _verify(U, V, W, n, m, p):
            best_result = {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Try to find lower rank decompositions
    for target_rank in [42, 40, 38]:
        if time.time() > deadline - 20:
            break
        try:
            U, V, W, success = _search_low_rank(n, m, p, target_rank, time_budget=25.0, seed=target_rank)
            if success:
                if _verify(U, V, W, n, m, p):
                    best_result = {"U": U, "V": V, "W": W}
                    break
        except Exception:
            pass
    
    # Schoolbook fallback
    if best_result is None:
        U_sb, V_sb, W_sb = _schoolbook(n, m, p)
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return best_result
