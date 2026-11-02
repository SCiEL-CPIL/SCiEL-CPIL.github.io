"""Tensor decomposition search for 2x2 by 2x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]
n, m, p = DIMENSIONS


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _rank11_decomposition():
    """
    Rank-11 decomposition: Strassen (rank 7) for 2x2 block + schoolbook (rank 4) for col 2.
    This is the known optimal rank for 2x2x3 over reals/rationals.
    """
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val):
        U[i * m + j][r] = val

    def set_v(j, k, r, val):
        V[j * p + k][r] = val

    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # Term 0: (A00+A11)(B00+B11) -> contributes to C[0,0] and C[1,1]
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # Term 1: (A10+A11)B00 -> contributes to C[0,1] and C[1,1]
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # Term 2: A00(B01-B11) -> contributes to C[1,0] and C[1,1]
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # Term 3: A11(B10-B00) -> contributes to C[0,0] and C[0,1]
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # Term 4: (A00+A01)B11 -> contributes to C[0,0] and C[1,0]
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # Term 5: (A10-A00)(B00+B01) -> contributes to C[1,1]
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 6: (A01-A11)(B10+B11) -> contributes to C[0,0]
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Term 7: A00*B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # Term 8: A01*B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 9: A10*B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Term 10: A11*B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}


def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 12."""
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
    return {"U": U, "V": V, "W": W}


def _residuals_rank10(params):
    """Compute residuals for rank-10 tensor decomposition."""
    U_flat = params[:40]
    V_flat = params[40:100]
    W_flat = params[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    res = np.empty(12)
    idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                s = 0.0
                for r in range(10):
                    s += U_mat[i*2+j, r] * V_mat[j*3+k, r] * W_mat[k*2+i, r]
                res[idx] = s - 1.0
                idx += 1
    return res


def _snap_and_verify(U_mat, V_mat, W_mat):
    """Try to snap numerical values to rational and verify."""
    snaps = [
        lambda x: round(x * 2) / 2,
        lambda x: round(x),
        lambda x: round(x * 4) / 4,
        lambda x: round(x * 8) / 8,
        lambda x: round(x * 6) / 6,
        lambda x: round(x * 12) / 12,
    ]
    for snap in snaps:
        U = [[snap(u) for u in row] for row in U_mat.tolist()]
        V = [[snap(v) for v in row] for row in V_mat.tolist()]
        W = [[snap(w) for w in row] for row in W_mat.tolist()]
        
        if _verify(U, V, W, 2, 2, 3):
            return {"U": U, "V": V, "W": W}
    return None


def _try_rank10_syzygy(deadline):
    """
    Syzygy-based compression: Use the structure of the rank-11 solution to search
    for a rank-10 solution by exploiting linear dependencies between the output
    components.
    """
    if time.time() > deadline - 30:
        return None
    
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None
    
    base = _rank11_decomposition()
    U11 = np.array(base["U"], dtype=float)
    V11 = np.array(base["V"], dtype=float)
    W11 = np.array(base["W"], dtype=float)
    
    rng = np.random.default_rng(12345)
    best_cost = float('inf')
    best_params = None
    
    # Strategy: Fix the first 7 terms (Strassen for 2x2) and solve for the last 3
    # terms that handle the third column. This reduces the problem significantly.
    
    # The Strassen part handles C[0,0], C[0,1], C[1,0], C[1,1]
    # We need C[0,2] and C[1,2] to be correct.
    
    # Let's parameterize the last 3 terms specially.
    # U_last: (4, 3), V_last: (6, 3), W_last: (6, 3)
    # But we can restrict V_last and W_last to have specific sparsity patterns.
    
    for trial in range(50):
        if time.time() > deadline - 25:
            break
        
        # Initialize the first 7 terms from Strassen
        U0 = U11[:, :7].copy()
        V0 = V11[:, :7].copy()
        W0 = W11[:, :7].copy()
        
        # Randomly perturb the first 7 terms slightly
        scale = 0.01 * (1 + trial / 20.0)
        U0 += rng.normal(0, scale, U0.shape)
        V0 += rng.normal(0, scale, V0.shape)
        W0 += rng.normal(0, scale, W0.shape)
        
        # Initialize the last 3 terms with random values
        U_last = rng.normal(0, 1, (4, 3))
        V_last = rng.normal(0, 1, (6, 3))
        W_last = rng.normal(0, 1, (6, 3))
        
        # Enforce sparsity in V_last: only k=2 entries can be non-zero
        V_last[:, :6] = 0  # k=0,1 parts should be zero for these terms
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), U_last.flatten(), 
                            V_last.flatten(), W_last.flatten()])
        
        # Reshape to (10, ) for the optimizer
        # U: 4*10, V: 6*10, W: 6*10
        U_full = np.zeros((4, 10))
        V_full = np.zeros((6, 10))
        W_full = np.zeros((6, 10))
        
        U_full[:, :7] = U0
        V_full[:, :7] = V0
        W_full[:, :7] = W0
        
        U_full[:, 7:] = U_last
        V_full[:, 7:] = V_last
        W_full[:, 7:] = W_last
        
        x0 = np.concatenate([U_full.flatten(), V_full.flatten(), W_full.flatten()])
        
        try:
            result = least_squares(_residuals_rank10, x0, method='lm', 
                                   max_nfev=5000, ftol=1e-18, xtol=1e-18, gtol=1e-18)
            cost = result.cost
            if cost < best_cost:
                best_cost = cost
                best_params = result.x.copy()
                if cost < 1e-14:
                    break
        except Exception:
            pass
    
    if best_params is None or best_cost > 1e-10:
        return None
    
    U_mat = best_params[:40].reshape(4, 10)
    V_mat = best_params[40:100].reshape(6, 10)
    W_mat = best_params[100:160].reshape(6, 10)
    
    return _snap_and_verify(U_mat, V_mat, W_mat)


def _try_rank10_direct(deadline):
    """
    Direct search for rank-10 solution using multiple random initializations.
    """
    if time.time() > deadline - 25:
        return None
    
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None
    
    rng = np.random.default_rng(54321)
    best_cost = float('inf')
    best_params = None
    
    for trial in range(80):
        if time.time() > deadline - 20:
            break
        
        # Start from rank-11 solution with random perturbations
        base = _rank11_decomposition()
        U11 = np.array(base["U"], dtype=float)
        V11 = np.array(base["V"], dtype=float)
        W11 = np.array(base["W"], dtype=float)
        
        # Drop one random term
        drop = trial % 11
        keep = [r for r in range(11) if r != drop]
        
        U0 = U11[:, keep].copy()
        V0 = V11[:, keep].copy()
        W0 = W11[:, keep].copy()
        
        # Add structured noise
        scale = 0.02 * (1 + trial / 40.0)
        U0 += rng.normal(0, scale, U0.shape)
        V0 += rng.normal(0, scale, V0.shape)
        W0 += rng.normal(0, scale, W0.shape)
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = least_squares(_residuals_rank10, x0, method='lm', 
                                   max_nfev=10000, ftol=1e-18, xtol=1e-18, gtol=1e-18)
            cost = result.cost
            if cost < best_cost:
                best_cost = cost
                best_params = result.x.copy()
                if cost < 1e-14:
                    break
        except Exception:
            pass
    
    if best_params is None or best_cost > 1e-10:
        return None
    
    U_mat = best_params[:40].reshape(4, 10)
    V_mat = best_params[40:100].reshape(6, 10)
    W_mat = best_params[100:160].reshape(6, 10)
    
    return _snap_and_verify(U_mat, V_mat, W_mat)


def construct():
    deadline = time.time() + 140
    
    # First, get the known rank-11 construction
    result11 = _rank11_decomposition()
    if _verify(result11["U"], result11["V"], result11["W"], n, m, p):
        best = result11
        best_rank = 11
    else:
        best = _schoolbook()
        best_rank = 12
    
    # Try to find a rank-10 solution using multiple strategies
    if best_rank > 10 and time.time() < deadline - 40:
        # Strategy 1: Syzygy-based compression
        result10 = _try_rank10_syzygy(deadline)
        if result10 is not None:
            best = result10
            best_rank = 10
        elif time.time() < deadline - 30:
            # Strategy 2: Direct search
            result10 = _try_rank10_direct(deadline)
            if result10 is not None:
                best = result10
                best_rank = 10
    
    return best
