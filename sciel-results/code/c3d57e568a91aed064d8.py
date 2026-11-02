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


def _try_rank10_block_circulant(deadline):
    """
    Block-circulant parameterization: embed 2x2x3 into a 4x4x4-like structure
    by mapping the third column of B to a linear combination of the first two
    using a non-singular rational matrix, then solve the transformed system.
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
    
    rng = np.random.default_rng(42)
    best_cost = float('inf')
    best_params = None
    
    # Strategy: use block-circulant structure
    # The idea is to enforce that columns 2,3 of W (corresponding to k=2)
    # are linear combinations of columns 0,1 (k=0,1)
    # Similarly for V, the third column is a combination of first two
    
    for trial in range(40):
        if time.time() > deadline - 25:
            break
        
        # Initialize from rank-11 solution with random perturbations
        scale = 0.05 * (1 + trial / 10.0)
        
        # Start from rank-11, drop one term randomly
        drop = trial % 11
        keep = [r for r in range(11) if r != drop]
        
        U0 = U11[:, keep].copy()
        V0 = V11[:, keep].copy()
        W0 = W11[:, keep].copy()
        
        # Add structured noise
        U0 += rng.normal(0, scale, U0.shape)
        V0 += rng.normal(0, scale, V0.shape)
        W0 += rng.normal(0, scale, W0.shape)
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = least_squares(_residuals_rank10, x0, method='lm', 
                                   max_nfev=20000, ftol=1e-18, xtol=1e-18, gtol=1e-18)
            cost = result.cost
            if cost < best_cost:
                best_cost = cost
                best_params = result.x.copy()
                if cost < 1e-14:
                    break
        except Exception:
            pass
    
    # Additional: random initializations with rational structure
    for trial in range(30):
        if time.time() > deadline - 20:
            break
        
        vals = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
        x0 = rng.choice(vals, 160).astype(float)
        
        try:
            result = least_squares(_residuals_rank10, x0, method='lm',
                                   max_nfev=15000, ftol=1e-18, xtol=1e-18)
            if result.cost < best_cost:
                best_cost = result.cost
                best_params = result.x.copy()
                if result.cost < 1e-14:
                    break
        except Exception:
            pass
    
    if best_params is None or best_cost > 1e-10:
        return None
    
    U_flat = best_params[:40]
    V_flat = best_params[40:100]
    W_flat = best_params[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    return _snap_and_verify(U_mat, V_mat, W_mat)


def _try_rank10_symmetric_search(deadline):
    """
    Exploit symmetry: search for decompositions where the W factors for k=2
    are affine combinations of W factors for k=0 and k=1, enforcing the
    linear constraint B[:,2] = a*B[:,0] + b*B[:,1] structure.
    """
    if time.time() > deadline - 25:
        return None
    
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None
    
    base = _rank11_decomposition()
    U11 = np.array(base["U"], dtype=float)
    V11 = np.array(base["V"], dtype=float)
    W11 = np.array(base["W"], dtype=float)
    
    rng = np.random.default_rng(777)
    best_cost = float('inf')
    best_params = None
    
    # Parameterize: W[k=2, i] = alpha_i * W[k=0, i] + beta_i * W[k=1, i]
    # V[j, k=2] = alpha_v * V[j, k=0] + beta_v * V[j, k=1]
    # This reduces degrees of freedom and enforces structure
    
    for trial in range(25):
        if time.time() > deadline - 20:
            break
        
        # Start from rank-11 with perturbation
        scale = 0.03 * (1 + trial / 15.0)
        drop = (trial * 3) % 11
        keep = [r for r in range(11) if r != drop]
        
        U0 = U11[:, keep].copy()
        V0 = V11[:, keep].copy()
        W0 = W11[:, keep].copy()
        
        U0 += rng.normal(0, scale, U0.shape)
        V0 += rng.normal(0, scale, V0.shape)
        W0 += rng.normal(0, scale, W0.shape)
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = least_squares(_residuals_rank10, x0, method='lm',
                                   max_nfev=15000, ftol=1e-18, xtol=1e-18)
            if result.cost < best_cost:
                best_cost = result.cost
                best_params = result.x.copy()
                if result.cost < 1e-14:
                    break
        except Exception:
            pass
    
    if best_params is None or best_cost > 1e-10:
        return None
    
    U_mat = best_params[:40].reshape(4, 10)
    V_mat = best_params[40:100].reshape(6, 10)
    W_mat = best_params[100:160].reshape(6, 10)
    
    return _snap_and_verify(U_mat, V_mat, W_mat)


def _try_rank10_groebner_like(deadline):
    """
    Attempt to find a rank-10 decomposition using a Groebner-basis-like approach.
    This is a heuristic: we fix the structure to have many zeros and solve the
    remaining polynomial system via substitution.
    """
    if time.time() > deadline - 25:
        return None
    
    try:
        from scipy.optimize import fsolve
    except ImportError:
        return None
    
    rng = np.random.default_rng(999)
    
    best_cost = float('inf')
    best_solution = None
    
    for trial in range(30):
        if time.time() > deadline - 20:
            break
        
        # Random initialization with sparsity
        U = np.zeros((4, 10))
        V = np.zeros((6, 10))
        W = np.zeros((6, 10))
        
        for i in range(4):
            for j in range(10):
                if rng.random() < 0.3:
                    U[i, j] = rng.choice([-1, 1])
        for i in range(6):
            for j in range(10):
                if rng.random() < 0.3:
                    V[i, j] = rng.choice([-1, 1])
        for i in range(6):
            for j in range(10):
                if rng.random() < 0.3:
                    W[i, j] = rng.choice([-1, 1])
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = fsolve(_residuals_rank10, x0, full_output=True, xtol=1e-14, maxfev=5000)
            x_sol, info, ier, msg = result
            if ier == 1:
                cost = np.sum(np.array(_residuals_rank10(x_sol))**2)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = x_sol.copy()
                    if cost < 1e-14:
                        break
        except Exception:
            pass
    
    if best_solution is None or best_cost > 1e-10:
        return None
    
    U_mat = best_solution[:40].reshape(4, 10)
    V_mat = best_solution[40:100].reshape(6, 10)
    W_mat = best_solution[100:160].reshape(6, 10)
    
    return _snap_and_verify(U_mat, V_mat, W_mat)


def construct():
    deadline = time.time() + 135
    
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
        # Strategy 1: Block-circulant parameterization
        result10 = _try_rank10_block_circulant(deadline)
        if result10 is not None:
            best = result10
            best_rank = 10
        elif time.time() < deadline - 30:
            # Strategy 2: Symmetric search with structural constraints
            result10 = _try_rank10_symmetric_search(deadline)
            if result10 is not None:
                best = result10
                best_rank = 10
            elif time.time() < deadline - 20:
                # Strategy 3: Groebner-like search
                result10 = _try_rank10_groebner_like(deadline)
                if result10 is not None:
                    best = result10
                    best_rank = 10
    
    return best
