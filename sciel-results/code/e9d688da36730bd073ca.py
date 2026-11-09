import numpy as np
import time

def _matrix_exp_hermitian(H, t):
    """Compute exp(-1j * t * H) for Hermitian H."""
    vals, vecs = np.linalg.eigh(H)
    # exp(-1j * t * vals)
    exp_vals = np.exp(-1j * t * vals)
    return vecs @ np.diag(exp_vals) @ vecs.conj().T

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    n = 6
    dim = 2**n
    delta = 0.8
    h = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    Jz = 1.0
    Jxy = delta * Jz
    
    def add_bond(H, i, j):
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sz
        ops[j] = sz
        H += Jz * kron_all(ops)
        
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sx
        ops[j] = sx
        H += (Jxy / 2) * kron_all(ops)
        
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sy
        ops[j] = sy
        H += (Jxy / 2) * kron_all(ops)
    
    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        add_bond(A, i, j)
    
    for (i, j) in [(1, 2), (3, 4)]:
        add_bond(B, i, j)
    
    for i in range(n):
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sz
        field_term = h * 0.5 * kron_all(ops)
        A += field_term
        B += field_term
    
    return A, B

def _spectral_norm_error(a, b, A, B, t):
    """Compute spectral norm of U(t) - exp(-i t (A+B))."""
    dim = A.shape[0]
    U = np.eye(dim, dtype=complex)
    for k in range(len(a)):
        # Order: exp(-i b_k t B) @ exp(-i a_k t A)
        # Note: The task says U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) in order k=1..12
        # This usually means U = (M_12 ... M_1) where M_k = exp(-i b_k t B) exp(-i a_k t A)
        # Or is it U = M_1 M_2 ... M_12?
        # "in order k = 1..12" implies the product is formed by multiplying from left to right or right to left?
        # Standard convention for "product formula" U = prod_{k=1}^N U_k often means U = U_N ... U_1 if acting on a state vector from the right, or U_1 U_2 ... U_N if the first factor is applied first.
        # Let's check the seed code:
        # U = expm(...) @ expm(...) @ U
        # This accumulates on the left. So U_final = M_12 ... M_1.
        # This means M_1 is applied first to the state.
        U = _matrix_exp_hermitian(B, -b[k] * t) @ _matrix_exp_hermitian(A, -a[k] * t) @ U
        
    U_exact = _matrix_exp_hermitian(A + B, -t)
    diff = U - U_exact
    s = np.linalg.svd(diff, compute_uv=False)
    return s[0]

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 140.0
    
    A, B = _build_hamiltonians()
    dim = A.shape[0]
    times = [0.25, 0.5, 1.0]
    
    # Symmetric parameterization: 6 unique a's, 6 unique b's
    # a_full = [a1..a6, a6..a1]
    # b_full = [b1..b6, b6..b1]
    
    def get_full(half_params):
        a_half = half_params[:6]
        b_half = half_params[6:12]
        a_full = np.concatenate([a_half, a_half[::-1]])
        b_full = np.concatenate([b_half, b_half[::-1]])
        # Normalize
        a_sum = np.sum(a_full)
        b_sum = np.sum(b_full)
        if abs(a_sum) > 1e-12: a_full = a_full / a_sum
        else: a_full = np.ones(12)/12
        if abs(b_sum) > 1e-12: b_full = b_full / b_sum
        else: b_full = np.ones(12)/12
        return a_full, b_full

    def objective(params):
        a_full, b_full = get_full(params)
        total_err = 0.0
        for t in times:
            total_err += _spectral_norm_error(a_full, b_full, A, B, t)
        return total_err / len(times)

    # Initial guess: Uniform
    x0 = np.ones(12) / 12
    
    # Simple Nelder-Mead implementation
    def nelder_mead(func, x0, max_iter=1000, tol=1e-6):
        n = len(x0)
        # Initial simplex
        simplex = [x0.copy() for _ in range(n+1)]
        simplex[0] = x0
        for i in range(1, n+1):
            simplex[i] = x0.copy()
            simplex[i][i-1] += 0.1 if x0[i-1] == 0 else 0.1 * x0[i-1]
        
        fvals = [func(s) for s in simplex]
        
        for it in range(max_iter):
            if time.time() > deadline - 10: break
            # Sort
            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]
            
            if abs(fvals[0] - fvals[-1]) < tol:
                break
                
            # Centroid of best n points
            centroid = np.mean(simplex[:n], axis=0)
            
            # Reflection
            xr = centroid + (centroid - simplex[-1])
            fr = func(xr)
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + 2 * (centroid - simplex[-1])
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + 0.5 * (centroid - simplex[-1])
                else:
                    # Inside contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                fc = func(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        
        return simplex[0], fvals[0]

    best_x = x0.copy()
    best_obj = objective(x0)
    
    # Run Nelder-Mead
    try:
        res_x, res_f = nelder_mead(objective, x0, max_iter=200)
        if res_f < best_obj:
            best_obj = res_f
            best_x = res_x
    except Exception:
        pass
        
    # Try a few random restarts if time permits
    if time.time() < deadline - 30:
        for _ in range(3):
            if time.time() > deadline - 10: break
            x_rand = best_x + 0.05 * np.random.randn(12)
            try:
                res_x, res_f = nelder_mead(objective, x_rand, max_iter=100)
                if res_f < best_obj:
                    best_obj = res_f
                    best_x = res_x
            except:
                pass

    a_full, b_full = get_full(best_x)
    
    # Clip and renormalize one last time
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)
    a_sum = np.sum(a_full)
    b_sum = np.sum(b_full)
    if abs(a_sum) > 1e-12: a_full = a_full / a_sum
    if abs(b_sum) > 1e-12: b_full = b_full / b_sum

    return [float(x) for x in a_full], [float(x) for x in b_full]
