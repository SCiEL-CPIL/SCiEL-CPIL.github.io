"""Optimized product formula design for Heisenberg-chain time evolution."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START

def _build_hamiltonians():
    N = 6
    aniso = 0.8
    field = 0.3
    dim = 2 ** N

    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    def kron_all(ops):
        out = np.array([[1.0]], dtype=complex)
        for op in ops:
            out = np.kron(out, op)
        return out

    def bond_ham(i):
        ops_x = [I2] * N
        ops_x[i] = sx
        ops_x[i + 1] = sx
        ops_y = [I2] * N
        ops_y[i] = sy
        ops_y[i + 1] = sy
        ops_z = [I2] * N
        ops_z[i] = sz
        ops_z[i + 1] = sz
        return kron_all(ops_x) + kron_all(ops_y) + aniso * kron_all(ops_z)

    field_full = np.zeros((dim, dim), dtype=complex)
    for s in range(N):
        ops = [I2] * N
        ops[s] = sz
        field_full += kron_all(ops)
    half_field = 0.5 * field_full

    A = half_field.copy()
    for i in (0, 2, 4):
        A += bond_ham(i)
    B = half_field.copy()
    for i in (1, 3):
        B += bond_ham(i)

    return A, B, A + B

def _precompute_eigh(op):
    w, V = np.linalg.eigh(op)
    return w, V

def _expm_from_eigh(w, V, scale):
    # exp(-i * scale * op) where op has eigenvalues w and eigenvectors V
    return V @ np.diag(np.exp(-1j * scale * w)) @ V.conj().T

def _objective_fast(a_full, b_full, A_eig, B_eig, H_eig, eval_times, dim):
    """Compute mean spectral norm error using Frobenius norm as smooth surrogate."""
    wH, VH = H_eig
    total_err = 0.0
    for t in eval_times:
        # Exact propagator
        U_exact = VH @ np.diag(np.exp(-1j * t * wH)) @ VH.conj().T
        # Product formula
        U = np.eye(dim, dtype=complex)
        wA, VA = A_eig
        wB, VB = B_eig
        for k in range(12):
            Eb = _expm_from_eigh(wB, VB, b_full[k] * t)
            Ea = _expm_from_eigh(wA, VA, a_full[k] * t)
            U = Eb @ Ea @ U
        E = U - U_exact
        # Use Frobenius norm squared as smooth proxy (cheaper than SVD)
        total_err += np.sum(np.abs(E) ** 2)
    return total_err / len(eval_times)

def _spectral_error_final(a_full, b_full, A_eig, B_eig, H_eig, eval_times, dim):
    """Compute actual mean spectral norm error for final scoring."""
    wH, VH = H_eig
    total_err = 0.0
    for t in eval_times:
        U_exact = VH @ np.diag(np.exp(-1j * t * wH)) @ VH.conj().T
        U = np.eye(dim, dtype=complex)
        wA, VA = A_eig
        wB, VB = B_eig
        for k in range(12):
            Eb = _expm_from_eigh(wB, VB, b_full[k] * t)
            Ea = _expm_from_eigh(wA, VA, a_full[k] * t)
            U = Eb @ Ea @ U
        E = U - U_exact
        s = np.linalg.svd(E, compute_uv=False)
        total_err += s[0]
    return total_err / len(eval_times)

def _nelder_mead(f, x0, bounds, max_iter=200, tol=1e-8, time_limit=None, start_time=None):
    """Pure NumPy Nelder-Mead simplex optimizer with bounds handling."""
    n = len(x0)
    # Initialize simplex
    simplex = [x0.copy()]
    f_vals = [f(x0)]
    
    # Create initial simplex with small perturbations
    for i in range(n):
        xi = x0.copy()
        step = 0.01 * max(1.0, abs(x0[i]))
        xi[i] += step
        xi = np.clip(xi, bounds[0], bounds[1])
        simplex.append(xi)
        f_vals.append(f(xi))
    
    best_idx = np.argmin(f_vals)
    best_x = simplex[best_idx].copy()
    best_f = f_vals[best_idx]
    
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink
    
    for it in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Sort simplex by function value
        order = np.argsort(f_vals)
        simplex = [simplex[i].copy() for i in order]
        f_vals = [f_vals[i] for i in order]
        
        if abs(f_vals[-1] - f_vals[0]) < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, bounds[0], bounds[1])
        f_r = f(x_r)
        
        if f_r < f_vals[0]:
            # Try expansion
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, bounds[0], bounds[1])
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
        elif f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, bounds[0], bounds[1])
            f_c = f(x_c)
            if f_c < f_vals[-1]:
                simplex[-1] = x_c
                f_vals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], bounds[0], bounds[1])
                    f_vals[i] = f(simplex[i])
        
        # Update best
        current_best_idx = np.argmin(f_vals)
        if f_vals[current_best_idx] < best_f:
            best_f = f_vals[current_best_idx]
            best_x = simplex[current_best_idx].copy()
    
    return best_x, best_f

def coefficients(stages):
    """
    12-stage product formula for a 6-site open XXZ Heisenberg chain
    (anisotropy 0.8, field 0.3) split into odd (A) and even (B) bonds.
    Uses Nelder-Mead optimization on a smooth Frobenius-norm surrogate.
    """
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 120s kill
    
    # Build Hamiltonians
    A, B, H = _build_hamiltonians()
    dim = 2 ** 6
    
    # Precompute eigendecompositions
    A_eig = _precompute_eigh(A)
    B_eig = _precompute_eigh(B)
    H_eig = _precompute_eigh(H)
    
    eval_times = [0.25, 0.5, 1.0]
    
    # Symmetric 12-stage parameterization:
    # a = [a1,a2,a3,a4,a5,a6,a6,a5,a4,a3,a2,a1]
    # b = [b1,b2,b3,b4,b5,b6,b6,b5,b4,b3,b2,b1]
    # 12 free params total
    
    def build_vectors(p):
        a = p[0:6]
        b = p[6:12]
        a_full = np.array([a[0], a[1], a[2], a[3], a[4], a[5],
                           a[5], a[4], a[3], a[2], a[1], a[0]], dtype=float)
        b_full = np.array([b[0], b[1], b[2], b[3], b[4], b[5],
                           b[5], b[4], b[3], b[2], b[1], b[0]], dtype=float)
        sa = a_full.sum()
        sb = b_full.sum()
        if abs(sa) < 1e-12:
            sa = 1e-12
        if abs(sb) < 1e-12:
            sb = 1e-12
        a_full = a_full / sa
        b_full = b_full / sb
        return a_full, b_full
    
    # Objective function for optimization (Frobenius norm surrogate)
    def objective(p):
        a_full, b_full = build_vectors(p)
        return _objective_fast(a_full, b_full, A_eig, B_eig, H_eig, eval_times, dim)
    
    # Initial point: 4th-order Suzuki-Trotter coefficients
    two_thirds = 2.0 ** (1.0 / 3.0)
    a1_st = 1.0 / (4.0 - two_thirds)
    a2_st = 1.0 / (2.0 * (4.0 - two_thirds))
    b1_st = 1.0 / (2.0 - two_thirds)
    b2_st = 1.0 / (2.0 * (2.0 - two_thirds))
    
    a_init = [a1_st, a2_st, a2_st, a2_st, a2_st, a2_st]
    b_init = [b1_st, b2_st, b2_st, b2_st, b2_st, b2_st]
    p0 = np.array(a_init + b_init, dtype=float)
    
    # Bounds for parameters (loose, will be clipped in build_vectors)
    bounds = (np.full(12, -10.0), np.full(12, 10.0))
    
    # Run Nelder-Mead with multiple restarts
    rng = np.random.default_rng(42)
    best_p = p0.copy()
    best_f = objective(p0)
    
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        if r == 0:
            p_cur = p0.copy()
        else:
            p_cur = p0 + rng.normal(0, 0.1, size=12)
            p_cur = np.clip(p_cur, bounds[0], bounds[1])
        
        time_limit = max(1.0, deadline - time.time())
        p_cur, f_cur = _nelder_mead(objective, p_cur, bounds, max_iter=100, 
                                      time_limit=time_limit, start_time=t_start)
        if f_cur < best_f:
            best_f = f_cur
            best_p = p_cur.copy()
    
    # Final polish with tighter tolerance if time remains
    if time.time() < deadline - 2.0:
        time_limit = max(1.0, deadline - time.time())
        best_p, best_f = _nelder_mead(objective, best_p, bounds, max_iter=50,
                                        tol=1e-10, time_limit=time_limit, start_time=t_start)
    
    # Build final vectors
    a_full, b_full = build_vectors(best_p)
    a_list = [float(x) for x in a_full]
    b_list = [float(x) for x in b_full]
    
    # Enforce bounds and sums
    a_list = [max(-50.0, min(50.0, x)) for x in a_list]
    b_list = [max(-50.0, min(50.0, x)) for x in b_list]
    sa = sum(a_list)
    sb = sum(b_list)
    if abs(sa) > 1e-12:
        a_list = [x / sa for x in a_list]
    if abs(sb) > 1e-12:
        b_list = [x / sb for x in b_list]
    
    # Feasibility check
    assert len(a_list) == 12 and len(b_list) == 12
    assert abs(sum(a_list) - 1.0) < 1e-8
    assert abs(sum(b_list) - 1.0) < 1e-8
    assert all(-50.0 <= x <= 50.0 for x in a_list)
    assert all(-50.0 <= x <= 50.0 for x in b_list)
    
    return a_list, b_list
# EVOLVE-BLOCK-END
