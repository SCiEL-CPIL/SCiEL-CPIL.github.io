"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _build_hamiltonians():
    """Build A (even bonds + half field) and B (odd bonds + half field) for 6-site XXZ chain."""
    N = 6
    anisotropy = 0.8
    field = 0.3
    J = 1.0
    
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    ident = np.eye(2, dtype=complex)
    
    def tensor_product(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def site_op(site, op):
        ops = [ident] * N
        ops[site] = op
        return tensor_product(ops)
    
    def bond_op(i, j, op1, op2):
        ops = [ident] * N
        ops[i] = op1
        ops[j] = op2
        return tensor_product(ops)
    
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    even_bonds = [(0, 1), (2, 3), (4, 5)]
    odd_bonds = [(1, 2), (3, 4)]
    
    for (i, j) in even_bonds:
        A += J * (bond_op(i, j, sx, sx) + bond_op(i, j, sy, sy) + anisotropy * bond_op(i, j, sz, sz))
    
    for (i, j) in odd_bonds:
        B += J * (bond_op(i, j, sx, sx) + bond_op(i, j, sy, sy) + anisotropy * bond_op(i, j, sz, sz))
    
    for site in range(N):
        A += 0.5 * field * site_op(site, sz)
        B += 0.5 * field * site_op(site, sz)
    
    return A, B

def _compute_error(a, b, A, B, t_values):
    """Compute mean spectral norm error."""
    from scipy.linalg import expm
    errors = []
    H = A + B
    dim = A.shape[0]
    for t in t_values:
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            U = expm(-1j * b[k] * t * B) @ U
            U = expm(-1j * a[k] * t * A) @ U
        U_exact = expm(-1j * t * H)
        diff = U - U_exact
        s = np.linalg.svd(diff, compute_uv=False)
        errors.append(s[0])
    return np.mean(errors)

def _is_valid(a, b):
    if len(a) != 12 or len(b) != 12:
        return False
    for x in a + b:
        if not np.isfinite(x) or x < -50 or x > 50:
            return False
    if abs(sum(a) - 1.0) > 1e-10 or abs(sum(b) - 1.0) > 1e-10:
        return False
    return True

def coefficients(stages):
    """Optimize coefficients for the 12-stage product formula."""
    start_time = time.time()
    deadline = start_time + 135
    
    A, B = _build_hamiltonians()
    t_values = [0.25, 0.5, 1.0]
    
    # Initial guess: Strang-like
    best_a = [1.0/12] * 12
    best_b = [1.0/12] * 12
    best_error = _compute_error(best_a, best_b, A, B, t_values)
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        a = list(params[:11])
        a.append(1.0 - sum(a))
        b = list(params[11:22])
        b.append(1.0 - sum(b))
        if not _is_valid(a, b):
            return 1e10
        return _compute_error(a, b, A, B, t_values)
    
    x0 = best_a[:11] + best_b[:11]
    
    from scipy.optimize import minimize
    
    best_params = x0
    best_obj = objective(x0)
    
    # Multiple restarts with Nelder-Mead
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 15:
            break
        if r == 0:
            x_start = x0
        else:
            # Perturb initial guess
            x_start = x0 + np.random.normal(0, 0.05, 22)
        
        result = minimize(objective, x_start, method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    
    # Polish with tighter tolerance
    if time.time() < deadline - 5:
        result = minimize(objective, best_params, method='Nelder-Mead',
                         options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    
    a = list(best_params[:11])
    a.append(1.0 - sum(a))
    b = list(best_params[11:22])
    b.append(1.0 - sum(b))
    
    if not _is_valid(a, b):
        return best_a, best_b
    
    final_error = _compute_error(a, b, A, B, t_values)
    if final_error < best_error:
        return a, b
    else:
        return best_a, best_b
# EVOLVE-BLOCK-END
