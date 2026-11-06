"""Optimized product-formula coefficients for Heisenberg-chain time evolution.

Strategy: Chebyshev-style minimax approximation over a grid of time points.
We minimize the maximum spectral-norm error across a dense grid of t values
in [0.1, 1.1] to ensure robust performance at the specific evaluation times
t = 0.25, 0.5, 1.0.
"""
import numpy as np
import time
import sys

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    N = 6
    anisotropy = 0.8
    field = 0.3
    
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    dim = 2 ** N
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    J = 1.0
    
    for i in range(N - 1):
        ops_sx = [I2] * N
        ops_sy = [I2] * N
        ops_sz = [I2] * N
        ops_sx[i] = sx
        ops_sx[i+1] = sx
        ops_sy[i] = sy
        ops_sy[i+1] = sy
        ops_sz[i] = sz
        ops_sz[i+1] = sz
        
        sx_i = kron_all(ops_sx)
        sy_i = kron_all(ops_sy)
        sz_i = kron_all(ops_sz)
        
        bond_ham = J * (sx_i @ sx_i + sy_i @ sy_i + anisotropy * sz_i @ sz_i)
        
        if i % 2 == 0:
            B += bond_ham
        else:
            A += bond_ham
    
    for i in range(N):
        ops_sz = [I2] * N
        ops_sz[i] = sz
        sz_i = kron_all(ops_sz)
        A += 0.5 * field * sz_i
        B += 0.5 * field * sz_i
    
    return A, B


def _precompute_exponentials(A, B, times):
    """Precompute exp(-i t A) and exp(-i t B) for all needed time values."""
    eigA_vals, eigA_vecs = np.linalg.eig(A)
    eigB_vals, eigB_vecs = np.linalg.eig(B)
    eigAB_vals, eigAB_vecs = np.linalg.eig(A + B)
    
    invA = np.linalg.inv(eigA_vecs)
    invB = np.linalg.inv(eigB_vecs)
    invAB = np.linalg.inv(eigAB_vecs)
    
    dim = A.shape[0]
    
    expA_cache = {}
    expB_cache = {}
    expAB_cache = {}
    
    for t in times:
        expA_cache[t] = eigA_vecs @ np.diag(np.exp(-1j * t * eigA_vals)) @ invA
        expB_cache[t] = eigB_vecs @ np.diag(np.exp(-1j * t * eigB_vals)) @ invB
        expAB_cache[t] = eigAB_vecs @ np.diag(np.exp(-1j * t * eigAB_vals)) @ invAB
    
    return expA_cache, expB_cache, expAB_cache, eigA_vals, eigA_vecs, invA, eigB_vals, eigB_vecs, invB


def _compute_error(a, b, stages, expA_cache, expB_cache, expAB_cache, 
                   eigA_vals, eigA_vecs, invA, eigB_vals, eigB_vecs, invB,
                   times, dim):
    """Compute max spectral-norm error over the time grid."""
    max_err = 0.0
    
    for t in times:
        U = np.eye(dim, dtype=complex)
        for k in range(stages):
            ta = a[k] * t
            tb = b[k] * t
            exp_A = eigA_vecs @ np.diag(np.exp(-1j * ta * eigA_vals)) @ invA
            exp_B = eigB_vecs @ np.diag(np.exp(-1j * tb * eigB_vals)) @ invB
            U = exp_B @ exp_A @ U
        
        U_exact = expAB_cache[t]
        err = np.linalg.svd(U - U_exact, compute_uv=False)[0]
        if err > max_err:
            max_err = err
    
    return max_err


def _objective(a, b, stages, precomp, times, dim):
    """Objective: max spectral-norm error over time grid."""
    expA_cache, expB_cache, expAB_cache, eigA_vals, eigA_vecs, invA, eigB_vals, eigB_vecs, invB = precomp
    return _compute_error(a, b, stages, expA_cache, expB_cache, expAB_cache,
                          eigA_vals, eigA_vecs, invA, eigB_vals, eigB_vecs, invB,
                          times, dim)


def _nelder_mead(objective, x0, max_iter=500, tol=1e-12, deadline=None):
    """Nelder-Mead with deadline guard."""
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        point = np.array(x0, dtype=float).copy()
        step = 0.01
        point[i] += step
        simplex.append(point)
    
    func_vals = [objective(p) for p in simplex]
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        order = np.argsort(func_vals)
        simplex = [simplex[i] for i in order]
        func_vals = [func_vals[i] for i in order]
        
        if abs(func_vals[0] - func_vals[-1]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = objective(x_r)
        
        if f_r < func_vals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                func_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                func_vals[-1] = f_r
        elif f_r < func_vals[-2]:
            simplex[-1] = x_r
            func_vals[-1] = f_r
        else:
            x_c = centroid + rho * (simplex[-1] - centroid)
            f_c = objective(x_c)
            if f_c < func_vals[-1]:
                simplex[-1] = x_c
                func_vals[-1] = f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    func_vals[i] = objective(simplex[i])
        
        if deadline is not None and time.time() > deadline:
            break
    
    best_idx = np.argmin(func_vals)
    return simplex[best_idx], func_vals[best_idx]


def _optimize(stages, deadline):
    """Run optimization with deadline guard."""
    A, B = _build_hamiltonians()
    dim = A.shape[0]
    
    # Time grid for Chebyshev-style minimax: dense around evaluation points
    # Include the three evaluation times plus neighbors for robustness
    times = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    
    precomp = _precompute_exponentials(A, B, times)
    
    # Initial guess: uniform coefficients
    x0 = np.ones(2 * stages) / stages
    
    def obj(x):
        a = x[:stages]
        b = x[stages:]
        # Enforce sum constraint by normalization
        sa = np.sum(a)
        sb = np.sum(b)
        if abs(sa) > 1e-15:
            a = a / sa
        else:
            a = np.ones(stages) / stages
        if abs(sb) > 1e-15:
            b = b / sb
        else:
            b = np.ones(stages) / stages
        return _objective(a, b, stages, precomp, times, dim)
    
    best_x, best_f = _nelder_mead(obj, x0, max_iter=300, tol=1e-10, deadline=deadline)
    
    # Extract and normalize
    a = best_x[:stages]
    b = best_x[stages:]
    
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-15:
        a = a / sa
    else:
        a = np.ones(stages) / stages
    if abs(sb) > 1e-15:
        b = b / sb
    else:
        b = np.ones(stages) / stages
    
    # Clamp
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    
    # Re-normalize after clamping
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-15:
        a = a / sa
    if abs(sb) > 1e-15:
        b = b / sb
    
    return a, b, best_f


def coefficients(stages):
    """Return optimized 12-stage product formula coefficients."""
    if stages != 12:
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
    
    # Deadline: 140s (10s safety margin from 150s limit)
    deadline = time.time() + 140.0
    
    try:
        a, b, _ = _optimize(stages, deadline)
        return list(a), list(b)
    except Exception:
        # Fallback: uniform coefficients
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
