"""Optimized product-formula coefficients for 6-site XXZ Heisenberg chain."""
import time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.linalg import expm

_START = time.time()
_DEADLINE = _START + 130.0  # safety margin before 150s kill

def _build_matrices():
    """Build A and B for 6-site XXZ chain with Δ=0.8, h=0.3."""
    N = 6
    dim = 2 ** N
    delta = 0.8
    h = 0.3
    
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def site_op(op, site):
        ops = [I] * N
        ops[site] = op
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def bond_op(i, j):
        xi = site_op(X, i)
        xj = site_op(X, j)
        yi = site_op(Y, i)
        yj = site_op(Y, j)
        zi = site_op(Z, i)
        zj = site_op(Z, j)
        return 0.25 * (xi @ xj + yi @ yj) + delta * 0.25 * (zi @ zj)
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    even_bonds = [(0, 1), (2, 3), (4, 5)]
    odd_bonds = [(1, 2), (3, 4)]
    
    for i, j in even_bonds:
        A += bond_op(i, j)
    for i, j in odd_bonds:
        B += bond_op(i, j)
    
    for site in range(N):
        field_term = 0.5 * h * site_op(Z, site)
        A += field_term
        B += field_term
    
    return A, B

def coefficients(stages):
    """Return (a, b) lists of length `stages` that minimize the mean spectral norm error."""
    if not hasattr(coefficients, '_A'):
        coefficients._A = None
        coefficients._B = None
        coefficients._U_exact = None
        coefficients._best_a = None
        coefficients._best_b = None
        coefficients._best_err = np.inf
        coefficients._initialized = False
    
    if not coefficients._initialized:
        A, B = _build_matrices()
        coefficients._A = A
        coefficients._B = B
        times = [0.25, 0.5, 1.0]
        coefficients._U_exact = [expm(-1j * t * (A + B)) for t in times]
        coefficients._initialized = True
    
    A = coefficients._A
    B = coefficients._B
    U_exacts = coefficients._U_exact
    times = [0.25, 0.5, 1.0]
    
    # Initial guess: equal weights
    a_init = np.ones(stages) / stages
    b_init = np.ones(stages) / stages
    
    def _objective(x):
        if time.time() > _DEADLINE:
            return np.inf
        a_free = x[:stages-1]
        b_free = x[stages-1:2*stages-1]
        a = np.concatenate([a_free, [1.0 - np.sum(a_free)]])
        b = np.concatenate([b_free, [1.0 - np.sum(b_free)]])
        
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e10
        
        errors = []
        for idx, t in enumerate(times):
            U = np.eye(A.shape[0], dtype=complex)
            for k in range(stages):
                U = expm(-1j * b[k] * t * B) @ U
                U = expm(-1j * a[k] * t * A) @ U
            diff = U - U_exacts[idx]
            err = np.linalg.norm(diff, 2)
            errors.append(err)
        return np.mean(errors)
    
    # Try optimization
    x_init = np.concatenate([a_init[:stages-1], b_init[:stages-1]])
    
    best_a = a_init.copy()
    best_b = b_init.copy()
    best_err = _objective(x_init)
    
    try:
        result = differential_evolution(
            _objective,
            bounds=[(-3, 3)] * (2*stages-2),
            maxiter=30,
            popsize=10,
            tol=1e-6,
            seed=42,
            polish=True
        )
        
        a_opt_free = result.x[:stages-1]
        b_opt_free = result.x[stages-1:2*stages-1]
        a_opt = np.concatenate([a_opt_free, [1.0 - np.sum(a_opt_free)]])
        b_opt = np.concatenate([b_opt_free, [1.0 - np.sum(b_opt_free)]])
        
        # Clamp and renormalize if needed
        if np.any(a_opt < -50) or np.any(a_opt > 50):
            a_opt = np.clip(a_opt, -50, 50)
            if np.abs(np.sum(a_opt)) > 1e-12:
                a_opt = a_opt / np.sum(a_opt)
        if np.any(b_opt < -50) or np.any(b_opt > 50):
            b_opt = np.clip(b_opt, -50, 50)
            if np.abs(np.sum(b_opt)) > 1e-12:
                b_opt = b_opt / np.sum(b_opt)
        
        err_opt = _objective(result.x)
        if err_opt < best_err:
            best_a = a_opt
            best_b = b_opt
            best_err = err_opt
            
    except Exception:
        pass
    
    # Store best
    coefficients._best_a = best_a
    coefficients._best_b = best_b
    coefficients._best_err = best_err
    
    return list(best_a), list(best_b)
