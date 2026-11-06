"""Optimized 12-stage product formula for Heisenberg chain time evolution."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import sys

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    # Spin-1/2 operators
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    n = 6
    dim = 2 ** n
    delta = 0.8
    field = 0.3
    
    # Initialize A and B as zero matrices
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Bond list: (0,1), (1,2), (2,3), (3,4), (4,5)
    # Odd bonds (0-indexed bond index): bonds 0, 2, 4 -> (0,1), (2,3), (4,5)
    # Even bonds: bonds 1, 3 -> (1,2), (3,4)
    
    bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    
    for bond_idx, (i, j) in enumerate(bonds):
        # Build bond Hamiltonian on sites i and j
        # H_bond = sx_i sx_j + sy_i sy_j + delta * sz_i sz_j
        
        # Helper: apply operator on site k
        def op_on_site(op, site):
            """Return operator op on site 'site' in n-site chain."""
            result = np.array([[1.0]], dtype=complex)
            for s in range(n):
                if s == site:
                    result = np.kron(result, op)
                else:
                    result = np.kron(result, np.eye(2, dtype=complex))
            return result
        
        sx_i = op_on_site(sx, i)
        sx_j = op_on_site(sx, j)
        sy_i = op_on_site(sy, i)
        sy_j = op_on_site(sy, j)
        sz_i = op_on_site(sz, i)
        sz_j = op_on_site(sz, j)
        
        H_bond = sx_i @ sx_j + sy_i @ sy_j + delta * sz_i @ sz_j
        
        if bond_idx % 2 == 0:  # odd bonds (0, 2, 4)
            A += H_bond
        else:  # even bonds (1, 3)
            B += H_bond
    
    # Add half the uniform field to each
    # Field term: field * sum_i S_i^z
    # Split half to A, half to B
    field_term = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        sz_i = op_on_site(sz, i)
        field_term += sz_i
    
    A += 0.5 * field * field_term
    B += 0.5 * field * field_term
    
    return A, B


def _compute_error(coeffs, A, B, times):
    """Compute mean spectral norm error for given coefficients."""
    a, b = coeffs[:12], coeffs[12:24]
    errors = []
    for t in times:
        U = np.eye(2**6, dtype=complex)
        for k in range(12):
            # exp(-i b_k t B) @ exp(-i a_k t A)
            # Use eigendecomposition for matrix exponential
            eigvals_b, eigvecs_b = np.linalg.eig(B)
            exp_b = eigvecs_b @ np.diag(np.exp(-1j * b[k] * t * eigvals_b)) @ np.linalg.inv(eigvecs_b)
            
            eigvals_a, eigvecs_a = np.linalg.eig(A)
            exp_a = eigvecs_a @ np.diag(np.exp(-1j * a[k] * t * eigvals_a)) @ np.linalg.inv(eigvecs_a)
            
            U = exp_b @ exp_a @ U
        
        # Exact propagator
        H = A + B
        eigvals_h, eigvecs_h = np.linalg.eig(H)
        U_exact = eigvecs_h @ np.diag(np.exp(-1j * t * eigvals_h)) @ np.linalg.inv(eigvecs_h)
        
        diff = U - U_exact
        # Spectral norm = largest singular value
        s = np.linalg.svd(diff, compute_uv=False)
        errors.append(s[0])
    return np.mean(errors)


def coefficients(stages):
    """Optimized 12-stage product formula coefficients."""
    if stages != 12:
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    start_time = time.time()
    deadline = start_time + 130  # 130s out of 150s budget
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    # Precompute eigendecompositions
    eigvals_a, eigvecs_a = np.linalg.eig(A)
    eigvals_b, eigvecs_b = np.linalg.eig(B)
    H = A + B
    eigvals_h, eigvecs_h = np.linalg.eig(H)
    
    inv_eigvecs_a = np.linalg.inv(eigvecs_a)
    inv_eigvecs_b = np.linalg.inv(eigvecs_b)
    inv_eigvecs_h = np.linalg.inv(eigvecs_h)
    
    def fast_error(coeffs_flat):
        """Fast error computation using precomputed eigendecomps."""
        a = coeffs_flat[:12]
        b = coeffs_flat[12:24]
        errors = []
        for t in times:
            U = np.eye(2**6, dtype=complex)
            for k in range(12):
                exp_b = eigvecs_b @ np.diag(np.exp(-1j * b[k] * t * eigvals_b)) @ inv_eigvecs_b
                exp_a = eigvecs_a @ np.diag(np.exp(-1j * a[k] * t * eigvals_a)) @ inv_eigvecs_a
                U = exp_b @ exp_a @ U
            
            U_exact = eigvecs_h @ np.diag(np.exp(-1j * t * eigvals_h)) @ inv_eigvecs_h
            diff = U - U_exact
            s = np.linalg.svd(diff, compute_uv=False)
            errors.append(s[0])
        return np.mean(errors)
    
    # Initial guess: 4th-order Suzuki-Trotter padded to 12
    c = 2.0 ** (1.0 / 3.0)
    c1 = 1.0 / (2.0 - c)
    c2 = -c / (2.0 - c)
    
    a_init = [c1/2, 0, c1/2, c2/2, 0, c2/2, c1/2, 0, c1/2, 0, 0, 0]
    b_init = [0, c1, 0, 0, c2, 0, 0, c1, 0, 0, 0, 0]
    
    x0 = np.array(a_init + b_init)
    
    # Constraint: sum(a) = 1, sum(b) = 1
    # Use penalty approach or reparameterization
    
    # Use a penalty-based approach with Nelder-Mead
    def penalized_error(x):
        a = x[:12]
        b = x[12:24]
        base_err = fast_error(x)
        penalty_a = (np.sum(a) - 1.0) ** 2 * 1000
        penalty_b = (np.sum(b) - 1.0) ** 2 * 1000
        return base_err + penalty_a + penalty_b
    
    best_x = x0.copy()
    best_err = penalized_error(x0)
    
    # Try Nelder-Mead
    try:
        result = minimize(penalized_error, x0, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if result.fun < best_err:
            best_err = result.fun
            best_x = result.x
        
        if time.time() > deadline:
            a = best_x[:12].tolist()
            b = best_x[12:24].tolist()
            # Normalize to ensure sums are exactly 1
            sa = sum(a)
            sb = sum(b)
            if abs(sa) > 1e-15:
                a = [x / sa for x in a]
            if abs(sb) > 1e-15:
                b = [x / sb for x in b]
            return a, b
    except Exception:
        pass
    
    # Try L-BFGS-B with bounds
    try:
        bounds = [(-50, 50)] * 24
        result2 = minimize(penalized_error, best_x, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 3000, 'ftol': 1e-15})
        if result2.fun < best_err:
            best_err = result2.fun
            best_x = result2.x
    except Exception:
        pass
    
    # Additional refinement with more iterations if time allows
    if time.time() < deadline - 10:
        try:
            result3 = minimize(penalized_error, best_x, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if result3.fun < best_err:
                best_err = result3.fun
                best_x = result3.x
        except Exception:
            pass
    
    a = best_x[:12].tolist()
    b = best_x[12:24].tolist()
    
    # Normalize to ensure sums are exactly 1
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-15:
        a = [x / sa for x in a]
    if abs(sb) > 1e-15:
        b = [x / sb for x in b]
    
    # Clamp to [-50, 50]
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    return a, b
# EVOLVE-BLOCK-END
