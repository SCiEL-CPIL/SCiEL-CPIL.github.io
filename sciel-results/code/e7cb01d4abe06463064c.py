"""Optimized 12-stage product formula for Heisenberg chain time evolution."""
import numpy as np
import scipy.linalg
import time

def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for 6-site XXZ chain."""
    n = 6
    anisotropy = 0.8
    field = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def tensor_product(*mats):
        result = mats[0]
        for m in mats[1:]:
            result = np.kron(result, m)
        return result
    
    def site_op(site, op):
        """Create operator acting on given site (0-indexed)."""
        ops = [I2] * n
        ops[site] = op
        return tensor_product(*ops)
    
    def bond_op(i, op1, op2):
        """Create operator op1 on site i, op2 on site i+1."""
        ops = [I2] * n
        ops[i] = op1
        ops[i + 1] = op2
        return tensor_product(*ops)
    
    # XXZ Hamiltonian: J * sum [aniso * Sz_i Sz_{i+1} + (Sx_i Sx_{i+1} + Sy_i Sy_{i+1})] - h * sum Sz_i
    # Split into odd bonds (A) and even bonds (B), each with half the field
    
    J = 1.0
    A = np.zeros((2**n, 2**n), dtype=complex)
    B = np.zeros((2**n, 2**n), dtype=complex)
    
    # Odd bonds: (0,1), (2,3), (4,5)
    for i in [0, 2, 4]:
        A += J * (anisotropy * bond_op(i, sz, sz) + bond_op(i, sx, sx) + bond_op(i, sy, sy))
    
    # Even bonds: (1,2), (3,4)
    for i in [1, 3]:
        B += J * (anisotropy * bond_op(i, sz, sz) + bond_op(i, sx, sx) + bond_op(i, sy, sy))
    
    # Half field on each
    field_term = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(n):
        field_term += field * site_op(i, sz)
    
    A += 0.5 * field_term
    B += 0.5 * field_term
    
    return A, B

def _error(a, b, A, B, times):
    """Compute mean spectral norm error."""
    dim = A.shape[0]
    total_err = 0.0
    for t in times:
        # Build product formula
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            # exp(-i b_k t B)
            Bk = b[k] * t * B
            expB = scipy.linalg.expm(-1j * Bk)
            # exp(-i a_k t A)
            Ak = a[k] * t * A
            expA = scipy.linalg.expm(-1j * Ak)
            U = expB @ expA @ U
        
        # Exact propagator
        H = A + B
        U_exact = scipy.linalg.expm(-1j * t * H)
        
        # Spectral norm of error
        err = U - U_exact
        total_err += np.linalg.norm(err, 2)
    
    return total_err / len(times)

def coefficients(stages):
    """Optimize 12-stage product formula coefficients."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    # Start with symmetric 6th-order coefficients
    # For 12 stages, use known high-order splitting
    a0 = np.array([1/6, -1/12, 1/6, 1/6, -1/12, 1/6, 1/6, -1/12, 1/6, -1/12, 1/6, 1/6])
    b0 = np.array([1/6, -1/12, 1/6, 1/6, -1/12, 1/6, 1/6, -1/12, 1/6, -1/12, 1/6, 1/6])
    
    # Ensure sums are 1
    a0 = a0 / np.sum(a0)
    b0 = b0 / np.sum(b0)
    
    best_a = a0.copy()
    best_b = b0.copy()
    best_err = _error(best_a, best_b, A, B, times)
    
    # Use scipy optimization with constraint handling
    # We'll optimize over unconstrained parameters and project to satisfy sum=1
    from scipy.optimize import minimize
    
    def objective(params):
        # params: 24 values, first 12 for a, last 12 for b
        a_raw = params[:12]
        b_raw = params[12:]
        
        # Project to sum=1 by adding offset
        a = a_raw + (1.0 - np.sum(a_raw)) / 12
        b = b_raw + (1.0 - np.sum(b_raw)) / 12
        
        # Clip to valid range
        a = np.clip(a, -50, 50)
        b = np.clip(b, -50, 50)
        
        # Re-normalize after clipping
        if np.sum(a) != 0:
            a = a / np.sum(a)
        else:
            a = np.ones(12) / 12
        if np.sum(b) != 0:
            b = b / np.sum(b)
        else:
            b = np.ones(12) / 12
        
        return _error(a, b, A, B, times)
    
    x0 = np.concatenate([best_a, best_b])
    
    # Try L-BFGS-B
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 50, 'maxfun': 50})
        if result.fun < best_err:
            a_opt = result.x[:12]
            b_opt = result.x[12:]
            a_opt = a_opt / np.sum(a_opt)
            b_opt = b_opt / np.sum(b_opt)
            best_a = a_opt
            best_b = b_opt
            best_err = result.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_a.tolist(), best_b.tolist()
    
    # Try another optimization from a different starting point
    # Symmetric structure: a_k = a_{13-k}, b_k = b_{13-k}
    # Only 6 free parameters for a, 6 for b
    def objective_sym(params):
        # params: 12 values, first 6 for a (symmetric), last 6 for b (symmetric)
        a_half = params[:6]
        b_half = params[6:]
        
        # Expand to full symmetric
        a_full = np.concatenate([a_half, a_half[::-1]])
        b_full = np.concatenate([b_half, b_half[::-1]])
        
        # Normalize
        a_sum = np.sum(a_full)
        b_sum = np.sum(b_full)
        if abs(a_sum) > 1e-12:
            a_full = a_full / a_sum
        else:
            a_full = np.ones(12) / 12
        if abs(b_sum) > 1e-12:
            b_full = b_full / b_sum
        else:
            b_full = np.ones(12) / 12
        
        return _error(a_full, b_full, A, B, times)
    
    # Start from symmetric version of best
    a_sym = best_a[:6]
    b_sym = best_b[:6]
    x0_sym = np.concatenate([a_sym, b_sym])
    
    try:
        result2 = minimize(objective_sym, x0_sym, method='L-BFGS-B',
                          options={'maxiter': 30, 'maxfun': 30})
        if result2.fun < best_err:
            a_half = result2.x[:6]
            b_half = result2.x[6:]
            a_full = np.concatenate([a_half, a_half[::-1]])
            b_full = np.concatenate([b_half, b_half[::-1]])
            a_full = a_full / np.sum(a_full)
            b_full = b_full / np.sum(b_full)
            best_a = a_full
            best_b = b_full
            best_err = result2.fun
    except Exception:
        pass
    
    return best_a.tolist(), best_b.tolist()
