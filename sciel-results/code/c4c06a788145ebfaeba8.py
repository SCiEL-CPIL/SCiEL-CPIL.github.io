import time
import numpy as np

_DEADLINE = time.time() + 140  # 10s safety margin

_HAMILTONIANS = None
_EIGS = None

def _init_system():
    global _HAMILTONIANS, _EIGS
    if _HAMILTONIANS is not None:
        return
    
    N = 6
    gamma = 0.8
    h = 0.3
    
    dim = 2 ** N
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    ident = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
        
    # Field terms
    for j in range(N):
        ops = [ident] * N
        ops[j] = sz
        field_term = (h / 2.0) * kron_all(ops)
        A += field_term
        B += field_term
        
    # Bond terms
    for j in range(N - 1):
        bond_ops = [ident] * N
        bond_ops[j] = sx
        bond_ops[j+1] = sx
        term_xx = kron_all(bond_ops)
        
        bond_ops = [ident] * N
        bond_ops[j] = sy
        bond_ops[j+1] = sy
        term_yy = kron_all(bond_ops)
        
        bond_ops = [ident] * N
        bond_ops[j] = sz
        bond_ops[j+1] = sz
        term_zz = gamma * kron_all(bond_ops)
        
        bond_term = term_xx + term_yy + term_zz
        
        if j % 2 == 0:
            B += bond_term
        else:
            A += bond_term
            
    _HAMILTONIANS = (A, B)
    
    # Pre-compute eigendecompositions
    wA, vA = np.linalg.eigh(A)
    wB, vB = np.linalg.eigh(B)
    H = A + B
    wH, vH = np.linalg.eigh(H)
    
    _EIGS = (wA, vA, wB, vB, wH, vH)

def _fast_mat_exp_diag(w, v, scale):
    """Compute exp(scale * M) given M = V diag(w) V^H"""
    diag_exp = np.exp(scale * w)
    # v * diag_exp is equivalent to v @ diag(diag_exp)
    # Then @ v.conj().T
    return (v * diag_exp) @ v.conj().T

def _objective(x):
    """
    Objective function.
    x = [a1..a12, b1..b12]
    """
    wA, vA, wB, vB, wH, vH = _EIGS
    
    a = x[:12]
    b = x[12:24]
    
    # Check bounds
    if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
        return 1e10
    
    # Check sums (consistency)
    if abs(np.sum(a) - 1.0) > 1e-6 or abs(np.sum(b) - 1.0) > 1e-6:
        return 1e10

    t_values = [0.25, 0.5, 1.0]
    errors = []
    
    for t in t_values:
        U_exact = _fast_mat_exp_diag(wH, vH, -1j * t)
        
        U_prod = np.eye(64, dtype=complex)
        for k in range(12):
            # U_prod = exp(-i b_k t B) @ exp(-i a_k t A) @ U_prod
            U_B = _fast_mat_exp_diag(wB, vB, -1j * b[k] * t)
            U_A = _fast_mat_exp_diag(wA, vA, -1j * a[k] * t)
            U_prod = U_B @ U_A @ U_prod
            
        err = np.linalg.norm(U_prod - U_exact, 2)
        errors.append(err)
        
    return np.mean(errors)

def _nelder_mead(f, x0, max_iter=1000, tol=1e-6, deadline=None):
    """Simple Nelder-Mead implementation."""
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    # Initialize simplex
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(1, n + 1):
        simplex[i] = x0.copy()
        simplex[i, i-1] += 0.1 * (1.0 if x0[i-1] == 0 else x0[i-1])
        
    fvals = np.array([f(x) for x in simplex])
    best_x = simplex[0].copy()
    best_f = np.min(fvals)
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Sort simplex
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        if fvals[0] < best_f:
            best_f = fvals[0]
            best_x = simplex[0].copy()
            
        # Check convergence
        if np.max(fvals) - np.min(fvals) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:n], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[n])
        f_r = f(x_r)
        
        if f_r < fvals[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = f(x_e)
            if f_e < f_r:
                simplex[n] = x_e
                fvals[n] = f_e
            else:
                simplex[n] = x_r
                fvals[n] = f_r
        elif f_r < fvals[n-1]:
            simplex[n] = x_r
            fvals[n] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[n] - centroid)
            f_c = f(x_c)
            if f_c < fvals[n]:
                simplex[n] = x_c
                fvals[n] = f_c
            else:
                # Shrink
                for j in range(1, n + 1):
                    simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                    fvals[j] = f(simplex[j])
                    
    return best_x, best_f

def coefficients(stages):
    _init_system()
    
    # Initial guess: Uniform split (1/12)
    # Or better: Suzuki 4th order coefficients scaled
    # Suzuki 4th order alpha: [1.3512, -1.7024, 1.3512]
    # For 12 steps, we can repeat a 3-step block 4 times, scaling by 1/4?
    # No, the coefficients must sum to 1.
    # If we use the 3-step coefficients directly, they sum to 1.
    # If we repeat them 4 times, they sum to 4.
    # We need to scale them by 1/4 to make the total sum 1.
    
    alpha1 = 1.0 / (2.0 - 2.0 ** (1.0/3.0))
    alpha2 = 1.0 - 2.0 * alpha1
    alpha3 = alpha1
    
    a_block = np.array([alpha1, alpha2, alpha1])
    b_block = np.array([alpha1, alpha2, alpha1])
    
    a_init = np.tile(a_block, 4) / 4.0
    b_init = np.tile(b_block, 4) / 4.0
    
    x0 = np.concatenate([a_init, b_init])
    
    # Optimize
    try:
        deadline = time.time() + 130 # 20s margin
        x_opt, f_opt = _nelder_mead(_objective, x0, max_iter=2000, deadline=deadline)
        
        a_opt = x_opt[:12]
        b_opt = x_opt[12:24]
        
        # Ensure constraints
        if np.any(a_opt < -50) or np.any(a_opt > 50) or np.any(b_opt < -50) or np.any(b_opt > 50):
            return list(a_init), list(b_init)
            
        # Re-normalize to ensure sum=1 exactly
        a_opt = a_opt / np.sum(a_opt)
        b_opt = b_opt / np.sum(b_opt)
        
        return list(a_opt), list(b_opt)
    except Exception:
        pass
        
    return list(a_init), list(b_init)
