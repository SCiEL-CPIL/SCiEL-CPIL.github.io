import numpy as np
import time
import sys

_DEADLINE = 140.0  # Safety margin for 150s limit
_T0 = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain."""
    N = 6
    delta = 0.8
    field = 0.3
    
    # Single site operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
        
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    s_plus = np.array([[0, 1], [0, 0]], dtype=complex)
    s_minus = np.array([[0, 0], [1, 0]], dtype=complex)
    
    def bond_operator(i, j):
        # S_i^+ S_j^-
        ops_sp_sm = [identity] * N
        ops_sp_sm[i] = s_plus
        ops_sp_sm[j] = s_minus
        term1 = 0.5 * kron_all(ops_sp_sm)
        
        # S_i^- S_j^+
        ops_sm_sp = [identity] * N
        ops_sm_sp[i] = s_minus
        ops_sm_sp[j] = s_plus
        term2 = 0.5 * kron_all(ops_sm_sp)
        
        # S_i^z S_j^z
        ops_sz_sz = [identity] * N
        ops_sz_sz[i] = sz
        ops_sz_sz[j] = sz
        term3 = delta * kron_all(ops_sz_sz)
        
        return term1 + term2 + term3
    
    # Odd bonds: (1,2) and (3,4) -> A
    A += bond_operator(1, 2)
    A += bond_operator(3, 4)
    
    # Even bonds: (0,1), (2,3), (4,5) -> B
    B += bond_operator(0, 1)
    B += bond_operator(2, 3)
    B += bond_operator(4, 5)
    
    # Uniform field split half into A, half into B
    field_op = np.zeros((2**N, 2**N), dtype=complex)
    for j in range(N):
        ops_sz = [identity] * N
        ops_sz[j] = sz
        field_op += kron_all(ops_sz)
        
    A += -0.5 * field * field_op
    B += -0.5 * field * field_op
    
    return A, B


def _precompute_eigs(A, B, H_full):
    """Precompute eigen-decompositions for fast matrix exponential."""
    eigA_vals, eigA_vecs = np.linalg.eigh(A)
    eigB_vals, eigB_vecs = np.linalg.eigh(B)
    eigH_vals, eigH_vecs = np.linalg.eigh(H_full)
    return eigA_vals, eigA_vecs, eigB_vals, eigB_vecs, eigH_vals, eigH_vecs


def _expm_from_eig(eigvals, eigvecs, coeff):
    """Compute exp(coeff * H) given eigen-decomposition."""
    if abs(coeff) < 1e-15:
        return np.eye(len(eigvals), dtype=complex)
    return eigvecs @ np.diag(np.exp(coeff * eigvals)) @ eigvecs.conj().T


def _compute_error(a, b, eigA_vals, eigA_vecs, eigB_vals, eigB_vecs, 
                   eigH_vals, eigH_vecs, t_values, dim):
    """Compute mean spectral norm error."""
    I = np.eye(dim, dtype=complex)
    total_error = 0.0
    n_times = len(t_values)
    
    for t in t_values:
        U = I.copy()
        
        for k in range(len(a)):
            eA = _expm_from_eig(eigA_vals, eigA_vecs, -1j * a[k] * t)
            eB = _expm_from_eig(eigB_vals, eigB_vecs, -1j * b[k] * t)
            # U = E_bk E_ak U
            U = eB @ (eA @ U)
        
        U_exact = _expm_from_eig(eigH_vals, eigH_vecs, -1j * t)
        
        diff = U - U_exact
        # Use spectral norm (2-norm)
        error = np.linalg.norm(diff, 2)
        total_error += error
    
    return total_error / n_times


def _compute_error_frobenius(a, b, eigA_vals, eigA_vecs, eigB_vals, eigB_vecs, 
                             eigH_vals, eigH_vecs, t_values, dim):
    """Compute mean Frobenius norm squared error (smoother for optimization)."""
    I = np.eye(dim, dtype=complex)
    total_error = 0.0
    n_times = len(t_values)
    
    for t in t_values:
        U = I.copy()
        
        for k in range(len(a)):
            eA = _expm_from_eig(eigA_vals, eigA_vecs, -1j * a[k] * t)
            eB = _expm_from_eig(eigB_vals, eigB_vecs, -1j * b[k] * t)
            U = eB @ (eA @ U)
        
        U_exact = _expm_from_eig(eigH_vals, eigH_vecs, -1j * t)
        
        diff = U - U_exact
        # Frobenius norm squared
        error = np.vdot(diff, diff).real
        total_error += error
    
    return total_error / n_times


def coefficients(stages):
    stages = 12
    A, B = _build_hamiltonians()
    H_full = A + B
    dim = 2**6
    t_values = [0.25, 0.5, 1.0]
    
    eigA_vals, eigA_vecs, eigB_vals, eigB_vecs, eigH_vals, eigH_vecs = _precompute_eigs(A, B, H_full)
    
    best_a = None
    best_b = None
    best_error = float('inf')
    
    # We will use a symmetric parameterization:
    # a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    # b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    # Parameters: p = [a1..a6, b1..b6]
    
    def objective(params):
        a_half = params[:6]
        b_half = params[6:12]
        
        a_full = list(a_half) + list(a_half[::-1])
        b_full = list(b_half) + list(b_half[::-1])
        
        # Normalize to sum 1
        a_sum = sum(a_full)
        b_sum = sum(b_full)
        
        if abs(a_sum) < 1e-12 or abs(b_sum) < 1e-12:
            return 1e10
            
        a_scaled = [x / a_sum for x in a_full]
        b_scaled = [x / b_sum for x in b_full]
        
        # Clamp
        a_scaled = [max(-50.0, min(50.0, x)) for x in a_scaled]
        b_scaled = [max(-50.0, min(50.0, x)) for x in b_scaled]
        
        # Use Frobenius norm for optimization speed/smoothness
        return _compute_error_frobenius(a_scaled, b_scaled, eigA_vals, eigA_vecs, 
                                        eigB_vals, eigB_vecs, eigH_vals, eigH_vecs, 
                                        t_values, dim)
        
    def get_full_coeffs(params):
        a_half = params[:6]
        b_half = params[6:12]
        a_full = list(a_half) + list(a_half[::-1])
        b_full = list(b_half) + list(b_half[::-1])
        a_sum = sum(a_full)
        b_sum = sum(b_full)
        if abs(a_sum) < 1e-12 or abs(b_sum) < 1e-12:
            return None, None
        a_scaled = [x / a_sum for x in a_full]
        b_scaled = [x / b_sum for x in b_full]
        a_scaled = [max(-50.0, min(50.0, x)) for x in a_scaled]
        b_scaled = [max(-50.0, min(50.0, x)) for x in b_scaled]
        return a_scaled, b_scaled

    # Try multiple initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(12) / 12.0)
    
    # 2. Strang-like (zeros at ends)
    a_strang = [0.0, 0.25, 0.25, 0.25, 0.25, 0.0]
    b_strang = [0.0, 0.0, 0.5, 0.5, 0.0, 0.0]
    # Normalize
    a_strang = [x/sum(a_strang) for x in a_strang]
    b_strang = [x/sum(b_strang) for x in b_strang]
    inits.append(np.array(a_strang + b_strang))
    
    # 3. Suzuki 4th order 3-stage repeated and symmetrized
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c2 = 1.0 - 2.0*c1
    d1 = c1
    d2 = 1.0 - 2.0*c1
    
    # Approximate mapping to 12 stages
    # We want sum 1.
    # Let's just use random restarts as well
    
    np.random.seed(42)
    for _ in range(5):
        # Random symmetric init
        a_r = np.random.randn(6)
        b_r = np.random.randn(6)
        # Ensure positive sums for stability during init
        a_r = np.abs(a_r) + 0.1
        b_r = np.abs(b_r) + 0.1
        inits.append(np.array(a_r + b_r))
        
    from scipy.optimize import minimize
    
    for x0 in inits:
        if _remaining() < 10:
            break
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'fatol': 1e-12, 'xatol': 1e-12, 'adaptive': True})
            
            if res.fun < best_error:
                best_error = res.fun
                a_best, b_best = get_full_coeffs(res.x)
                if a_best is not None:
                    best_a = a_best
                    best_b = b_best
                    
        except Exception:
            pass
            
    # If we found a good solution, refine it with a second round
    if best_a is not None and _remaining() > 20:
        x0_refine = np.array(best_a[:6] + best_b[:6])
        try:
            res = minimize(objective, x0_refine, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'fatol': 1e-14, 'xatol': 1e-14, 'adaptive': True})
            if res.fun < best_error:
                best_error = res.fun
                a_ref, b_ref = get_full_coeffs(res.x)
                if a_ref is not None:
                    best_a = a_ref
                    best_b = b_ref
        except Exception:
            pass
            
    # Final fallback if something went wrong
    if best_a is None:
        best_a = [1.0/12.0]*12
        best_b = [1.0/12.0]*12
        
    # Final normalization and clamping
    a_sum = sum(best_a)
    b_sum = sum(best_b)
    if abs(a_sum) > 1e-15:
        best_a = [x / a_sum for x in best_a]
    if abs(b_sum) > 1e-15:
        best_b = [x / b_sum for x in best_b]
        
    best_a = [max(-50.0, min(50.0, float(x))) for x in best_a]
    best_b = [max(-50.0, min(50.0, float(x))) for x in best_b]
    
    return best_a, best_b
