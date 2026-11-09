"""Optimized product-formula coefficients for six-site Heisenberg chain splitting."""
import numpy as np
from numpy.linalg import norm
import time

# Deadline guard
_DEADLINE = 135.0  # Safety margin from 150s limit
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _expm_hermitian(H, t):
    """Compute exp(-i t H) for Hermitian H."""
    D, V = np.linalg.eigh(H)
    exp_D = np.exp(-1j * t * D)
    return V @ np.diag(exp_D) @ V.conj().T

# Precomputation variables
_A = None
_B = None
_V_A = None
_D_A = None
_V_B = None
_D_B = None
_exact_propagators = {}
_t_values = [0.25, 0.5, 1.0]

def _build_hamiltonians():
    global _A, _B, _V_A, _D_A, _V_B, _D_B
    if _A is not None:
        return
    
    N = 6
    J = 1.0
    gamma_field = 0.3
    anisotropy = 0.8
    
    # Build Heisenberg XXZ chain Hamiltonian
    # H = sum J (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + anisotropy Sz_i Sz_{i+1}) - gamma_field sum Sz_i
    # Split into even and odd bonds.
    # A = odd bonds + half field
    # B = even bonds + half field
    
    # Spin operators
    sx = np.array([[0, 0.5], [0.5, 0]])
    sy = np.array([[0, -0.5j], [0.5j, 0]])
    sz = np.array([[0.5, 0], [0, -0.5]])
    I = np.eye(2)
    
    def kron_all(ops):
        res = np.array([[1]])
        for op in ops:
            res = np.kron(res, op)
        return res
        
    H_A = np.zeros((2**N, 2**N), dtype=complex)
    H_B = np.zeros((2**N, 2**N), dtype=complex)
    
    for i in range(N):
        ops = [I] * N
        ops[i] = sz
        field_op = kron_all(ops)
        # Each half gets half the field
        H_A += -0.5 * gamma_field * field_op
        H_B += -0.5 * gamma_field * field_op
        
    def add_bond(H, i, j):
        ops_x = [I] * N
        ops_x[i] = sx
        ops_x[j] = sx
        H += J * kron_all(ops_x)
        
        ops_y = [I] * N
        ops_y[i] = sy
        ops_y[j] = sy
        H += J * kron_all(ops_y)
        
        ops_z = [I] * N
        ops_z[i] = sz
        ops_z[j] = sz
        H += J * anisotropy * kron_all(ops_z)
        
    # A bonds: (0,1), (2,3), (4,5)
    add_bond(H_A, 0, 1)
    add_bond(H_A, 2, 3)
    add_bond(H_A, 4, 5)
    
    # B bonds: (1,2), (3,4)
    add_bond(H_B, 1, 2)
    add_bond(H_B, 3, 4)
    
    _A = H_A
    _B = H_B
    
    # Diagonalize
    _D_A, _V_A = np.linalg.eigh(_A)
    _D_B, _V_B = np.linalg.eigh(_B)
    
    # Precompute exact propagators
    for t in _t_values:
        _exact_propagators[t] = _expm_hermitian(_A + _B, t)

def _normalize_and_clip(x, lower=-50.0, upper=50.0):
    """Clip to bounds and normalize sum to 1."""
    x = np.clip(x, lower, upper)
    s = x.sum()
    if abs(s) < 1e-15:
        # If sum is zero, return uniform
        return np.ones_like(x) / len(x)
    x = x / s
    return x

def _spectral_norm_error(a, b):
    """
    Computes mean spectral norm error for the 3 time points.
    a, b are lists of 12 floats.
    """
    if _A is None:
        _build_hamiltonians()
        
    total_err = 0.0
    dim = _A.shape[0]
    I = np.eye(dim)
    
    for t in _t_values:
        U = I.copy()
        
        for k in range(12):
            # exp(-i b_k t B)
            exp_B = np.diag(np.exp(-1j * b[k] * t * _D_B))
            M_B = _V_B @ exp_B @ _V_B.conj().T
            
            # exp(-i a_k t A)
            exp_A = np.diag(np.exp(-1j * a[k] * t * _D_A))
            M_A = _V_A @ exp_A @ _V_A.conj().T
            
            M_k = M_B @ M_A
            U = U @ M_k
            
        U_exact = _exact_propagators[t]
        err = norm(U - U_exact, 2)
        total_err += err
        
    return total_err / len(_t_values)

def _nelder_mead_simplex(objective, x0, bounds=(-50.0, 50.0), max_iter=100, tol=1e-8, time_limit=None):
    """
    Nelder-Mead simplex optimization with time limit.
    objective: function that takes a 2D array (2 x 12) and returns scalar
    x0: initial point, shape (2, 12)
    """
    n = x0.size  # 24 variables
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    # Initialize simplex
    simplex = []
    fvals = []
    
    x0 = np.array(x0, dtype=float)
    simplex.append(x0.copy())
    fvals.append(objective(x0))
    
    # Create other vertices with small perturbations
    for i in range(n):
        xi = x0.copy()
        step = 0.01 if x0[i] != 0 else 0.01
        xi[i] += step
        simplex.append(xi)
        fvals.append(objective(xi))
    
    # Sort by function value
    order = np.argsort(fvals)
    simplex = [simplex[i] for i in order]
    fvals = [fvals[i] for i in order]
    
    best_x = simplex[0].copy()
    best_f = fvals[0]
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - time_limit) > 0:
            break
        
        # Reflection
        xw = simplex[-1]
        xc = np.mean(simplex[:-1], axis=0)
        xr = xc + alpha * (xc - xw)
        fr = objective(r)
        
        if fr < fvals[0]:
            # Expansion
            xe = xc + gamma * (xc - xw)
            fe = objective(xe)
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
                xc = np.mean(simplex[:-1], axis=0)
                xc = xc + 0.5 * (xr - xc)
                fc = objective(xc)
                if fc < fr:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])
            else:
                # Inside contraction
                xc = np.mean(simplex[:-1], axis=0)
                xc = xc + 0.5 * (xc - xw)
                fc = objective(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])
        
        # Sort
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Update best
        if fvals[0] < best_f:
            best_f = fvals[0]
            best_x = simplex[0].copy()
        
        # Convergence check
        if max(fvals) - min(fvals) < tol:
            break
    
    return best_x, best_f

def _objective_from_params(params):
    """
    Convert 24-parameter vector to (a, b) and compute error.
    params: shape (2, 12)
    """
    a = _normalize_and_clip(params[0])
    b = _normalize_and_clip(params[1])
    return _spectral_norm_error(a, b)

def coefficients(stages):
    if _A is None:
        _build_hamiltonians()
        
    # Initial guess: Uniform coefficients
    a0 = np.ones(12) / 12
    b0 = np.ones(12) / 12
    x0 = np.vstack([a0, b0])
    
    # Compute initial error
    init_err = _spectral_norm_error(a0, b0)
    
    # Try Nelder-Mead optimization if we have time
    if _remaining() > 30:
        try:
            best_x, best_f = _nelder_mead_simplex(
                _objective_from_params,
                x0,
                max_iter=50,
                tol=1e-6,
                time_limit=time.time() + _remaining() - 10
            )
            a_opt = _normalize_and_clip(best_x[0])
            b_opt = _normalize_and_clip(best_x[1])
            
            # Verify improvement
            if best_f < init_err:
                a = a_opt.tolist()
                b = b_opt.tolist()
                return a, b
        except Exception:
            pass
    
    # Fallback to uniform
    return a0.tolist(), b0.tolist()
