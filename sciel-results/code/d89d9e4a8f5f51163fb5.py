"""Optimized product formula for Heisenberg chain time evolution."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

_start_time = None

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain."""
    n = 6
    dim = 2 ** n
    anisotropy = 0.8
    field = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def tensor_op(op, site):
        """Apply op at 'site' (0-indexed from left), identity elsewhere."""
        result = np.array([[1]], dtype=complex)
        for s in range(n):
            if s == site:
                result = np.kron(result, op)
            else:
                result = np.kron(result, np.eye(2, dtype=complex))
        return result
    
    # Build odd bonds: (0,1), (2,3), (4,5)
    A = np.zeros((dim, dim), dtype=complex)
    for bond in [(0, 1), (2, 3), (4, 5)]:
        i, j = bond
        sx_i = tensor_op(sx, i)
        sx_j = tensor_op(sx, j)
        sy_i = tensor_op(sy, i)
        sy_j = tensor_op(sy, j)
        sz_i = tensor_op(sz, i)
        sz_j = tensor_op(sz, j)
        A += anisotropy * 0.5 * (sx_i @ sx_j + sy_i @ sy_j) + sz_i @ sz_j
    
    # Half uniform field on even sites: 0, 2, 4
    for site in [0, 2, 4]:
        sz_site = tensor_op(sz, site)
        A += 0.5 * field * sz_site
    
    # Build even bonds: (1,2), (3,4)
    B = np.zeros((dim, dim), dtype=complex)
    for bond in [(1, 2), (3, 4)]:
        i, j = bond
        sx_i = tensor_op(sx, i)
        sx_j = tensor_op(sx, j)
        sy_i = tensor_op(sy, i)
        sy_j = tensor_op(sy, j)
        sz_i = tensor_op(sz, i)
        sz_j = tensor_op(sz, j)
        B += anisotropy * 0.5 * (sx_i @ sx_j + sy_i @ sy_j) + sz_i @ sz_j
    
    # Half uniform field on odd sites: 1, 3, 5
    for site in [1, 3, 5]:
        sz_site = tensor_op(sz, site)
        B += 0.5 * field * sz_site
    
    return A, B

def _objective(params, A, B, times):
    """Compute mean spectral norm error."""
    stages = 12
    a = params[:stages]
    b = params[stages:2*stages]
    
    error_sum = 0.0
    for t in times:
        U = np.eye(2**6, dtype=complex)
        for k in range(stages):
            U = expm(-1j * b[k] * t * B) @ U
            U = expm(-1j * a[k] * t * A) @ U
        
        H = A + B
        U_exact = expm(-1j * t * H)
        
        diff = U - U_exact
        sv = np.linalg.svd(diff, compute_uv=False)
        error_sum += sv[0] if len(sv) > 0 else 0.0
    
    return error_sum / len(times)

def coefficients(stages):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    deadline = _start_time + 100.0  # Safety margin
    
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    # Fallback: 4th-order Yoshida
    c = 2.0 ** (1.0/3.0)
    x1 = 1.0 / (2.0 - c)
    x2 = -c / (2.0 - c)
    x3 = 1.0 - x1
    a_fallback = [x3, x1, x3, x1, x3, x1]
    b_fallback = [x2, x2, x2, x2, x2, x2]
    sa = sum(a_fallback)
    sb = sum(b_fallback)
    a_fallback = [ai/sa for ai in a_fallback]
    b_fallback = [bi/sb for bi in b_fallback]
    
    # Parameterization: 22 free params
    def constrained_objective(x):
        if time.time() > deadline:
            return 1e10  # Signal timeout
        a = list(x[:stages-1])
        a.append(1.0 - sum(a))
        b = list(x[stages-1:2*(stages-1)])
        b.append(1.0 - sum(b))
        a = [max(-50.0, min(50.0, ai)) for ai in a]
        b = [max(-50.0, min(50.0, bi)) for bi in b]
        sa = sum(a)
        sb = sum(b)
        if abs(sa) > 1e-15:
            a = [ai / sa for ai in a]
        if abs(sb) > 1e-15:
            b = [bi / sb for bi in b]
        return _objective(a + b, A, B, times)
    
    x0 = [1.0/stages] * (2*(stages-1))
    
    try:
        result = minimize(constrained_objective, x0, method='Nelder-Mead',
                         options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7,
                                  'adaptive': True})
        
        a = list(result.x[:stages-1])
        a.append(1.0 - sum(a))
        b = list(result.x[stages-1:2*(stages-1)])
        b.append(1.0 - sum(b))
        
        a = [max(-50.0, min(50.0, ai)) for ai in a]
        b = [max(-50.0, min(50.0, bi)) for bi in b]
        
        # Verify sums are close to 1
        if abs(sum(a) - 1.0) > 0.1 or abs(sum(b) - 1.0) > 0.1:
            a = a_fallback
            b = b_fallback
            
    except Exception:
        a = a_fallback
        b = b_fallback
    
    if len(a) != stages or len(b) != stages:
        a = [1.0/stages] * stages
        b = [1.0/stages] * stages
    
    return a, b
# EVOLVE-BLOCK-END
