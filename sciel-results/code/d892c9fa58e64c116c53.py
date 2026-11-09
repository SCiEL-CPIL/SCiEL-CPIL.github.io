import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 130.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _build_hamiltonians():
    N = 6
    delta = 0.8
    field = 0.3
    
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1.0+0j]])
        for op in ops:
            result = np.kron(result, op)
        return result

    # Build A (odd bonds + half field) and B (even bonds + half field)
    # Open chain: bonds 1-2, 2-3, 3-4, 4-5, 5-6
    # Odd bonds: 1-2, 3-4, 5-6 (indices 0, 2, 4 in 0-based bond list? No, sites 0..5)
    # Bond i is between site i and i+1.
    # Odd bonds: (0,1), (2,3), (4,5)
    # Even bonds: (1,2), (3,4)
    
    A = np.zeros((64, 64), dtype=complex)
    B = np.zeros((64, 64), dtype=complex)
    
    # Field term: sum sz / 2
    # Half field in A, half in B
    field_A = 0.0
    field_B = 0.0
    for i in range(N):
        ops = [I2] * N
        ops[i] = sz
        field_A += 0.15 * kron_all(ops) # 0.3 / 2
        field_B += 0.15 * kron_all(ops)
        
    # XXZ Interaction: delta * (sx sx + sy sy) + sz sz
    # = 0.5 * delta * (sigma+ sigma- + sigma- sigma+) + sz sz
    
    def bond_op(i, j):
        # i, j are site indices
        ops_xx = [I2] * N
        ops_yy = [I2] * N
        ops_zz = [I2] * N
        ops_plus = [I2] * N
        ops_minus = [I2] * N
        
        ops_xx[i] = sx
        ops_xx[j] = sx
        ops_yy[i] = sy
        ops_yy[j] = sy
        ops_zz[i] = sz
        ops_zz[j] = sz
        
        # sigma+ = sx + i sy, sigma- = sx - i sy
        # sigma+ sigma- = (sx + i sy)(sx - i sy) = sx^2 + sy^2 + i(sy sx - sx sy) = I/2 + i (i sz) = I/2 - sz
        # This is getting complicated. Let's just use sx sx + sy sy.
        
        return 0.5 * delta * (kron_all(ops_xx) + kron_all(ops_yy)) + kron_all(ops_zz)

    # Odd bonds: (0,1), (2,3), (4,5)
    A += bond_op(0, 1)
    A += bond_op(2, 3)
    A += bond_op(4, 5)
    
    # Even bonds: (1,2), (3,4)
    B += bond_op(1, 2)
    B += bond_op(3, 4)
    
    A += field_A
    B += field_B
    
    return A, B

def _compute_error(coeffs, A, B, t):
    a, b = coeffs
    U = np.eye(64, dtype=complex)
    # Order k=1..12
    for k in range(12):
        # exp(-i b_k t B)
        B_step = -1j * b[k] * t * B
        # exp(-i a_k t A)
        A_step = -1j * a[k] * t * A
        
        # Use matrix exponential
        # For small t, we can use Taylor series or scipy.linalg.expm
        # scipy.linalg.expm is robust
        from scipy.linalg import expm
        U = expm(B_step) @ U
        U = expm(A_step) @ U
        
    # Exact propagator
    H = A + B
    U_exact = expm(-1j * t * H)
    
    # Spectral norm of difference
    diff = U - U_exact
    # Singular values
    s = np.linalg.svd(diff, compute_uv=False)
    return s[0]

def coefficients(stages):
    if _remaining() < 10.0:
        # Fallback to Strang
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    A, B = _build_hamiltonians()
    
    # Initial guess: Strang
    substeps = 6
    a0 = [0.5 / substeps] * 12
    b0 = []
    for _ in range(substeps):
        b0 += [1.0 / substeps, 0.0]
    
    x0 = np.array(a0 + b0)
    
    def objective(x):
        a = x[:12]
        b = x[12:]
        # Consistency check
        if abs(np.sum(a) - 1.0) > 1e-6 or abs(np.sum(b) - 1.0) > 1e-6:
            return 1e10
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e10
            
        # Evaluate at t=1.0 (most sensitive)
        return _compute_error((a, b), A, B, 1.0)
        
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.success or res.fun < objective(x0):
            a_opt = res.x[:12]
            b_opt = res.x[12:]
            # Clip to range
            a_opt = np.clip(a_opt, -50, 50)
            b_opt = np.clip(b_opt, -50, 50)
            
            # Re-normalize to ensure sum=1
            sum_a = np.sum(a_opt)
            sum_b = np.sum(b_opt)
            if sum_a != 0:
                a_opt = a_opt / sum_a
            if sum_b != 0:
                b_opt = b_opt / sum_b
                
            return list(a_opt), list(b_opt)
        else:
            return a0, b0
    except Exception:
        return a0, b0

