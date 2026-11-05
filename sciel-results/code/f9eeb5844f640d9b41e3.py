import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import expm

def _build_operators():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    N = 6
    anisotropy = 0.8
    field = 0.3
    dim = 2 ** N
    
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1.0]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    def bond_operator(site1, site2, op1, op2):
        ops = [identity] * N
        ops[site1] = op1
        ops[site2] = op2
        return kron_all(ops)
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    gamma = anisotropy
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    even_bonds = [(1, 2), (3, 4)]
    
    for (j, j1) in odd_bonds:
        A += bond_operator(j, j1, sx, sx) + bond_operator(j, j1, sy, sy) + gamma * bond_operator(j, j1, sz, sz)
    
    for (j, j1) in even_bonds:
        B += bond_operator(j, j1, sx, sx) + bond_operator(j, j1, sy, sy) + gamma * bond_operator(j, j1, sz, sz)
    
    half_field = field / 2.0
    field_op = np.zeros((dim, dim), dtype=complex)
    for j in range(N):
        ops = [identity] * N
        ops[j] = sz
        field_op += kron_all(ops)
    
    A += half_field * field_op
    B += half_field * field_op
    
    return A, B

def _spectral_norm_error(a, b, A, B, t_values):
    """Compute mean spectral norm error over t_values."""
    errors = []
    for t in t_values:
        U = np.eye(2 ** 6, dtype=complex)
        for k in range(12):
            if abs(b[k]) > 1e-15:
                U = expm(-1j * b[k] * t * B) @ U
            if abs(a[k]) > 1e-15:
                U = expm(-1j * a[k] * t * A) @ U
        
        H = A + B
        U_exact = expm(-1j * t * H)
        
        diff = U - U_exact
        s = np.linalg.svd(diff, compute_uv=False)
        errors.append(s[0])
    
    return np.mean(errors)

def coefficients(stages):
    """Optimize coefficients for the 12-stage product formula."""
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    A, B = _build_operators()
    t_values = [0.25, 0.5, 1.0]
    
    # Initial guess
    a0 = np.ones(12) / 12.0
    b0 = np.ones(12) / 12.0
    
    def objective(params):
        a_params = params[:11]
        b_params = params[11:22]
        
        a = np.zeros(12)
        b = np.zeros(12)
        a[:11] = a_params
        a[11] = 1.0 - np.sum(a_params)
        b[:11] = b_params
        b[11] = 1.0 - np.sum(b_params)
        
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e6
        
        return _spectral_norm_error(a, b, A, B, t_values)
    
    x0 = np.concatenate([a0[:11], b0[:11]])
    best_a = a0.copy()
    best_b = b0.copy()
    best_score = objective(x0)
    
    # Optimization
    bounds = [(-50, 50)] * 22
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if result.fun < best_score:
            best_score = result.fun
            a_params = result.x[:11]
            b_params = result.x[11:22]
            best_a = np.zeros(12)
            best_b = np.zeros(12)
            best_a[:11] = a_params
            best_a[11] = 1.0 - np.sum(a_params)
            best_b[:11] = b_params
            best_b[11] = 1.0 - np.sum(b_params)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        a_list = [float(x) for x in best_a]
        b_list = [float(x) for x in best_b]
        a_sum = sum(a_list)
        b_sum = sum(b_list)
        if abs(a_sum) > 1e-15:
            a_list = [x / a_sum for x in a_list]
        if abs(b_sum) > 1e-15:
            b_list = [x / b_sum for x in b_list]
        a_list = [max(-50, min(50, x)) for x in a_list]
        b_list = [max(-50, min(50, x)) for x in b_list]
        return a_list, b_list
    
    # Second optimization: symmetric pattern
    def objective_sym(params):
        a_sym = params[:6]
        b_sym = params[6:12]
        
        a = np.zeros(12)
        b = np.zeros(12)
        for i in range(6):
            a[i] = a_sym[i]
            a[11-i] = a_sym[i]
            b[i] = b_sym[i]
            b[11-i] = b_sym[i]
        
        a_sum = np.sum(a)
        b_sum = np.sum(b)
        if abs(a_sum) > 1e-10:
            a = a / a_sum
        if abs(b_sum) > 1e-10:
            b = b / b_sum
        
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e6
        
        return _spectral_norm_error(a, b, A, B, t_values)
    
    a_sym0 = np.ones(6) / 6.0
    b_sym0 = np.ones(6) / 6.0
    x0_sym = np.concatenate([a_sym0, b_sym0])
    
    try:
        result_sym = minimize(objective_sym, x0_sym, method='L-BFGS-B', 
                             bounds=[(-50, 50)] * 12,
                             options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if result_sym.fun < best_score:
            best_score = result_sym.fun
            a_sym = result_sym.x[:6]
            b_sym = result_sym.x[6:12]
            best_a = np.zeros(12)
            best_b = np.zeros(12)
            for i in range(6):
                best_a[i] = a_sym[i]
                best_a[11-i] = a_sym[i]
                best_b[i] = b_sym[i]
                best_b[11-i] = b_sym[i]
            a_sum = np.sum(best_a)
            b_sum = np.sum(best_b)
            if abs(a_sum) > 1e-10:
                best_a = best_a / a_sum
            if abs(b_sum) > 1e-10:
                best_b = best_b / b_sum
    except Exception:
        pass
    
    a_list = [float(x) for x in best_a]
    b_list = [float(x) for x in best_b]
    
    a_sum = sum(a_list)
    b_sum = sum(b_list)
    if abs(a_sum) > 1e-15:
        a_list = [x / a_sum for x in a_list]
    if abs(b_sum) > 1e-15:
        b_list = [x / b_sum for x in b_list]
    
    a_list = [max(-50, min(50, x)) for x in a_list]
    b_list = [max(-50, min(50, x)) for x in b_list]
    
    return a_list, b_list
