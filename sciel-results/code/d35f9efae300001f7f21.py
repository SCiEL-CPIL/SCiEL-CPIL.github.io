import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

_A = None
_B = None
_AB = None

def get_hamiltonians():
    global _A, _B, _AB
    if _A is not None:
        return _A, _B, _AB
    
    N = 6
    lam = 0.8
    h = 0.3
    
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        out = np.array([[1.0]], dtype=complex)
        for op in ops:
            out = np.kron(out, op)
        return out

    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    field_term = 0.0
    for j in range(N):
        ops = [I]*N
        ops[j] = sz
        field_term += kron_all(ops)
    
    A += 0.5 * h * field_term
    B += 0.5 * h * field_term
    
    for i in range(N-1):
        ops_x = [I]*N
        ops_x[i] = sx
        ops_x[i+1] = sx
        
        ops_y = [I]*N
        ops_y[i] = sy
        ops_y[i+1] = sy
        
        ops_z = [I]*N
        ops_z[i] = sz
        ops_z[i+1] = sz
        
        H_bond = kron_all(ops_x) + kron_all(ops_y) + lam * kron_all(ops_z)
        
        if i % 2 == 0:
            A += H_bond
        else:
            B += H_bond
            
    _A = A
    _B = B
    _AB = A + B
    return _A, _B, _AB

def objective_sym(x):
    a1_5 = x[:5]
    b1_5 = x[5:10]
    
    a6 = 0.5 - np.sum(a1_5)
    b6 = 0.5 - np.sum(b1_5)
    
    a_half = np.append(a1_5, a6)
    b_half = np.append(b1_5, b6)
    
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])
    
    A, B, AB = get_hamiltonians()
    times = [0.25, 0.5, 1.0]
    total_err = 0.0
    
    for t in times:
        U_prod = np.eye(2**6, dtype=complex)
        
        exp_A = [expm(-1j * a_full[k] * t * A) for k in range(12)]
        exp_B = [expm(-1j * b_full[k] * t * B) for k in range(12)]
        
        for k in range(12):
            U_prod = U_prod @ exp_B[k] @ exp_A[k]
            
        U_exact = expm(-1j * t * AB)
        
        diff = U_prod - U_exact
        s = np.linalg.svd(diff, compute_uv=False)
        total_err += s[0]
        
    return total_err / 3.0

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial guess: uniform
    # a_i = 1/12, b_i = 1/12
    # Symmetric: a_1..a_6 = 1/12? Sum = 6/12 = 0.5. Correct.
    x0 = np.full(10, 1.0/12.0)
    
    bounds = [(-50, 50)] * 10
    
    best_x = x0
    best_f = objective_sym(x0)
    
    try:
        res = minimize(objective_sym, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if time.time() < deadline:
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
                
    except Exception:
        pass
        
    a1_5 = best_x[:5]
    b1_5 = best_x[5:10]
    
    a6 = 0.5 - np.sum(a1_5)
    b6 = 0.5 - np.sum(b1_5)
    
    a_half = np.append(a1_5, a6)
    b_half = np.append(b1_5, b6)
    
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])
    
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)
    
    if not np.isclose(np.sum(a_full), 1.0):
        a_full = a_full / np.sum(a_full)
    if not np.isclose(np.sum(b_full), 1.0):
        b_full = b_full / np.sum(b_full)
        
    return list(a_full), list(b_full)
