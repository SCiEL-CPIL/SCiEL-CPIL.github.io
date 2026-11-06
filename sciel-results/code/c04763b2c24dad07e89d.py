import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

_A = None
_B = None
_H = None

def _get_matrices():
    global _A, _B, _H
    if _A is not None:
        return _A, _B, _H
    
    Sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    Sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    Sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    N = 6
    dim = 2 ** N
    
    def kron_list(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for j in range(N):
        ops = [I] * N
        ops[j] = Sz
        field_term = 0.15 * kron_list(ops)
        A += field_term
        B += field_term
        
    for j in range(0, N-1, 2):
        ops = [I] * N
        ops[j] = Sx; ops[j+1] = Sx
        A += kron_list(ops)
        
        ops = [I] * N
        ops[j] = Sy; ops[j+1] = Sy
        A += kron_list(ops)
        
        ops = [I] * N
        ops[j] = Sz; ops[j+1] = Sz
        A += 0.8 * kron_list(ops)
        
    for j in range(1, N-1, 2):
        ops = [I] * N
        ops[j] = Sx; ops[j+1] = Sx
        B += kron_list(ops)
        
        ops = [I] * N
        ops[j] = Sy; ops[j+1] = Sy
        B += kron_list(ops)
        
        ops = [I] * N
        ops[j] = Sz; ops[j+1] = Sz
        B += 0.8 * kron_list(ops)
        
    _A = A
    _B = B
    _H = A + B
    return _A, _B, _H

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 50.0
    
    A, B, H = _get_matrices()
    
    def objective(params):
        u = params[:5]
        v = params[5:10]
        
        a6 = 0.5 - np.sum(u)
        b6 = 0.5 - np.sum(v)
        
        a = np.array([u[0], u[1], u[2], u[3], u[4], a6])
        b = np.array([v[0], v[1], v[2], v[3], v[4], b6])
        
        if np.any(np.abs(a) > 50) or np.any(np.abs(b) > 50):
            return 1e10
            
        full_a = np.concatenate([a, a[::-1]])
        full_b = np.concatenate([b, b[::-1]])
        
        error = 0.0
        for t in [0.25, 0.5, 1.0]:
            U = np.eye(24, dtype=complex)
            for k in range(12):
                U = np.dot(U, expm(-1j * t * full_b[k] * B))
                U = np.dot(U, expm(-1j * t * full_a[k] * A))
                
            U_exact = expm(-1j * t * H)
            error += np.linalg.norm(U - U_exact, 2)
            
        return error / 3.0

    x0 = np.full(10, 1.0/12.0)
    
    best_x = x0
    best_val = objective(x0)
    
    try:
        res = minimize(objective, x0, method='Powell', options={'maxiter': 300, 'disp': False})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
        
    if time.time() > deadline:
        pass

    u = best_x[:5]
    v = best_x[5:10]
    a6 = 0.5 - np.sum(u)
    b6 = 0.5 - np.sum(v)
    
    a = [u[0], u[1], u[2], u[3], u[4], a6]
    b = [v[0], v[1], v[2], v[3], v[4], b6]
    
    full_a = a + a[::-1]
    full_b = b + b[::-1]
    
    return list(full_a), list(full_b)
