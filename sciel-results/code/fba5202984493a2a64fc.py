import numpy as np
import time
import scipy.linalg as la
import scipy.optimize as opt

_A, _B = None, None
_VA, _DA, _VB, _DB = None, None, None, None

def build_hamiltonians():
    n = 6
    dim = 2 ** n
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    I = np.eye(dim, dtype=complex)
    
    def op(site, mat):
        op = I
        for s in range(n):
            if s == site:
                op = np.kron(op, mat)
            else:
                op = np.kron(op, I)
        return op
        
    def op2(site1, site2, mat1, mat2):
        op = I
        for s in range(n):
            if s == site1:
                op = np.kron(op, mat1)
            elif s == site2:
                op = np.kron(op, mat2)
            else:
                op = np.kron(op, I)
        return op

    delta = 0.8
    h_field = 0.3
    half_h = h_field / 2.0
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for i in range(n):
        A += half_h * op(i, sz)
        B += half_h * op(i, sz)
        
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    even_bonds = [(1, 2), (3, 4)]
    
    for i, j in odd_bonds:
        A += op2(i, j, sx, sx)
        A += op2(i, j, sy, sy)
        A += delta * op2(i, j, sz, sz)
        
    for i, j in even_bonds:
        B += op2(i, j, sx, sx)
        B += op2(i, j, sy, sy)
        B += delta * op2(i, j, sz, sz)
        
    return A, B

def get_hamiltonians_and_eigs():
    global _A, _B, _VA, _DA, _VB, _DB
    if _A is None:
        _A, _B = build_hamiltonians()
        _VA, _DA = la.eigh(_A) # eigh for Hermitian
        _VB, _DB = la.eigh(_B)
    return _A, _B, _VA, _DA, _VB, _DB

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 50.0
    
    A, B, VA, DA, VB, DB = get_hamiltonians_and_eigs()
    AB = A + B
    
    ts = [0.25, 0.5, 1.0]
    exact_propagators = [la.expm(-1j * t * AB) for t in ts]
    
    n_params = 2 * stages
    x0 = np.full(n_params, 1.0 / stages)
    
    def objective(x):
        if time.time() > deadline:
            return np.inf
        
        a = x[:stages]
        b = x[stages:]
        
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        if sum_a == 0 or sum_b == 0:
            return np.inf
        a = a / sum_a
        b = b / sum_b
        
        total_error = 0.0
        for i, t in enumerate(ts):
            U = np.eye(A.shape[0], dtype=complex)
            for k in range(stages):
                # exp(-i b_k t B)
                # B = VB DB VB^H
                # exp(-i c t B) = VB exp(-i c t DB) VB^H
                c_b = b[k] * t
                exp_DB = np.exp(-1j * c_b * DB)
                exp_B = (VB * exp_DB) @ VB.conj().T # VB * exp_DB scales columns
                
                c_a = a[k] * t
                exp_DA = np.exp(-1j * c_a * DA)
                exp_A = (VA * exp_DA) @ VA.conj().T
                
                F_k = exp_B @ exp_A
                U = U @ F_k
                
            error = la.norm(U - exact_propagators[i], 2)
            total_error += error
        return total_error / len(ts)

    best_x = x0
    best_val = objective(best_x)
    
    try:
        # Use Nelder-Mead or Powell
        # Powell is good for this
        res = opt.minimize(objective, x0, method='Powell', 
                           options={'maxfev': 200, 'disp': False})
        
        if time.time() < deadline:
            if res.fun < best_val:
                best_x = res.x
                best_val = res.fun
    except Exception:
        pass
        
    a = best_x[:stages]
    b = best_x[stages:]
    
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_a != 0:
        a = a / sum_a
    else:
        a = np.ones(stages) / stages
        
    if sum_b != 0:
        b = b / sum_b
    else:
        b = np.ones(stages) / stages
        
    return list(map(float, a)), list(map(float, b))
