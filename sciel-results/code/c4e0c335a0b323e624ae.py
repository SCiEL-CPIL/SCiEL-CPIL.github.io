import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import time

def coefficients(stages):
    start_time = time.time()
    time_limit = 140.0
    
    # 1. Build Hamiltonians
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2.0
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2.0
    sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2.0
    s0 = np.eye(2, dtype=complex)
    
    A = np.zeros((64, 64), dtype=complex)
    B = np.zeros((64, 64), dtype=complex)
    
    def kron_all(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res
    
    def add_bond(H, i, j, anisotropy):
        def make_term(op_i, op_j):
            ops = []
            for k in range(6):
                if k == i:
                    ops.append(op_i)
                elif k == j:
                    ops.append(op_j)
                else:
                    ops.append(s0)
            return kron_all(ops)
            
        H += make_term(sx, sx)
        H += make_term(sy, sy)
        H += anisotropy * make_term(sz, sz)
        
    add_bond(A, 0, 1, 0.8)
    add_bond(A, 2, 3, 0.8)
    add_bond(A, 4, 5, 0.8)
    
    add_bond(B, 1, 2, 0.8)
    add_bond(B, 3, 4, 0.8)
    
    field_strength = 0.3
    for j in range(6):
        ops = []
        for k in range(6):
            if k == j:
                ops.append(sz)
            else:
                ops.append(s0)
        H_field_j = kron_all(ops)
        A += 0.5 * field_strength * H_field_j
        B += 0.5 * field_strength * H_field_j

    # 2. Precompute exact propagators
    t_values = [0.25, 0.5, 1.0]
    H_total = A + B
    exact_props = [la.expm(-1j * t * H_total) for t in t_values]
    
    # 3. Define cost function with symmetry
    # Variables: a1..a5, b1..b5
    # a6 = 0.5 - sum(a1..a5)
    # b6 = 0.5 - sum(b1..b5)
    # Full a: [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    
    def cost(x):
        a1, a2, a3, a4, a5 = x[0:5]
        b1, b2, b3, b4, b5 = x[5:10]
        
        a6 = 0.5 - (a1 + a2 + a3 + a4 + a5)
        b6 = 0.5 - (b1 + b2 + b3 + b4 + b5)
        
        a_full = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
        b_full = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
        
        # Check bounds
        if np.any(np.abs(a_full) > 50) or np.any(np.abs(b_full) > 50):
            return 1e10
            
        err = 0.0
        for i, t in enumerate(t_values):
            U = np.eye(64, dtype=complex)
            for k in range(12):
                U = la.expm(-1j * b_full[k] * t * B) @ U
                U = la.expm(-1j * a_full[k] * t * A) @ U
            diff = U - exact_props[i]
            err += np.linalg.norm(diff, 2)
        return err / 3.0

    # 4. Optimize
    x0 = np.zeros(10)
    # Initial guess: equal split 1/12
    # a1..a5 = 1/12, a6 = 0.5 - 5/12 = 1/12.
    # So all are 1/12.
    x0 = np.full(10, 1.0/12.0)
    
    best_x = x0
    best_cost = cost(x0)
    
    # Time check
    if time.time() - start_time > time_limit - 20:
        pass # Skip optimization if building took too long (unlikely)
    else:
        try:
            # Use Nelder-Mead
            # maxiter 500 should be safe
            res = opt.minimize(cost, x0, method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_cost:
                best_x = res.x
                best_cost = res.fun
                
        except Exception:
            pass
            
    # Check time again
    if time.time() - start_time > time_limit - 5:
        # Return best found
        pass

    a1, a2, a3, a4, a5 = best_x[0:5]
    b1, b2, b3, b4, b5 = best_x[5:10]
    
    a6 = 0.5 - (a1 + a2 + a3 + a4 + a5)
    b6 = 0.5 - (b1 + b2 + b3 + b4 + b5)
    
    a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    
    # Ensure finite
    a = [float(v) for v in a]
    b = [float(v) for v in b]
    
    return a, b
