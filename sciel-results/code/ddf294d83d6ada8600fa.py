import numpy as np
from scipy.optimize import minimize
import time

def _hermite_coeffs_in_x2(n):
    if n % 2 != 0:
        raise ValueError("Only even n supported")
    k = n // 2
    h_prev = [1]
    if k == 0: return h_prev
    h_curr = [0, 2]
    for m in range(2, n + 1):
        h_next = [0] * (m + 1)
        for i in range(len(h_curr)):
            h_next[i + 1] += 2 * h_curr[i]
        for i in range(len(h_prev)):
            h_next[i] -= 2 * (m - 1) * h_prev[i]
        h_prev = h_curr
        h_curr = h_next
    coeffs_y = []
    for j in range(k + 1):
        coeffs_y.append(h_curr[2*j])
    return coeffs_y

C0 = _hermite_coeffs_in_x2(0)
C4 = _hermite_coeffs_in_x2(4)
C8 = _hermite_coeffs_in_x2(8)
C12 = _hermite_coeffs_in_x2(12)

def _compute_value(c0, c1, c2):
    c3 = (c0 - 3*c1 + 105*c2) / 10395.0
    def pad(v, length):
        return v + [0]*(length-len(v))
    L = 7
    C0_p = pad(C0, L)
    C4_p = pad(C4, L)
    C8_p = pad(C8, L)
    C12_p = pad(C12, L)
    P = c0*C0_p + c1*C4_p + c2*C8_p + c3*C12_p
    if abs(P[0]) < 1e-10:
        P = P[1:]
    if len(P) == 0:
        return np.inf
    try:
        roots = np.roots(P[::-1])
    except Exception:
        return np.inf
    max_y = 0.0
    for r in roots:
        if abs(np.imag(r)) < 1e-8:
            real_part = np.real(r)
            if real_part > 1e-8:
                if real_part > max_y:
                    max_y = real_part
    if max_y == 0.0:
        return np.inf
    return max_y

def construct():
    start_time = time.time()
    deadline = start_time + 180
    best_value = np.inf
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    
    def objective(x):
        return _compute_value(x[0], x[1], x[2])
        
    # 1. Grid search c0=1
    n_grid = 100
    c1_min, c1_max = -2.0, 2.0
    c2_min, c2_max = -0.05, 0.05
    
    c1_vals = np.linspace(c1_min, c1_max, n_grid)
    c2_vals = np.linspace(c2_min, c2_max, n_grid)
    
    for i in range(n_grid):
        if time.time() > deadline - 30:
            break
        c1 = c1_vals[i]
        for j in range(n_grid):
            if time.time() > deadline - 30:
                break
            c2 = c2_vals[j]
            c0 = 1.0
            val = _compute_value(c0, c1, c2)
            if val < best_value:
                best_value = val
                best_coeffs = np.array([c0, c1, c2])
                
    # 2. Grid search c0=0, c1=1
    n_grid_1d = 200
    c2_vals_1d = np.linspace(-0.5, 0.5, n_grid_1d)
    for c2 in c2_vals_1d:
        if time.time() > deadline - 30:
            break
        val = _compute_value(0.0, 1.0, c2)
        if val < best_value:
            best_value = val
            best_coeffs = np.array([0.0, 1.0, c2])
            
    # 3. Local refinement
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_coeffs, method='Nelder-Mead', 
                           options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 2000})
            if res.fun < best_value:
                best_value = res.fun
                best_coeffs = res.x
        except Exception:
            pass
            
    return list(best_coeffs)
