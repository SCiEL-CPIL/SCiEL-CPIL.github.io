import numpy as np
import time
from scipy.optimize import minimize

def compute_value(coeffs):
    c0, c1, c2 = coeffs
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 1680.0) / 4084080.0
    
    F = 4096.0 * c3
    E = -122880.0 * c3
    D = 256.0 * c2 + 1397760.0 * c3
    C = -3072.0 * c2 - 6812160.0 * c3
    B = 16.0 * c1 + 11520.0 * c2 + 13977600.0 * c3
    A = -48.0 * c1 - 13440.0 * c2 - 11827200.0 * c3
    
    coeffs_q = [F, E, D, C, B, A]
    # Remove leading zeros? np.roots handles it, but let's be safe
    # Find roots
    roots = np.roots(coeffs_q)
    
    valid_roots = []
    for root in roots:
        if abs(root.imag) < 1e-9:
            r = root.real
            if r > 0:
                eps = 1e-7
                q_minus = np.polyval(coeffs_q, r - eps)
                q_plus = np.polyval(coeffs_q, r + eps)
                if q_minus * q_plus < 0:
                    valid_roots.append(r)
            
    if not valid_roots:
        # If no sign-changing positive roots, what then?
        # The task implies there is one. 
        # If g is positive everywhere, maybe r is infinity? Or 0?
        # But seed has a value.
        # Let's return a large number to discourage this.
        return 1e9
        
    r_max = max(valid_roots)
    return r_max ** 2 / (2.0 * np.pi)

def construct():
    start_time = time.time()
    deadline = start_time + 170.0
    
    # Strategy: Force c3 = 0 to reduce Q to cubic.
    # c3 = 0 => c0 + 12 c1 + 1680 c2 = 0
    # So c0 = -12 c1 - 1680 c2
    # We optimize c1, c2.
    
    def obj_cubic(params):
        c1, c2 = params
        c0 = -12.0 * c1 - 1680.0 * c2
        coeffs = [c0, c1, c2]
        return compute_value(coeffs)
    
    # Initial guesses
    # Seed: c0=-1.13, c1=0.04, c2=0.0003
    # c3_seed = -(-1.13 + 0.48 + 0.504)/4084080 != 0
    # So seed is not in this subspace.
    # We need to find a good starting point in the c3=0 subspace.
    
    best_val = 1e9
    best_params = [0.0, 0.0]
    
    rng = np.random.default_rng(42)
    
    # Try random starts
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        c1 = rng.uniform(-10, 10)
        c2 = rng.uniform(-10, 10)
        val = obj_cubic([c1, c2])
        if val < best_val:
            best_val = val
            best_params = [c1, c2]
            
    # Local optimization from best start
    if time.time() < deadline - 10:
        try:
            res = minimize(obj_cubic, best_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except:
            pass
            
    c1_opt, c2_opt = best_params
    c0_opt = -12.0 * c1_opt - 1680.0 * c2_opt
    
    # Check if this is better than seed
    seed_val = compute_value([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    if best_val < seed_val:
        return np.array([c0_opt, c1_opt, c2_opt])
    else:
        return np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
