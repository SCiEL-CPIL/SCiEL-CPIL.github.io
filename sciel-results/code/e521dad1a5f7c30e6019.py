# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Precompute Hermite polynomial coefficients (highest degree first)
def _hermite_coeffs(n):
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([2.0, 0.0])
    h_prev = np.array([1.0])
    h_curr = np.array([2.0, 0.0])
    for k in range(1, n):
        new_len = len(h_curr) + 1
        h_2x = np.zeros(new_len)
        h_2x[:-1] = 2.0 * h_curr
        h_2k = np.zeros(new_len)
        pad = new_len - len(h_prev)
        h_2k[pad:] = 2.0 * k * h_prev
        h_next = h_2x - h_2k
        h_prev = h_curr
        h_curr = h_next
    return h_curr

H0_c = _hermite_coeffs(0)
H4_c = _hermite_coeffs(4)
H8_c = _hermite_coeffs(8)
H12_c = _hermite_coeffs(12)

H0_0 = H0_c[-1]   # 1
H4_0 = H4_c[-1]   # 12
H8_0 = H8_c[-1]   # 20160
H12_0 = H12_c[-1] # 462

def _build_q_from_coeffs(c0, c1, c2):
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    g = np.zeros(13)
    for c, hc in [(c0, H0_c), (c1, H4_c), (c2, H8_c), (c3, H12_c)]:
        pad = 13 - len(hc)
        hc_p = np.concatenate([np.zeros(pad), hc])
        g += c * hc_p
    # g has degree 12, coefficients [x^12, x^11, ..., x^0] in g[0..12]
    # q = g/x^2 has degree 10, coefficients [x^10, x^9, ..., x^0]
    # q[i] = g[i+2], so q = g[2:13]
    q = g[2:13]  # degree 10, length 11
    return q

def _compute_value(c0, c1, c2):
    q = _build_q_from_coeffs(c0, c1, c2)
    roots = np.roots(q)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    if len(real_roots) == 0:
        return float('inf')
    valid_roots = []
    for r in real_roots:
        eps = 1e-9 * max(1.0, abs(r))
        q_left = np.polyval(q, r - eps)
        q_right = np.polyval(q, r + eps)
        if q_left * q_right < 0:
            valid_roots.append(r)
    if not valid_roots:
        return float('inf')
    r = max(valid_roots)
    return r**2 / (2.0 * np.pi)

def construct():
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 210s limit
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _compute_value(*seed)
    best_params = seed.copy()
    
    # Structural Change: High-volume Random Restart + Powell Method
    # Powell is robust for non-smooth, low-dimensional problems.
    np.random.seed(42)
    
    # Phase 1: Broad Exploration with Random Restarts
    n_restarts = 200
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    for i in range(n_restarts):
        if time.time() > deadline - 20:
            break
            
        # Generate random starting point
        # Mix of uniform and Gaussian perturbations around best/seed
        if i == 0:
            trial = best_params.copy()
        elif i == 1:
            trial = seed.copy()
        else:
            if np.random.random() < 0.5:
                trial = np.random.uniform(bounds[0][0], bounds[0][1], 3)
            else:
                # Perturb current best
                trial = best_params + np.random.normal(0, 1.0, 3)
                # Clip to reasonable bounds to avoid overflow
                trial = np.clip(trial, -50, 50)
        
        try:
            # Powell's method
            res = minimize(
                lambda p: _compute_value(p[0], p[1], p[2]),
                trial,
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Local Refinement around the Best Found
    if time.time() < deadline - 10:
        # Run several Powell runs starting from very close to the best
        for i in range(50):
            if time.time() > deadline - 10:
                break
            trial = best_params + np.random.normal(0, 1e-4, 3)
            try:
                res = minimize(
                    lambda p: _compute_value(p[0], p[1], p[2]),
                    trial,
                    method='Powell',
                    options={'maxiter': 500, 'xtol': 1e-15, 'ftol': 1e-15}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass

    return np.array(best_params)
# EVOLVE-BLOCK-END
