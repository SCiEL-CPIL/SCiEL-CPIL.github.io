"""Constrained optimization: minimize largest sign-changing root of q(x)=g(x)/x^2
subject to g(x) >= 0 for all x >= 0, using penalty-based Nelder-Mead."""
import numpy as np
import time
import sys

def _hermite_coeffs(n):
    """Return coefficients of physicists' Hermite polynomial H_n(x) as a list
    from highest degree to constant term (for numpy.polyval)."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [2.0, 0.0]
    h_prev = [1.0]
    h_curr = [2.0, 0.0]
    for k in range(1, n):
        shifted = [2.0 * c for c in h_curr] + [0.0]
        diff = len(shifted) - len(h_prev)
        padded = [0.0] * diff + [-2.0 * k * c for c in h_prev]
        h_next = [a + b for a, b in zip(shifted, padded)]
        h_prev, h_curr = h_curr, h_next
    return h_curr

_H0 = np.array(_hermite_coeffs(0))
_H4 = np.array(_hermite_coeffs(4))
_H8 = np.array(_hermite_coeffs(8))
_H12 = np.array(_hermite_coeffs(12))

_MAX_LEN = 13
def _pad(arr):
    if len(arr) >= _MAX_LEN:
        return arr[:_MAX_LEN]
    return np.concatenate([arr, np.zeros(_MAX_LEN - len(arr))])

_P0 = _pad(_H0)
_P4 = _pad(_H4)
_P8 = _pad(_H8)
_P12 = _pad(_H12)

_H0_0 = 1.0
_H4_0 = -12.0
_H8_0 = 1680.0
_H12_0 = -1330560.0

def _build_g_coefficients(c0, c1, c2):
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    g_coeffs = c0 * _P0 + c1 * _P4 + c2 * _P8 + c3 * _P12
    return g_coeffs, c3

def _q_coefficients(c0, c1, c2):
    g_coeffs, c3 = _build_g_coefficients(c0, c1, c2)
    q_coeffs = g_coeffs[:11]
    return q_coeffs

def _find_largest_sign_changing_root(q_coeffs):
    try:
        roots = np.roots(q_coeffs)
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-9 * max(1.0, abs(r.real)):
                real_roots.append(r.real)
        
        if not real_roots:
            return None
        
        best_r = None
        for r in real_roots:
            if r <= 0:
                continue
            eps = 1e-7 * max(1.0, abs(r))
            val_left = np.polyval(q_coeffs, r - eps)
            val_right = np.polyval(q_coeffs, r + eps)
            if val_left * val_right < 0:
                if best_r is None or r > best_r:
                    best_r = r
        
        return best_r
    except Exception:
        return None

def _check_nonnegativity(g_coeffs, n_points=1000, x_max=50.0):
    x_grid = np.linspace(0.001, x_max, n_points)
    g_vals = np.polyval(g_coeffs, x_grid)
    return np.min(g_vals)

def _compute_value(c0, c1, c2, strict=False):
    q_coeffs = _q_coefficients(c0, c1, c2)
    g_coeffs, _ = _build_g_coefficients(c0, c1, c2)
    
    if strict:
        min_val = _check_nonnegativity(g_coeffs, n_points=2000, x_max=60.0)
    else:
        min_val = _check_nonnegativity(g_coeffs, n_points=800, x_max=50.0)
    
    if min_val < -1e-8:
        return None
    
    r = _find_largest_sign_changing_root(q_coeffs)
    if r is None or r <= 0:
        return None
    
    value = r**2 / (2.0 * np.pi)
    return value

def _objective(params):
    c0, c1, c2 = params
    q_coeffs = _q_coefficients(c0, c1, c2)
    g_coeffs, _ = _build_g_coefficients(c0, c1, c2)
    
    min_val = _check_nonnegativity(g_coeffs, n_points=600, x_max=50.0)
    violation = max(0.0, -min_val)
    penalty = 1000.0 * violation * violation + 100.0 * violation
    
    r = _find_largest_sign_changing_root(q_coeffs)
    if r is None or r <= 0:
        return 1e10 + penalty
    
    value = r**2 / (2.0 * np.pi)
    return value + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 195.0
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0], dtype=np.float64)
    best_coeffs = seed.copy()
    best_value = _compute_value(*seed, strict=True)
    if best_value is None:
        best_value = float('inf')
    
    from scipy.optimize import minimize
    
    rng = np.random.RandomState(42)
    
    # Multiple starting points with varied perturbations
    starting_points = [seed.copy()]
    for i in range(8):
        if time.time() > deadline:
            break
        pert = rng.normal(0, 0.03, size=3)
        starting_points.append(seed + pert)
    
    for x0 in starting_points:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 800,
                    'xatol': 1e-9,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            candidate = result.x
            val = _compute_value(*candidate, strict=False)
            if val is not None and val < best_value:
                best_value = val
                best_coeffs = candidate.copy()
        except Exception:
            pass
    
    # Multi-scale local refinement
    if time.time() < deadline:
        current = best_coeffs.copy()
        current_value = best_value
        
        for scale in [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]:
            if time.time() > deadline:
                break
            for _ in range(400):
                if time.time() > deadline:
                    break
                delta = rng.normal(0, scale, size=3)
                new_coeffs = current + delta
                val = _compute_value(*new_coeffs, strict=False)
                if val is not None and val < current_value:
                    current = new_coeffs
                    current_value = val
                    if current_value < best_value:
                        best_value = current_value
                        best_coeffs = current.copy()
        
        # Directional search in key directions
        directions = [
            np.array([1, 0, 0]), np.array([-1, 0, 0]),
            np.array([0, 1, 0]), np.array([0, -1, 0]),
            np.array([0, 0, 1]), np.array([0, 0, -1]),
            np.array([1, 1, 0]), np.array([1, -1, 0]),
            np.array([-1, 1, 0]), np.array([1, 0, 1]),
            np.array([1, 0, -1]), np.array([0, 1, 1]),
            np.array([0, 1, -1]), np.array([1, 1, 1]),
            np.array([1, 1, -1]), np.array([1, -1, 1]),
            np.array([1, -1, -1]), np.array([-1, -1, 1]),
            np.array([-1, -1, -1]),
        ]
        
        for direction in directions:
            if time.time() > deadline:
                break
            direction = direction / np.linalg.norm(direction)
            
            for scale in [0.005, 0.002, 0.001, 0.0005]:
                if time.time() > deadline:
                    break
                for step in range(40):
                    if time.time() > deadline:
                        break
                    new_coeffs = current + scale * step * 0.1 * direction
                    val = _compute_value(*new_coeffs, strict=False)
                    if val is not None and val < current_value:
                        current = new_coeffs
                        current_value = val
                        if current_value < best_value:
                            best_value = current_value
                            best_coeffs = current.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 8:
        for scale in [0.00001, 0.000005, 0.000002, 0.000001]:
            if time.time() > deadline - 8:
                break
            for _ in range(300):
                if time.time() > deadline - 8:
                    break
                delta = rng.normal(0, scale, size=3)
                new_coeffs = best_coeffs + delta
                val = _compute_value(*new_coeffs, strict=True)
                if val is not None and val < best_value:
                    best_value = val
                    best_coeffs = new_coeffs.copy()
    
    # Final verification
    final_val = _compute_value(*best_coeffs, strict=True)
    if final_val is not None and final_val < best_value:
        best_value = final_val
    
    return best_coeffs
