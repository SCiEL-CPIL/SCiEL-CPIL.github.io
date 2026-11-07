import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize, differential_evolution

# ---------- exact Hermite coefficients (physicists' H_n) ----------
def _hermite_coeffs(n):
    if n == 0:
        return [Fraction(1)]
    if n == 1:
        return [Fraction(0), Fraction(2)]
    c1 = _hermite_coeffs(n - 1)
    c2 = _hermite_coeffs(n - 2)
    a = [Fraction(0)] + [Fraction(2) * v for v in c1]
    b = [Fraction(-2 * (n - 1)) * v for v in c2]
    m = max(len(a), len(b))
    res = [Fraction(0)] * m
    for i, v in enumerate(a):
        res[i] += v
    for i, v in enumerate(b):
        res[i] += v
    while len(res) > 1 and res[-1] == 0:
        res.pop()
    return res

_HC = {n: _hermite_coeffs(n) for n in (0, 4, 8, 12)}
_HF = {n: np.array([float(v) for v in c], dtype=np.float64) for n, c in _HC.items()}
_H0_0 = float(_HC[0][0])
_H4_0 = float(_HC[4][0])
_H8_0 = float(_HC[8][0])
_H12_0 = float(_HC[12][0])

# ---------- coefficient extraction ----------
def _build_q_coeffs(c1, c2):
    """q(x)=g(x)/x^2 with c0=1. Returns ascending coeffs of q in powers x^{2k}."""
    c3 = -(_H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    g = 1.0 * _HF[0] + c1 * _HF[4] + c2 * _HF[8] + c3 * _HF[12]
    # g is even, so g[2], g[4], ... are the coefficients of x^2, x^4, etc.
    # q(x) = g(x)/x^2. If g(x) = sum a_i x^i, q(x) = sum a_{i+2} x^{i-2}.
    # Since g is even, only even powers.
    # g[0] is coeff of x^0 (must be 0).
    # g[2] is coeff of x^2. q[0] = g[2].
    # g[4] is coeff of x^4. q[1] = g[4].
    # ...
    # g[12] is coeff of x^12. q[5] = g[12].
    q = g[2:13:2]
    while len(q) > 1 and q[-1] == 0.0:
        q = q[:-1]
    return np.asarray(q, dtype=np.float64)

def _find_largest_sign_change_root(q_coeffs):
    """Find the largest real root of q(y) where q changes sign.
    q_coeffs: ascending powers of y (q = sum q[i] * y^i)
    Returns y_value or None if no valid root found.
    """
    if len(q_coeffs) < 2:
        return None
    
    # Convert to descending powers for np.roots
    q_desc = q_coeffs[::-1]
    
    # Use companion matrix eigenvalue method for better numerical stability
    try:
        roots = np.roots(q_desc)
    except Exception:
        return None
    
    # Filter for real, positive roots
    real_pos_roots = []
    for r in roots:
        if abs(r.imag) <= 1e-12 * max(1.0, abs(r.real)) and r.real > 1e-14:
            real_pos_roots.append(float(r.real))
    
    if not real_pos_roots:
        return None
    
    # Sort in descending order
    real_pos_roots.sort(reverse=True)
    
    # Define q(y) and its derivative for sign change detection
    def q_val(y):
        return np.polyval(q_desc, y)
    
    def q_deriv(y):
        # Derivative of q(y) = sum q[i] * y^i
        # q'(y) = sum i * q[i] * y^(i-1)
        if len(q_coeffs) < 2:
            return 0.0
        deriv_coeffs = np.array([i * q_coeffs[i] for i in range(1, len(q_coeffs))])
        return np.polyval(deriv_coeffs[::-1], y)
    
    # Check each root from largest to smallest for sign change
    for y in real_pos_roots:
        # Use multiple step sizes for robustness
        for h in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
            h_scaled = h * max(1.0, abs(y))
            if y - h_scaled <= 0:
                continue
            left_val = q_val(y - h_scaled)
            right_val = q_val(y + h_scaled)
            if left_val * right_val < 0:
                # Refine the root using bisection
                a, b = y - h_scaled, y + h_scaled
                fa, fb = left_val, right_val
                for _ in range(60):
                    mid = 0.5 * (a + b)
                    fm = q_val(mid)
                    if fm == 0:
                        a = b = mid
                        break
                    if fa * fm < 0:
                        b, fb = mid, fm
                    else:
                        a, fa = mid, fm
                return 0.5 * (a + b)
    
    return None

def _value_from(c1, c2):
    try:
        q = _build_q_coeffs(c1, c2)
        if len(q) < 2:
            return np.inf
        
        best_y = _find_largest_sign_change_root(q)
        
        if best_y is None:
            return 1e10
            
        return best_y / (2.0 * np.pi)
    except Exception:
        return np.inf

def construct():
    start = time.time()
    DEADLINE = 205.0  # Safety margin for 210s limit

    seed_c0 = -113.0 / 100.0
    seed_c1 = 1.0 / 25.0
    seed_c2 = 1.0 / 3240.0
    seed_c1r = seed_c1 / seed_c0
    seed_c2r = seed_c2 / seed_c0

    best_c1 = seed_c1r
    best_c2 = seed_c2r
    best_val = _value_from(best_c1, best_c2)

    def time_left():
        return DEADLINE - (time.time() - start)

    # Phase 1: Global Search using Differential Evolution
    bounds = [(-1.0, 1.0), (-0.1, 0.1)]
    
    if time_left() > 60:
        try:
            def obj_de(params):
                if time_left() < 10:
                    return best_val
                return _value_from(params[0], params[1])
            
            res_de = differential_evolution(obj_de, bounds, maxiter=100, popsize=10, 
                                             tol=1e-6, seed=42, polish=False)
            if res_de.success or (np.isfinite(res_de.fun) and res_de.fun < best_val):
                best_c1, best_c2 = res_de.x
                best_val = _value_from(best_c1, best_c2)
        except Exception:
            pass

    # Phase 2: Local Optimization from multiple starts
    def obj_exact(params):
        if time_left() < 5:
            return best_val
        return _value_from(params[0], params[1])

    rng = np.random.RandomState(42)
    starts = [[best_c1, best_c2], [seed_c1r, seed_c2r]]
    
    # Add more diverse random starts
    for _ in range(10):
        starts.append([best_c1 + rng.randn() * 0.2, best_c2 + rng.randn() * 0.02])
        
    for s in starts:
        if time_left() < 15:
            break
        try:
            res = minimize(obj_exact, s, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = float(res.fun)
                best_c1, best_c2 = float(res.x[0]), float(res.x[1])
        except Exception:
            pass

    # Phase 3: Final Polish with tighter tolerances
    if time_left() > 10:
        try:
            res = minimize(obj_exact, [best_c1, best_c2], method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = float(res.fun)
                best_c1, best_c2 = float(res.x[0]), float(res.x[1])
        except Exception:
            pass

    if not np.isfinite(best_val):
        return np.array([seed_c0, seed_c1, seed_c2])

    return np.array([1.0, float(best_c1), float(best_c2)])
