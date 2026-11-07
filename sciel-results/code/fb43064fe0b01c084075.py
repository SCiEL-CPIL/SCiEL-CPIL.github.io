"""Optimize the uncertainty-inequality constant via model-and-optimize approach."""
import time
import numpy as np
from fractions import Fraction


def _hermite_coeffs(n):
    """Return list of Fractions for H_n(x) = sum a_k x^k (k=0..n)."""
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


def _build_q_coeffs(c1, c2):
    """
    Given c1, c2 (with c0=1), return float coefficient array of q(x) = g(x)/x^2.
    q(x) is degree 10: q(x) = sum_{k=0}^{5} g_{2(k+1)} x^{2k}
    Coefficients ascending: [q_0, q_1, ..., q_5] where q_i is coeff of x^{2i}.
    """
    c3 = -(_H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    g_coeffs = 1.0 * _HF[0] + c1 * _HF[4] + c2 * _HF[8] + c3 * _HF[12]
    # g_coeffs[2] is coeff of x^2, g_coeffs[4] of x^4, etc.
    q = g_coeffs[2:13:2]
    while len(q) > 1 and q[-1] == 0.0:
        q = q[:-1]
    return q


def _value_from(c1, c2):
    """Compute r^2/(2*pi). Returns inf if no valid sign-change root."""
    try:
        q = _build_q_coeffs(c1, c2)
        if len(q) < 2:
            return np.inf
        
        # Find roots of q(x) = 0. q is in x^{2k} so let y = x^2, Q(y) = q(sqrt(y))
        # Q(y) = q_0 + q_1*y + ... + q_5*y^5
        Q = q  # same coefficients, just variable change
        
        try:
            roots_y = np.roots(Q[::-1])  # np.roots expects descending order
        except Exception:
            return np.inf
        
        # Filter: real, positive roots
        real_pos = []
        for r in roots_y:
            if abs(r.imag) <= 1e-10 * max(1.0, abs(r.real)):
                if r.real > 1e-12:
                    real_pos.append(float(r.real))
        
        if not real_pos:
            return 1e10
        
        def Qval(y):
            return np.polyval(Q[::-1], y)
        
        # Check sign change at each root
        best_y = None
        for y in sorted(real_pos, reverse=True):
            h = 1e-7 * max(1.0, abs(y))
            left = Qval(y - h)
            right = Qval(y + h)
            if left * right < 0:
                best_y = y
                break
        
        if best_y is None:
            for y in sorted(real_pos, reverse=True):
                for h in [1e-6, 1e-5, 1e-4, 1e-3]:
                    h = h * max(1.0, abs(y))
                    left = Qval(y - h)
                    right = Qval(y + h)
                    if left * right < 0:
                        best_y = y
                        break
                if best_y is not None:
                    break
        
        if best_y is None:
            return 1e10
        
        return best_y / (2.0 * np.pi)
    except Exception:
        return np.inf


def _value_from_smooth(c1, c2, beta=50.0):
    """
    Smooth surrogate: approximate min over sign-change roots of y_i/(2*pi).
    Uses log-sum-exp to get a smooth lower-bound-like estimate.
    Actually we want the LARGEST sign-change root, so we use max via log-sum-exp.
    """
    try:
        q = _build_q_coeffs(c1, c2)
        if len(q) < 2:
            return 1e10
        
        Q = q
        try:
            roots_y = np.roots(Q[::-1])
        except Exception:
            return 1e10
        
        def Qval(y):
            return np.polyval(Q[::-1], y)
        
        sign_change_ys = []
        for r in roots_y:
            if abs(r.imag) <= 1e-10 * max(1.0, abs(r.real)):
                y = float(r.real)
                if y > 1e-12:
                    h = 1e-7 * max(1.0, abs(y))
                    left = Qval(y - h)
                    right = Qval(y + h)
                    if left * right < 0:
                        sign_change_ys.append(y)
        
        if not sign_change_ys:
            return 1e10
        
        # We want the largest, so use soft-max
        ys = np.array(sign_change_ys)
        if len(ys) == 1:
            return ys[0] / (2.0 * np.pi)
        
        # Soft max: (1/beta) * log(sum(exp(beta * y_i)))
        max_y = np.max(ys)
        log_sum = max_y + np.log(np.sum(np.exp(beta * (ys - max_y))))
        soft_max_y = log_sum / beta
        
        return soft_max_y / (2.0 * np.pi)
    except Exception:
        return 1e10


def construct():
    """Return three coefficients [c0, c1, c2] minimizing r^2/(2*pi)."""
    start = time.time()
    DEADLINE = 205.0  # 5s margin
    
    seed_c0 = -113.0 / 100.0
    seed_c1 = 1.0 / 25.0
    seed_c2 = 1.0 / 3240.0
    
    # Normalize to c0 = 1
    seed_c1r = seed_c1 / seed_c0
    seed_c2r = seed_c2 / seed_c0
    
    best_c1 = seed_c1r
    best_c2 = seed_c2r
    best_val = _value_from(best_c1, best_c2)
    
    def time_left():
        return DEADLINE - (time.time() - start)
    
    # Grid search over c1, c2
    c1_range = np.linspace(-3.0, 3.0, 150)
    c2_range = np.linspace(-0.3, 0.3, 150)
    
    found = False
    for c1 in c1_range:
        if time_left() < 10:
            break
        for c2 in c2_range:
            if time_left() < 5:
                break
            v = _value_from(c1, c2)
            if v < best_val:
                best_val = v
                best_c1 = c1
                best_c2 = c2
                found = True
    
    # Local refinement using smooth surrogate with Nelder-Mead
    if time_left() > 10:
        from scipy.optimize import minimize
        
        def objective_smooth(params):
            return _value_from_smooth(params[0], params[1], beta=50.0)
        
        try:
            res = minimize(
                objective_smooth,
                [best_c1, best_c2],
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16},
            )
            if np.isfinite(res.fun):
                v_exact = _value_from(res.x[0], res.x[1])
                if v_exact < best_val:
                    best_val = v_exact
                    best_c1, best_c2 = res.x
        except Exception:
            pass
    
    # Direct Nelder-Mead on exact value
    if time_left() > 10:
        from scipy.optimize import minimize
        
        def objective(params):
            return _value_from(params[0], params[1])
        
        try:
            res = minimize(
                objective,
                [best_c1, best_c2],
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16},
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = float(res.fun)
                best_c1, best_c2 = res.x
        except Exception:
            pass
    
    # COBYLA polish
    if time_left() > 5:
        from scipy.optimize import minimize
        def objective(params):
            return _value_from(params[0], params[1])
        try:
            res = minimize(
                objective,
                [best_c1, best_c2],
                method='COBYLA',
                options={'maxiter': 10000, 'rhobeg': 1e-10, 'tol': 1e-18},
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = float(res.fun)
                best_c1, best_c2 = res.x
        except Exception:
            pass
    
    # Multi-start with smooth surrogate
    if time_left() > 15:
        from scipy.optimize import minimize
        
        def objective_smooth(params):
            return _value_from_smooth(params[0], params[1], beta=100.0)
        
        rng = np.random.RandomState(42)
        for _ in range(20):
            if time_left() < 5:
                break
            init = [best_c1 + rng.randn() * 0.5, best_c2 + rng.randn() * 0.1]
            try:
                res = minimize(
                    objective_smooth,
                    init,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16},
                )
                if np.isfinite(res.fun):
                    v_exact = _value_from(res.x[0], res.x[1])
                    if v_exact < best_val:
                        best_val = v_exact
                        best_c1, best_c2 = res.x
            except Exception:
                pass
    
    # Final high-precision polish
    if time_left() > 3:
        from scipy.optimize import minimize
        def objective(params):
            return _value_from(params[0], params[1])
        try:
            res = minimize(
                objective,
                [best_c1, best_c2],
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-18},
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = float(res.fun)
                best_c1, best_c2 = res.x
        except Exception:
            pass
    
    if not np.isfinite(best_val):
        return np.array([seed_c0, seed_c1, seed_c2])
    
    c0_out = 1.0
    c1_out = float(best_c1)
    c2_out = float(best_c2)
    
    return np.array([c0_out, c1_out, c2_out])
