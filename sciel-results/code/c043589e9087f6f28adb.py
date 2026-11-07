"""Optimize the uncertainty-inequality constant via direct polynomial root optimization."""
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


def _build_Q_coeffs(c1, c2):
    """
    Given c1, c2 (with c0=1), return float coefficient array of Q(y) where
    Q(y) = g(x)/x^2 with y = x^2. Q is degree 5 in y.
    Returns coefficients [q_0, q_1, ..., q_5] for Q(y) = sum q_i * y^i.
    """
    c3 = -(_H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    g_coeffs = 1.0 * _HF[0] + c1 * _HF[4] + c2 * _HF[8] + c3 * _HF[12]
    Q = g_coeffs[2:13:2].copy()
    while len(Q) > 1 and Q[-1] == 0.0:
        Q = Q[:-1]
    return Q


def _largest_sign_change_root_Q(Q):
    """
    Find the largest positive real root of Q(y) at which Q changes sign.
    Returns y_max or None if no such root exists.
    """
    if len(Q) < 2:
        return None
    
    # Find all roots of Q(y) = 0
    try:
        roots_y = np.roots(Q[::-1])  # np.roots expects descending order
    except Exception:
        return None
    
    # Filter: real, positive roots
    real_pos = []
    for r in roots_y:
        if abs(r.imag) <= 1e-8 * max(1.0, abs(r.real)):
            if r.real > 1e-12:
                real_pos.append(float(r.real))
    
    if not real_pos:
        return None
    
    def Qval(y):
        return float(np.polyval(Q[::-1], y))
    
    # Check sign change at each root, starting from largest
    best_y = None
    for y in sorted(real_pos, reverse=True):
        # Use multiple step sizes to robustly detect sign change
        found = False
        for h_mult in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            h = h_mult * max(1.0, abs(y))
            left = Qval(y - h)
            right = Qval(y + h)
            if left * right < 0:
                best_y = y
                found = True
                break
        if found:
            break
    
    return best_y


def _value_from_smooth(c1, c2):
    """
    Smooth surrogate for the nonsmooth objective. Uses a soft-max over
    positive real roots with sign-change detection to create a differentiable
    objective for gradient-based optimization.
    """
    try:
        Q = _build_Q_coeffs(c1, c2)
        if len(Q) < 2:
            return 1e10
        
        roots_y = np.roots(Q[::-1])
        
        # Filter real positive roots
        real_pos = []
        for r in roots_y:
            if abs(r.imag) <= 1e-8 * max(1.0, abs(r.real)):
                if r.real > 1e-12:
                    real_pos.append(float(r.real))
        
        if not real_pos:
            return 1e10
        
        def Qval(y):
            return float(np.polyval(Q[::-1], y))
        
        # Identify sign-change roots and compute soft-max
        sc_roots = []
        for y in real_pos:
            h = 1e-5 * max(1.0, abs(y))
            left = Qval(y - h)
            right = Qval(y + h)
            if left * right < 0:
                sc_roots.append(y)
        
        if not sc_roots:
            return 1e10
        
        # Use log-sum-exp for soft-max to get a smooth approximation of max
        arr = np.array(sc_roots)
        max_val = np.max(arr)
        # Soft-max with temperature parameter
        temp = 0.01
        log_weights = (arr - max_val) / temp
        log_weights = np.clip(log_weights, -100, 100)
        exp_weights = np.exp(log_weights)
        log_sum_exp = max_val / temp + np.log(np.sum(exp_weights))
        soft_max = temp * log_sum_exp
        
        return soft_max / (2.0 * np.pi)
    except Exception:
        return np.inf


def _value_from(c1, c2):
    """Compute r^2/(2*pi). Returns inf if no valid sign-change root."""
    try:
        Q = _build_Q_coeffs(c1, c2)
        y_max = _largest_sign_change_root_Q(Q)
        if y_max is None:
            return 1e10
        return y_max / (2.0 * np.pi)
    except Exception:
        return np.inf


def construct():
    """Return three coefficients [c0, c1, c2] minimizing r^2/(2*pi)."""
    start = time.time()
    DEADLINE = 200.0  # 10s margin before 210s kill
    
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
    
    # Phase 1: Broad differential evolution search
    if time_left() > 80:
        from scipy.optimize import differential_evolution
        
        def objective_de(params):
            return _value_from(params[0], params[1])
        
        bounds = [(-3.0, 3.0), (-0.5, 0.5)]
        
        try:
            result_de = differential_evolution(
                objective_de,
                bounds,
                maxiter=200,
                popsize=20,
                tol=1e-8,
                seed=42,
                polish=True,
                workers=1,
            )
            if np.isfinite(result_de.fun) and result_de.fun < best_val:
                best_val = float(result_de.fun)
                best_c1, best_c2 = result_de.x
        except Exception:
            pass
    
    # Phase 2: Multi-start Nelder-Mead from diverse points
    if time_left() > 40:
        from scipy.optimize import minimize
        
        def objective(params):
            return _value_from(params[0], params[1])
        
        # Generate diverse starting points
        np.random.seed(123)
        n_starts = 30
        c1_starts = np.linspace(-2.0, 2.0, n_starts)
        c2_starts = np.linspace(-0.3, 0.3, n_starts)
        
        # Include the current best and seed
        all_starts = [(best_c1, best_c2), (seed_c1r, seed_c2r)]
        
        for i in range(n_starts):
            for j in range(n_starts // 3):
                all_starts.append((c1_starts[i], c2_starts[j]))
        
        # Shuffle and try
        np.random.shuffle(all_starts)
        
        for c1_init, c2_init in all_starts:
            if time_left() < 35:
                break
            try:
                res = minimize(
                    objective,
                    [c1_init, c2_init],
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14},
                )
                if np.isfinite(res.fun) and res.fun < best_val:
                    best_val = float(res.fun)
                    best_c1, best_c2 = res.x
            except Exception:
                pass
    
    # Phase 3: Gradient-based polish using smooth surrogate
    if time_left() > 20:
        from scipy.optimize import minimize
        
        def objective_smooth(params):
            return _value_from_smooth(params[0], params[1])
        
        try:
            res = minimize(
                objective_smooth,
                [best_c1, best_c2],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 50},
            )
            if np.isfinite(res.fun):
                # Verify with exact objective
                exact_val = _value_from(res.x[0], res.x[1])
                if np.isfinite(exact_val) and exact_val < best_val:
                    best_val = float(exact_val)
                    best_c1, best_c2 = res.x
        except Exception:
            pass
        
        # Also try COBYLA on the exact objective
        if time_left() > 10:
            def objective_exact(params):
                return _value_from(params[0], params[1])
            
            try:
                res = minimize(
                    objective_exact,
                    [best_c1, best_c2],
                    method='COBYLA',
                    options={'maxiter': 5000, 'rhobeg': 1e-12, 'tol': 1e-16, 'catol': 1e-14},
                )
                if np.isfinite(res.fun) and res.fun < best_val:
                    best_val = float(res.fun)
                    best_c1, best_c2 = res.x
            except Exception:
                pass
    
    # Phase 4: Final ultra-fine local search
    if time_left() > 5:
        from scipy.optimize import minimize
        
        def objective(params):
            return _value_from(params[0], params[1])
        
        try:
            res = minimize(
                objective,
                [best_c1, best_c2],
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True},
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = float(res.fun)
                best_c1, best_c2 = res.x
        except Exception:
            pass
    
    # Phase 5: Try to find a configuration with NO sign-change root (ideal)
    # If we can make Q(y) have no positive real roots where it changes sign,
    # then there's no valid r, which might be interpreted as infinite value.
    # But actually, we want the LARGEST sign-change root to be as small as possible.
    # So we want to push all sign-change roots to small y values.
    if time_left() > 3:
        # Try extreme values that might eliminate large roots
        extreme_candidates = [
            (0.0, 0.0),
            (1.0, 0.1),
            (-1.0, 0.1),
            (2.0, 0.2),
            (-2.0, 0.2),
            (0.5, -0.05),
            (-0.5, -0.05),
            (best_c1 * 1.5, best_c2 * 1.5),
            (best_c1 * 0.5, best_c2 * 0.5),
        ]
        
        for c1_t, c2_t in extreme_candidates:
            if time_left() < 2:
                break
            v = _value_from(c1_t, c2_t)
            if np.isfinite(v) and v < best_val:
                best_val = v
                best_c1, best_c2 = c1_t, c2_t
    
    if not np.isfinite(best_val):
        return np.array([seed_c0, seed_c1, seed_c2])
    
    c0_out = 1.0
    c1_out = float(best_c1)
    c2_out = float(best_c2)
    
    return np.array([c0_out, c1_out, c2_out])



