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


def _find_double_root(c1, c2):
    """
    Find a nearby (c1, c2) such that Q has a double root at some positive y.
    Uses Newton's method on the system Q(y)=0, Q'(y)=0 where (c1,c2,y) are unknowns.
    We fix y and solve for c1,c2 that make Q(y)=0 and Q'(y)=0.
    Then we check if this gives a smaller value.
    """
    # Q(y) = q0 + q1*y + q2*y^2 + q3*y^3 + q4*y^4 + q5*y^5
    # where qi are linear in c1, c2 (since c3 is determined by c1,c2)
    # Let's compute the coefficient structure
    # c3 = -(H0_0 + c1*H4_0 + c2*H8_0) / H12_0
    # g(x) = H0 + c1*H4 + c2*H8 + c3*H12
    # Q(y) = g(x)/x^2, so Q(y) = sum_{k=0}^{5} g_{2k+2} * y^k
    # g_{2k+2} = H0_{2k+2} + c1*H4_{2k+2} + c2*H8_{2k+2} + c3*H12_{2k+2}
    
    # For each degree 2k+2 (k=0..5), the coefficient is:
    # a_k + c1*b_k + c2*d_k + c3*e_k
    # where c3 = -(H0_0 + c1*H4_0 + c2*H8_0) / H12_0
    # So coeff_k = a_k + c1*(b_k - H4_0*e_k/H12_0) + c2*(d_k - H8_0*e_k/H12_0)
    # Let A_k = a_k, B_k = b_k - H4_0*e_k/H12_0, C_k = d_k - H8_0*e_k/H12_0
    
    H0 = _HF[0]
    H4 = _HF[4]
    H8 = _HF[8]
    H12 = _HF[12]
    
    # g coefficients at even degrees 2,4,6,8,10,12
    A = np.array([H0[2], H0[4], H0[6], H0[8], H0[10], H0[12]], dtype=np.float64)
    B = np.array([H4[2], H4[4], H4[6], H4[8], H4[10], H4[12]], dtype=np.float64)
    D = np.array([H8[2], H8[4], H8[6], H8[8], H8[10], H8[12]], dtype=np.float64)
    E = np.array([H12[2], H12[4], H12[6], H12[8], H12[10], H12[12]], dtype=np.float64)
    
    h4_0 = H4[0]
    h8_0 = H8[0]
    h12_0 = H12[0]
    
    B_adj = B - h4_0 * E / h12_0
    C_adj = D - h8_0 * E / h12_0
    
    # Q(y) = sum_{k=0}^{5} (A_k + c1*B_adj_k + c2*C_adj_k) * y^k
    # Q'(y) = sum_{k=1}^{5} k*(A_k + c1*B_adj_k + c2*C_adj_k) * y^{k-1}
    
    # For a given y, solve linear system for c1, c2:
    # sum_k (A_k + c1*B_adj_k + c2*C_adj_k) y^k = 0
    # sum_k k*(A_k + c1*B_adj_k + c2*C_adj_k) y^{k-1} = 0
    
    def solve_for_double_root(y_target):
        """Solve for c1, c2 such that Q(y_target)=0 and Q'(y_target)=0."""
        y = y_target
        ys = np.array([y**k for k in range(6)])
        ys_deriv = np.array([k * y**(k-1) for k in range(6)])
        ys_deriv[0] = 0  # k=0 term vanishes in derivative
        
        # Linear system: [B_adj·ys, C_adj·ys; B_adj·ys_deriv, C_adj·ys_deriv] * [c1;c2] = -[A·ys; A·ys_deriv]
        M = np.array([
            [np.dot(B_adj, ys), np.dot(C_adj, ys)],
            [np.dot(B_adj, ys_deriv), np.dot(C_adj, ys_deriv)]
        ])
        rhs = -np.array([np.dot(A, ys), np.dot(A, ys_deriv)])
        
        det = np.linalg.det(M)
        if abs(det) < 1e-15:
            return None
        
        try:
            sol = np.linalg.solve(M, rhs)
            c1_sol, c2_sol = sol
            # Verify
            Qval = np.dot(A + c1_sol * B_adj + c2_sol * C_adj, ys)
            Qdval = np.dot(A + c1_sol * B_adj + c2_sol * C_adj, ys_deriv)
            if abs(Qval) < 1e-10 * max(1, np.sum(np.abs(A + c1_sol * B_adj + c2_sol * C_adj) * ys)) and \
               abs(Qdval) < 1e-10 * max(1, np.sum(np.abs(A + c1_sol * B_adj + c2_sol * C_adj) * ys_deriv)):
                return (float(c1_sol), float(c2_sol))
        except np.linalg.LinAlgError:
            pass
        return None
    
    # Try a grid of y targets
    best = None
    best_val = 1e10
    for y_target in np.logspace(-3, 3, 50):
        sol = solve_for_double_root(y_target)
        if sol is None:
            continue
        c1_t, c2_t = sol
        v = _value_from(c1_t, c2_t)
        if np.isfinite(v) and v < best_val:
            best_val = v
            best = (c1_t, c2_t, y_target)
    
    return best, best_val


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
    
    # Phase 1: Double-root search (tangent points give local minima of the largest root)
    if time_left() > 150:
        dr_best, dr_val = _find_double_root(seed_c1r, seed_c2r)
        if dr_best is not None and dr_val < best_val:
            best_val = dr_val
            best_c1, best_c2 = dr_best
    
    # Phase 2: Broad differential evolution search
    if time_left() > 100:
        from scipy.optimize import differential_evolution
        
        def objective_de(params):
            return _value_from(params[0], params[1])
        
        bounds = [(-5.0, 5.0), (-1.0, 1.0)]
        
        try:
            result_de = differential_evolution(
                objective_de,
                bounds,
                maxiter=300,
                popsize=25,
                tol=1e-10,
                seed=42,
                polish=True,
                workers=1,
            )
            if np.isfinite(result_de.fun) and result_de.fun < best_val:
                best_val = float(result_de.fun)
                best_c1, best_c2 = result_de.x
        except Exception:
            pass
        
        # Try different seeds and bounds
        for seed_try in [1, 7, 13, 99]:
            if time_left() < 60:
                break
            try:
                result_de2 = differential_evolution(
                    objective_de,
                    bounds,
                    maxiter=150,
                    popsize=20,
                    tol=1e-10,
                    seed=seed_try,
                    polish=True,
                    workers=1,
                )
                if np.isfinite(result_de2.fun) and result_de2.fun < best_val:
                    best_val = float(result_de2.fun)
                    best_c1, best_c2 = result_de2.x
            except Exception:
                pass
    
    # Phase 3: Multi-start Nelder-Mead from diverse points
    if time_left() > 50:
        from scipy.optimize import minimize
        
        def objective(params):
            return _value_from(params[0], params[1])
        
        # Generate diverse starting points around best and seed
        np.random.seed(123)
        all_starts = [(best_c1, best_c2), (seed_c1r, seed_c2r)]
        
        # Add points from double-root candidates
        for scale in [0.5, 1.5, 2.0]:
            all_starts.append((best_c1 * scale, best_c2 * scale))
        
        # Random perturbations
        for _ in range(40):
            c1_r = best_c1 + np.random.randn() * 0.5
            c2_r = best_c2 + np.random.randn() * 0.1
            all_starts.append((c1_r, c2_r))
        
        # Grid points
        for c1_g in np.linspace(-3.0, 3.0, 15):
            for c2_g in np.linspace(-0.5, 0.5, 10):
                all_starts.append((c1_g, c2_g))
        
        np.random.shuffle(all_starts)
        
        for c1_init, c2_init in all_starts:
            if time_left() < 40:
                break
            try:
                res = minimize(
                    objective,
                    [c1_init, c2_init],
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True},
                )
                if np.isfinite(res.fun) and res.fun < best_val:
                    best_val = float(res.fun)
                    best_c1, best_c2 = res.x
            except Exception:
                pass
    
    # Phase 4: High-precision local polish
    if time_left() > 15:
        from scipy.optimize import minimize
        
        def objective(params):
            return _value_from(params[0], params[1])
        
        # Multiple refinement passes with tighter tolerances
        for tol_level in [1e-12, 1e-14, 1e-16]:
            if time_left() < 10:
                break
            try:
                res = minimize(
                    objective,
                    [best_c1, best_c2],
                    method='Nelder-Mead',
                    options={'maxiter': 20000, 'xatol': tol_level, 'fatol': tol_level * 0.01, 'adaptive': True},
                )
                if np.isfinite(res.fun) and res.fun < best_val:
                    best_val = float(res.fun)
                    best_c1, best_c2 = res.x
            except Exception:
                pass
        
        # COBYLA for potential escape from local minima
        if time_left() > 8:
            try:
                res = minimize(
                    objective,
                    [best_c1, best_c2],
                    method='COBYLA',
                    options={'maxiter': 10000, 'rhobeg': 1e-14, 'tol': 1e-18, 'catol': 1e-16},
                )
                if np.isfinite(res.fun) and res.fun < best_val:
                    best_val = float(res.fun)
                    best_c1, best_c2 = res.x
            except Exception:
                pass
    
    # Phase 5: Final fine-tuning with small perturbations
    if time_left() > 5:
        def objective(params):
            return _value_from(params[0], params[1])
        
        # Try very small perturbations in random directions
        np.random.seed(456)
        for _ in range(200):
            if time_left() < 3:
                break
            direction = np.random.randn(2)
            direction /= np.linalg.norm(direction)
            for step in [1e-3, 1e-4, 1e-5, 1e-6]:
                trial = np.array([best_c1, best_c2]) + step * direction
                v = _value_from(trial[0], trial[1])
                if np.isfinite(v) and v < best_val:
                    best_val = v
                    best_c1, best_c2 = float(trial[0]), float(trial[1])
    
    if not np.isfinite(best_val):
        return np.array([seed_c0, seed_c1, seed_c2])
    
    c0_out = 1.0
    c1_out = float(best_c1)
    c2_out = float(best_c2)
    
    return np.array([c0_out, c1_out, c2_out])
