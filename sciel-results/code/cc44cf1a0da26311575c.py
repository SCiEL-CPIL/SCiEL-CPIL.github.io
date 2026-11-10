import time
import math
import numpy as np
import sympy as sp
from scipy.optimize import minimize

def construct():
    """Return 12 double-root locations z_1..z_12."""
    
    # Helper to compute Laguerre polynomial L_n^(alpha)(x) exactly using sympy
    def laguerre_poly(n, alpha, x):
        return sp.laguerre(n, alpha, x)
    
    # Precompute symbolic Laguerre polynomials and their derivatives
    alpha = sp.Rational(-1, 2)
    x = sp.symbols('x')
    Ls = [sp.laguerre(2*k, alpha, x) for k in range(26)]
    Ls_diff = [sp.diff(L, x) for L in Ls]
    
    def _score_exact(z_list):
        try:
            z_arr = np.array(sorted(z_list), dtype=float)
            if np.any(z_arr <= 0) or np.any(~np.isfinite(z_arr)):
                return float('inf')
            if len(np.unique(z_arr)) < len(z_arr):
                return float('inf')
            
            # Convert z_i to exact rationals
            z_sympy = [sp.Rational(float(zi)) for zi in z_arr]
            
            # Setup linear system A c = b
            c_syms = sp.symbols('c0:26')
            
            # Build A and b
            A_rows = []
            b_vals = []
            
            # Row 0: g(0) = 0
            A_rows.append([Ls[k].subs(x, 0) for k in range(26)])
            b_vals.append(sp.Rational(0))
            
            # Row 1: g'(0) = 1
            A_rows.append([Ls_diff[k].subs(x, 0) for k in range(26)])
            b_vals.append(sp.Rational(1))
            
            for i in range(12):
                zi = z_sympy[i]
                # Row 2+2i: g(zi) = 0
                A_rows.append([Ls[k].subs(x, zi) for k in range(26)])
                b_vals.append(sp.Rational(0))
                
                # Row 3+2i: g'(zi) = 0
                A_rows.append([Ls_diff[k].subs(x, zi) for k in range(26)])
                b_vals.append(sp.Rational(0))
                
            A_mat = sp.Matrix(A_rows)
            b_vec = sp.Matrix(b_vals)
            
            # Check for singularity quickly?
            # LUsolve will raise an error if singular.
            try:
                c_sol = A_mat.LUsolve(b_vec)
            except Exception:
                return float('inf')
            
            # Construct g(x) with solved coefficients
            g_sol = sum(c_sol[k] * Ls[k] for k in range(26))
            
            # Construct D(x) = x * prod (x - z_i)^2
            D_expr = x
            for zi in z_sympy:
                D_expr *= (x - zi)**2
            
            # q(x) = g(x) / D(x)
            q_poly, rem = sp.div(g_sol, D_expr, x)
            
            if rem != 0:
                if sp.simplify(rem) != 0:
                    return float('inf')
            
            # Find real roots of q(x)
            try:
                q_roots = sp.real_roots(q_poly, x)
            except Exception:
                return float('inf')
            
            max_r = 0.0
            eps = sp.Rational(1, 10)**198
            
            for r in q_roots:
                q_minus = q_poly.subs(x, r - eps)
                q_plus = q_poly.subs(x, r + eps)
                
                prod = q_minus * q_plus
                if prod < 0:
                    r_val = float(r)
                    if r_val > max_r:
                        max_r = r_val
            
            B = max(0, max_r)
            return B / (2 * math.pi)
            
        except Exception:
            return float('inf')

    # Initial guess: Roots of L_12^{-1/2}(x)
    try:
        roots_lag = sp.nroots(sp.laguerre(12, alpha, x))
        z_init = np.array(sorted([float(r.real) for r in roots_lag if abs(r.imag) < 1e-10]))
        if len(z_init) != 12:
            # Fallback to Bessel zeros if nroots fails
            def bessel_zero_J0(k):
                x_val = (k - 0.25) * math.pi
                if x_val < 0.1:
                    return 2.4048255576957728
                return x_val - 1.0/(8.0*x_val) + 9.0/(128.0*x_val**3) - 225.0/(3072.0*x_val**5)
            z_init = np.array([bessel_zero_J0(i)**2 for i in range(1, 13)])
    except Exception:
        def bessel_zero_J0(k):
            x_val = (k - 0.25) * math.pi
            if x_val < 0.1:
                return 2.4048255576957728
            return x_val - 1.0/(8.0*x_val) + 9.0/(128.0*x_val**3) - 225.0/(3072.0*x_val**5)
        z_init = np.array([bessel_zero_J0(i)**2 for i in range(1, 13)])

    best_z = z_init.copy()
    best_score = _score_exact(best_z)
    
    deadline = time.time() + 580.0
    
    # Use scipy.optimize.minimize with Nelder-Mead
    def objective(z):
        if time.time() > deadline:
            raise TimeoutError
        # Ensure positivity and distinctness
        if np.any(z <= 0):
            return 1e10
        z_sorted = np.sort(z)
        if np.any(np.diff(z_sorted) < 1e-8):
            return 1e10
        return _score_exact(z_sorted)
    
    try:
        res = minimize(objective, z_init, method='Nelder-Mead', 
                       options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        if res.success and res.fun < best_score:
            best_z = res.x
            best_score = res.fun
    except TimeoutError:
        pass
    except Exception:
        pass

    # If time remains, do a few random perturbations on the best found
    if time.time() < deadline - 30:
        current_z = best_z.copy()
        current_score = best_score
        np.random.seed(42)
        
        for _ in range(10):
            if time.time() > deadline:
                break
            candidate = current_z.copy()
            idx = np.random.randint(0, 12)
            factor = np.random.uniform(0.9, 1.1)
            candidate[idx] *= factor
            
            if np.any(candidate <= 0):
                continue
            c_sorted = np.sort(candidate)
            if np.any(np.diff(c_sorted) < 1e-6):
                continue
                
            score = _score_exact(c_sorted)
            if score < current_score:
                current_score = score
                current_z = c_sorted
                best_z = c_sorted
                best_score = score

    return list(best_z)
