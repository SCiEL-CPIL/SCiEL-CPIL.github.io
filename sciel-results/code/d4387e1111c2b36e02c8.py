"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_score = 1e10

    def om_a(a, Om0, w0, wa):
        """Matter density parameter at scale factor a for CPL dark energy."""
        a_safe = np.maximum(a, 1e-10)
        E2 = Om0 * a_safe**(-3) + (1 - Om0) * a_safe**(-3*(1+w0)) * np.exp(3*wa*(a_safe - 1))
        return Om0 * a_safe**(-3) / E2

    def solve_ode_reference(a_vals, Om0, w0, wa):
        """Solve growth ODE for reference values."""
        a_min = max(min(a_vals), 1e-5)
        a_max = max(a_vals)
        
        def ode(x, y):
            a_x = np.exp(x)
            om_x = om_a(a_x, Om0, w0, wa)
            f_x = y[0]
            D_x = y[1]
            dfdx = -f_x**2 - 1.5*(1 - om_x)*f_x + 1.5
            dDdx = f_x * D_x
            return [dfdx, dDdx]
        
        x_start = np.log(a_min)
        x_end = 0.0
        om_1 = om_a(1.0, Om0, w0, wa)
        y0 = [om_1**0.55, 1.0]
        
        t_eval = np.log(a_vals)
        sol = solve_ivp(ode, [x_start, x_end], y0, rtol=1e-9, atol=1e-12, 
                       t_eval=t_eval)
        
        if sol.success:
            f_vals = sol.y[0]
            D_vals = sol.y[1]
            D_vals = D_vals / D_vals[-1]
            return f_vals, D_vals
        return None, None

    def gamma_vec(c, a_vals, Om0, w0, wa):
        """Compute gamma for vector of a values."""
        om = om_a(a_vals, Om0, w0, wa)
        q = 1 - om
        u = 1 + w0 + wa*(1 - a_vals)
        return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a_vals + c[5]*u*q

    def f_approx_vec(c, a_vals, Om0, w0, wa):
        """Approximate f for vector of a values."""
        g = gamma_vec(c, a_vals, Om0, w0, wa)
        om = om_a(a_vals, Om0, w0, wa)
        return np.power(om, g)

    def D_approx_vec(c, a_vals, Om0, w0, wa):
        """Approximate normalized D for vector of a values using cumulative integration."""
        a_sorted = np.sort(a_vals)
        a_sorted = np.maximum(a_sorted, 1e-5)
        
        # Dense grid on log scale for integration
        n_grid = 2000
        x_grid = np.linspace(np.log(a_sorted[0]), 0.0, n_grid)
        a_grid = np.exp(x_grid)
        f_grid = f_approx_vec(c, a_grid, Om0, w0, wa)
        
        # Cumulative integral from x_i to 0: -integral = -(total - cumsum up to i)
        dx = np.diff(x_grid)
        f_mid = 0.5 * (f_grid[:-1] + f_grid[1:])
        seg_int = f_mid * dx
        
        # cumsum gives integral from start to each point
        cum = np.concatenate([[0.0], np.cumsum(seg_int)])
        total_int = cum[-1]
        
        # For each a_i, integral from ln(a_i) to 0 = total - integral from start to ln(a_i)
        # Find index for each a_i in x_grid
        x_i = np.log(a_sorted)
        idx = np.searchsorted(x_grid, x_i, side='right') - 1
        idx = np.clip(idx, 0, n_grid - 1)
        integral_from_start = cum[idx]
        integral_to_0 = total_int - integral_from_start
        
        D_vals = np.exp(-integral_to_0)
        
        # Map back to original order
        sort_idx = np.argsort(a_vals)
        D_original = np.empty_like(D_vals)
        D_original[sort_idx] = D_vals
        return D_original

    # Validation cosmologies (disjoint public set)
    test_models = [
        (0.30, -1.0, 0.0),
        (0.30, -1.0, 0.5),
        (0.30, -0.5, -0.5),
        (0.25, -1.0, 0.3),
        (0.35, -1.2, 0.4),
        (0.30, -0.8, -0.3),
        (0.28, -1.0, 0.2),
        (0.32, -0.9, -0.4),
    ]
    a_values = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])

    # Pre-compute reference data
    print("Computing reference data...")
    ref_data = []
    for Om0, w0, wa in test_models:
        f_ref, D_ref = solve_ode_reference(a_values, Om0, w0, wa)
        if f_ref is not None:
            ref_data.append((Om0, w0, wa, f_ref, D_ref))
    
    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # LOG-SPACE LINEAR LEAST SQUARES PRE-FIT
    # ln(f) = gamma * ln(Om) where gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # So ln(f) = (c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q) * ln(Om)
    # This is linear in c!
    A_rows = []
    b_vec = []
    for Om0, w0, wa, f_ref, D_ref in ref_data:
        om = om_a(a_values, Om0, w0, wa)
        q = 1 - om
        u = 1 + w0 + wa*(1 - a_values)
        ln_om = np.log(np.maximum(om, 1e-15))
        ln_f_ref = np.log(np.maximum(f_ref, 1e-15))
        
        # Each row: [1, q, q^2, u, wa*a, u*q] * ln_om = ln_f
        features = np.column_stack([
            np.ones_like(a_values),
            q,
            q**2,
            u,
            wa * a_values,
            u * q
        ])
        A_rows.append(features * ln_om[:, np.newaxis])
        b_vec.append(ln_f_ref)
    
    A_all = np.vstack(A_rows)
    b_all = np.concatenate(b_vec)
    
    # Solve least squares
    try:
        c_ls, residuals, rank, sv = np.linalg.lstsq(A_all, b_all, rcond=None)
        c_ls = np.clip(c_ls, -10, 10)
        print(f"LSQ pre-fit: {c_ls}")
    except Exception:
        c_ls = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    def objective(c):
        """Compute 100 * max relative error with smooth approximation."""
        if time.time() > deadline - 3:
            return 1e10
        
        max_err = 0.0
        all_errs = []
        
        for Om0, w0, wa, f_ref, D_ref in ref_data:
            f_app = f_approx_vec(c, a_values, Om0, w0, wa)
            D_app = D_approx_vec(c, a_values, Om0, w0, wa)
            
            # Check gamma bounds
            g = gamma_vec(c, a_values, Om0, w0, wa)
            if np.any(g < 0.1) or np.any(g > 1.2):
                return 1e10
            
            # Relative errors
            err_f = np.abs(f_app - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
            err_D = np.abs(D_app - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
            
            all_errs.extend(err_f.tolist())
            all_errs.extend(err_D.tolist())
        
        if not all_errs:
            return 1e10
        
        err_arr = np.array(all_errs)
        # Smooth max approximation
        beta = 30.0
        log_max = np.log(np.sum(np.exp(beta * err_arr))) / beta
        return 100.0 * log_max

    # Start from LSQ solution
    c0 = c_ls.copy()
    bounds = [(-10, 10)] * 6

    try:
        # Phase 1: L-BFGS-B from LSQ start
        result = minimize(objective, c0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30})
        
        if time.time() < deadline - 15 and result.fun < best_score:
            best_score = result.fun
            best_c = result.x
        
        # Phase 2: Nelder-Mead polish
        if time.time() < deadline - 10:
            result2 = minimize(objective, best_c, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True})
            if time.time() < deadline - 3 and result2.fun < best_score:
                best_score = result2.fun
                best_c = result2.x
        
        # Phase 3: Final polish with tighter tolerances
        if time.time() < deadline - 5:
            result3 = minimize(objective, best_c, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True})
            if result3.fun < best_score:
                best_score = result3.fun
                best_c = result3.x
        
        best_c = np.clip(best_c, -10, 10)
        return list(best_c)
        
    except Exception:
        # Fallback: return LSQ solution if it's valid
        g_check = gamma_vec(c_ls, a_values, 0.3, -1.0, 0.0)
        if np.all(g_check >= 0.1) and np.all(g_check <= 1.2):
            return list(np.clip(c_ls, -10, 10))
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

# EVOLVE-BLOCK-END
