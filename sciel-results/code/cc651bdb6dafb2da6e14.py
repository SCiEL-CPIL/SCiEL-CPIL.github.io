"""Improved growth-index ansatz with direct optimization from task structure."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

# EVOLVE-BLOCK-START

# Deadline guard: evaluator kills after 90s; use 80s safety margin
_DEADLINE = time.time() + 78.0

def _check_deadline():
    return time.time() > _DEADLINE

def construct():
    """Return six coefficients [c0..c5] for the growth-index ansatz.
    
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    f = Om(a)^gamma
    D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
    """
    # Seed: constant gamma = 0.55
    best_coeff = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_score = np.inf
    
    # Validation cosmologies (typical CPL dark energy models)
    val_cosmos = [
        (0.3, -1.0, 0.0),      # Lambda CDM
        (0.3, -1.0, 0.1),      # mild phantom/accel
        (0.3, -1.0, -0.1),     # quintessence-like
        (0.3, -0.9, 0.0),      # early-decaying
        (0.3, -1.1, 0.0),      # phantom
        (0.3, -1.0, 0.2),      # strong wa
        (0.28, -1.0, 0.0),     # slightly lower Om0
        (0.32, -1.0, 0.0),     # slightly higher Om0
        (0.3, -0.8, 0.3),      # more extreme
        (0.3, -1.2, -0.2),     # phantom with negative wa
    ]
    
    # Scale factor grid for evaluation
    a_grid = np.logspace(np.log10(0.1), 0, 150)
    
    def _Om_a_num(a_val, Om0, w0, wa):
        """Compute Om(a) at a single point using exact CPL integral."""
        if a_val <= 0:
            return Om0
        # integral_0^{ln a} [1+w0+wa*(1-e^x)] dx = ln(a)*(1+w0+wa) - wa*(a-1)
        integral = np.log(a_val)*(1+w0+wa) - wa*(a_val-1)
        rho_de_ratio = np.exp(-3 * integral)
        rho_m_ratio = a_val**(-3)
        Om_a = Om0 * rho_m_ratio / (Om0 * rho_m_ratio + (1-Om0) * rho_de_ratio)
        return Om_a
    
    def _compute_Om_array(a_arr, Om0, w0, wa):
        """Vectorized Om(a) computation."""
        return np.array([_Om_a_num(ai, Om0, w0, wa) for ai in a_arr])
    
    def _compute_reference(Om0, w0, wa, a_eval):
        """Compute reference f and D using ODE integration."""
        # Integrate f ODE: f' = f(f+1.5) - 1.5*Om(e^x) where x = ln a
        # Use solve_ivp for accuracy
        def ode_f(x, y):
            a_val = np.exp(x)
            Om_val = _Om_a_num(a_val, Om0, w0, wa)
            return [y[0] * (y[0] + 1.5) - 1.5 * Om_val]
        
        # Initial condition at a=1 (x=0): use perturbation theory
        # f(1) ≈ Om0^0.55 is a good approximation, but let's use a better one
        # For Lambda CDM, f(1) ≈ 0.55 + 0.05*(1-Om0)
        f_init = 0.55 + 0.05 * (1.0 - Om0)
        
        # Integrate from x=0 to x=ln(a_min)
        x_min = np.log(0.1)
        x_max = 0.0
        
        # We need f at each point in a_eval
        x_eval = np.log(a_eval)
        
        # Solve ODE from 0 to x_min (backward)
        try:
            sol = solve_ivp(ode_f, [0, x_min], [f_init], 
                           t_eval=np.sort(x_eval)[::-1],  # from 0 to x_min
                           method='DOP853', rtol=1e-10, atol=1e-12)
            
            if not sol.success:
                # Fallback to simple Euler
                f_ref = np.zeros(len(a_eval))
                D_ref = np.ones(len(a_eval))
                f_ref[-1] = f_init
                for i in range(len(a_eval)-2, -1, -1):
                    dx = np.log(a_eval[i]) - np.log(a_eval[i+1])
                    Om_i = _Om_a_num(a_eval[i+1], Om0, w0, wa)
                    f_prime = f_ref[i+1]*(f_ref[i+1] + 1.5) - 1.5*Om_i
                    f_ref[i] = f_ref[i+1] + dx * f_prime
                # Compute D
                for i in range(len(a_eval)):
                    if i == len(a_eval)-1:
                        D_ref[i] = 1.0
                    else:
                        integral = np.trapz(f_ref[i:], np.log(a_eval[i:]))
                        D_ref[i] = np.exp(-integral)
                return f_ref, D_ref
            
            # sol.y[0] contains f values at t_eval points
            # We need to reorder to match a_eval order
            x_eval_sorted = np.sort(x_eval)[::-1]
            f_sorted = sol.y[0]
            
            # Create mapping
            f_ref = np.zeros(len(a_eval))
            for i, xi in enumerate(x_eval):
                # Find closest point in sorted array
                idx = np.argmin(np.abs(x_eval_sorted - xi))
                f_ref[i] = f_sorted[idx]
            
            # Compute D_ref
            D_ref = np.ones(len(a_eval))
            for i in range(len(a_eval)):
                if i == len(a_eval)-1:
                    D_ref[i] = 1.0
                else:
                    integral = np.trapz(f_ref[i:], np.log(a_eval[i:]))
                    D_ref[i] = np.exp(-integral)
            
            return f_ref, D_ref
            
        except Exception:
            # Fallback to simple Euler
            f_ref = np.zeros(len(a_eval))
            D_ref = np.ones(len(a_eval))
            f_ref[-1] = f_init
            for i in range(len(a_eval)-2, -1, -1):
                dx = np.log(a_eval[i]) - np.log(a_eval[i+1])
                Om_i = _Om_a_num(a_eval[i+1], Om0, w0, wa)
                f_prime = f_ref[i+1]*(f_ref[i+1] + 1.5) - 1.5*Om_i
                f_ref[i] = f_ref[i+1] + dx * f_prime
            for i in range(len(a_eval)):
                if i == len(a_eval)-1:
                    D_ref[i] = 1.0
                else:
                    integral = np.trapz(f_ref[i:], np.log(a_eval[i:]))
                    D_ref[i] = np.exp(-integral)
            return f_ref, D_ref
    
    # Precompute reference solutions for all validation cosmologies
    ref_data = {}
    for (Om0, w0, wa) in val_cosmos:
        if _check_deadline():
            break
        f_ref, D_ref = _compute_reference(Om0, w0, wa, a_grid)
        ref_data[(Om0, w0, wa)] = (f_ref, D_ref)
    
    def _compute_errors(coeff, a_grid, val_cosmos, ref_data):
        """Compute max relative error for f and D over all validation cosmologies."""
        c0, c1, c2, c3, c4, c5 = coeff
        max_err_f = 0.0
        max_err_D = 0.0
        
        for (Om0, w0, wa) in val_cosmos:
            if (Om0, w0, wa) not in ref_data:
                continue
            f_ref, D_ref = ref_data[(Om0, w0, wa)]
            
            # Compute Om(a) on grid
            Om_arr = _compute_Om_array(a_grid, Om0, w0, wa)
            
            # Compute q, u, gamma
            q = 1.0 - Om_arr
            u = 1.0 + w0 + wa * (1.0 - a_grid)
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_grid + c5*u*q
            
            # Check gamma bounds
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return np.inf
            
            # f = Om(a)^gamma
            f_ours = np.power(np.maximum(Om_arr, 1e-15), gamma)
            
            # Compute D_ours
            D_ours = np.ones(len(a_grid))
            for i in range(len(a_grid)):
                if i == len(a_grid)-1:
                    D_ours[i] = 1.0
                else:
                    integral = np.trapz(f_ours[i:], np.log(a_grid[i:]))
                    D_ours[i] = np.exp(-integral)
            
            # Relative errors - skip points too close to 0
            valid = a_grid > 0.15
            if np.any(valid):
                rel_f = np.abs(f_ours[valid] - f_ref[valid]) / np.maximum(np.abs(f_ref[valid]), 1e-10)
                rel_D = np.abs(D_ours[valid] - D_ref[valid]) / np.maximum(np.abs(D_ref[valid]), 1e-10)
                max_err_f = max(max_err_f, np.max(rel_f) if len(rel_f) > 0 else 0)
                max_err_D = max(max_err_D, np.max(rel_D) if len(rel_D) > 0 else 0)
        
        return max_err_f + max_err_D
    
    def objective(coeff):
        if _check_deadline():
            return np.inf
        try:
            return _compute_errors(coeff, a_grid, val_cosmos, ref_data)
        except Exception:
            return np.inf
    
    # Try Nelder-Mead optimization from multiple starting points
    starting_points = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.50, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.1, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.1],
        [0.50, 0.1, 0.05, 0.05, 0.05, 0.05],
        [0.52, 0.08, 0.02, 0.03, 0.01, 0.02],
        [0.58, -0.05, 0.05, 0.02, -0.01, 0.03],
    ]
    
    for sp in starting_points:
        if _check_deadline():
            break
        try:
            res = minimize(
                objective,
                np.array(sp),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True}
            )
            if _check_deadline():
                break
            if res.fun < best_score and np.all(np.abs(res.x) <= 10):
                # Check gamma bounds on a test grid
                test_a = np.logspace(np.log10(0.1), 0, 50)
                valid = True
                for (Om0, w0, wa) in val_cosmos:
                    Om_t = _compute_Om_array(test_a, Om0, w0, wa)
                    q_t = 1.0 - Om_t
                    u_t = 1.0 + w0 + wa*(1.0 - test_a)
                    g_t = (res.x[0] + res.x[1]*q_t + res.x[2]*q_t**2 + 
                           res.x[3]*u_t + res.x[4]*wa*test_a + res.x[5]*u_t*q_t)
                    if np.any(g_t < 0.1) or np.any(g_t > 1.2):
                        valid = False
                        break
                if valid:
                    best_coeff = res.x
                    best_score = res.fun
        except Exception:
            pass
    
    # Final polish with tighter tolerance if time permits
    if not _check_deadline() and best_score < np.inf:
        try:
            res = minimize(
                objective,
                best_coeff,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
            )
            if _check_deadline():
                pass
            elif res.fun < best_score and np.all(np.abs(res.x) <= 10):
                test_a = np.logspace(np.log10(0.1), 0, 50)
                valid = True
                for (Om0, w0, wa) in val_cosmos:
                    Om_t = _compute_Om_array(test_a, Om0, w0, wa)
                    q_t = 1.0 - Om_t
                    u_t = 1.0 + w0 + wa*(1.0 - test_a)
                    g_t = (res.x[0] + res.x[1]*q_t + res.x[2]*q_t**2 + 
                           res.x[3]*u_t + res.x[4]*wa*test_a + res.x[5]*u_t*q_t)
                    if np.any(g_t < 0.1) or np.any(g_t > 1.2):
                        valid = False
                        break
                if valid:
                    best_coeff = res.x
                    best_score = res.fun
        except Exception:
            pass
    
    # Clamp to [-10, 10]
    best_coeff = np.clip(best_coeff, -10.0, 10.0)
    
    return best_coeff.tolist()

# EVOLVE-BLOCK-END
