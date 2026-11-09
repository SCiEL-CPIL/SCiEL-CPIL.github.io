"""Improved cosmology growth-index fitting with vectorized minimax optimization."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.integrate import cumulative_trapezoid

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # 1. Define validation cosmologies - broader coverage
    om_m_values = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    w0_values = [-1.3, -1.1, -1.0, -0.9, -0.7]
    wa_values = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]
    
    test_models = []
    for om in om_m_values:
        for w0 in w0_values:
            for wa in wa_values:
                test_models.append((om, w0, wa))
    
    # 2. Pre-compute reference data
    a_eval = np.linspace(0.05, 1.0, 300)
    ln_a_eval = np.log(a_eval)
    
    ref_f_list = []
    ref_D_list = []
    om_a_list = []
    q_list = []
    u_list = []
    wa_a_list = []
    
    for om_m, w0, wa in test_models:
        def ode_rhs(ln_a, f):
            a = np.exp(ln_a)
            exp_wa = 3.0 * (1.0 + w0 + wa)
            de_term = (1.0 - om_m) * (a ** exp_wa)
            mat_term = om_m * (a ** 3.0)
            e2 = mat_term + de_term
            om_a = mat_term / e2 if e2 > 1e-15 else 0.0
            
            coeff = 1.5 * (1.0 + om_a) + (2.0/3.0) * w0 + (2.0/3.0) * wa * (1.0 - a)
            return -f**2 - f * coeff + 0.5 * om_a

        a_init = 0.05
        ln_a_init = np.log(a_init)
        exp_wa_init = 3.0 * (1.0 + w0 + wa)
        de_term_init = (1.0 - om_m) * (a_init ** exp_wa_init)
        mat_term_init = om_m * (a_init ** 3.0)
        e2_init = mat_term_init + de_term_init
        om_a_init = mat_term_init / e2_init if e2_init > 1e-15 else 0.0
        f_init = om_a_init ** 0.55 if om_a_init > 0 else 0.55
        
        t_span = (ln_a_init, 0.0)
        t_eval = ln_a_eval
        
        try:
            sol = solve_ivp(ode_rhs, t_span, [f_init], t_eval=t_eval, 
                           rtol=1e-9, atol=1e-12, method='RK45')
            if not sol.success:
                continue
                
            f_ref = sol.y[0]
            if np.any(f_ref <= 0) or np.any(~np.isfinite(f_ref)):
                continue
            
            # Calculate D_ref: D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
            # = exp(integral from 0 to ln(a) of f(exp(x)) dx)  ... wait, let me re-read
            # D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
            # So integral from ln(a) to 0, which for a<1 means ln(a)<0, so integral from negative to 0
            # = - integral from 0 to ln(a)
            # So D/D(1) = exp(-integral from ln(a) to 0) = exp(integral from 0 to ln(a))
            # But ln(a) < 0 for a < 1, so integral from 0 to ln(a) is negative of integral from ln(a) to 0
            
            # Let me compute: integral from ln(a) to 0 of f(exp(x)) dx
            # For a grid from ln(0.05) to 0, this is the integral from each point to 0
            # Use cumulative trapezoid in reverse
            
            f_rev = f_ref[::-1]
            ln_a_rev = ln_a_eval[::-1]
            
            # cumulative_trapezoid gives integral from start
            int_cum = cumulative_trapezoid(f_rev, ln_a_rev, initial=0.0)
            # int_cum[i] = integral from ln_a_rev[0] to ln_a_rev[i] = integral from 0 to ln_a_eval[rev]
            # We want integral from ln(a) to 0 = - integral from 0 to ln(a)
            # int_cum[rev] = integral from 0 to ln_a_eval[rev]
            # So integral from ln(a) to 0 = -int_cum[rev]
            # D/D(1) = exp(-integral from ln(a) to 0) = exp(int_cum[rev])
            
            int_vals_rev = int_cum[::-1]  # integral from 0 to each ln(a)
            # D/D(1) = exp(-integral from ln(a) to 0) = exp(integral from 0 to ln(a))
            D_ref = np.exp(int_vals_rev)
            
            if np.any(~np.isfinite(D_ref)) or np.any(D_ref <= 0):
                continue
            
            # Pre-compute om_a, q, u, wa*a
            exp_wa = 3.0 * (1.0 + w0 + wa)
            de_term = (1.0 - om_m) * (a_eval ** exp_wa)
            mat_term = om_m * (a_eval ** 3.0)
            e2 = mat_term + de_term
            om_a = mat_term / e2
            om_a = np.clip(om_a, 1e-15, 1.0)
            
            q = 1.0 - om_a
            u = 1.0 + w0 + wa * (1.0 - a_eval)
            wa_a = wa * a_eval
            
            ref_f_list.append(f_ref)
            ref_D_list.append(D_ref)
            om_a_list.append(om_a)
            q_list.append(q)
            u_list.append(u)
            wa_a_list.append(wa_a)
            
        except Exception:
            continue
    
    if len(ref_f_list) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Stack into arrays for vectorized computation
    ref_f = np.array(ref_f_list)       # shape: (n_models, n_points)
    ref_D = np.array(ref_D_list)
    om_a_arr = np.array(om_a_list)
    q_arr = np.array(q_list)
    u_arr = np.array(u_list)
    wa_a_arr = np.array(wa_a_list)
    
    n_models = ref_f.shape[0]
    n_points = ref_f.shape[1]
    
    # Pre-compute ln for log-space operations
    ln_ref_f = np.log(ref_f)
    ln_ref_D = np.log(ref_D)
    
    # Pre-compute reversed arrays for D integration
    ref_f_rev = ref_f[:, ::-1]
    ln_a_rev = ln_a_eval[::-1]
    
    def compute_errors(coeffs):
        """Vectorized computation of relative errors for f and D."""
        c0, c1, c2, c3, c4, c5 = coeffs
        
        # Compute gamma for all models and points
        gamma = c0 + c1*q_arr + c2*q_arr**2 + c3*u_arr + c4*wa_a_arr + c5*u_arr*q_arr
        
        # Check gamma bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return None
        
        # Compute f_approx
        f_approx = om_a_arr ** gamma
        
        # Relative error in f
        err_f = np.abs(f_approx - ref_f) / np.abs(ref_f)
        
        # Compute D_approx
        # Need to integrate f_approx in log-space
        f_approx_rev = f_approx[:, ::-1]
        int_cum = cumulative_trapezoid(f_approx_rev, ln_a_rev, axis=1, initial=0.0)
        int_vals_rev = int_cum[:, ::-1]
        D_approx = np.exp(int_vals_rev)
        
        # Relative error in D
        err_D = np.abs(D_approx - ref_D) / np.abs(ref_D)
        
        max_err = max(np.max(err_f), np.max(err_D))
        return max_err
    
    def smooth_objective(coeffs, beta=200.0):
        """Smooth surrogate using log-sum-exp in log-space."""
        if time.time() > deadline:
            return 1e10
        
        c0, c1, c2, c3, c4, c5 = coeffs
        
        gamma = c0 + c1*q_arr + c2*q_arr**2 + c3*u_arr + c4*wa_a_arr + c5*u_arr*q_arr
        
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
        
        f_approx = om_a_arr ** gamma
        
        # Log-space errors for f
        log_err_f = np.abs(np.log(f_approx) - ln_ref_f)
        
        # Compute D_approx in log-space
        f_approx_rev = f_approx[:, ::-1]
        int_cum = cumulative_trapezoid(f_approx_rev, ln_a_rev, axis=1, initial=0.0)
        int_vals_rev = int_cum[:, ::-1]
        log_err_D = np.abs(int_vals_rev - ln_ref_D)
        
        # Combine all errors
        all_errors = np.concatenate([log_err_f.ravel(), log_err_D.ravel()])
        
        max_err = np.max(all_errors)
        if max_err < 1e-15:
            return 0.0
        
        # Log-sum-exp smooth max
        shifted = all_errors - max_err
        # Clip to avoid overflow
        shifted = np.clip(shifted, -50, 50)
        lse = max_err + (1.0 / beta) * np.log(np.sum(np.exp(beta * shifted)))
        
        return 100.0 * lse
    
    def exact_objective(coeffs):
        """Exact minimax objective."""
        if time.time() > deadline:
            return 1e10
        errs = compute_errors(coeffs)
        if errs is None:
            return 1e10
        return 100.0 * errs
    
    # 3. Multi-stage optimization
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    bounds = [(-10, 10)] * 6
    
    best_x = x0.copy()
    best_score = exact_objective(x0)
    
    # Stage 1: Coarse Nelder-Mead with smooth objective
    if time.time() < deadline - 35.0:
        try:
            res = minimize(smooth_objective, x0, args=(50,), method='Nelder-Mead',
                          bounds=bounds, options={'maxiter': 300, 'xatol': 1e-3, 
                          'fatol': 1e-3, 'adaptive': True, 'maxfev': 500})
            if res.fun < best_score:
                exact_s = exact_objective(res.x)
                if exact_s < best_score:
                    best_score = exact_s
                    best_x = res.x.copy()
        except Exception:
            pass
    
    # Stage 2: Medium smoothness
    if time.time() < deadline - 25.0:
        try:
            res = minimize(smooth_objective, best_x, args=(100,), method='Nelder-Mead',
                          bounds=bounds, options={'maxiter': 250, 'xatol': 1e-4,
                          'fatol': 1e-4, 'adaptive': True, 'maxfev': 400})
            if res.fun < best_score:
                exact_s = exact_objective(res.x)
                if exact_s < best_score:
                    best_score = exact_s
                    best_x = res.x.copy()
        except Exception:
            pass
    
    # Stage 3: L-BFGS-B with higher smoothness
    if time.time() < deadline - 18.0:
        try:
            res = minimize(smooth_objective, best_x, args=(200,), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10,
                          'gtol': 1e-8, 'maxfun': 200})
            if res.fun < best_score:
                exact_s = exact_objective(res.x)
                if exact_s < best_score:
                    best_score = exact_s
                    best_x = res.x.copy()
        except Exception:
            pass
    
    # Stage 4: Very high smoothness NM
    if time.time() < deadline - 12.0:
        try:
            res = minimize(smooth_objective, best_x, args=(500,), method='Nelder-Mead',
                          bounds=bounds, options={'maxiter': 200, 'xatol': 1e-5,
                          'fatol': 1e-5, 'adaptive': True, 'maxfev': 300})
            if res.fun < best_score:
                exact_s = exact_objective(res.x)
                if exact_s < best_score:
                    best_score = exact_s
                    best_x = res.x.copy()
        except Exception:
            pass
    
    # Stage 5: Exact minimax NM
    if time.time() < deadline - 8.0:
        try:
            res = minimize(exact_objective, best_x, method='Nelder-Mead',
                          bounds=bounds, options={'maxiter': 200, 'xatol': 1e-6,
                          'fatol': 1e-6, 'adaptive': True, 'maxfev': 300})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Stage 6: Final polish
    if time.time() < deadline - 4.0:
        try:
            res = minimize(exact_objective, best_x, method='Nelder-Mead',
                          bounds=bounds, options={'maxiter': 150, 'xatol': 1e-8,
                          'fatol': 1e-8, 'adaptive': True, 'maxfev': 200})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Clip to valid range
    best_x = np.clip(best_x, -10, 10)
    
    # Final verification
    final_score = exact_objective(best_x)
    if final_score < best_score:
        best_score = final_score
    
    return list(best_x)

# EVOLVE-BLOCK-END
