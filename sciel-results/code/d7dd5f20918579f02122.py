import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline, 15s safety margin
    
    # Define validation cosmologies (diverse set)
    om0_vals = [0.25, 0.30, 0.35, 0.40]
    w0_vals = [-1.2, -1.0, -0.8, -0.5, 0.0]
    wa_vals = [-0.5, -0.2, 0.0, 0.2, 0.5]
    
    # a grid for evaluation
    a_vals = np.linspace(0.05, 0.999, 200)
    
    def om_a(a, om0, w0, wa):
        de = (1 - om0) * np.exp(3 * (1 + w0) * (1 - a) - 1.5 * wa * (1 - a**2))
        return om0 * a**3 / (om0 * a**3 + de)
    
    def f_exact(a, om0, w0, wa):
        # Solve ODE for f: df/da = f * (1.5*Om(a) - 1 + 0.5*(1+w(a)))
        # Actually the standard growth equation: f' + H*f = ... 
        # Let me use the standard form: f = Om^gamma, and the ODE for f
        # The growth equation in terms of a:
        # f' = f * (3/2 * Om(a) - 1/2 * (1 + w_eff(a)))
        # where w_eff is the effective dark energy equation of state
        # More precisely: f' = f * (3/2 * Om(a) - 1/2 * (1 + w(a)))
        # where w(a) = w0 + wa*(1-a)
        
        def ode(a, y):
            om = om_a(a, om0, w0, wa)
            w = w0 + wa * (1 - a)
            return y * (1.5 * om - 0.5 * (1 + w))
        
        # Integrate from a=1 (where f=1) backwards to a
        # Or integrate forward from small a
        # Better: integrate from a=1 to a for each a, but that's expensive
        # Instead, integrate once from a_min to 1 and store
        
        # For efficiency, integrate from a=0.05 to a=1
        a_grid = np.linspace(0.05, 1.0, 1000)
        sol = solve_ivp(ode, [0.05, 1.0], [1.0], t_eval=a_grid, rtol=1e-10, atol=1e-12)
        
        # Interpolate to get f at our a_vals
        f_vals = np.interp(a_vals, a_grid, sol.y[0])
        return f_vals
    
    # Pre-compute exact f for all cosmologies
    print("Computing reference solutions...")
    ref_f = {}
    for om0 in om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                key = (om0, w0, wa)
                ref_f[key] = f_exact(a_vals, om0, w0, wa)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Compute ideal gamma for each cosmology
    ideal_gamma = {}
    for key, f_vals in ref_f.items():
        om0, w0, wa = key
        om_vals = om_a(a_vals, om0, w0, wa)
        # gamma = ln(f) / ln(Om)
        mask = (om_vals > 0.01) & (om_vals < 0.99) & (np.abs(om_vals - 1) > 0.01)
        if np.sum(mask) > 10:
            gamma_ideal = np.log(f_vals[mask]) / np.log(om_vals[mask])
            ideal_gamma[key] = (a_vals[mask], gamma_ideal, om0, w0, wa)
    
    # Set up least-squares problem
    def design_matrix(a, om0, w0, wa):
        om = om_a(a, om0, w0, wa)
        q = 1 - om
        u = 1 + w0 + wa * (1 - a)
        A = np.column_stack([
            np.ones_like(a),
            q,
            q**2,
            u,
            wa * a,
            u * q
        ])
        return A
    
    # Collect all data points
    all_A = []
    all_b = []
    for key, (a_masked, gamma_ideal, om0, w0, wa) in ideal_gamma.items():
        A = design_matrix(a_masked, om0, w0, wa)
        all_A.append(A)
        all_b.append(gamma_ideal)
    
    if len(all_A) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    A_all = np.vstack(all_A)
    b_all = np.concatenate(all_b)
    
    # Least squares fit
    c_ls, residuals, rank, sv = np.linalg.lstsq(A_all, b_all, rcond=None)
    c_ls = np.clip(c_ls, -10, 10)
    
    print(f"LS solution: {c_ls}")
    
    # Define the objective function: max relative error of f and D
    def compute_errors(c, max_a_eval=50):
        c = np.clip(c, -10, 10)
        c0, c1, c2, c3, c4, c5 = c
        
        max_err_f = 0.0
        max_err_d = 0.0
        
        for key, f_ref in ref_f.items():
            om0, w0, wa = key
            om_vals = om_a(a_vals, om0, w0, wa)
            q = 1 - om_vals
            u = 1 + w0 + wa * (1 - a_vals)
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
            
            # Check gamma constraint
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 100.0  # Penalty for invalid gamma
            
            f_approx = om_vals ** gamma
            
            # Relative error in f
            mask_f = np.abs(f_ref) > 1e-6
            if np.any(mask_f):
                rel_err_f = np.abs(f_approx[mask_f] - f_ref[mask_f]) / np.abs(f_ref[mask_f])
                max_err_f = max(max_err_f, np.max(rel_err_f))
            
            # Compute D(a)/D(1) = exp(-∫_ln(a)^0 f(exp(x)) dx)
            # Change variable: x = ln(a), so a = exp(x), da = exp(x)dx
            # ∫ f(a) da from a to 1 = ∫ f(exp(x)) exp(x) dx from ln(a) to 0
            # So D(a)/D(1) = exp(-∫_{ln(a)}^0 f(exp(x)) exp(x) dx)
            
            # Use the approximate f
            x_grid = np.log(a_vals)
            integrand = f_approx * a_vals  # f(exp(x)) * exp(x)
            
            # Integrate from x to 0 (i.e., from a to 1)
            # cumulative integral from a_vals[i] to 1
            # We need ∫_{a_i}^1 f(a) da
            # Use cumulative trapezoidal integration from 1 down to a_i
            
            # Sort a_vals in decreasing order for integration from 1 to a
            # a_vals is already in increasing order (0.05 to 0.999)
            # ∫_{a_i}^1 f(a) da = -∫_1^{a_i} f(a) da
            
            # Cumulative integral from 1 to a_i
            # Use np.cumtrapz
            f_da = f_approx * a_vals  # Wait, no. The integral is ∫ f(a) da, not ∫ f(a) a da
            # Let me re-read: D(a)/D(1) = exp(-∫_{ln(a)}^0 f(exp(x)) dx)
            # With x = ln(a), a = exp(x), da = exp(x) dx
            # ∫ f(a) da = ∫ f(exp(x)) exp(x) dx
            # So ∫_{ln(a)}^0 f(exp(x)) dx = ∫_a^1 f(a') (1/a') da' ... no
            # Actually: ∫_{ln(a)}^0 f(exp(x)) dx, let t = exp(x), dt = exp(x) dx = t dx, so dx = dt/t
            # = ∫_a^1 f(t) (dt/t)
            # Hmm, that's different. Let me re-derive.
            # The growth factor satisfies D' = f D (in terms of ln a)
            # So D(a)/D(1) = exp(∫_1^{ln a} f(exp(x)) dx) = exp(-∫_{ln a}^1 f(exp(x)) dx)
            # Wait, the task says: D(a)/D(1)=exp(-integral_ln(a)^0 f(exp(x)) dx)
            # So it's exp(-∫_{ln(a)}^0 f(exp(x)) dx)
            # Let t = exp(x), dt = t dx, dx = dt/t
            # ∫_{ln(a)}^0 f(exp(x)) dx = ∫_a^1 f(t) (dt/t)
            # So D(a)/D(1) = exp(-∫_a^1 f(t)/t dt)
            
            # OK so I need to compute ∫_a^1 f(t)/t dt
            integrand_d = f_approx / a_vals
            # Cumulative integral from a_i to 1
            # Reverse the array for integration from 1 down to a_i
            a_rev = a_vals[::-1]
            integrand_rev = integrand_d[::-1]
            cum_int = np.cumsum(integrand_rev) * np.diff(np.concatenate([[a_rev[0]], a_rev]))
            # Actually, let me use scipy.integrate.cumulative_trapezoid
            from scipy.integrate import cumulative_trapezoid
            cum_int_rev = cumulative_trapezoid(integrand_rev, a_rev, initial=0)
            cum_int = cum_int_rev[::-1]  # Now cum_int[i] = ∫_{a_vals[i]}^1 f(t)/t dt
            
            d_approx = np.exp(-cum_int)
            
            # Compute reference D
            integrand_d_ref = f_ref / a_vals
            cum_int_rev_ref = cumulative_trapezoid(integrand_d_ref[::-1], a_vals[::-1], initial=0)
            cum_int_ref = cum_int_rev_ref[::-1]
            d_ref = np.exp(-cum_int_ref)
            
            # Relative error in D
            mask_d = np.abs(d_ref) > 1e-6
            if np.any(mask_d):
                rel_err_d = np.abs(d_approx[mask_d] - d_ref[mask_d]) / np.abs(d_ref[mask_d])
                max_err_d = max(max_err_d, np.max(rel_err_d))
        
        return 100 * max(max_err_f, max_err_d)
    
    # Optimize using Nelder-Mead
    print("Optimizing...")
    best_c = c_ls.copy()
    best_score = compute_errors(best_c)
    print(f"Initial score: {best_score}")
    
    # Try a few different starting points
    start_points = [
        c_ls,
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.6, -0.1, 0.0, 0.0, 0.0, 0.0],
    ]
    
    for sp in start_points:
        sp = np.clip(sp, -10, 10)
        try:
            result = minimize(compute_errors, sp, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_score:
                best_score = result.fun
                best_c = np.clip(result.x, -10, 10)
                print(f"Improved score: {best_score}, c: {best_c}")
        except Exception as e:
            print(f"Optimization error: {e}")
        
        if time.time() > deadline:
            break
    
    print(f"Final score: {best_score}, c: {best_c}")
    return list(np.clip(best_c, -10, 10))
