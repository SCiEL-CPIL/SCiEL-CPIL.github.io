import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin
    
    # Representative cosmologies (Om, w0, wa)
    # We assume the validation set is similar to standard LCDM-like or CPL models
    cosmologies = [
        (0.3, -1.0, 0.0),   # LCDM
        (0.3, -0.9, 0.1),   # CPL
        (0.25, -1.0, 0.0),  # LCDM low Om
        (0.35, -1.0, 0.0),  # LCDM high Om
        (0.3, -0.5, 0.5),   # Quintessence-like
        (0.3, -1.2, 0.5),   # Phantom-like
        (0.3, 0.0, 0.0),    # Matter-like (w=0)
        (0.3, -0.8, 0.2),   # General CPL
    ]
    
    # Scale factors to check
    a_vals = np.logspace(-4, 0, 50)
    
    # Pre-compute reference solutions for each cosmology
    # D_ref[i] is the normalized growth factor D(a)/D(1)
    # f_ref[i] is the growth rate f(a) = d ln D / d ln a
    ref_data = []
    for i, (om, w0, wa) in enumerate(cosmologies):
        if time.time() - start_time > timeout:
            break
        try:
            # Define H^2(a) normalized to H0^2
            # E(a)^2 = Om a^-3 + (1-Om) a^(-3(1+w0+wa))
            # Note: if 1+wa is negative, a^(-3(1+w0+wa)) might be tricky at a=0, but a>=1e-4 is fine.
            
            def E_sq(a):
                de_term = (1.0 - om) * np.power(a, -3.0 * (1.0 + w0 + wa))
                return om * np.power(a, -3.0) + de_term
            
            def dE_sq_da(a):
                # Derivative of E^2 with respect to a
                # d/da (om a^-3) = -3 om a^-4
                # d/da ((1-om) a^(-3(1+w0+wa))) = -3(1+w0+wa)(1-om) a^(-3(1+w0+wa)-1)
                term_m = -3.0 * om * np.power(a, -4.0)
                term_de = -3.0 * (1.0 + w0 + wa) * (1.0 - om) * np.power(a, -3.0 * (1.0 + w0 + wa) - 1.0)
                return term_m + term_de
            
            def ode(x, y):
                a = np.exp(x)
                D, Dp = y
                # y = [D, D'] where D' = dD/da
                # D'' = (3/2) (Om(a)/a^2) D - (3 + H'/H) D'
                # H'/H = (1/2) (E^2)' / E^2
                e2 = E_sq(a)
                de2_da = dE_sq_da(a)
                if e2 <= 0:
                    return [0, 0]
                h_prime_over_h = 0.5 * de2_da / e2
                om_a = om * np.power(a, -3.0) / e2
                Dpp = 1.5 * om_a / (a * a) * D - (3.0 + h_prime_over_h) * Dp
                return [Dp, Dpp]
            
            # Integrate from x_min = ln(1e-4) to x_max = 0
            x_min = np.log(1e-4)
            x_max = 0.0
            # Initial conditions at x_min: D ~ a, D' ~ 1
            a0 = np.exp(x_min)
            y0 = [a0, 1.0]
            
            sol = solve_ivp(ode, [x_min, x_max], y0, rtol=1e-10, atol=1e-12, dense_output=True)
            
            if not sol.success:
                continue
                
            # Evaluate on a_vals
            D_arr = sol.sol(np.log(a_vals))[0]
            Dp_arr = sol.sol(np.log(a_vals))[1]
            
            # Normalize D such that D(1) = 1
            D_norm_factor = D_arr[-1]
            D_norm = D_arr / D_norm_factor
            # f = d ln D / d ln a = a D' / D
            f_arr = a_vals * Dp_arr / D_arr
            
            ref_data.append((om, w0, wa, D_norm, f_arr))
        except Exception:
            continue

    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    def objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        
        # Clip coefficients to [-10, 10] to satisfy constraint
        c0 = np.clip(c0, -10, 10)
        c1 = np.clip(c1, -10, 10)
        c2 = np.clip(c2, -10, 10)
        c3 = np.clip(c3, -10, 10)
        c4 = np.clip(c4, -10, 10)
        c5 = np.clip(c5, -10, 10)
        
        max_err = 0.0
        
        for om, w0, wa, D_ref, f_ref in ref_data:
            # Compute approximate values
            # q = 1 - Om(a)
            # u = 1 + w0 + wa*(1-a)
            
            # We need Om(a) for each a
            e2 = om * np.power(a_vals, -3.0) + (1.0 - om) * np.power(a_vals, -3.0 * (1.0 + w0 + wa))
            om_a = om * np.power(a_vals, -3.0) / e2
            
            q = 1.0 - om_a
            u = 1.0 + w0 + wa * (1.0 - a_vals)
            
            gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a_vals + c5 * u * q
            
            # Check gamma constraint [0.1, 1.2]
            # If violated, penalty
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e6 # Large penalty
            
            f_app = np.power(om_a, gamma)
            
            # D_app: exp(-integral f dx) from 0 to x
            # x = ln(a)
            # We integrate from x=0 (a=1) backwards to x=a_vals
            # D(a)/D(1) = exp( - ∫_{ln(a)}^0 f(exp(x)) dx )
            # Let's use cumulative trapz from end to start
            x_vals = np.log(a_vals)
            # f_app is defined at a_vals. 
            # We need integral from x_i to 0.
            # np.cumtrapz integrates from start to current.
            # So we compute integral from start to end, then subtract from total.
            # Or just flip.
            # Let's compute integral of f_app dx from x_min to x_max
            # I_i = ∫_{x_min}^{x_i} f dx
            # We want ∫_{x_i}^{x_max} f dx = I_total - I_i
            # Note: x_max = 0.
            
            # Use simple trapezoidal rule
            dx = np.diff(x_vals)
            f_avg = (f_app[:-1] + f_app[1:]) / 2.0
            int_steps = f_avg * dx
            int_cum = np.concatenate(([0.0], np.cumsum(int_steps)))
            total_int = int_cum[-1]
            
            # Integral from x_i to x_max is total_int - int_cum[i]
            # Since x increases, and we want ∫_{x_i}^{0}, and x_i < 0.
            # The integral ∫_{x_i}^{0} f dx is positive.
            # My int_cum is ∫_{x_min}^{x_i}.
            # So ∫_{x_i}^{0} = total_int - int_cum[i].
            
            exponents = -(total_int - int_cum)
            D_app = np.exp(exponents)
            
            # Relative errors
            # f error
            err_f = np.abs(f_app - f_ref) / np.maximum(f_ref, 1e-10)
            # D error
            err_D = np.abs(D_app - D_ref) / np.maximum(D_ref, 1e-10)
            
            local_max_err = np.max(np.concatenate([err_f, err_D]))
            if local_max_err > max_err:
                max_err = local_max_err
                
        return max_err

    # Initial guess: constant gamma 0.55
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    best_coeffs = x0
    best_score = objective(x0)
    
    # Optimization
    try:
        # Use Nelder-Mead for robustness without gradients
        # Or L-BFGS-B with bounds
        bounds = [(-10, 10)] * 6
        
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success and res.fun < best_score:
            best_score = res.fun
            best_coeffs = res.x
        
        # Try Nelder-Mead as well, starting from best found
        res2 = minimize(objective, best_coeffs, method='Nelder-Mead', options={'maxiter': 500, 'fatol': 1e-12, 'xatol': 1e-8})
        if res2.fun < best_score:
            best_score = res2.fun
            best_coeffs = res2.x
            
    except Exception:
        pass
    
    # Final check and clipping
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    return best_coeffs.tolist()
