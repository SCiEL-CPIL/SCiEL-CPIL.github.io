import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

def _compute_reference(Om0, w0, wa, a_vals):
    """
    Computes reference f and D for given cosmology and a values.
    Uses the ODE:
    dD/dx = D'
    D'' + (3 + 2 w_eff) D' + 1.5 * Om(a) D = 0
    where x = ln(a)
    """
    # We integrate from x_start = ln(a_min) to x_end = ln(a_max)
    # Initial conditions at a=1 (x=0): D(1)=1, D'(1)=f(1)
    # f(1) is close to 1 for matter dominated at high z, but at a=1,
    # f(1) can be computed or approximated. For LambdaCDM, f(1) ~ 0.6-0.8.
    # A good initial guess for f(1) is Om0^0.55 or solving the ODE backwards?
    # Better: Integrate from a small a (high z) where D ~ a to a=1?
    # Or integrate from a=1 downwards?
    # Let's integrate from a=1 to a_min.
    # We need f(1). For a flat universe, f(1) is not 1.
    # We can find f(1) by solving the ODE from a high redshift where f~1.
    # Let's start at a_start = 0.01, D(a_start) = a_start (linear growth in matter era), f(a_start) = 1.
    # Then integrate to a=1 to get D(1) and f(1).
    # Then normalize D(1)=1 and use f(1) as initial condition for the main range.
    
    a_start = 0.01
    a_end = 1.0
    x_start = np.log(a_start)
    x_end = 0.0
    
    def ode_func(x, y):
        a = np.exp(x)
        # Background evolution
        # Om(a) = Om0 * a^3 / (Om0 * a^3 + (1-Om0) * (1-a)^(-3*wa))
        # Wait, the DE factor is (1-a)^(-3(1+w0+wa)) is incorrect for general wa?
        # rho_de(a) = rho_de(1) * (1-a)^(-3(1+w0+wa))
        # E(a)^2 = Om0/a^3 + (1-Om0)*(1-a)^(-3(1+w0+wa))
        
        de_term = (1-a)**(-3*(1+w0+wa))
        E2 = Om0/(a**3) + (1-Om0)*de_term
        Om_a = (Om0/(a**3)) / E2
        de_a = 1 - Om_a
        
        w_a_val = w0 + wa*(1-a)
        w_eff = w_a_val * (de_a / 1.0) # Om_tot is 1 in flat universe
        
        D = y[0]
        Dp = y[1]
        
        Dpp = -(3 + 2*w_eff)*Dp - 1.5*Om_a*D
        
        return [Dp, Dpp]
    
    # Integrate from a_start to a=1
    try:
        sol = solve_ivp(ode_func, [x_start, x_end], [a_start, 1.0], 
                        method='RK45', rtol=1e-9, atol=1e-12, max_step=0.01)
        if not sol.success:
            return None, None
            
        D_1 = sol.y[0, -1]
        Dp_1 = sol.y[1, -1]
        f_1 = Dp_1 / D_1
        
        # Now integrate from a=1 to a_min using normalized D(1)=1, D'(1)=f_1
        # Actually, we just need f and D/D(1) for the given a_vals.
        # We can use the solution from the first pass if a_vals are in [a_start, 1].
        # But a_vals in the task are likely 0.05 to 0.99.
        # We can just evaluate the ODE at the required points.
        
        # To be precise, let's integrate again from a=1 to min(a_vals) if needed, 
        # or just use the dense output if available.
        # solve_ivp with dense_output=True is better.
        sol2 = solve_ivp(ode_func, [x_end, np.log(0.05)], [1.0, f_1], 
                         method='RK45', rtol=1e-9, atol=1e-12, max_step=0.01, dense_output=True)
        if not sol2.success:
            return None, None
            
        x_vals = np.log(a_vals)
        # Note: sol2 integrates from x_end=0 to negative x.
        # The direction is backwards, which is fine for RK45.
        # But the initial condition is at x=0.
        # We need to handle the case where a_vals > 1? No, a <= 1.
        
        Y = sol2.sol(x_vals)
        D_vals = Y[0]
        Dp_vals = Y[1]
        f_vals = Dp_vals / D_vals
        
        # Normalize D to D(1)=1. Since we started with D(1)=1, this is already done.
        return f_vals, D_vals
        
    except Exception:
        return None, None

def _growth_index_error(coeffs):
    c0, c1, c2, c3, c4, c5 = coeffs
    
    test_models = [
        (0.3, -1.0, 0.0),
        (0.3, -0.9, -0.3),
        (0.3, -1.2, 0.5),
        (0.3, -0.8, 0.0),
        (0.2, -1.0, 0.0),
        (0.4, -1.0, 0.0),
        (0.3, -1.0, -1.0),
        (0.3, -1.0, 1.0),
        (0.3, -0.5, 0.0),
        (0.3, -1.5, 0.5),
    ]
    
    a_values = np.linspace(0.05, 0.99, 40)
    
    max_rel_err = 0.0
    
    for Om0, w0, wa in test_models:
        f_ref, D_ref = _compute_reference(Om0, w0, wa, a_values)
        if f_ref is None:
            continue
            
        for i, a in enumerate(a_values):
            # Compute Om(a)
            de_term = (1-a)**(-3*(1+w0+wa))
            E2 = Om0/(a**3) + (1-Om0)*de_term
            Om_a = (Om0/(a**3)) / E2
            
            q = 1 - Om_a
            u = 1 + w0 + wa*(1-a)
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
            
            if gamma < 0.1 or gamma > 1.2:
                return 1000.0
                
            f_approx = Om_a**gamma
            
            rel_err_f = abs(f_approx - f_ref[i]) / max(abs(f_ref[i]), 1e-10)
            max_rel_err = max(max_rel_err, rel_err_f)
            
            # D error
            # D_approx(a)/D_approx(1) = exp(-int_a^1 f dx)
            # We need to compute this integral.
            # To avoid expensive integration per error call, we can approximate D_approx
            # using the same gamma? No, D is integral of f.
            # However, the score is max error of f AND D.
            # Computing D_approx requires numerical integration of the approximate f.
            # This is slow.
            # Alternative: Use the fact that D_approx is close to a^gamma_eff?
            # Or just pre-compute the integral for the current coeffs?
            # Since coeffs change, we must re-integrate.
            # This is too slow for 1000+ optimization steps.
            
            # Strategy: Only check f error in the fast loop?
            # But the task says "worst relative error across both observables".
            # We must include D.
            
            # Optimization: 
            # We can compute D_approx by integrating f_approx from a to 1.
            # f_approx = Om(x)^gamma(x).
            # We can do this with a simple trapezoidal rule on the grid a_values.
            # Since we have a grid, we can precompute the integral for the whole grid
            # for a given set of coeffs.
            
            pass # This loop structure is inefficient for D.

    # Better Approach for Error Function:
    # 1. Iterate models.
    # 2. For each model, get f_ref, D_ref.
    # 3. Compute f_approx for all a_values.
    # 4. Compute D_approx by integrating f_approx.
    # 5. Update max error.
    
    # Let's rewrite the inner loop to be vectorized where possible.
    
    max_rel_err = 0.0
    for Om0, w0, wa in test_models:
        f_ref, D_ref = _compute_reference(Om0, w0, wa, a_values)
        if f_ref is None: continue
        
        # Compute Om(a) for all a
        a = a_values
        de_term = (1-a)**(-3*(1+w0+wa))
        E2 = Om0/(a**3) + (1-Om0)*de_term
        Om_a = (Om0/(a**3)) / E2
        
        q = 1 - Om_a
        u = 1 + w0 + wa*(1-a)
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
        
        # Check bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1000.0
            
        f_approx = Om_a**gamma
        
        # D_approx
        # Integrate f_approx from a to 1.
        # x = ln(a). dx = da/a.
        # Integral = int_{ln(a)}^{0} f_approx(x) dx.
        # We have values at discrete a_i.
        # Use np.trapz on x grid.
        x = np.log(a)
        # f_approx is defined at x.
        # We need integral from x_i to 0.
        # Since x is decreasing (0.05 to 0.99 -> -3 to -0.01), 
        # and we integrate to 0, we can compute cumulative integral.
        
        # Sort x ascending for integration?
        # x is already ascending (ln 0.05 < ln 0.99).
        # We want int_{x_i}^0 f dx.
        # Total integral int_{x_min}^0 f dx = I_total.
        # int_{x_i}^0 f dx = I_total - int_{x_min}^{x_i} f dx.
        
        # Calculate cumulative integral from x_min to x_i
        # trapz requires y and x.
        # cum_trapz is not in standard np, but we can do:
        # dy = f_approx * dx.
        # cum_int = np.concatenate(([0], np.cumsum(0.5*(f_approx[1:]+f_approx[:-1]) * np.diff(x))))
        
        dx = np.diff(x)
        trapz_steps = 0.5 * (f_approx[1:] + f_approx[:-1]) * dx
        cum_int = np.concatenate(([0.0], np.cumsum(trapz_steps)))
        
        # Integral from x_i to x_min is cum_int[i].
        # Integral from x_i to 0 is (Integral from x_min to 0) - cum_int[i].
        # Integral from x_min to 0 is cum_int[-1] + 0.5*(f_approx[-1]+f_approx_at_0)*dx_last?
        # Actually, our grid ends at 0.99, not 1.0.
        # We can append a=1.0, f(1)=Om(1)^gamma(1) to the grid for integration.
        
        a_ext = np.append(a, 1.0)
        x_ext = np.log(a_ext)
        Om_1 = Om0
        q_1 = 1 - Om0
        u_1 = 1 + w0
        gamma_1 = c0 + c1*q_1 + c2*q_1**2 + c3*u_1 + c4*wa*1.0 + c5*u_1*q_1
        f_1 = Om_1**gamma_1
        f_approx_ext = np.append(f_approx, f_1)
        
        dx_ext = np.diff(x_ext)
        trapz_ext = 0.5 * (f_approx_ext[1:] + f_approx_ext[:-1]) * dx_ext
        cum_int_ext = np.concatenate(([0.0], np.cumsum(trapz_ext)))
        
        total_int = cum_int_ext[-1]
        # D_approx(a_i) / D(1) = exp(- (total_int - cum_int_ext[i]))
        # Note: cum_int_ext[i] is integral from a_min to a_i.
        # Integral from a_i to 1 is total_int - cum_int_ext[i].
        
        D_approx = np.exp(-(total_int - cum_int_ext[:-1]))
        
        # Errors
        rel_err_f = np.abs(f_approx - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        rel_err_D = np.abs(D_approx - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        
        max_rel_err = max(max_rel_err, np.max(rel_err_f), np.max(rel_err_D))
        
    return max_rel_err * 100

def construct():
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_x = x0
    best_err = _growth_index_error(x0)
    
    # Nelder-Mead
    for i in range(50):
        if time.time() > deadline:
            break
            
        res = minimize(_growth_index_error, best_x, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if res.fun < best_err:
            best_err = res.fun
            best_x = res.x
        else:
            # If no improvement, try a random restart?
            # Or break if stuck?
            if i > 5:
                break

    best_x = np.clip(best_x, -10, 10)
    return list(best_x)
