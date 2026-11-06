"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin

    # Representative cosmologies for calibration
    # (Om0, w0, wa)
    cosmologies = [
        (0.30, -1.0, 0.0),   # LambdaCDM
        (0.30, -1.0, 0.5),   # CPL wa>0
        (0.30, -1.0, -0.5),  # CPL wa<0
        (0.25, -1.0, 0.0),   # Lower Om
        (0.35, -1.0, 0.0),   # Higher Om
        (0.30, -0.9, 0.0),   # w0 > -1
        (0.30, -1.1, 0.0),   # w0 < -1
    ]
    
    # Dense grid for a
    a_grid = np.linspace(0.1, 1.0, 100)
    
    # Pre-compute reference solutions
    # We store f_ref and D_ref for each cosmology and a point
    # D_ref is normalized such that D(1)=1
    ref_data = []
    
    for Om0, w0, wa in cosmologies:
        if time.time() > deadline:
            break
            
        Om_DE0 = 1.0 - Om0
        
        def Om_of_a(a):
            # Vectorized Om(a)
            w_a = w0 + wa * (1.0 - a)
            Om_DE = Om_DE0 * np.power(a, -3.0 * (1.0 + w_a))
            Om_m = Om0 * np.power(a, 3.0)
            return Om_m / (Om_m + Om_DE)
        
        def H_prime_over_H(a):
            # H'/H = (1/2) d ln H^2 / d ln a
            # H^2 = H0^2 (Om_m a^-3 + Om_DE a^(-3(1+w)))
            # For CPL, this is complex. We use numerical differentiation on ln a.
            # Let x = ln a. d/dx = a d/da.
            # H^2(x) = Om0 e^{-3x} + Om_DE0 e^{-3(1+w(x))x} where w(x) = w0 + wa(1-e^x)
            # This is messy. Let's just use a central difference on a.
            da = 1e-6
            a_plus = np.minimum(a + da, 1.0)
            a_minus = np.maximum(a - da, 0.01)
            
            def H2(a_val):
                w_val = w0 + wa * (1.0 - a_val)
                Om_DE_val = Om_DE0 * np.power(a_val, -3.0 * (1.0 + w_val))
                Om_m_val = Om0 * np.power(a_val, 3.0)
                # H^2/H0^2 = Om_m/a^3 + Om_DE/a^3(1+w) ... wait
                # Standard form: H^2 = H0^2 [ Om_m a^-3 + Om_k a^-2 + Om_DE a^{-3(1+w)} ]
                # Assuming flat: Om_k=0.
                return Om0 * np.power(a_val, -3.0) + Om_DE_val * np.power(a_val, -3.0 * (1.0 + w_val))
            
            H2_plus = H2(a_plus)
            H2_minus = H2(a_minus)
            
            # d ln H / d ln a = (a/2) d ln H^2 / da
            # d ln H^2 / da = (H2_plus - H2_minus) / (a_plus - a_minus) / H2
            dlnH2_da = (H2_plus - H2_minus) / (a_plus - a_minus) / ((H2_plus + H2_minus) / 2.0)
            return 0.5 * a * dlnH2_da

        # Solve ODE: D'' + (3 + H'/H) D' - (3/2) Om D = 0
        # Let y = [D, D']
        # y0' = y1
        # y1' = -(3 + H'/H) y1 + 1.5 Om y0
        
        def ode_func(x, y):
            a_val = np.exp(x)
            HpH_val = H_prime_over_H(a_val)
            Om_val = Om_of_a(a_val)
            return [y[1], -(3.0 + HpH_val) * y[1] + 1.5 * Om_val * y[0]]
        
        # Initial conditions at a=1 (x=0)
        # D(1) = 1
        # D'(1) = f(1) = Om(1)^gamma(1). Om(1)=1, so f(1)=1.
        y0 = [1.0, 1.0]
        
        try:
            sol = solve_ivp(
                ode_func, [0.0, np.log(0.1)], y0, method='RK45', 
                rtol=1e-8, atol=1e-10, dense_output=True
            )
            if not sol.success:
                continue
                
            # Evaluate at grid points
            x_grid = np.log(a_grid)
            y_grid = sol.sol(x_grid)
            D_ref = y_grid[0]
            f_ref = y_grid[1]
            
            # Store
            ref_data.append((a_grid, f_ref, D_ref, Om0, w0, wa))
            
        except Exception:
            continue

    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Define the error function for optimization
    def compute_error(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        max_rel_err = 0.0
        
        for a_grid, f_ref, D_ref, Om0, w0, wa in ref_data:
            # Compute Om(a)
            w_a = w0 + wa * (1.0 - a_grid)
            Om_DE = (1.0 - Om0) * np.power(a_grid, -3.0 * (1.0 + w_a))
            Om_m = Om0 * np.power(a_grid, 3.0)
            Om = Om_m / (Om_m + Om_DE)
            
            # Compute ansatz variables
            q = 1.0 - Om
            u = 1.0 + w0 + wa * (1.0 - a_grid)
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_grid + c5*u*q
            
            # Check bounds
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e6 # Large penalty
            
            f_approx = np.power(Om, gamma)
            
            # Compute D_approx via integration of f
            # D(a)/D(1) = exp( integral_1^a f d ln a ) = exp( integral_0^{ln a} f d x )
            # We have f on a_grid. x_grid = ln(a_grid).
            # Use cumulative trapezoidal integration
            x_grid = np.log(a_grid)
            # f is a function of x. We have f at x_grid.
            # Integrate from 0 to x_grid. Note x_grid is negative (0.1 to 1.0 -> -2.3 to 0)
            # We need integral from 0 to x.
            # Let's reverse the grid to integrate from 0 downwards or just use cumtrapz and adjust.
            # Easier: integrate from x_grid[0] to x_grid[i] and add constant.
            # D(1) = 1 => integral from 0 to 0 is 0.
            # We need I(x) = int_0^x f(t) dt.
            # Since x < 0, I(x) = - int_x^0 f(t) dt.
            
            # Let's compute cumulative integral from the end (a=1, x=0) backwards.
            # f_rev = f_approx[::-1]
            # x_rev = x_grid[::-1]
            # dx = np.diff(x_rev)
            # int_rev = np.concatenate(([0], np.cumsum(0.5*(f_rev[:-1]+f_rev[1:])*dx)))
            # This gives integral from x_rev[0] (which is x_grid[-1]=0) to x_rev[i].
            # So int_rev[i] = int_{0}^{x_rev[i]} f dt.
            # We want I(x) = int_0^x f dt.
            
            f_rev = f_approx[::-1]
            x_rev = x_grid[::-1]
            dx = np.diff(x_rev)
            # trapezoidal rule
            int_rev = np.zeros_like(x_rev)
            int_rev[1:] = np.cumsum(0.5 * (f_rev[:-1] + f_rev[1:]) * dx)
            # int_rev[i] is integral from x_rev[0] (0) to x_rev[i].
            # So D_approx_rev[i] = exp(int_rev[i])
            D_approx_rev = np.exp(int_rev)
            D_approx = D_approx_rev[::-1]
            
            # Relative errors
            rel_err_f = np.abs(f_approx - f_ref) / np.abs(f_ref)
            rel_err_D = np.abs(D_approx - D_ref) / np.abs(D_ref)
            
            # Mask out very small values to avoid division by zero issues
            mask_f = np.abs(f_ref) > 1e-8
            mask_D = np.abs(D_ref) > 1e-8
            
            if np.any(mask_f):
                max_rel_err = max(max_rel_err, np.max(rel_err_f[mask_f]))
            if np.any(mask_D):
                max_rel_err = max(max_rel_err, np.max(rel_err_D[mask_D]))
                
        return max_rel_err

    # Optimization
    # Initial guess: constant gamma 0.55
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    best_x = x0
    best_err = compute_error(x0)
    
    # Use Nelder-Mead for robustness
    try:
        res = minimize(
            compute_error, x0, method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True}
        )
        
        if res.success or res.fun < best_err:
            best_x = res.x
            best_err = res.fun
            
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        return [float(c) for c in best_x]
    
    # Refine with L-BFGS-B if time permits
    try:
        res2 = minimize(
            compute_error, best_x, method='L-BFGS-B',
            bounds=[(0.1, 1.0), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)],
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        if res2.fun < best_err:
            best_x = res2.x
            best_err = res2.fun
    except Exception:
        pass

    # Final check on bounds for coefficients [-10, 10]
    final_coeffs = [float(np.clip(c, -10, 10)) for c in best_x]
    
    return final_coeffs

# EVOLVE-BLOCK-END
