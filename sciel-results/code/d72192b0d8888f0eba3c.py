import numpy as np
import scipy.optimize
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin for 90s limit

    # 1. Define Training Cosmologies
    # We need to cover the likely validation space.
    # Om_m0 in [0.2, 0.4], w0 in [-1.5, -0.5], wa in [-1.0, 1.0]
    # Use a representative grid.
    om_m0_vals = [0.25, 0.30, 0.35]
    w0_vals = [-1.2, -1.0, -0.8]
    wa_vals = [-0.5, 0.0, 0.5]
    
    cosmologies = []
    for om in om_m0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                cosmologies.append((om, w0, wa))

    # 2. High-resolution grid for a values
    # Log-spaced from 0.05 to 1.0. 200 points is sufficient for accurate integration.
    a_grid = np.logspace(np.log10(0.05), 0.0, 200)
    x_grid = np.log(a_grid)
    dx = x_grid[1] - x_grid[0] # Constant step in log space

    # 3. Fast Reference Generation using Finite Differences
    # This is much faster than solve_ivp and vectorized.
    ref_data = []
    
    for (om_m0, w0, wa) in cosmologies:
        om_l0 = 1.0 - om_m0
        
        # Calculate Om_m(a) and w(a) on the grid
        a = a_grid
        de_term = om_l0 * (a ** (-3.0 * (1.0 + w0))) * np.exp(-3.0 * wa * (a - 1.0))
        m_term = om_m0 * (a ** -3.0)
        om_m = m_term / (m_term + de_term)
        w_a = w0 + wa * (1.0 - a)
        
        # Initial conditions at a=1 (index -1)
        # D(1) = 1.0
        # D'(1) = f(1) = Om_m0^0.55 (Standard seed)
        D = np.zeros_like(a_grid)
        Dp = np.zeros_like(a_grid)
        D[-1] = 1.0
        Dp[-1] = om_m0 ** 0.55
        
        # Integrate backwards from a=1 to a_min using 2nd order finite differences
        # D'' = - (1.5 + 1.5 w_a) D' + 1.5 om_m D
        # We step from i to i-1.
        # Let h = dx.
        # D_{i-1} = 2D_i - D_{i+1} - h^2 D''_i (approx, but we need to be careful with signs)
        # Actually, let's use the standard recurrence for D''(x) = F(x, D, D')
        # D(x-h) = 2D(x) - D(x+h) - h^2 F(x, D, D')
        # Here x is decreasing as we go from end to start? 
        # a_grid is increasing (0.05 -> 1.0). x_grid is increasing (-ln20 -> 0).
        # We start at x=0 (index -1) and go to x<0 (index 0).
        # So we are stepping in negative x direction.
        # Let x_k = x_grid[k]. We know D_k, D_{k+1}. We want D_{k-1}.
        # D_{k-1} = 2D_k - D_{k+1} - dx^2 * D''_k
        # D'_{k-1} = 2D'_k - D'_{k+1} - dx^2 * D'''_k? No, simpler to use central difference for D' update or just recompute D' from D.
        # Recomputing D' from D is safer for stability in simple FD.
        # D'_k ~ (D_{k+1} - D_{k-1}) / (2dx) -> This requires D_{k-1} which we are solving for.
        # Alternative: Use 4th order Runge-Kutta vectorized? No, FD is fine if dx is small enough.
        # Let's use a simple 2nd order FD for D and calculate D' via central difference after the pass, 
        # or use the relation D'_{k-1} = D'_k - dx * D''_k (1st order) which might be unstable.
        
        # Better: Use the standard "shooting" with higher precision? 
        # Actually, for 200 points, dx ~ 0.03. 2nd order FD error is O(dx^2) ~ 1e-3. 
        # This might be too noisy for "worst error" minimization if the reference itself is noisy.
        # Let's use 4th order Runge-Kutta but vectorized? No, RK is sequential.
        
        # Let's stick to solve_ivp but with a very tight budget and fallback.
        # Wait, the prompt says "tightly integrated general-relativistic second-order ODE provides an independent numerical reference".
        # The current program uses solve_ivp. It scored 1.0. 
        # The "inspiration" mentions 0.6 -> 1.24 using Linder approximation.
        # The key is that the *approximation* (gamma formula) is what we are fitting.
        # If the reference is noisy, we fit the noise.
        # Let's use a more accurate reference: 4th order Runge-Kutta is standard.
        # To speed it up, we can reduce the number of cosmologies or grid points if needed, 
        # but 9*200 points is small. The bottleneck in the current code is likely the 
        # Python loop inside `objective` for calculating errors and the integration.
        
        # Let's re-implement the reference generation with a fast, accurate method.
        # We will use scipy.integrate.solve_ivp but with relaxed tolerances to speed it up, 
        # as we only need the shape of the error, not 1e-9 precision.
        # rtol=1e-6, atol=1e-8 should be sufficient for finding the best coefficients.
        
        def ode_system(x, y):
            a_val = np.exp(x)
            # Recalculate om_m and w at this specific x for the ODE solver
            # This is slow. We should pre-compute on a fine grid and interpolate?
            # Or just accept the cost. 9 cosmologies * 1 integration = 9 calls.
            # Each call takes ~0.1s? 0.9s total. Acceptable.
            
            # Calculate om_m(a_val)
            de_t = om_l0 * (a_val ** (-3.0 * (1.0 + w0))) * np.exp(-3.0 * wa * (a_val - 1.0))
            m_t = om_m0 * (a_val ** -3.0)
            om_val = m_t / (m_t + de_t)
            w_val = w0 + wa * (1.0 - a_val)
            
            dD = y[1]
            dDp = -1.5 * (1.0 + w_val) * y[1] + 1.5 * om_val * y[0]
            return [dD, dDp]
        
        x_min = x_grid[0]
        f_seed = om_m0 ** 0.55
        try:
            sol = scipy.integrate.solve_ivp(ode_system, [0, x_min], [1.0, f_seed], 
                                            method='RK45', rtol=1e-6, atol=1e-8,
                                            dense_output=True)
            if sol.success:
                sol_grid = sol.sol(x_grid)
                D_ref = sol_grid[0]
                Dp_ref = sol_grid[1]
                f_ref = Dp_ref / D_ref
            else:
                # Fallback to analytic approx if fail
                gamma_const = 0.55
                f_ref = om_m ** gamma_const
                # Integrate f_ref to get D_ref
                # D = exp(-int f dx)
                # Use cumulative sum on x_grid (reversed)
                f_rev = f_ref[::-1]
                dx_val = x_grid[1] - x_grid[0]
                int_rev = np.cumsum(f_rev * dx_val)
                int_rev = int_rev[::-1]
                D_ref = np.exp(-int_rev)
                D_ref /= D_ref[-1]
        except:
            gamma_const = 0.55
            f_ref = om_m ** gamma_const
            f_rev = f_ref[::-1]
            dx_val = x_grid[1] - x_grid[0]
            int_rev = np.cumsum(f_rev * dx_val)
            int_rev = int_rev[::-1]
            D_ref = np.exp(-int_rev)
            D_ref /= D_ref[-1]
            
        ref_data.append((a_grid, D_ref, f_ref, om_m0, w0, wa))
        
    # Check time
    if time.time() > deadline - 20.0:
        # Not enough time for optimization, return seed
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # 4. Optimization Strategy
    # We want to minimize the MAX relative error.
    # The max function is non-differentiable. 
    # However, `scipy.optimize.minimize` with method='Nelder-Mead' or 'Powell' can handle non-smooth functions.
    # Alternatively, we can use a soft-max approximation for the objective to make it smooth,
    # then polish with a direct max search.
    
    # Let's use a hybrid:
    # 1. Minimize sum of squared errors (smooth) to get close.
    # 2. Minimize max error (non-smooth) using Nelder-Mead from that point.
    
    def compute_errors(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        sum_sq_err = 0.0
        n_points = 0
        
        for (a_grid, D_ref, f_ref, om_m0, w0, wa) in ref_data:
            # Calculate om_m on grid
            de_term = (1.0 - om_m0) * (a_grid ** (-3.0 * (1.0 + w0))) * np.exp(-3.0 * wa * (a_grid - 1.0))
            m_term = om_m0 * (a_grid ** -3.0)
            om_m_grid = m_term / (m_term + de_term)
            
            q_grid = 1.0 - om_m_grid
            u_grid = 1.0 + w0 + wa * (1.0 - a_grid)
            
            gamma_grid = c0 + c1*q_grid + c2*q_grid**2 + c3*u_grid + c4*wa*a_grid + c5*u_grid*q_grid
            
            # Check bounds
            if np.any(gamma_grid < 0.1) or np.any(gamma_grid > 1.2):
                return 1e10, 1e10
            
            f_approx = om_m_grid ** gamma_grid
            
            # Calculate D_approx
            # D(a)/D(1) = exp(- \int_0^{\ln a} f(e^x) dx)
            # x_grid is increasing. Integral from 0 to x_i.
            # Since x_grid[0] < 0 and x_grid[-1] = 0.
            # We integrate backwards from 0.
            f_rev = f_approx[::-1]
            # Trapezoidal rule cumulative integral
            # dx is constant
            dx_val = x_grid[1] - x_grid[0]
            int_rev = np.zeros_like(f_rev)
            # int_rev[0] = 0
            # int_rev[i] = int_rev[i-1] + 0.5*(f[i-1]+f[i])*dx
            # Vectorized cumulative trapezoid
            # np.cumsum(0.5 * (f_rev[:-1] + f_rev[1:]) * dx_val)
            trapezoids = 0.5 * (f_rev[:-1] + f_rev[1:]) * dx_val
            int_rev[1:] = np.cumsum(trapezoids)
            int_rev = int_rev[::-1]
            
            D_approx = np.exp(-int_rev)
            
            # Errors
            err_f = np.abs((f_approx - f_ref) / f_ref)
            err_D = np.abs((D_approx - D_ref) / D_ref)
            
            err_vec = np.concatenate([err_f, err_D])
            max_err_c = np.max(err_vec)
            if max_err_c > max_err:
                max_err = max_err_c
            
            sum_sq_err += np.sum(err_vec**2)
            n_points += len(err_vec)
            
        return max_err, sum_sq_err / n_points

    # Objective for smooth optimization (sum of squares)
    def obj_smooth(coeffs):
        _, mse = compute_errors(coeffs)
        return mse

    # Objective for max error optimization
    def obj_max(coeffs):
        max_err, _ = compute_errors(coeffs)
        return max_err

    # Initial guess
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_coeffs = x0
    best_max_err, _ = compute_errors(x0)
    
    # Phase 1: Smooth optimization
    if time.time() < deadline - 15.0:
        try:
            res1 = scipy.optimize.minimize(obj_smooth, x0, method='L-BFGS-B', 
                                           bounds=[(-10, 10)]*6, 
                                           options={'maxiter': 100, 'ftol': 1e-10})
            if res1.fun < obj_smooth(x0):
                best_coeffs = res1.x
                best_max_err, _ = compute_errors(best_coeffs)
        except Exception:
            pass

    # Phase 2: Max error optimization (Nelder-Mead)
    # This directly targets the metric we care about.
    if time.time() < deadline - 5.0:
        try:
            # Use a slightly larger initial simplex for NM to escape local minima
            res2 = scipy.optimize.minimize(obj_max, best_coeffs, method='Nelder-Mead',
                                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res2.fun < best_max_err:
                best_coeffs = res2.x
                best_max_err, _ = compute_errors(best_coeffs)
        except Exception:
            pass

    # Final validation
    c = np.clip(best_coeffs, -10, 10)
    
    # Check gamma constraints on a dense grid for all cosmologies
    valid = True
    test_a = np.logspace(np.log10(0.01), 0.0, 500)
    for (om_m0, w0, wa) in cosmologies:
        de_term = (1.0 - om_m0) * (test_a ** (-3.0 * (1.0 + w0))) * np.exp(-3.0 * wa * (test_a - 1.0))
        m_term = om_m0 * (test_a ** -3.0)
        om_m_grid = m_term / (m_term + de_term)
        q_grid = 1.0 - om_m_grid
        u_grid = 1.0 + w0 + wa * (1.0 - test_a)
        gamma_grid = c[0] + c[1]*q_grid + c[2]*q_grid**2 + c[3]*u_grid + c[4]*wa*test_a + c[5]*u_grid*q_grid
        if np.any(gamma_grid < 0.1) or np.any(gamma_grid > 1.2):
            valid = False
            break
            
    if not valid:
        # Revert to safe constant
        c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    return c.tolist()
# EVOLVE-BLOCK-END
