import numpy as np
import scipy.optimize as so
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    # 1. Define a set of representative validation cosmologies
    # (Om, w0, wa)
    models = [
        (0.30, -1.0, 0.0),   # LCDM
        (0.30, -0.9, 0.2),   # CPL
        (0.25, -1.1, 0.5),   # CPL
        (0.70, -1.0, 0.0),   # High Om
        (0.10, -1.0, 0.0),   # Low Om
        (0.30, -0.5, -0.5),  # Thawing DE
        (0.30, -1.5, 1.0),   # Phantom
    ]
    
    # 2. Pre-compute reference solutions using ODE integration
    # We integrate D'' + (3 + 2(1 + w0 + wa*(1-a))/a) D' - 1.5 Om(a)/a D = 0 ?
    # Actually, standard form in x = ln a:
    # D'' + (3 + 2 H'/H) D' - 1.5 Om(a) D = 0
    # H'/H = 0.5 Om_m + 0.5 Om_de (1 + 3w)
    # Om_de = 1 - Om_m
    # w = w0 + wa(1-a)
    
    # Let's use a robust ODE solver for the reference.
    # We define the system: y = [D, D']
    # y' = [D', D'']
    
    def compute_reference(Om_m, w0, wa, a_grid):
        """
        Computes reference f and D for a given model on a_grid.
        a_grid is sorted ascending.
        """
        # We integrate from a=1 backwards to a_min.
        # Let x = ln a. x goes from 0 to ln(a_min).
        # It is easier to integrate forward from x_min to 0 if we reverse time, 
        # but standard ODE solvers integrate forward.
        # Let's integrate from x=ln(a_min) to x=0.
        # Boundary conditions at x=0 (a=1):
        # D(0) = 1
        # D'(0) = f(0) = Om_m^gamma_ref. 
        # However, we don't know gamma_ref exactly without an approximation.
        # For the reference, we can use the standard approximation gamma_ref = 0.55 + 0.5*(1-Om_m) for the initial guess of D'(0), 
        # or better, just integrate the ODE with a high-precision constant gamma approximation for the *reference*?
        # No, the reference should be the "true" GR solution.
        # The ODE is linear. We can solve it for two basis solutions or just one with specific ICs.
        # Let's use the IC: D(1)=1, D'(1)=f(1).
        # What is f(1)? In LCDM, f(1) = Om_m. In general, f(1) is close to Om_m^0.55.
        # To be independent of the approximation, we can define the reference D such that D(1)=1.
        # The value of D'(1) is determined by the ODE history.
        # Actually, the ODE determines D up to two constants.
        # We fix D(1)=1. The second constant D'(1) is the growth rate at present.
        # We can compute D'(1) by integrating backwards from a small a where we assume matter domination?
        # Or simpler: Use a high-order numerical integration of the ODE from a=1e-4 to a=1.
        
        a_min = 1e-4
        # Create grid for integration
        x_grid = np.linspace(np.log(a_min), 0, 1000)
        
        def ode_system(x, y):
            a = np.exp(x)
            Om_a = Om_m * a**3 / (Om_m * a**3 + (1 - Om_m) * np.exp(3 * (1 + w0 + wa*(1-a)) * a) / np.exp(3 * (1 + w0 + wa*(1-a)) * a)) # This is wrong for CPL energy density
            
            # Correct CPL density:
            # rho_de = rho_de0 * exp(3 * integral w da'/a')
            # integral w da'/a' from 1 to a = w0(1-a) + wa/2 (1-a^2) ? No.
            # w(a) = w0 + wa(1-a)
            # integral_1^a (w0 + wa(1-a'))/a' da' = w0(1-a) + wa(1 - a ln a - (1-a)) ?
            # Let's use the standard formula:
            # E(z)^2 = Om_m (1+z)^3 + (1-Om_m) exp(3(1+w0)(1-a) - 3wa(1-a^2)/2)
            
            z = 1/a - 1
            E2 = Om_m * (1+z)**3 + (1 - Om_m) * np.exp(3 * (1 + w0) * (1 - a) - 1.5 * wa * (1 - a**2))
            H2 = E2 # assuming H0=1
            
            # Om_a = Om_m (1+z)^3 / E2
            Om_a = Om_m * (1+z)**3 / E2
            
            # H'/H = (d ln H / d ln a) = 0.5 (Om_a + Om_de_a (1+3w))
            # Om_de_a = 1 - Om_a
            w_a_val = w0 + wa * (1 - a)
            H_prime_over_H = 0.5 * (Om_a + (1 - Om_a) * (1 + 3 * w_a_val))
            
            # ODE: D'' + (3 + 2 H'/H) D' - 1.5 Om_a D = 0
            D, D_prime = y
            D_double_prime = - (3 + 2 * H_prime_over_H) * D_prime + 1.5 * Om_a * D
            return [D_prime, D_double_prime]
        
        # We need D(1)=1 and D'(1)=f_ref(1).
        # We don't know f_ref(1) a priori.
        # However, the shape of D(a) is what matters for the ratio D(a)/D(1).
        # We can solve for D with D(1)=1 and D'(1)=1 (arbitrary scale), then normalize.
        # The relative error in D is scale invariant.
        # The relative error in f depends on D'.
        # f = D'/D.
        # So we need the *correct* D'(1).
        # In the limit a->0, D ~ a^gamma.
        # This is complicated.
        
        # Alternative: Use the fact that for the *reference*, we can just use a very accurate numerical integration 
        # and assume the "true" f(1) is what comes out of the integration if we match the asymptotic behavior?
        # Actually, the standard "reference" in these tasks is often the numerical solution of the ODE 
        # with ICs D(1)=1, D'(1)=f(1) where f(1) is calculated from a high-precision gamma?
        # No, the ODE is linear. If we pick D'(1)=1, we get a solution D_num.
        # The true solution is C * D_num.
        # D(1)=1 implies C=1/D_num(1)=1. So D_num(1) must be 1.
        # So we set D(1)=1. What about D'(1)?
        # The ODE doesn't fix D'(1) without boundary conditions at early times.
        # However, for the purpose of minimizing error against a *fixed* reference, 
        # the reference must be well-defined.
        # Usually, the reference is generated by integrating the ODE from a=1e-4 to a=1 
        # with matter domination initial conditions D ~ a^1.15 or similar, 
        # then rescaling so D(1)=1.
        
        # Let's do that:
        # 1. Integrate from x_min to 0 with D(x_min)=1, D'(x_min)=1.15 (approx).
        # 2. Get D(0) and D'(0).
        # 3. Normalize: D_norm(x) = D(x) / D(0).
        # 4. f_norm(x) = D'_norm(x) / D_norm(x) = D'(x) / D(x).
        
        sol = so.integrate.solve_ivp(ode_system, [x_grid[0], x_grid[-1]], [1.0, 1.15], method='RK45', rtol=1e-9, atol=1e-12)
        if not sol.success:
            return None, None
            
        D_ref_raw = sol.y[0]
        Dp_ref_raw = sol.y[1]
        
        # Interpolate to a_grid
        # x_grid corresponds to a_grid_int = exp(x_grid)
        a_int = np.exp(x_grid)
        
        D_ref = np.interp(a_grid, a_int, D_ref_raw)
        Dp_ref = np.interp(a_grid, a_int, Dp_ref_raw)
        
        # Normalize D so D(1)=1
        # Find index closest to a=1
        idx_1 = np.argmin(np.abs(a_grid - 1.0))
        D_ref = D_ref / D_ref[idx_1]
        
        f_ref = Dp_ref / D_ref
        
        return f_ref, D_ref

    # 3. Pre-compute references for all models
    a_eval = np.logspace(-4, 0, 200) # 0.0001 to 1.0
    refs = []
    for m in models:
        f_ref, D_ref = compute_reference(m[0], m[1], m[2], a_eval)
        if f_ref is not None:
            refs.append((m, f_ref, D_ref))
            
    if not refs:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # 4. Define the objective function
    def objective(c):
        c0, c1, c2, c3, c4, c5 = c
        
        max_err = 0.0
        penalty = 0.0
        
        for m, f_ref, D_ref in refs:
            Om_m, w0, wa = m
            
            # Compute q, u, gamma
            # q = 1 - Om(a)
            # Need Om(a) for each a in a_eval
            z = 1/a_eval - 1
            E2 = Om_m * (1+z)**3 + (1 - Om_m) * np.exp(3 * (1 + w0) * (1 - a_eval) - 1.5 * wa * (1 - a_eval**2))
            Om_a = Om_m * (1+z)**3 / E2
            
            q = 1 - Om_a
            u = 1 + w0 + wa * (1 - a_eval)
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_eval + c5*u*q
            
            # Constraint: gamma in [0.1, 1.2]
            # Soft penalty
            gamma_violation = np.maximum(0, 0.1 - gamma) + np.maximum(0, gamma - 1.2)
            penalty += np.sum(gamma_violation**2)
            
            # Compute f_app
            f_app = Om_a ** gamma
            
            # Compute D_app
            # D(a)/D(1) = exp(-int_ln(a)^0 f(exp(x)) dx)
            # We have f_app on a_grid. We need integral of f_app(x) dx where x=ln a.
            # dx = da/a.
            # Integral from a to 1 of f(a') da'/a'
            # Let's compute cumulative integral from 1 down to a.
            # Let x = ln a. 
            # We can use np.cumtrapz on x_grid.
            x_eval = np.log(a_eval)
            # f_app is defined on a_eval. 
            # Integral I(a) = int_a^1 f(a') da'/a'
            # Let's compute J(x) = int_x^0 f(e^t) dt
            # J(x) = - int_0^x f(e^t) dt
            # We have points (x_i, f_i).
            # Use cumulative trapezoidal integration from the end (x=0) backwards.
            # Or just integrate from x_min to x, then subtract from total.
            
            # Simpler: Compute integral of f_app / a da from a_min to a_max.
            # int = np.cumtrapz(f_app / a_eval, a_eval)
            # This gives int from a_min to a_i.
            # We want int from a_i to 1.
            # Total int = int from a_min to 1.
            # int(a_i to 1) = Total - int(a_min to a_i)
            
            integrand = f_app / a_eval
            cum_int = np.cumtrapz(integrand, a_eval, initial=0.0)
            total_int = cum_int[-1]
            integral_a_to_1 = total_int - cum_int
            
            D_app = np.exp(-integral_a_to_1)
            
            # Relative errors
            err_f = np.abs(f_app - f_ref) / np.abs(f_ref)
            err_D = np.abs(D_app - D_ref) / np.abs(D_ref)
            
            # Smooth max
            k = 50.0
            max_err_f = np.logaddexp.reduce(k * err_f) / k
            max_err_D = np.logaddexp.reduce(k * err_D) / k
            max_err = max(max_err_f, max_err_D)
            
        return max_err + 10.0 * penalty

    # 5. Optimization
    best_c = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_score = objective(best_c)
    
    starts = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.6, -0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],
    ]
    
    bounds = [(-10, 10)] * 6
    
    for start in starts:
        if time.time() > deadline:
            break
            
        try:
            res = so.minimize(objective, start, method='L-BFGS-B', bounds=bounds, 
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_c = res.x
        except Exception:
            pass
            
    # Final check
    if time.time() > deadline - 5:
         return [float(x) for x in best_c]
         
    return [float(x) for x in best_c]

# EVOLVE-BLOCK-END
