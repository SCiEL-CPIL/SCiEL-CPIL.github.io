"""Compact dark-energy growth approximation: six-coefficient gamma model."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# EVOLVE-BLOCK-START
def construct():
    # Internal calibration of the 6 coefficients using representative models
    # This approach uses numerical optimization to fit the ansatz to high-fidelity
    # ODE solutions for a set of cosmologies, rather than using hard-coded constants.
    
    # Representative cosmologies to cover the likely validation space
    # (Omega_m0, w0, wa)
    models = [
        (0.3, -1.0, 0.0),   # LambdaCDM
        (0.3, -1.0, 0.5),   # CPL
        (0.3, -0.5, -1.0),  # CPL
        (0.3, -1.5, 1.0),   # CPL
        (0.25, -1.0, 0.0),  # Low Omega_m
        (0.35, -1.0, 0.0),  # High Omega_m
    ]
    
    a_min = 0.1
    a_max = 1.0
    n_points = 50
    
    def compute_reference(om_m0, w0, wa):
        """Solve the ODE for f(a) and D(a) for a given cosmology."""
        # Define H(a) and Om(a)
        def get_om_m(a):
            # Flat universe: Om_m(a) = Om_m0 * a^-3 / (Om_m0 * a^-3 + Om_DE(a))
            # Om_DE(a) = (1 - Om_m0) * (a0/a)^(3(1+w0+wa)) ? No, CPL is a0=1.
            # For CPL: Om_DE(a) = (1 - Om_m0) * a^(-3(1+w0+wa))
            # Wait, standard CPL definition: p = w0 + wa(1-a).
            # rho_DE(a) = rho_DE0 * a^(-3(1+w0+wa)) is only for constant w.
            # For CPL, the density evolution is:
            # rho_DE(a) = rho_DE0 * exp(-3 * integral_1^a (1+w(a'))/a' da')
            # This is complex. Let's use the standard numerical integration for Om_m(a).
            
            # Actually, for the ODE, we need Om_m(a) and w(a) at each step.
            # It is easier to define the system in terms of a.
            pass
        
        # Let's implement the ODE solver carefully.
        # Variables: y = [ln(D), f] or just D and f?
        # dD/da = D * f / a
        # df/da = (3/2 * Om_m - f * (1 + Hdot/H) - f^2) / a
        
        # We need Om_m(a) and Hdot/H(a).
        # In a flat universe:
        # Om_m(a) = Om_m0 * a^-3 / (Om_m0 * a^-3 + Om_DE(a))
        # Om_DE(a) = (1 - Om_m0) * (a/a0)^(-3(1+w0+wa)) is WRONG for CPL.
        # Correct CPL density:
        # rho_DE(a) = rho_DE0 * exp( -3 * ∫_1^a (1 + w0 + wa(1-a'))/a' da' )
        # ∫ (1+w0)/a' da' + ∫ wa(1-a')/a' da' = (1+w0) ln a + wa ∫ (1/a' - 1) da'
        # = (1+w0) ln a + wa (ln a - (a - 1))
        # = (1+w0+wa) ln a - wa(a - 1)
        # So rho_DE(a) ∝ a^(1+w0+wa) * exp(-wa(a-1))
        
        def om_m_of_a(a):
            # a is scalar or array
            if np.isscalar(a):
                a_arr = np.array([a])
            else:
                a_arr = a
            exp_term = np.exp(-wa * (a_arr - 1))
            a_term = a_arr ** (1 + w0 + wa)
            om_de = (1 - om_m0) * a_term * exp_term
            om_m = om_m0 * a_arr ** -3
            denom = om_m + om_de
            return om_m / denom
        
        def hdot_h_of_a(a):
            # 1 + Hdot/H = 1 - 0.5*Om_m - 1.5*Om_DE*(1+w)
            # w = w0 + wa*(1-a)
            om_m = om_m_of_a(a)
            om_de = 1 - om_m
            w = w0 + wa * (1 - a)
            return 1 - 0.5 * om_m - 1.5 * om_de * (1 + w)
        
        def ode_system(a, y):
            d = y[0]
            f = y[1]
            om_m = om_m_of_a(a)
            hdot_h = hdot_h_of_a(a)
            
            # dD/da = D * f / a
            d_d = d * f / a
            # df/da = (1.5*Om_m - f*(1+Hdot/H) - f^2) / a
            d_f = (1.5 * om_m - f * (1 + hdot_h) - f**2) / a
            
            return [d_d, d_f]
        
        # Initial conditions at a=1 (present)
        # f(1) is not exactly 0.55, it's the current growth rate.
        # For LambdaCDM, f(1) ~ 0.5.
        # We can shoot from a=1 to a=a_min.
        # D(1) = 1. f(1) needs to be determined.
        # Actually, the ODE is singular at a=0 but stable at a=1.
        # Let's integrate from a=1 to a=a_min.
        
        # We need a good initial guess for f(1).
        # f(1) = Om_m0^gamma. Gamma ~ 0.55.
        f_init = om_m0 ** 0.55
        
        # Integrate backwards? Or forwards from a_min to 1?
        # The ODE is stiff near a=0. Integrating from 1 to a_min is safer.
        # But D(a) grows as a increases.
        # Let's integrate from a=1 to a=a_min.
        
        # solve_ivp expects t_span = [t0, tf]
        # We want a from 1 to a_min.
        a_grid = np.linspace(1, a_min, n_points)
        
        # We need to handle the array evaluation in ode_system.
        # scipy solve_ivp handles scalar a in the function call.
        
        sol = solve_ivp(ode_system, [1, a_min], [1.0, f_init], 
                        t_eval=a_grid, method='DOP853', rtol=1e-8, atol=1e-10)
        
        if not sol.success:
            return None, None
            
        D_ref = sol.y[0]
        f_ref = sol.y[1]
        
        # Normalize D such that D(1)=1. The solution starts at 1, so it is already normalized.
        # However, the direction is 1 -> a_min.
        # The evaluator computes D(a)/D(1).
        # Our D_ref[0] is D(1)=1. D_ref[-1] is D(a_min).
        # We need to return D and f on a grid from a_min to 1 for comparison?
        # The error metric is max relative error.
        
        return a_grid, D_ref, f_ref

    def error_function(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        
        for om_m0, w0, wa in models:
            a_grid, D_ref, f_ref = compute_reference(om_m0, w0, wa)
            if D_ref is None:
                return 1e6 # Penalty for failure
            
            # Compute ansatz f and D
            # a is the grid from 1 to a_min
            a = a_grid
            
            # Compute q and u
            # Need Om_m(a) for the ansatz too?
            # The ansatz uses q = 1 - Om(a).
            # We need Om_m(a) for the ansatz as well.
            # We can reuse the om_m_of_a logic, but it's nested in compute_reference.
            # Let's refactor to make om_m_of_a accessible or recompute.
            
            # Recalculate Om_m for the ansatz
            exp_term = np.exp(-wa * (a - 1))
            a_term = a ** (1 + w0 + wa)
            om_de = (1 - om_m0) * a_term * exp_term
            om_m_ansatz = om_m0 * a ** -3
            om_m_total = om_m_ansatz + om_de
            om_m = om_m_ansatz / om_m_total
            
            q = 1 - om_m
            u = 1 + w0 + wa * (1 - a)
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
            
            # Check gamma bounds
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e6 + np.max(np.abs(gamma) - 1.2) # Penalty
            
            f_ansatz = om_m ** gamma
            
            # Error in f
            err_f = np.max(np.abs(f_ansatz - f_ref) / np.abs(f_ref))
            
            # Error in D
            # The evaluator computes D(a)/D(1) = exp(-int f dx)
            # We have D_ref from ODE.
            # D_ansatz should be normalized to 1 at a=1.
            # The ODE solution D_ref is normalized to 1 at a=1.
            # So we can compare D_ansatz and D_ref directly.
            # But we need to compute D_ansatz from f_ansatz.
            # D_ansatz(a) = exp(-int_1^a f_ansatz(a')/a' da') ?
            # dD/da = D f / a => ln D = int f/a da.
            # D(a)/D(1) = exp( int_1^a f(a')/a' da' ).
            # Since we are going from 1 to a_min, the integral is negative.
            
            # Calculate integral of f_ansatz / a
            # Using trapezoidal rule on the grid
            integrand = f_ansatz / a
            # a_grid is decreasing.
            # np.trapz(integrand, a_grid)
            integral = np.trapz(integrand, a_grid)
            
            # D_ansatz at each point?
            # We need cumulative integral.
            # Let's compute D_ansatz at each point.
            D_ansatz = np.ones_like(a_grid)
            D_ansatz[0] = 1.0
            # Integrate from 1 to a_i
            # Since a_grid is sorted 1 -> a_min, we can accumulate.
            # dx = a[i] - a[i-1]
            # ln D[i] = ln D[i-1] + f[i-1]/a[i-1] * (a[i]-a[i-1])
            
            for i in range(1, len(a_grid)):
                da = a_grid[i] - a_grid[i-1]
                # Simple Euler or Trapezoid?
                # Trapezoid is better.
                avg_f_over_a = 0.5 * (f_ansatz[i-1]/a_grid[i-1] + f_ansatz[i]/a_grid[i])
                D_ansatz[i] = D_ansatz[i-1] * np.exp(avg_f_over_a * da)
                
            # Compare D_ansatz and D_ref
            err_D = np.max(np.abs(D_ansatz - D_ref) / np.abs(D_ref))
            
            local_max_err = max(err_f, err_D)
            if local_max_err > max_err:
                max_err = local_max_err
                
        return max_err

    # Initial guess
    x0 = [0.55, -0.05, 0.0, -0.02, 0.0, 0.0]
    
    # Optimize
    # Use Nelder-Mead as it doesn't require gradients and is robust for small N
    res = minimize(error_function, x0, method='Nelder-Mead', 
                   options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
    
    if res.success:
        return res.x
    else:
        return x0

# EVOLVE-BLOCK-END
