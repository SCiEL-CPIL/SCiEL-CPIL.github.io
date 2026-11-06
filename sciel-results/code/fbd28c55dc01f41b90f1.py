import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin for 90s limit
    
    # Define the ansatz functions
    def get_gamma(c, a, Om, w0, wa):
        q = 1.0 - Om
        u = 1.0 + w0 + wa * (1.0 - a)
        gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
        return gamma

    def f_approx(c, a, Om, w0, wa):
        gamma = get_gamma(c, a, Om, w0, wa)
        # Ensure Om is positive for power
        if Om <= 0:
            return 0.0
        return Om ** gamma

    def D_approx(c, a, Om, w0, wa):
        # D(a)/D(1) = exp(-int_ln(a)^0 f(exp(x)) dx)
        # Let x = ln(a'), a' = exp(x). dx = da'/a'
        # Integral from a to 1 of f(a')/a' da'
        # We integrate from x=ln(a) to x=0
        if a >= 1.0:
            return 1.0
        x0 = np.log(a)
        x1 = 0.0
        
        # Use numerical integration
        def integrand(x):
            a_prime = np.exp(x)
            Om_prime = Om / (Om + (1.0 - Om) * a_prime**(-3.0 * (1.0 + w0 + wa)))
            # Wait, Om(a) formula for CPL:
            # E(a)^2 = Om a^-3 + (1-Om) a^-3(1+w0+wa)
            # Om(a) = Om a^-3 / E(a)^2
            # Let's define Om(a) properly
            return 0.0 # placeholder
        
        # Better to define Om(a) explicitly
        def Om_of_a(a_val, Om0, w0, wa):
            E2 = Om0 * a_val**(-3.0) + (1.0 - Om0) * a_val**(-3.0 * (1.0 + w0 + wa))
            if E2 <= 0:
                return 0.0
            return Om0 * a_val**(-3.0) / E2

        def integrand(x):
            a_prime = np.exp(x)
            Om_prime = Om_of_a(a_prime, Om, w0, wa)
            if Om_prime <= 0:
                return 0.0
            gamma_prime = get_gamma(c, a_prime, Om_prime, w0, wa)
            return Om_prime ** gamma_prime

        # Integrate from x0 to 0
        # solve_ivp is overkill for 1D integral, use quad
        from scipy.integrate import quad
        val, _ = quad(integrand, x0, x1)
        return np.exp(-val)

    # Reference calculation
    def calc_reference(Om0, w0, wa, a_grid):
        """
        Solve ODE for D(a) and compute f_ref and D_ref normalized.
        D'' + (2 + H'/H) D' - 1.5 Om(a) D = 0
        H'/H = -1.5 (1 + w_eff(a))
        w_eff = w0 + wa(1-a)
        So H'/H = -1.5 (1 + w0 + wa(1-a))
        """
        def ode_system(a, y):
            D = y[0]
            Dp = y[1]
            w_eff = w0 + wa * (1.0 - a)
            # H'/H = -1.5 * (1 + w_eff)
            term = 2.0 - 1.5 * (1.0 + w_eff)
            Om_a = Om_of_a(a, Om0, w0, wa)
            Dpp = 1.5 * Om_a * D - term * Dp
            return [Dp, Dpp]

        # Integrate from a_min to 1
        a_min = 0.05
        # Initial conditions at a_min
        # In matter dominated era, D ~ a^1.5
        # D(a_min) = a_min^1.5
        # D'(a_min) = 1.5 * a_min^0.5
        a_start = a_min
        D_start = a_start ** 1.5
        Dp_start = 1.5 * a_start ** 0.5
        y0 = [D_start, Dp_start]
        
        # Solve forward
        sol = solve_ivp(ode_system, [a_start, 1.0], y0, t_eval=a_grid, method='RK45', rtol=1e-9, atol=1e-12)
        
        if not sol.success:
            return None, None

        D_vals = sol.y[0]
        Dp_vals = sol.y[1]
        
        # Normalize D so D(1) = 1
        D_norm = D_vals / D_vals[-1]
        
        # f_ref = D'/D
        # Avoid division by zero
        f_ref = np.zeros_like(D_vals)
        mask = np.abs(D_vals) > 1e-10
        f_ref[mask] = Dp_vals[mask] / D_vals[mask]
        
        return f_ref, D_norm

    # Objective function
    def objective(coeffs):
        c = coeffs
        max_err = 0.0
        gamma_min = 1.0
        gamma_max = 0.0
        
        # Check gamma bounds
        # Use a representative set of a values for gamma check
        a_check = np.linspace(0.1, 1.0, 20)
        for Om0, w0, wa in validation_set:
            for a in a_check:
                Om_a = Om_of_a(a, Om0, w0, wa)
                g = get_gamma(c, a, Om_a, w0, wa)
                if g < gamma_min: gamma_min = g
                if g > gamma_max: gamma_max = g
                if g < 0.1 or g > 1.2:
                    return 1000.0 # Penalty

        # Evaluate error on grid
        a_grid = np.linspace(0.05, 1.0, 50)
        
        for Om0, w0, wa in validation_set:
            f_ref, D_ref = calc_reference(Om0, w0, wa, a_grid)
            if f_ref is None:
                continue
            
            for i, a in enumerate(a_grid):
                Om_a = Om_of_a(a, Om0, w0, wa)
                f_app = f_approx(c, a, Om_a, w0, wa)
                D_app = D_approx(c, a, Om_a, w0, wa)
                
                # Relative errors
                if abs(f_ref[i]) > 1e-6:
                    err_f = abs(f_app - f_ref[i]) / abs(f_ref[i])
                else:
                    err_f = abs(f_app - f_ref[i])
                
                if abs(D_ref[i]) > 1e-6:
                    err_D = abs(D_app - D_ref[i]) / abs(D_ref[i])
                else:
                    err_D = abs(D_app - D_ref[i])
                    
                max_err = max(max_err, err_f, err_D)
                
        return 100.0 * max_err

    # Validation set
    validation_set = [
        (0.3, -1.0, 0.0),      # LCDM
        (0.27, -1.0, 0.0),
        (0.3, -1.0, 0.1),      # w_a > 0
        (0.3, -1.0, -0.1),     # w_a < 0
        (0.2, -1.1, 0.0),      # Low Om, w0 < -1
        (0.4, -0.9, 0.0),      # High Om, w0 > -1
        (0.3, -0.9, 0.2),      # Mixed
        (0.3, -1.1, -0.2),     # Mixed
        (0.25, -1.05, 0.15),
        (0.35, -0.95, -0.15),
    ]

    # Initial guess
    c0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    best_c = c0.copy()
    best_score = objective(best_c)
    
    # Optimization
    # Use Nelder-Mead
    try:
        res = minimize(objective, best_c, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False})
        if res.fun < best_score:
            best_c = res.x
            best_score = res.fun
    except Exception as e:
        print(f"Optimization error: {e}", file=sys.stderr)
        
    # Check time
    if time.time() - start_time > timeout:
        return best_c

    # Try Powell as well if time permits
    if time.time() - start_time < timeout - 10:
        try:
            res2 = minimize(objective, best_c, method='Powell', 
                            options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-10, 'disp': False})
            if res2.fun < best_score:
                best_c = res2.x
                best_score = res2.fun
        except Exception as e:
            print(f"Powell error: {e}", file=sys.stderr)

    # Final check on gamma bounds for best_c
    # If invalid, fall back to seed
    valid = True
    a_check = np.linspace(0.05, 1.0, 50)
    for Om0, w0, wa in validation_set:
        for a in a_check:
            Om_a = Om_of_a(a, Om0, w0, wa)
            g = get_gamma(best_c, a, Om_a, w0, wa)
            if g < 0.1 or g > 1.2:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    return best_c

# EVOLVE-BLOCK-END
