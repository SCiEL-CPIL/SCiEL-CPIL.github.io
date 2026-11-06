"""Improved growth-index approximation for CPL dark energy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

def construct():
    """
    Return six coefficients for the growth-index formula.
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    where q = 1-Om(a), u = 1+w0+wa*(1-a)
    """
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s limit

    # Representative validation cosmologies (Om0, w0, wa)
    test_models = [
        (0.30, 0.0, 0.0),    # LCDM
        (0.30, -1.0, 0.0),   # Cosmological Constant
        (0.30, -0.5, 0.5),   # CPL 1
        (0.30, -0.8, 0.3),   # CPL 2
        (0.30, -0.6, -0.2),  # CPL 3
        (0.25, -0.7, 0.4),   # Low Om, CPL
        (0.35, -1.2, 0.8),   # High Om, w0 < -1
    ]

    # Grid for a values (log-spaced for better early-time resolution)
    a_grid = np.geomspace(0.01, 1.0, 100)

    # Pre-compute reference values for each model
    references = []
    for Om0, w0, wa in test_models:
        def ode_system(a, y):
            D, Dp = y
            E2 = Om0 * a**-3 + (1-Om0) * np.exp(3*wa*(1-a))
            Om_a = Om0 * a**-3 / E2
            E2_prime = -3*Om0*a**-4 + 3*wa*(1-Om0)*np.exp(3*wa*(1-a))
            HpH = 0.5 * a * E2_prime / E2
            Dpp = -(3 + HpH) * Dp + 1.5 * Om_a * D
            return [Dp, Dpp]

        a_start = 0.01
        y0 = [a_start, 1.0]
        
        sol = solve_ivp(ode_system, [a_start, 1.0], y0, method='RK45', 
                        rtol=1e-8, atol=1e-10, t_eval=a_grid)
        
        if not sol.success:
            continue
            
        D_ref = sol.y[0]
        Dp_ref = sol.y[1]
        
        # Normalize D(1) = 1
        D_ref_norm = D_ref / D_ref[-1]
        Dp_ref_norm = Dp_ref / D_ref[-1]
        f_ref = Dp_ref_norm / D_ref_norm
        
        # Pre-compute q, u, Om_a for this model
        E2 = Om0 * a_grid**-3 + (1-Om0) * np.exp(3*wa*(1-a_grid))
        Om_a = Om0 * a_grid**-3 / E2
        q = 1 - Om_a
        u = 1 + w0 + wa * (1 - a_grid)
        
        references.append({
            'Om0': Om0, 'w0': w0, 'wa': wa,
            'a_grid': a_grid, 'f_ref': f_ref, 'D_ref_norm': D_ref_norm,
            'q': q, 'u': u, 'Om_a': Om_a, 'wa_a': wa * a_grid, 'u_q': u * q
        })

    if not references:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    def compute_error(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        
        for ref in references:
            a_grid = ref['a_grid']
            f_ref = ref['f_ref']
            D_ref_norm = ref['D_ref_norm']
            q = ref['q']
            u = ref['u']
            Om_a = ref['Om_a']
            wa_a = ref['wa_a']
            u_q = ref['u_q']
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa_a + c5*u_q
            
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e6
            
            f_approx = Om_a ** gamma
            
            # Compute D_approx_norm via cumulative trapezoid in log-space
            h_grid = np.log(a_grid)
            J = np.zeros_like(a_grid)
            for i in range(len(a_grid)-2, -1, -1):
                J[i] = J[i+1] + 0.5 * (f_approx[i] + f_approx[i+1]) * (h_grid[i+1] - h_grid[i])
            
            D_approx_norm = np.exp(-J)
            
            err_f = np.abs(f_approx - f_ref) / np.abs(f_ref)
            err_D = np.abs(D_approx_norm - D_ref_norm) / np.abs(D_ref_norm)
            
            max_err = max(max_err, np.max(err_f), np.max(err_D))
        
        return max_err

    # Initial guess (seed)
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_x = x0
    best_score = compute_error(x0)

    # Nelder-Mead optimization
    max_iter = 300
    options = {'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6}
    
    try:
        res = minimize(compute_error, x0, method='Nelder-Mead', options=options)
        if res.fun < best_score:
            best_x = res.x
            best_score = res.fun
    except Exception:
        pass

    # Polish with tighter tolerance if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        try:
            options2 = {'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
            res2 = minimize(compute_error, best_x, method='Nelder-Mead', options=options2)
            if res2.fun < best_score:
                best_x = res2.x
                best_score = res2.fun
        except Exception:
            pass

    # Ensure coefficients are finite and in [-10, 10]
    best_x = np.clip(best_x, -10, 10)
    if not np.all(np.isfinite(best_x)):
        best_x = x0

    return best_x.tolist()
# EVOLVE-BLOCK-END
