import numpy as np
from scipy.integrate import solve_ivp, minimize, cumulative_trapezoid
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Grid definition
    om_m0_vals = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    w0_vals = np.array([-1.5, -1.0, -0.5, 0.0])
    wa_vals = np.array([-0.5, 0.0, 0.5])
    a_vals = np.geomspace(0.01, 1.0, 40)
    
    # Flatten the parameter grid
    om_m0_flat, w0_flat, wa_flat = np.meshgrid(om_m0_vals, w0_vals, wa_vals, indexing='ij')
    om_m0_flat = om_m0_flat.ravel()
    w0_flat = w0_flat.ravel()
    wa_flat = wa_flat.ravel()
    
    n_cosmo = len(om_m0_flat)
    n_a = len(a_vals)
    
    # Pre-compute reference data
    f_refs = np.zeros((n_cosmo, n_a))
    D_refs = np.zeros((n_cosmo, n_a))
    
    for i in range(n_cosmo):
        om_m0 = om_m0_flat[i]
        w0 = w0_flat[i]
        wa = wa_flat[i]
        om_de0 = 1.0 - om_m0
        
        def E2(a):
            term_m = om_m0 * a**(-3)
            term_de = om_de0 * a**(-3*(1+w0+wa)) * np.exp(-3*wa*(1-a))
            return term_m + term_de
            
        def E2_prime(a):
            term_m = om_m0 * a**(-3)
            term_de = om_de0 * a**(-3*(1+w0+wa)) * np.exp(-3*wa*(1-a))
            d_term_m = -3 * term_m / a
            d_term_de = term_de * (-3*(1+w0+wa)/a - 3*wa)
            return d_term_m + d_term_de

        def ode(t, y):
            a = t
            d = y[0]
            dp = y[1]
            e2 = E2(a)
            e2_p = E2_prime(a)
            h_prime_over_h = e2_p / (2 * e2)
            om_m_a = om_m0 * a**(-3) / e2
            dpp = -(3 + h_prime_over_h) * dp + 1.5 * om_m_a * d
            return [dp, dpp]
            
        a_start = 0.01
        a_end = 1.0
        y0 = [a_start, 1.0]
        
        try:
            sol = solve_ivp(ode, [a_start, a_end], y0, method='RK45', rtol=1e-8, atol=1e-10, dense_output=False)
            if not sol.success:
                continue
                
            D_1 = sol.y[0, -1]
            D_norm = sol.y[0] / D_1
            D_prime_norm = sol.y[1] / D_1
            
            # Interpolate to a_vals? 
            # solve_ivp returns points on the solution grid. We need values at a_vals.
            # Since a_vals are within [0.01, 1.0], and we used dense_output=False, 
            # we should use dense_output=True or interpolate.
            # Let's re-run with dense_output=True for safety.
        except Exception:
            continue
            
    # Re-run with dense_output for accuracy
    for i in range(n_cosmo):
        om_m0 = om_m0_flat[i]
        w0 = w0_flat[i]
        wa = wa_flat[i]
        om_de0 = 1.0 - om_m0
        
        def E2(a):
            term_m = om_m0 * a**(-3)
            term_de = om_de0 * a**(-3*(1+w0+wa)) * np.exp(-3*wa*(1-a))
            return term_m + term_de
            
        def E2_prime(a):
            term_m = om_m0 * a**(-3)
            term_de = om_de0 * a**(-3*(1+w0+wa)) * np.exp(-3*wa*(1-a))
            d_term_m = -3 * term_m / a
            d_term_de = term_de * (-3*(1+w0+wa)/a - 3*wa)
            return d_term_m + d_term_de

        def ode(t, y):
            a = t
            d = y[0]
            dp = y[1]
            e2 = E2(a)
            e2_p = E2_prime(a)
            h_prime_over_h = e2_p / (2 * e2)
            om_m_a = om_m0 * a**(-3) / e2
            dpp = -(3 + h_prime_over_h) * dp + 1.5 * om_m_a * d
            return [dp, dpp]
            
        a_start = 0.01
        a_end = 1.0
        y0 = [a_start, 1.0]
        
        try:
            sol = solve_ivp(ode, [a_start, a_end], y0, method='RK45', rtol=1e-8, atol=1e-10, dense_output=True)
            if not sol.success:
                continue
                
            D_sol = sol.sol(a_vals)[0]
            Dp_sol = sol.sol(a_vals)[1]
            
            D_1 = D_sol[-1]
            D_norm = D_sol / D_1
            D_prime_norm = Dp_sol / D_1
            
            f_refs[i] = D_prime_norm / D_norm
            D_refs[i] = D_norm
            
        except Exception:
            continue
            
    if time.time() > deadline:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    def objective(c):
        max_err = 0.0
        if np.any(c < -10) or np.any(c > 10):
            return 100.0
            
        for i in range(n_cosmo):
            om_m0 = om_m0_flat[i]
            w0 = w0_flat[i]
            wa = wa_flat[i]
            om_de0 = 1.0 - om_m0
            
            term_m = om_m0 * a_vals**(-3)
            term_de = om_de0 * a_vals**(-3*(1+w0+wa)) * np.exp(-3*wa*(1-a_vals))
            e2 = term_m + term_de
            om_m_a = term_m / e2
            
            q = 1 - om_m_a
            u = 1 + w0 + wa * (1 - a_vals)
            
            gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a_vals + c[5]*u*q
            
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 100.0 + np.max(np.abs(gamma - np.clip(gamma, 0.1, 1.2)))
                
            f_app = om_m_a ** gamma
            
            try:
                cum_int = cumulative_trapezoid(f_app, a_vals, initial=0)
            except Exception:
                cum_int = np.zeros_like(a_vals)
                for j in range(1, n_a):
                    cum_int[j] = cum_int[j-1] + 0.5 * (f_app[j] + f_app[j-1]) * (a_vals[j] - a_vals[j-1])
                    
            int_a_to_1 = cum_int[-1] - cum_int
            D_app = np.exp(-int_a_to_1)
            
            err_f = np.abs(f_app - f_refs[i]) / f_refs[i]
            err_D = np.abs(D_app - D_refs[i]) / D_refs[i]
            
            max_err = max(max_err, np.max(err_f), np.max(err_D))
            
            if time.time() > deadline:
                return max_err * 100.0
                
        return max_err * 100.0

    c0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_c = c0
    best_score = objective(c0)
    
    if time.time() < deadline - 5:
        res = minimize(objective, c0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_score:
            best_score = res.fun
            best_c = res.x
            
    if time.time() < deadline - 10:
        c1 = [0.5, 0.1, -0.1, 0.0, 0.0, 0.0]
        res2 = minimize(objective, c1, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
        if res2.fun < best_score:
            best_score = res2.fun
            best_c = res2.x
            
    return best_c
