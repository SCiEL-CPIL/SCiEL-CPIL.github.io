import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, cumulative_trapezoid
import time

def construct():
    start_time = time.time()
    deadline = 80.0
    
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Grid
    w0_vals = np.linspace(-1.0, 0.5, 5)
    wa_vals = np.linspace(-0.5, 1.0, 5)
    Om0_vals = np.linspace(0.25, 0.35, 3)
    a_vals = np.linspace(0.1, 1.0, 10)
    
    grid_w0 = []
    grid_wa = []
    grid_Om0 = []
    for w0 in w0_vals:
        for wa in wa_vals:
            for Om0 in Om0_vals:
                grid_w0.append(w0)
                grid_wa.append(wa)
                grid_Om0.append(Om0)
                
    grid_w0 = np.array(grid_w0)
    grid_wa = np.array(grid_wa)
    grid_Om0 = np.array(grid_Om0)
    
    true_f_cache = {}
    true_D_cache = {}
    
    def get_true_values(w0, wa, Om0, a_list):
        key = (w0, wa, Om0)
        if key in true_f_cache:
            return true_f_cache[key], true_D_cache[key]
        
        def ode_rhs(lna, y):
            f = y[0]
            a = np.exp(lna)
            w = w0 + wa * (1 - a)
            Ode0 = 1.0 - Om0
            term_de = Ode0 * (1 + w0) * np.exp(wa * (1 - a))
            term_m = Om0 * a**-3
            Om_a = term_m / (term_m + term_de)
            
            df_dlna = f**2 - 1.5 * (1 + w) * Om_a - f * (1 - 1.5 * (1 + w) * Om_a)
            return [df_dlna]
        
        f_init = Om0**0.55
        lna_init = 0.0
        lna_vals = np.log(a_list)
        
        sol = solve_ivp(ode_rhs, (0.0, np.min(lna_vals)), [f_init], t_eval=lna_vals, rtol=1e-8, atol=1e-10, method='RK45')
        
        if not sol.success:
            f_true = np.array([Om0**0.55] * len(a_list))
            D_true = np.ones(len(a_list))
            true_f_cache[key] = f_true
            true_D_cache[key] = D_true
            return f_true, D_true
        
        f_true = sol.y[0]
        
        idx = np.argsort(lna_vals)
        lna_sorted = lna_vals[idx]
        f_sorted = f_true[idx]
        
        integral = cumulative_trapezoid(f_sorted, lna_sorted, initial=0)
        D_vals = np.exp(-(integral[-1] - integral))
        
        D_true = np.empty_like(D_vals)
        D_true[idx] = D_vals
        
        true_f_cache[key] = f_true
        true_D_cache[key] = D_true
        return f_true, D_true

    def objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        
        if np.any(coeffs < -10) or np.any(coeffs > 10):
            return 1e6
        
        max_err = 0.0
        for i in range(len(grid_w0)):
            w0 = grid_w0[i]
            wa = grid_wa[i]
            Om0 = grid_Om0[i]
            
            f_true, D_true = get_true_values(w0, wa, Om0, a_vals)
            
            a = a_vals
            Ode0 = 1.0 - Om0
            term_de = Ode0 * (1 + w0) * np.exp(wa * (1 - a))
            term_m = Om0 * a**-3
            Om_a = term_m / (term_m + term_de)
            
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a)
            
            gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
            
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e6 + (np.max(np.abs(gamma - 0.1)) + np.max(np.abs(gamma - 1.2)))
            
            f_approx = Om_a ** gamma
            
            idx = np.argsort(np.log(a))
            lna_sorted = np.log(a)[idx]
            f_sorted = f_approx[idx]
            
            integral = cumulative_trapezoid(f_sorted, lna_sorted, initial=0)
            D_approx_sorted = np.exp(-(integral[-1] - integral))
            
            D_approx = np.empty_like(D_approx_sorted)
            D_approx[idx] = D_approx_sorted
            
            err_f = np.abs(f_approx - f_true) / np.abs(f_true)
            err_D = np.abs(D_approx - D_true) / np.abs(D_true)
            
            max_err = max(max_err, np.max(err_f), np.max(err_D))
            
            if time.time() - start_time > deadline:
                break
                
        return max_err

    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
    
    if result.success and result.fun < objective(x0):
        best_coeffs = result.x
    else:
        best_coeffs = x0
        
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    return list(best_coeffs)
