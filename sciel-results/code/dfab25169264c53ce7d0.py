import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize
import time

def _Om_of_a(a, Om0, w0, wa):
    """Compute matter density parameter at scale factor a for CPL dark energy."""
    Omd0 = 1.0 - Om0
    if a <= 0:
        return np.nan
    ln_a = np.log(a)
    exponent = 3.0 * ((w0 + wa) * ln_a + wa * (1.0 - a))
    Omd_ratio = np.exp(exponent)
    Om_matter = Om0 * a**(-3)
    Om_de = Omd0 * Omd_ratio
    total = Om_matter + Om_de
    if total == 0:
        return 0.0
    return Om_matter / total

def _compute_reference(Om0, w0, wa, a_values):
    """Compute reference f and D/D(1) using ODE integration."""
    Omd0 = 1.0 - Om0
    
    def H2(a):
        ln_a = np.log(a)
        exponent = 3.0 * ((w0 + wa) * ln_a + wa * (1.0 - a))
        Omd_ratio = np.exp(exponent)
        Om_matter = Om0 * a**(-3)
        Om_de = Omd0 * Omd_ratio
        return Om_matter + Om_de
    
    def dH2_dlna(a):
        ln_a = np.log(a)
        exponent = 3.0 * ((w0 + wa) * ln_a + wa * (1.0 - a))
        Omd_ratio = np.exp(exponent)
        Om_matter = Om0 * a**(-3)
        Om_de = Omd0 * Omd_ratio
        
        dOm_matter_dlna = -3.0 * Om_matter
        dOm_de_dlna = Om_de * (3.0 * (w0 + wa) - 3.0 * wa * (1.0 - a))
        
        return dOm_matter_dlna + dOm_de_dlna
    
    def H_prime_over_H(a):
        H2_val = H2(a)
        if H2_val == 0:
            return 0.0
        return 0.5 * dH2_dlna(a) / H2_val
    
    def ode_rhs(x, state):
        D, f = state
        a = np.exp(x)
        if a <= 0:
            return [0, 0]
        Om_a = _Om_of_a(a, Om0, w0, wa)
        Hph = H_prime_over_H(a)
        Dpp = -(3.0 + Hph) * f + 1.5 * Om_a * D
        return [f, Dpp]
    
    # Integrate from x_start to x=0
    x_start = np.log(0.01)
    x_end = 0.0
    
    a_start = np.exp(x_start)
    Om_start = _Om_of_a(a_start, Om0, w0, wa)
    f_start = Om_start ** 0.55
    
    sol = solve_ivp(ode_rhs, [x_start, x_end], [1.0, f_start], 
                    dense_output=True, rtol=1e-10, atol=1e-12, max_step=0.05)
    
    if not sol.success:
        return None
    
    D_at_1, f_at_1 = sol.sol(0.0)
    D_norm = D_at_1
    
    f_vals = np.zeros(len(a_values))
    D_norm_vals = np.zeros(len(a_values))
    for i, a in enumerate(a_values):
        x = np.log(a)
        if x < x_start or x > x_end:
            f_vals[i] = np.nan
            D_norm_vals[i] = np.nan
            continue
        D_i, f_i = sol.sol(x)
        f_vals[i] = f_i
        D_norm_vals[i] = D_i / D_norm
    
    return f_vals, D_norm_vals

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    test_cosmos = [
        (0.3, -1.0, 0.0),
        (0.3, -0.9, 0.1),
        (0.3, -1.1, 0.2),
        (0.3, -0.8, -0.1),
        (0.25, -1.0, 0.0),
        (0.35, -1.0, 0.0),
        (0.3, -1.0, 0.5),
        (0.3, -1.0, -0.5),
        (0.3, -0.5, 0.0),
        (0.3, -1.5, 0.0),
    ]
    
    a_values = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    # Pre-compute references
    references = []
    for (Om0, w0, wa) in test_cosmos:
        ref = _compute_reference(Om0, w0, wa, a_values)
        if ref is not None:
            references.append((Om0, w0, wa, ref[0], ref[1]))
    
    if time.time() > deadline:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def error_func(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        
        for (Om0, w0, wa, f_ref, D_ref) in references:
            for i, a in enumerate(a_values):
                if np.isnan(f_ref[i]) or np.isnan(D_ref[i]):
                    continue
                
                Om_a = _Om_of_a(a, Om0, w0, wa)
                if Om_a <= 0 or Om_a >= 1:
                    continue
                
                q = 1.0 - Om_a
                u = 1.0 + w0 + wa * (1.0 - a)
                
                gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
                
                if gamma < 0.1 or gamma > 1.2:
                    max_err = max(max_err, 1.0)
                    continue
                
                f_approx = Om_a ** gamma
                
                if abs(f_ref[i]) > 1e-10:
                    err_f = abs(f_approx - f_ref[i]) / abs(f_ref[i])
                    max_err = max(max_err, err_f)
                
                # Compute D_approx
                def integrand(x):
                    a_x = np.exp(x)
                    Om_x = _Om_of_a(a_x, Om0, w0, wa)
                    if Om_x <= 0 or Om_x >= 1:
                        return 0.0
                    q_x = 1.0 - Om_x
                    u_x = 1.0 + w0 + wa * (1.0 - a_x)
                    gamma_x = c0 + c1 * q_x + c2 * q_x**2 + c3 * u_x + c4 * wa * a_x + c5 * u_x * q_x
                    if gamma_x < 0.1 or gamma_x > 1.2:
                        return 0.0
                    return Om_x ** gamma_x
                
                try:
                    integral_val, _ = quad(integrand, np.log(a), 0.0, limit=100)
                    D_approx = np.exp(-integral_val)
                    
                    if abs(D_ref[i]) > 1e-10:
                        err_D = abs(D_approx - D_ref[i]) / abs(D_ref[i])
                        max_err = max(max_err, err_D)
                except:
                    pass
        
        return max_err
    
    # Optimize
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_coeffs = x0.copy()
    best_err = error_func(best_coeffs)
    
    try:
        result = minimize(error_func, x0, method='L-BFGS-B', 
                         bounds=[(-10, 10)] * 6,
                         options={'maxiter': 50, 'maxfun': 100})
        if result.fun < best_err:
            best_err = result.fun
            best_coeffs = result.x
        
        if time.time() > deadline:
            return np.clip(best_coeffs, -10, 10).tolist()
        
        # More starting points
        starting_points = [
            np.array([0.55, 0.1, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.5, 0.1, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.55, 0.0, 0.05, 0.0, 0.0, 0.0]),
            np.array([0.55, 0.0, 0.0, 0.05, 0.0, 0.0]),
            np.array([0.55, 0.0, 0.0, 0.0, 0.05, 0.0]),
            np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.05]),
        ]
        
        for sp in starting_points:
            if time.time() > deadline:
                break
            try:
                result2 = minimize(error_func, sp, method='L-BFGS-B',
                                  bounds=[(-10, 10)] * 6,
                                  options={'maxiter': 30, 'maxfun': 60})
                if result2.fun < best_err:
                    best_err = result2.fun
                    best_coeffs = result2.x
            except:
                pass
    except:
        pass
    
    best_coeffs = np.clip(best_coeffs, -10, 10)
    return best_coeffs.tolist()
