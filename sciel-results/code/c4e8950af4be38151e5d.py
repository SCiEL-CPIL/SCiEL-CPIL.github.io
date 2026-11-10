"""Growth-index formula optimizer for CPL dark-energy cosmologies."""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize, least_squares
import time

_START_TIME = time.time()
_DEADLINE = 85.0

def _Om_of_a(a, Om0, w0, wa):
    """Compute Omega_matter(a) for CPL dark energy."""
    if a == 1.0:
        return Om0
    
    def integrand_de(x):
        return (1.0 + w0 + wa * (1.0 - x)) / x
    
    if a < 1.0:
        integral_de, _ = quad(integrand_de, a, 1.0, limit=200)
    else:
        integral_de, _ = quad(integrand_de, 1.0, a, limit=200)
    
    ratio_de = np.exp(-3.0 * integral_de)
    ratio_m = a ** (-3.0)
    
    rho_m_a = Om0 * ratio_m
    rho_de_a = (1.0 - Om0) * ratio_de
    
    return rho_m_a / (rho_m_a + rho_de_a)

def _compute_gamma_grid(Om0, w0, wa, a_values):
    """Solve the ODE once for a cosmology and return f(a) at each a."""
    n = len(a_values)
    f_vals = np.zeros(n)
    
    for i, a in enumerate(a_values):
        if a == 1.0:
            Om_a = Om0
            # At a=1, f = Om0 * (1 + Om0/2) / (1 + Om0/2) ... use ODE
            # Actually for flat universe at a=1: f = Om0^(gamma) but let's use ODE
            pass
        
        Om_a = _Om_of_a(a, Om0, w0, wa)
        
        def ode_rhs(lna, y):
            a_local = np.exp(lna)
            Om_local = _Om_of_a(a_local, Om0, w0, wa)
            u_local = 1.0 + w0 + wa * (1.0 - a_local)
            dlnH2_dlna = 3.0 * Om_local + 3.0 * (1.0 - Om_local) * u_local - 3.0 * (1.0 - Om_local) * wa
            Hprime_over_H = 0.5 * dlnH2_dlna
            f_local = y[0]
            df_dlna = -(3.0/2.0 + Hprime_over_H) * f_local + (3.0/2.0) * Om_local
            return [df_dlna]
        
        lna_target = np.log(a)
        lna_start = -5.0
        y0 = [2.0/3.0]
        
        sol = solve_ivp(ode_rhs, [lna_start, lna_target], y0, method='RK45', 
                       rtol=1e-10, atol=1e-12, max_step=0.1)
        if sol.success:
            f_vals[i] = sol.y[0, -1]
        else:
            f_vals[i] = Om_a ** 0.55
    
    return f_vals

def _gamma_from_f(f, Om_a):
    """Extract gamma = ln(f)/ln(Om_a) from f and Om_a."""
    gamma = np.zeros_like(f)
    for i in range(len(f)):
        if Om_a[i] > 1e-10 and Om_a[i] < 1.0 - 1e-10:
            gamma[i] = np.log(f[i]) / np.log(Om_a[i])
        else:
            gamma[i] = 0.55
    return gamma

def _build_design_matrix(a_values, Om0, w0, wa, coeffs):
    """Build the design matrix for linear least-squares in coefficients."""
    n = len(a_values)
    A = np.zeros((n, 6))
    for i, a in enumerate(a_values):
        Om_a = _Om_of_a(a, Om0, w0, wa)
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa * (1.0 - a)
        A[i, 0] = 1.0
        A[i, 1] = q
        A[i, 2] = q**2
        A[i, 3] = u
        A[i, 4] = wa * a
        A[i, 5] = u * q
    return A

def _fit_coefficients(cosmologies, a_values):
    """Fit 6 coefficients by least-squares on gamma(a)."""
    # Collect all gamma data points
    all_A = []
    all_gamma = []
    
    for Om0, w0, wa in cosmologies:
        f_vals = _compute_gamma_grid(Om0, w0, wa, a_values)
        Om_vals = np.array([_Om_of_a(a, Om0, w0, wa) for a in a_values])
        gamma_vals = _gamma_from_f(f_vals, Om_vals)
        
        A = _build_design_matrix(a_values, Om0, w0, wa, None)
        all_A.append(A)
        all_gamma.append(gamma_vals)
    
    A_full = np.vstack(all_A)
    gamma_full = np.concatenate(all_gamma)
    
    # Weight points: weight by 1/Om to emphasize low-a where gamma varies most
    weights = np.ones(len(gamma_full))
    # Slightly upweight early times
    for i, a in enumerate(a_values):
        for c in range(len(cosmologies)):
            idx = c * len(a_values) + i
            if a < 0.3:
                weights[idx] = 2.0
    
    # Solve weighted least squares
    Aw = A_full * weights[:, None]
    gw = gamma_full * weights
    
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(Aw, gw, rcond=None)
    except:
        coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Clamp to valid range
    coeffs = np.clip(coeffs, -10.0, 10.0)
    
    return coeffs

def _compute_max_error(coeffs, cosmologies, a_values):
    """Compute 100 * max relative error over all cosmologies and a values."""
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err = 0.0
    
    for Om0, w0, wa in cosmologies:
        f_ref = _compute_gamma_grid(Om0, w0, wa, a_values)
        
        for i, a in enumerate(a_values):
            Om_a = _Om_of_a(a, Om0, w0, wa)
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a)
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
            
            if gamma < 0.1 or gamma > 1.2:
                return 1e10
            
            f_approx = Om_a ** gamma
            f_r = f_ref[i]
            
            if abs(f_r) > 1e-10:
                err = abs(f_approx - f_r) / abs(f_r)
                max_err = max(max_err, err)
            
            # D error: compute both
            D_approx = _D_norm_approx(a, Om0, w0, wa, coeffs)
            D_ref = _D_norm_ref(a, Om0, w0, wa)
            
            if abs(D_ref) > 1e-10:
                err_D = abs(D_approx - D_ref) / abs(D_ref)
                max_err = max(max_err, err_D)
    
    return 100.0 * max_err

def _D_norm_approx(a, Om0, w0, wa, coeffs):
    """D(a)/D(1) using approximate gamma."""
    c0, c1, c2, c3, c4, c5 = coeffs
    if a == 1.0:
        return 1.0
    
    def integrand(x):
        a_local = np.exp(x)
        Om_local = _Om_of_a(a_local, Om0, w0, wa)
        q_local = 1.0 - Om_local
        u_local = 1.0 + w0 + wa * (1.0 - a_local)
        gamma_local = c0 + c1*q_local + c2*q_local**2 + c3*u_local + c4*wa*a_local + c5*u_local*q_local
        if gamma_local < 0.1 or gamma_local > 1.2:
            gamma_local = 0.55
        return Om_local ** gamma_local
    
    lna_target = np.log(a)
    if lna_target < 0:
        integral, _ = quad(integrand, lna_target, 0.0, limit=200)
        return np.exp(integral)
    else:
        integral, _ = quad(integrand, 0.0, lna_target, limit=200)
        return np.exp(-integral)

def _D_norm_ref(a, Om0, w0, wa):
    """D(a)/D(1) using constant gamma=0.55."""
    if a == 1.0:
        return 1.0
    
    def integrand(x):
        a_local = np.exp(x)
        Om_local = _Om_of_a(a_local, Om0, w0, wa)
        return Om_local ** 0.55
    
    lna_target = np.log(a)
    if lna_target < 0:
        integral, _ = quad(integrand, lna_target, 0.0, limit=200)
        return np.exp(integral)
    else:
        integral, _ = quad(integrand, 0.0, lna_target, limit=200)
        return np.exp(-integral)

def _objective(coeffs):
    """Objective: 100 * max relative error."""
    if time.time() > _DEADLINE:
        return 1e10
    
    cosmologies = [
        (0.3, -1.0, 0.0),
        (0.3, -1.0, 0.3),
        (0.3, -0.9, 0.0),
        (0.25, -1.0, 0.5),
        (0.35, -1.2, 0.0),
        (0.28, -1.1, 0.2),
    ]
    a_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    
    return _compute_max_error(coeffs, cosmologies, a_values)

def _objective_smooth(coeffs, beta=30.0):
    """Smooth surrogate using log-sum-exp."""
    if time.time() > _DEADLINE:
        return 1e10
    
    cosmologies = [
        (0.3, -1.0, 0.0),
        (0.3, -1.0, 0.3),
        (0.3, -0.9, 0.0),
        (0.25, -1.0, 0.5),
        (0.35, -1.2, 0.0),
    ]
    a_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    
    c0, c1, c2, c3, c4, c5 = coeffs
    errors = []
    
    for Om0, w0, wa in cosmologies:
        f_ref = _compute_gamma_grid(Om0, w0, wa, a_values)
        
        for i, a in enumerate(a_values):
            Om_a = _Om_of_a(a, Om0, w0, wa)
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a)
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
            
            if gamma < 0.1 or gamma > 1.2:
                return 1e10
            
            f_approx = Om_a ** gamma
            f_r = f_ref[i]
            
            if abs(f_r) > 1e-10:
                errors.append(abs(f_approx - f_r) / abs(f_r))
            
            D_approx = _D_norm_approx(a, Om0, w0, wa, coeffs)
            D_ref = _D_norm_ref(a, Om0, w0, wa)
            
            if abs(D_ref) > 1e-10:
                errors.append(abs(D_approx - D_ref) / abs(D_ref))
    
    if not errors:
        return 0.0
    
    errors_arr = np.array(errors)
    max_err = np.max(errors_arr)
    if max_err < 1e-12:
        return 0.0
    log_sum = np.log(np.sum(np.exp(beta * (errors_arr - max_err))))
    smooth_max = max_err + log_sum / beta
    return 100.0 * smooth_max

def construct():
    """Construct the six coefficients."""
    global _START_TIME, _DEADLINE
    
    cosmologies = [
        (0.3, -1.0, 0.0),
        (0.3, -1.0, 0.3),
        (0.3, -0.9, 0.0),
        (0.25, -1.0, 0.5),
        (0.35, -1.2, 0.0),
        (0.28, -1.1, 0.2),
    ]
    a_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    
    # Phase 1: Linear least-squares fit on gamma
    best_coeffs = _fit_coefficients(cosmologies, a_values)
    best_score = _compute_max_error(best_coeffs, cosmologies, a_values)
    
    # Phase 2: Nelder-Mead polish with smooth objective
    try:
        result = minimize(_objective_smooth, best_coeffs, method='Nelder-Mead',
                         options={'maxiter': 600, 'xatol': 1e-8, 'fatol': 1e-8,
                                 'adaptive': True})
        if result.fun < best_score:
            c = np.clip(result.x, -10, 10)
            s = _compute_max_error(c, cosmologies, a_values)
            if s < best_score:
                best_coeffs = c
                best_score = s
    except Exception:
        pass
    
    if time.time() > _DEADLINE - 20.0:
        return list(best_coeffs)
    
    # Phase 3: Multi-start Nelder-Mead
    start_points = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.6, -0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.05, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.05, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.05],
        list(best_coeffs),
    ]
    
    for sp in start_points:
        if time.time() > _DEADLINE - 25.0:
            break
        try:
            result = minimize(_objective_smooth, np.array(sp), method='Nelder-Mead',
                             options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-8,
                                     'adaptive': True})
            if result.fun < best_score:
                c = np.clip(result.x, -10, 10)
                s = _compute_max_error(c, cosmologies, a_values)
                if s < best_score:
                    best_coeffs = c
                    best_score = s
        except Exception:
            pass
    
    if time.time() > _DEADLINE - 10.0:
        return list(best_coeffs)
    
    # Phase 4: High-precision polish with exact objective
    try:
        result = minimize(_objective, best_coeffs, method='Nelder-Mead',
                         options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-10,
                                 'adaptive': True})
        if result.fun < best_score:
            c = np.clip(result.x, -10, 10)
            best_coeffs = c
            best_score = result.fun
    except Exception:
        pass
    
    return list(best_coeffs)
