import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import minimize
import time

def _Om_m(a, Om0, Ode0, Oh0, w0, wa):
    Om_m = Om0 * a**(-3)
    Ode = Ode0 * (1 + a * wa)**(3*(1 + w0 + wa))
    Oh = Oh0 * a**(-2)
    E2 = Om_m + Ode + Oh
    return Om_m / E2

def _H2(a, Om0, Ode0, Oh0, w0, wa):
    Om_m = Om0 * a**(-3)
    Ode = Ode0 * (1 + a * wa)**(3*(1 + w0 + wa))
    Oh = Oh0 * a**(-2)
    return Om_m + Ode + Oh

def _Hprime_over_H(a, Om0, Ode0, Oh0, w0, wa):
    n = 3*(1 + w0 + wa)
    term_m = -3 * Om0 * a**(-3)
    term_de = Ode0 * n * wa * (1 + a * wa)**(n - 1)
    term_h = -2 * Oh0 * a**(-2)
    dE2_dlna = term_m + term_de + term_h
    E2 = _H2(a, Om0, Ode0, Oh0, w0, wa)
    return 0.5 * dE2_dlna / E2

def _solve_growth(Om0, Ode0, Oh0, w0, wa, a_grid):
    def ode(lna, y):
        a = np.exp(lna)
        y1, y2 = y
        Hprime_H = _Hprime_over_H(a, Om0, Ode0, Oh0, w0, wa)
        Om_m_a = _Om_m(a, Om0, Ode0, Oh0, w0, wa)
        dy1 = y2
        dy2 = -(3/(2*a) + Hprime_H) * y2 + (3/2) * Om_m_a / a**2 * y1
        return [dy1, dy2]
    
    a_min = a_grid[0]
    lna_min = np.log(a_min)
    lna_max = 0.0
    lna_eval = np.linspace(lna_min, lna_max, len(a_grid))
    
    sol = solve_ivp(ode, [lna_min, lna_max], [1.0, 1.0], t_eval=lna_eval, 
                    method='DOP853', rtol=1e-10, atol=1e-10, max_step=0.01)
    if not sol.success:
        return None, None
    
    D_raw = sol.y[0]
    D_prime_raw = sol.y[1]
    D_norm = D_raw / D_raw[-1]
    D_prime_norm = D_prime_raw / D_raw[-1]
    f = D_prime_norm / D_norm
    D_normalized = D_norm
    return D_normalized, f

def _approx_gamma(a, c, Om0, Ode0, Oh0, w0, wa):
    Om_m_a = _Om_m(a, Om0, Ode0, Oh0, w0, wa)
    q = 1 - Om_m_a
    u = 1 + w0 + wa * (1 - a)
    c0, c1, c2, c3, c4, c5 = c
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    return gamma

def _approx_f(a, c, Om0, Ode0, Oh0, w0, wa):
    Om_m_a = _Om_m(a, Om0, Ode0, Oh0, w0, wa)
    gamma = _approx_gamma(a, c, Om0, Ode0, Oh0, w0, wa)
    return Om_m_a ** gamma

def _approx_D_normalized(a_grid, c, Om0, Ode0, Oh0, w0, wa):
    lna_grid = np.log(a_grid)
    f_vals = np.array([_approx_f(np.exp(x), c, Om0, Ode0, Oh0, w0, wa) for x in lna_grid])
    integral_cum = cumulative_trapezoid(f_vals, lna_grid, initial=0)
    total_integral = integral_cum[-1]
    integral_from_i_to_end = total_integral - integral_cum
    D_norm = np.exp(-integral_from_i_to_end)
    return D_norm

def _objective(c, ref_data, a_grid):
    max_err = 0.0
    for (Om0, Ode0, Oh0, w0, wa, D_ref, f_ref) in ref_data:
        f_approx = np.array([_approx_f(a, c, Om0, Ode0, Oh0, w0, wa) for a in a_grid])
        D_approx = _approx_D_normalized(a_grid, c, Om0, Ode0, Oh0, w0, wa)
        err_f = np.abs(f_approx - f_ref) / np.abs(f_ref)
        err_D = np.abs(D_approx - D_ref) / np.abs(D_ref)
        max_err = max(max_err, np.max(err_f), np.max(err_D))
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    cosmologies = []
    Om0_vals = [0.25, 0.30, 0.35]
    w0_vals = [-1.0, -1.2, -0.8]
    wa_vals = [0.0, 0.1, -0.1]
    
    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                Ode0 = 1.0 - Om0
                Oh0 = 0.0
                cosmologies.append((Om0, Ode0, Oh0, w0, wa))
    
    cosmologies.append((0.30, 0.65, 0.05, -1.0, 0.0))
    cosmologies.append((0.25, 0.70, 0.05, -1.2, 0.1))
    
    a_grid = np.logspace(np.log10(0.05), 0.0, 50)
    
    ref_data = []
    for (Om0, Ode0, Oh0, w0, wa) in cosmologies:
        D_ref, f_ref = _solve_growth(Om0, Ode0, Oh0, w0, wa, a_grid)
        if D_ref is not None:
            ref_data.append((Om0, Ode0, Oh0, w0, wa, D_ref, f_ref))
    
    c0_init = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_c = c0_init
    best_obj = _objective(c0_init, ref_data, a_grid)
    
    if time.time() < deadline:
        result = minimize(_objective, c0_init, args=(ref_data, a_grid), method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        if result.fun < best_obj:
            best_c = result.x
            best_obj = result.fun
    
    best_c = np.clip(best_c, -10, 10)
    
    valid = True
    for (Om0, Ode0, Oh0, w0, wa, _, _) in ref_data:
        for a in a_grid:
            g = _approx_gamma(a, best_c, Om0, Ode0, Oh0, w0, wa)
            if g < 0.1 or g > 1.2:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        best_c = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return list(best_c)
