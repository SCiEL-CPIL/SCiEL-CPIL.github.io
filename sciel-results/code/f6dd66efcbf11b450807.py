"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

_REF_CACHE = None

def _Om_of_a(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    a_safe = np.maximum(a, 1e-10)
    OmDE0 = 1.0 - Om0
    log_ratio = -3.0*(1.0+w0+wa)*np.log(a_safe) - 3.0*wa*(1.0 - a_safe)
    OmDE = OmDE0 * np.exp(log_ratio)
    Om = Om0 / (a_safe**3) / (Om0/(a_safe**3) + OmDE)
    return Om

def _init_cache():
    global _REF_CACHE
    if _REF_CACHE is not None:
        return
    
    Om0_vals = [0.25, 0.30, 0.35]
    w0_vals = [-1.2, -1.0, -0.8]
    wa_vals = [-0.5, -0.2, 0.0, 0.2]
    a_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    
    refs = []
    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                x_min = np.log(0.01)
                x_max = 0.0
                Om_min = _Om_of_a(0.01, Om0, w0, wa)
                F_init = Om_min**0.55
                D_init = 1.0
                
                def rhs(x, y, Om0=Om0, w0=w0, wa=wa):
                    a = np.exp(x)
                    D, F = y
                    Om = _Om_of_a(a, Om0, w0, wa)
                    OmDE = 1.0 - Om
                    w = w0 + wa*(1.0 - a)
                    weff = w * OmDE
                    dD = F
                    dF = -(1.5 + 1.5*weff)*F + 1.5*Om*D
                    return [dD, dF]
                
                sol = solve_ivp(rhs, [x_min, x_max], [D_init, F_init],
                               dense_output=True, rtol=1e-10, atol=1e-12, max_step=0.01)
                
                if not sol.success:
                    continue
                
                D_at_1 = sol.sol(0.0)[0]
                
                f_ref = np.zeros(len(a_vals))
                D_ref = np.zeros(len(a_vals))
                for i, av in enumerate(a_vals):
                    xv = np.log(av)
                    if xv < x_min:
                        f_ref[i] = _Om_of_a(av, Om0, w0, wa)**0.55
                        D_ref[i] = 1.0
                    else:
                        Dv = sol.sol(xv)[0] / D_at_1
                        Fv = sol.sol(xv)[1]
                        f_ref[i] = Fv / Dv if abs(Dv) > 1e-15 else 0.55
                        D_ref[i] = Dv
                
                refs.append({
                    'Om0': Om0, 'w0': w0, 'wa': wa,
                    'a_vals': a_vals, 'f_ref': f_ref, 'D_ref': D_ref
                })
    
    _REF_CACHE = refs

def _model_gamma(c, a, Om0, w0, wa):
    c0, c1, c2, c3, c4, c5 = c
    Om = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa*(1.0 - a)
    return c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q

def _model_f(c, a, Om0, w0, wa):
    gamma = _model_gamma(c, a, Om0, w0, wa)
    Om = np.maximum(_Om_of_a(a, Om0, w0, wa), 1e-10)
    return Om**gamma

def _model_D_vec(c, a_vals, Om0, w0, wa):
    D = np.zeros(len(a_vals))
    x_fine = np.linspace(np.log(0.05), 0.0, 100)
    a_fine = np.exp(x_fine)
    f_fine = _model_f(c, a_fine, Om0, w0, wa)
    
    cum_int = np.zeros(len(x_fine))
    for i in range(1, len(x_fine)):
        cum_int[i] = cum_int[i-1] + 0.5*(f_fine[i-1] + f_fine[i])*(x_fine[i]-x_fine[i-1])
    total_int = cum_int[-1]
    
    for i, av in enumerate(a_vals):
        if av >= 1.0:
            D[i] = 1.0
        else:
            xv = np.log(av)
            idx = np.searchsorted(x_fine, xv)
            if idx <= 0:
                integral = total_int
            elif idx >= len(x_fine):
                integral = 0.0
            else:
                x0, x1 = x_fine[idx-1], x_fine[idx]
                c0v, c1v = cum_int[idx-1], cum_int[idx]
                frac = (xv - x0) / (x1 - x0) if x1 != x0 else 0.5
                cum_at_xv = c0v + frac*(c1v - c0v)
                integral = total_int - cum_at_xv
            D[i] = np.exp(-integral)
    return D

def _objective(c):
    c = np.clip(c, -10.0, 10.0)
    max_err = 0.0
    if _REF_CACHE is None:
        _init_cache()
    
    for ref in _REF_CACHE:
        Om0, w0, wa = ref['Om0'], ref['w0'], ref['wa']
        a_vals, f_ref, D_ref = ref['a_vals'], ref['f_ref'], ref['D_ref']
        
        f_mod = _model_f(c, a_vals, Om0, w0, wa)
        D_mod = _model_D_vec(c, a_vals, Om0, w0, wa)
        
        gamma_vals = _model_gamma(c, a_vals, Om0, w0, wa)
        if np.any(gamma_vals < 0.1) or np.any(gamma_vals > 1.2):
            return 100.0
        
        mask_f = np.abs(f_ref) > 1e-10
        if np.any(mask_f):
            err_f = np.max(np.abs(f_mod[mask_f] - f_ref[mask_f]) / np.abs(f_ref[mask_f]))
            max_err = max(max_err, err_f)
        
        mask_D = np.abs(D_ref) > 1e-10
        if np.any(mask_D):
            err_D = np.max(np.abs(D_mod[mask_D] - D_ref[mask_D]) / np.abs(D_ref[mask_D]))
            max_err = max(max_err, err_D)
    
    return 100.0 * max_err

def construct():
    t_start = time.time()
    DEADLINE = 70.0
    
    _init_cache()
    
    c0_init = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_c = c0_init.copy()
    best_obj = _objective(c0_init)
    
    try:
        res = minimize(_objective, c0_init, method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7})
        if res.fun < best_obj:
            best_obj = res.fun
            best_c = res.x
    except:
        pass
    
    if time.time() - t_start < DEADLINE:
        try:
            res = minimize(_objective, best_c, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-10})
            if res.fun < best_obj:
                best_obj = res.fun
                best_c = res.x
        except:
            pass
    
    if time.time() - t_start < DEADLINE:
        try:
            res = minimize(_objective, best_c, method='Powell',
                          options={'maxiter': 100, 'xtol': 1e-7, 'ftol': 1e-9})
            if res.fun < best_obj:
                best_obj = res.fun
                best_c = res.x
        except:
            pass
    
    if time.time() - t_start < DEADLINE:
        try:
            res = minimize(_objective, best_c, method='Nelder-Mead',
                          options={'maxiter': 50, 'xatol': 1e-7, 'fatol': 1e-9,
                                   'initial_simplex': 0.01 * np.ones(6)})
            if res.fun < best_obj:
                best_obj = res.fun
                best_c = res.x
        except:
            pass
    
    best_c = np.clip(best_c, -10.0, 10.0)
    return list(best_c)

# EVOLVE-BLOCK-END
