"""Cosmology: compact dark-energy growth approximation.
Optimized DE + smooth surrogate + high-precision local polish.
"""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize

_START = time.time()
_DEADLINE = 85.0

def _elapsed():
    return time.time() - _START

def _check_deadline():
    if _elapsed() > _DEADLINE:
        raise TimeoutError()

def _om_matter_vec(a, om0):
    return om0 / (a ** 3)

def _gamma_of_a_vec(a, om0, w0, wa, c):
    c0, c1, c2, c3, c4, c5 = c
    om_a = _om_matter_vec(a, om0)
    q = 1.0 - om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma

def _f_of_a_vec(a, om0, w0, wa, c):
    om_a = _om_matter_vec(a, om0)
    gamma = _gamma_of_a_vec(a, om0, w0, wa, c)
    gamma = np.clip(gamma, 0.1, 1.2)
    return om_a ** gamma

def _solve_growth_ode(om0, w0, wa, a_min=0.01):
    def ode(x, y):
        a = np.exp(x)
        E2 = om0 / (a**3) + (1.0 - om0) * np.exp(3.0 * (1.0 + w0 + wa) * (1.0 - a))
        om_m = om0 / (a**3)
        om_de = (1.0 - om0) * np.exp(3.0 * (1.0 + w0 + wa) * (1.0 - a))
        w = w0 + wa * (1.0 - a)
        E = np.sqrt(E2)
        A = 3.0 / E * (1.0 + 0.5 * om_de * (1.0 + 3.0 * w) / om_m) + 2.0
        B = -3.0 * om_m / (2.0 * E2)
        return [y[1], -A * y[1] + B * y[0]]

    x_min = np.log(a_min)
    x_max = np.log(1.0)
    a_init = a_min
    D_init = a_init ** (6.0 / 5.0)
    Dp_init = (6.0 / 5.0) * a_init ** (1.0 / 5.0)
    
    sol = solve_ivp(ode, [x_min, x_max], [D_init, Dp_init],
                    method='DOP853', rtol=1e-10, atol=1e-12,
                    dense_output=True, max_step=0.01)
    
    if not sol.success:
        raise RuntimeError("ODE solve failed")
    return sol

def _get_reference(om0, w0, wa, a_grid):
    sol = _solve_growth_ode(om0, w0, wa)
    x_grid = np.log(a_grid)
    y = sol.sol(x_grid)
    D_raw = y[0]
    Dp_raw = y[1]
    D_ref = D_raw / D_raw[-1]
    f_ref = Dp_raw / D_raw
    D_ref = np.maximum(D_ref, 1e-10)
    f_ref = np.maximum(f_ref, 1e-10)
    return f_ref, D_ref

def _build_validation_set():
    om0_vals = [0.2, 0.25, 0.27, 0.3, 0.35, 0.4, 0.45]
    w0_vals = [-1.0, -0.95, -0.9, -0.85, -0.8, -1.05]
    wa_vals = [0.0, 0.2, 0.4, -0.2, -0.4]
    
    cosmologies = []
    for om0 in om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                cosmologies.append((om0, w0, wa))
    
    np.random.seed(42)
    if len(cosmologies) > 40:
        strata = {}
        for i, (om0, w0, wa) in enumerate(cosmologies):
            key = (om0, w0)
            if key not in strata:
                strata[key] = []
            strata[key].append(i)
        
        selected = set()
        for key, indices in strata.items():
            if len(indices) <= 3:
                selected.update(indices)
            else:
                sample_idx = np.random.choice(indices, 3, replace=False)
                selected.update(sample_idx.tolist())
        
        if len(selected) > 40:
            selected = set(np.random.choice(list(selected), 40, replace=False))
        
        cosmologies = [cosmologies[i] for i in sorted(selected)]
    
    for ref in [(0.27, -1.0, 0.0), (0.3, -1.0, 0.0), (0.27, -0.9, 0.2),
                (0.3, -0.9, 0.3), (0.25, -0.8, 0.4)]:
        if ref not in cosmologies:
            cosmologies.append(ref)
    
    a_grid = np.logspace(np.log10(0.02), 0.0, 150)
    a_grid = np.unique(a_grid)
    return cosmologies, a_grid

def _build_D_integration(a_grid, x_grid):
    n = len(x_grid)
    dx = np.diff(x_grid)
    M = np.zeros((n, n))
    for j in range(n - 1):
        w = 0.5 * dx[j]
        M[j, j] += w
        M[j, j+1] += w
    M_cum = np.zeros_like(M)
    for i in range(n):
        M_cum[i] = np.sum(M[i:], axis=0)
    return M_cum

def _evaluate_score(c, cosmologies, a_grid, f_refs, D_refs, x_grid, M_cum_list):
    max_err = 0.0
    n_a = len(a_grid)
    
    for i, (om0, w0, wa) in enumerate(cosmologies):
        f_ref = f_refs[i]
        D_ref = D_refs[i]
        
        f_model = _f_of_a_vec(a_grid, om0, w0, wa, c)
        err_f = np.abs(f_model - f_ref) / f_ref
        max_err = max(max_err, np.max(err_f))
        
        a_x = np.exp(x_grid)
        f_on_x = _f_of_a_vec(a_x, om0, w0, wa, c)
        cum_int = M_cum_list[i] @ f_on_x
        D_model = np.exp(-cum_int)
        
        err_D = np.abs(D_model - D_ref) / D_ref
        max_err = max(max_err, np.max(err_D))
    
    return 100.0 * max_err

def _objective_smooth(c, cosmologies, a_grid, f_refs, D_refs, x_grid, M_cum_list, beta=100.0):
    n_cosmo = len(cosmologies)
    n_a = len(a_grid)
    total_errs = n_cosmo * 2 * n_a
    errs = np.empty(total_errs)
    
    idx = 0
    for i, (om0, w0, wa) in enumerate(cosmologies):
        f_ref = f_refs[i]
        D_ref = D_refs[i]
        
        f_model = _f_of_a_vec(a_grid, om0, w0, wa, c)
        err_f = np.abs(f_model - f_ref) / f_ref
        errs[idx:idx+n_a] = err_f
        idx += n_a
        
        a_x = np.exp(x_grid)
        f_on_x = _f_of_a_vec(a_x, om0, w0, wa, c)
        cum_int = M_cum_list[i] @ f_on_x
        D_model = np.exp(-cum_int)
        err_D = np.abs(D_model - D_ref) / D_ref
        errs[idx:idx+n_a] = err_D
        idx += n_a
    
    if idx == 0:
        return 1e6
    
    errs = errs[:idx]
    m = np.max(errs)
    if m < 1e-15:
        return 0.0
    shifted = (errs - m) * beta
    shifted = np.clip(shifted, -500, 500)
    lse = m + np.log(np.sum(np.exp(shifted))) / beta
    return 100.0 * lse

def _local_polish(c0, cosmologies, a_grid, f_refs, D_refs, x_grid, M_cum_list, max_iter=500):
    c_best = c0.copy()
    score_best = _evaluate_score(c_best, cosmologies, a_grid, f_refs, D_refs, x_grid, M_cum_list)
    
    np.random.seed(123)
    for attempt in range(8):
        if _elapsed() > _DEADLINE - 5:
            break
        
        if attempt == 0:
            c_start = c_best.copy()
        else:
            c_start = c_best + np.random.randn(6) * 0.005
        
        c_start = np.clip(c_start, -10, 10)
        
        try:
            res = minimize(
                _evaluate_score,
                c_start,
                args=(cosmologies, a_grid, f_refs, D_refs, x_grid, M_cum_list),
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.success:
                c_try = np.clip(res.x, -10, 10)
                score_try = _evaluate_score(c_try, cosmologies, a_grid, f_refs, D_refs, x_grid, M_cum_list)
                if score_try < score_best:
                    c_best = c_try
                    score_best = score_try
        except Exception:
            pass
    
    return c_best

def construct():
    _check_deadline()
    
    cosmologies, a_grid = _build_validation_set()
    n_cosmo = len(cosmologies)
    x_grid = np.log(a_grid)
    
    f_refs = []
    D_refs = []
    for i, (om0, w0, wa) in enumerate(cosmologies):
        _check_deadline()
        f_ref, D_ref = _get_reference(om0, w0, wa, a_grid)
        f_refs.append(f_ref)
        D_refs.append(D_ref)
    
    _check_deadline()
    
    M_cum_list = []
    for i in range(n_cosmo):
        M_cum_list.append(_build_D_integration(a_grid, x_grid))
    
    _check_deadline()
    
    bounds = [(-10, 10)] * 6
    c_best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    score_init = _evaluate_score(c_best, cosmologies, a_grid, f_refs, D_refs, x_grid, M_cum_list)
    
    if _elapsed() < _DEADLINE - 35:
        try:
            de_result = differential_evolution(
                _objective_smooth,
                bounds,
                args=(cosmologies, a_grid, f_refs, D_refs, x_grid, M_cum_list, 100.0),
                maxiter=150,
                popsize=30,
                tol=1e-8,
                seed=42,
                polish=False,
                updating='immediate'
            )
            if de_result.success and _elapsed() < _DEADLINE - 15:
                c_de = de_result.x
                score_de = _evaluate_score(c_de, cosmologies, a_grid, f_refs, D_refs, x_grid, M_cum_list)
                if score_de < score_init:
                    c_best = c_de
                    score_init = score_de
        except Exception:
            pass
    
    _check_deadline()
    
    if _elapsed() < _DEADLINE - 10:
        c_polished = _local_polish(c_best, cosmologies, a_grid, f_refs, D_refs, x_grid, M_cum_list, max_iter=500)
        score_polished = _evaluate_score(c_polished, cosmologies, a_grid, f_refs, D_refs, x_grid, M_cum_list)
        if score_polished < score_init:
            c_best = c_polished
            score_init = score_polished
    
    _check_deadline()
    
    c_best = np.clip(c_best, -10, 10)
    if not np.all(np.isfinite(c_best)):
        c_best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    a_check = np.logspace(np.log10(0.01), 0.0, 200)
    gamma_ok = True
    for om0, w0, wa in cosmologies:
        g = _gamma_of_a_vec(a_check, om0, w0, wa, c_best)
        if np.any(g < 0.1) or np.any(g > 1.2):
            gamma_ok = False
            break
    
    if not gamma_ok:
        for scale in [0.8, 0.6, 0.4, 0.2, 0.0]:
            c_try = np.array([c_best[0], scale*c_best[1], scale*c_best[2],
                              scale*c_best[3], scale*c_best[4], scale*c_best[5]])
            gamma_ok = True
            for om0, w0, wa in cosmologies:
                g = _gamma_of_a_vec(a_check, om0, w0, wa, c_try)
                if np.any(g < 0.1) or np.any(g > 1.2):
                    gamma_ok = False
                    break
            if gamma_ok:
                c_best = c_try
                break
    
    if not gamma_ok:
        c_best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    return [float(x) for x in c_best]
