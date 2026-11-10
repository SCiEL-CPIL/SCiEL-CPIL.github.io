# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize, differential_evolution

_K = 10
_NPHI = 130
_NTH = 70
_NPTS = _NTH * _NPHI
_NC = 2 * _K + 1

def _build_quadrature():
    x, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI
    W = np.outer(w, np.full(_NPHI, dphi))

    Y = np.zeros((_NTH, _NPHI, _NC))
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        denom = 1.0
        for n in range(_K - am + 1, _K + am + 1):
            denom *= n
        fact_ratio = 1.0 / denom
        Nm = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * fact_ratio)
        P = lpmv(am, _K, x)
        if m == 0:
            Y[:, :, idx] = Nm * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.sin(am * phi)[None, :]
            
    return Y, W

_Y, _W = _build_quadrature()
_Wflat = _W.ravel()
_Yflat = _Y.reshape(_NPTS, _NC)

def _objective(c):
    f = _Yflat @ c
    f2 = f * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return 1e30
    f4 = f2 * f2
    l4sq = float(np.dot(_Wflat, f4))
    return l4sq / (l2sq * l2sq)

def _gradient(c):
    f = _Yflat @ c
    f2 = f * f
    f3 = f2 * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return np.zeros_like(c)
    l4sq = float(np.dot(_Wflat, f2 * f2))
    wf3 = _Wflat * f3
    wf = _Wflat * f
    grad_f3 = _Yflat.T @ wf3
    grad_f = _Yflat.T @ wf
    B = l2sq
    A = l4sq
    grad = (4.0 / (B * B)) * grad_f3 - (4.0 * A / (B * B * B)) * grad_f
    return grad

def _objective_sym(c_sym):
    c_full = np.zeros(_NC)
    c_full[_K] = c_sym[0]
    for m in range(1, _K + 1):
        c_full[_K + m] = c_sym[m]
    return _objective(c_full)

def _gradient_sym(c_sym):
    c_full = np.zeros(_NC)
    c_full[_K] = c_sym[0]
    for m in range(1, _K + 1):
        c_full[_K + m] = c_sym[m]
    grad_full = _gradient(c_full)
    grad_sym = np.zeros(len(c_sym))
    grad_sym[0] = grad_full[_K]
    for m in range(1, _K + 1):
        grad_sym[m] = grad_full[_K + m]
    return grad_sym

def _c_to_complex(c):
    z = np.zeros(_K + 1, dtype=np.complex128)
    z[0] = c[_K]
    for m in range(1, _K + 1):
        z[m] = c[_K + m] + 1j * c[_K - m]
    return z

def _complex_to_c(z):
    c = np.zeros(_NC)
    c[_K] = z[0].real
    for m in range(1, _K + 1):
        c[_K + m] = z[m].real
        c[_K - m] = z[m].imag
    return c

def _objective_complex(z):
    c = _complex_to_c(z)
    return _objective(c)

def _gradient_complex(z):
    c = _complex_to_c(z)
    grad_full = _gradient(c)
    gz = np.zeros(_K + 1, dtype=np.complex128)
    gz[0] = grad_full[_K]
    for m in range(1, _K + 1):
        gz[m] = 0.5 * (grad_full[_K + m] - 1j * grad_full[_K - m])
    return gz

def _quartic_lanczos(c0, max_iter=2000, tol=1e-14, deadline=None):
    """Lanczos-style iteration on the sphere for the quartic form.
    
    This performs projected gradient descent on the sphere with adaptive step size.
    The key idea: the quartic form c^T Q c^3 (where Q encodes the 4th moment)
    has stationary points that can be found by iterating c <- normalize(c - alpha * grad).
    """
    c = np.asarray(c0, dtype=np.float64).copy()
    nrm = np.linalg.norm(c)
    if nrm < 1e-15:
        return c0, _objective(c0)
    c = c / nrm
    
    f = _objective(c)
    best_c = c.copy()
    best_f = f
    
    g = _gradient(c)
    # Project gradient onto tangent space
    g_t = g - np.dot(g, c) * c
    
    p = -g_t.copy()
    step = 0.05
    p_norm = np.linalg.norm(p)
    if p_norm < 1e-15:
        return best_c, best_f
    p = p / p_norm
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Adaptive line search: try several step sizes
        best_trial_f = f
        best_step = 0.0
        for mult in [1.0, 0.5, 2.0, 0.25, 4.0, 0.1, 0.05]:
            trial_step = step * mult
            if trial_step <= 0:
                continue
            new_c = c + trial_step * p
            new_nrm = np.linalg.norm(new_c)
            if new_nrm < 1e-15:
                continue
            new_c = new_c / new_nrm
            trial_f = _objective(new_c)
            if trial_f < best_trial_f:
                best_trial_f = trial_f
                best_step = trial_step
        
        if best_step == 0.0:
            # Try very small steps
            for mult in [0.02, 0.01, 0.005, 0.002, 0.001]:
                trial_step = step * mult
                if trial_step <= 0:
                    continue
                new_c = c + trial_step * p
                new_nrm = np.linalg.norm(new_c)
                if new_nrm < 1e-15:
                    continue
                new_c = new_c / new_nrm
                trial_f = _objective(new_c)
                if trial_f < best_trial_f:
                    best_trial_f = trial_f
                    best_step = trial_step
        
        if best_step == 0.0:
            break
        
        new_c = c + best_step * p
        new_nrm = np.linalg.norm(new_c)
        if new_nrm < 1e-15:
            break
        new_c = new_c / new_nrm
        new_f = _objective(new_c)
        
        if new_f < f:
            if new_f < best_f:
                best_f = new_f
                best_c = new_c.copy()
            
            new_g = _gradient(new_c)
            new_g_t = new_g - np.dot(new_g, new_c) * new_c
            new_g_t_norm = np.linalg.norm(new_g_t)
            
            if new_g_t_norm < tol:
                c = new_c
                f = new_f
                break
            
            # Update conjugate direction (Polak-Ribiere)
            old_g_t_norm = np.linalg.norm(g_t)
            beta = np.dot(new_g_t, new_g_t - g_t) / max(old_g_t_norm * old_g_t_norm, 1e-30)
            beta = max(0.0, min(beta, 2.0))  # Clip for stability
            
            p = -new_g_t + beta * p
            p_norm = np.linalg.norm(p)
            if p_norm < 1e-15:
                p = -new_g_t
                p_norm = np.linalg.norm(p)
                if p_norm < 1e-15:
                    break
                p = p / p_norm
            else:
                p = p / p_norm
            
            step = best_step
            c = new_c
            f = new_f
            g_t = new_g_t
        else:
            step *= 0.5
            if step < 1e-16:
                break
            continue
    
    return best_c, best_f

def _polish(c0, deadline):
    c = np.asarray(c0, dtype=np.float64).copy()
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    
    c = c / nrm
    
    def fun(x):
        return _objective(x)
        
    def jac(x):
        return _gradient(x)
        
    best_c = c.copy()
    best_val = fun(c)
    
    try:
        constraint = {'type': 'eq', 'fun': lambda x: np.dot(x, x) - 1.0, 'jac': lambda x: 2.0 * x}
        res = minimize(fun, c, jac=jac, method='SLSQP',
                       constraints=[constraint],
                       options={'maxiter': 500, 'ftol': 1e-15, 'disp': False})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
            best_c /= np.linalg.norm(best_c)
    except Exception:
        pass
        
    nrm = np.linalg.norm(best_c)
    if nrm < 1e-14:
        return None
    return best_c / nrm, float(best_val)

def _polish_high_precision(c0, deadline):
    c = np.asarray(c0, dtype=np.float64).copy()
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    
    c = c / nrm
    
    def fun(x):
        return _objective(x)
        
    def jac(x):
        return _gradient(x)
        
    best_c = c.copy()
    best_val = fun(c)
    
    try:
        constraint = {'type': 'eq', 'fun': lambda x: np.dot(x, x) - 1.0, 'jac': lambda x: 2.0 * x}
        res = minimize(fun, c, jac=jac, method='SLSQP',
                       constraints=[constraint],
                       options={'maxiter': 1000, 'ftol': 1e-16, 'disp': False})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
            best_c /= np.linalg.norm(best_c)
    except Exception:
        pass
    
    if time.time() < deadline - 5:
        try:
            constraint = {'type': 'eq', 'fun': lambda x: np.dot(x, x) - 1.0, 'jac': lambda x: 2.0 * x}
            res2 = minimize(fun, best_c, jac=jac, method='SLSQP',
                            constraints=[constraint],
                            options={'maxiter': 1000, 'ftol': 1e-16, 'disp': False})
            if res2.fun < best_val:
                best_val = res2.fun
                best_c = res2.x.copy()
                best_c /= np.linalg.norm(best_c)
        except Exception:
            pass
    
    nrm = np.linalg.norm(best_c)
    if nrm < 1e-14:
        return None
    return best_c / nrm, float(best_val)

def _polish_sym(c0_sym, deadline):
    c_sym = np.asarray(c0_sym, dtype=np.float64).copy()
    nrm = np.linalg.norm(c_sym)
    if nrm < 1e-14:
        return None
    
    c_sym = c_sym / nrm
    
    def fun(x):
        return _objective_sym(x)
        
    def jac(x):
        return _gradient_sym(x)
        
    best_c_sym = c_sym.copy()
    best_val = fun(c_sym)
    
    try:
        constraint = {'type': 'eq', 'fun': lambda x: np.dot(x, x) - 1.0, 'jac': lambda x: 2.0 * x}
        res = minimize(fun, c_sym, jac=jac, method='SLSQP',
                       constraints=[constraint],
                       options={'maxiter': 500, 'ftol': 1e-15, 'disp': False})
        if res.fun < best_val:
            best_val = res.fun
            best_c_sym = res.x.copy()
            best_c_sym /= np.linalg.norm(best_c_sym)
    except Exception:
        pass
        
    c_full = np.zeros(_NC)
    c_full[_K] = best_c_sym[0]
    for m in range(1, _K + 1):
        c_full[_K + m] = best_c_sym[m]
        
    nrm = np.linalg.norm(c_full)
    if nrm < 1e-14:
        return None
    return c_full / nrm, float(best_val)

def _cg_complex_sphere(z0, max_iter=500, tol=1e-14, deadline=None):
    z = np.array(z0, dtype=np.complex128).copy()
    r = np.sqrt(np.sum(np.abs(z)**2))
    if r < 1e-15:
        return z0, _objective_complex(z0)
    z = z / r * r
    
    f = _objective_complex(z)
    best_z = z.copy()
    best_f = f
    
    g = _gradient_complex(z)
    
    z_dot = np.sum(np.abs(z)**2)
    g_dot_z = np.sum(g * np.conj(z))
    g_t = g - (g_dot_z / z_dot) * z
    
    p = -g_t.copy()
    step = 0.1
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        best_trial_f = f
        best_step = 0.0
        
        for trial_mult in [1.0, 0.5, 2.0, 0.25, 0.1, 4.0, 0.05]:
            trial_step = step * trial_mult
            if trial_step <= 0:
                continue
            new_z = z + step * p
            r_new = np.sqrt(np.sum(np.abs(new_z)**2))
            if r_new < 1e-15:
                continue
            new_z = new_z * (r / r_new)
            trial_f = _objective_complex(new_z)
            if trial_f < best_trial_f:
                best_trial_f = trial_f
                best_step = trial_step
        
        if best_step == 0.0:
            for trial_mult in [0.02, 0.01, 0.005, 0.002, 0.001]:
                trial_step = step * trial_mult
                if trial_step <= 0:
                    continue
                new_z = z + trial_step * p
                r_new = np.sqrt(np.sum(np.abs(new_z)**2))
                if r_new < 1e-15:
                    continue
                new_z = new_z * (r / r_new)
                trial_f = _objective_complex(new_z)
                if trial_f < best_trial_f:
                    best_trial_f = trial_f
                    best_step = trial_step
        
        if best_step == 0.0:
            break
        
        z_new = z + best_step * p
        r_new = np.sqrt(np.sum(np.abs(z_new)**2))
        if r_new < 1e-15:
            break
        z_new = z_new * (r / r_new)
        f_new = _objective_complex(z_new)
        
        if f_new < f:
            if f_new < best_f:
                best_f = f_new
                best_z = z_new.copy()
            
            g_new = _gradient_complex(z_new)
            z_dot_new = np.sum(np.abs(z_new)**2)
            g_dot_z_new = np.sum(g_new * np.conj(z_new))
            g_t_new = g_new - (g_dot_z_new / z_dot_new) * z_new
            
            g_norm_new = np.sqrt(np.sum(np.abs(g_t_new)**2))
            if g_norm_new < tol:
                z = z_new
                f = f_new
                break
            
            beta = np.sum(g_t_new * np.conj(g_t_new)) / max(np.sum(g_t * np.conj(g_t)), 1e-30)
            p = -g_t_new + beta * p
            step = best_step
            z = z_new
            f = f_new
            g_t = g_t_new
        else:
            step *= 0.5
            if step < 1e-15:
                break
            continue
    
    return best_z, best_f

def rudin_problem10() -> np.ndarray:
    start = time.time()
    deadline = start + 600.0
    
    best_c = None
    best_val = np.inf
    
    # 1. Global search in symmetric subspace (11D)
    bounds_sym = [(-10.0, 10.0) for _ in range(_K + 1)]
    
    try:
        if time.time() < deadline - 120.0:
            de_result_sym = differential_evolution(
                _objective_sym,
                bounds_sym,
                seed=42,
                maxiter=1500,
                popsize=25,
                tol=1e-12,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=False,
                init='sobol'
            )
            if de_result_sym.fun < best_val:
                best_val = de_result_sym.fun
                c_full = np.zeros(_NC)
                c_full[_K] = de_result_sym.x[0]
                for m in range(1, _K + 1):
                    c_full[_K + m] = de_result_sym.x[m]
                best_c = c_full.copy()
    except Exception:
        pass
    
    # 1b. Global search in full 21D space
    bounds_full = [(-10.0, 10.0) for _ in range(_NC)]
    
    try:
        if time.time() < deadline - 90.0:
            de_result_full = differential_evolution(
                _objective,
                bounds_full,
                seed=123,
                maxiter=800,
                popsize=20,
                tol=1e-12,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=False,
                init='sobol'
            )
            if de_result_full.fun < best_val:
                best_val = de_result_full.fun
                best_c = de_result_full.x.copy()
    except Exception:
        pass

    # 2. Quartic Lanczos polish from DE result
    if best_c is not None and time.time() < deadline - 60.0:
        c_lanczos, val_lanczos = _quartic_lanczos(best_c, max_iter=3000, tol=1e-15, deadline=deadline - 30)
        if val_lanczos < best_val:
            best_val = val_lanczos
            best_c = c_lanczos

    # 3. SLSQP polish
    if best_c is not None and time.time() < deadline - 40.0:
        polished = _polish(best_c, deadline - 10)
        if polished:
            c, val = polished
            if val < best_val:
                best_val = val
                best_c = c

    # 4. Complex sphere CG polish
    if best_c is not None and time.time() < deadline - 25.0:
        z0 = _c_to_complex(best_c)
        z_cg, val_cg = _cg_complex_sphere(z0, max_iter=800, tol=1e-15, deadline=deadline - 10)
        c_cg = _complex_to_c(z_cg)
        nrm = np.linalg.norm(c_cg)
        if nrm > 1e-14:
            c_cg = c_cg / nrm
            val_cg = _objective(c_cg)
            if val_cg < best_val:
                best_val = val_cg
                best_c = c_cg

    # 5. High-precision polish
    if best_c is not None and time.time() < deadline - 15.0:
        polished_hp = _polish_high_precision(best_c, deadline - 5)
        if polished_hp:
            c, val = polished_hp
            if val < best_val:
                best_val = val
                best_c = c
        
        if time.time() < deadline - 10.0:
            z0 = _c_to_complex(best_c)
            z_cg, val_cg = _cg_complex_sphere(z0, max_iter=500, tol=1e-16, deadline=deadline - 5)
            c_cg = _complex_to_c(z_cg)
            nrm = np.linalg.norm(c_cg)
            if nrm > 1e-14:
                c_cg = c_cg / nrm
                val_cg = _objective(c_cg)
                if val_cg < best_val:
                    best_val = val_cg
                    best_c = c_cg

    # 6. Random restarts with quartic Lanczos + SLSQP + CG
    rng = np.random.default_rng(123)
    
    while time.time() < deadline - 30.0:
        if rng.integers(0, 2) == 0:
            # Symmetric space restart
            c0_sym = rng.standard_normal(_K + 1)
            nrm = np.linalg.norm(c0_sym)
            if nrm < 1e-14:
                continue
            c0_sym /= nrm
            
            c_full_start = np.zeros(_NC)
            c_full_start[_K] = c0_sym[0]
            for m in range(1, _K + 1):
                c_full_start[_K + m] = c0_sym[m]
            
            # Quartic Lanczos
            c_lanczos, val_lanczos = _quartic_lanczos(c_full_start, max_iter=500, tol=1e-13, deadline=deadline - 15)
            if val_lanczos < best_val:
                best_val = val_lanczos
                best_c = c_lanczos
            
            # SLSQP
            if time.time() < deadline - 15.0:
                polished = _polish(c_full_start, deadline - 10)
                if polished:
                    c, val = polished
                    if val < best_val:
                        best_val = val
                        best_c = c
            
            # Complex CG
            if time.time() < deadline - 10.0:
                z0 = _c_to_complex(c_full_start)
                z_cg, val_cg = _cg_complex_sphere(z0, max_iter=300, tol=1e-14, deadline=deadline - 5)
                c_cg = _complex_to_c(z_cg)
                nrm = np.linalg.norm(c_cg)
                if nrm > 1e-14:
                    c_cg = c_cg / nrm
                    val_cg = _objective(c_cg)
                    if val_cg < best_val:
                        best_val = val_cg
                        best_c = c_cg
        else:
            # Full space restart
            c0_full = rng.standard_normal(_NC)
            nrm = np.linalg.norm(c0_full)
            if nrm < 1e-14:
                continue
            c0_full /= nrm
            
            # Quartic Lanczos
            c_lanczos, val_lanczos = _quartic_lanczos(c0_full, max_iter=500, tol=1e-13, deadline=deadline - 15)
            if val_lanczos < best_val:
                best_val = val_lanczos
                best_c = c_lanczos
            
            # SLSQP
            if time.time() < deadline - 15.0:
                polished = _polish(c0_full, deadline - 10)
                if polished:
                    c, val = polished
                    if val < best_val:
                        best_val = val
                        best_c = c
            
            # Complex CG
            if time.time() < deadline - 10.0:
                z0 = _c_to_complex(c0_full)
                z_cg, val_cg = _cg_complex_sphere(z0, max_iter=300, tol=1e-14, deadline=deadline - 5)
                c_cg = _complex_to_c(z_cg)
                nrm = np.linalg.norm(c_cg)
                if nrm > 1e-14:
                    c_cg = c_cg / nrm
                    val_cg = _objective(c_cg)
                    if val_cg < best_val:
                        best_val = val_cg
                        best_c = c_cg

    # Final safety
    if best_c is None:
        best_c = np.zeros(_NC, dtype=np.float64)
        best_c[_K] = 1.0
        
    best_c = np.asarray(best_c, dtype=np.float64)
    
    if not np.all(np.isfinite(best_c)):
        best_c = np.zeros(_NC, dtype=np.float64)
        best_c[_K] = 1.0
        
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)
        
    return best_c

# EVOLVE-BLOCK-END
