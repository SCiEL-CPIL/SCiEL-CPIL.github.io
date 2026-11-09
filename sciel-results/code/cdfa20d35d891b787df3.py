# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize

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
        # N_m = sqrt((2K+1)/(4pi) * (K-am)!/(K+am)!)
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


def _objective_L4(c):
    f = _Yflat @ c
    return float(np.dot(_Wflat, f * f * f * f))


def _gradient_L4(c):
    f = _Yflat @ c
    return 4.0 * (_Yflat.T @ (_Wflat * f * f * f))


def _objective_unconstrained(c):
    norm_sq = float(np.dot(c, c))
    if norm_sq < 1e-30:
        return 1e30
    f = _Yflat @ c
    H = float(np.dot(_Wflat, f * f * f * f))
    return H / (norm_sq * norm_sq)


def _gradient_unconstrained(c):
    norm_sq = float(np.dot(c, c))
    if norm_sq < 1e-30:
        return np.zeros(_NC)
    f = _Yflat @ c
    H = float(np.dot(_Wflat, f * f * f * f))
    dH = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
    grad = (dH - H * c / norm_sq) / (norm_sq * norm_sq)
    return grad


def _conjugate_gradient_sphere(c, max_iter=5000, tol=1e-17, deadline=None):
    """
    Conjugate gradient on the unit sphere with Fletcher-Reeves restart.
    Uses backtracking line search with Armijo condition.
    """
    c = np.asarray(c, dtype=np.float64)
    norm = np.linalg.norm(c)
    if norm < 1e-12:
        c = np.ones(_NC)
        c /= np.linalg.norm(c)
    else:
        c = c / norm

    best_val = _objective_L4(c)
    best_c = c.copy()

    # Compute initial gradient and project to tangent space
    f = _Yflat @ c
    grad = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
    grad_tan = grad - np.dot(grad, c) * c

    d = -grad_tan.copy()
    prev_grad_tan = grad_tan.copy()
    prev_grad_norm_sq = float(np.dot(prev_grad_tan, prev_grad_tan))

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 2.0:
            break

        # Backtracking line search with Armijo condition
        step = 1.0
        f_curr = best_val
        grad_dot_d = float(np.dot(grad_tan, d))
        success = False
        for _ in range(50):
            c_trial = c + step * d
            c_norm = np.linalg.norm(c_trial)
            if c_norm < 1e-12:
                step *= 0.5
                continue
            c_trial = c_trial / c_norm
            f_trial = _objective_L4(c_trial)
            if f_trial < f_curr - 1e-12 * step * abs(grad_dot_d):
                success = True
                break
            step *= 0.5

        if not success:
            break

        c = c_trial
        if f_trial < best_val:
            best_val = f_trial
            best_c = c.copy()

        # Compute new gradient
        f = _Yflat @ c
        grad = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
        grad_tan = grad - np.dot(grad, c) * c

        gnorm = np.linalg.norm(grad_tan)
        if gnorm < tol:
            break

        # Fletcher-Reeves update with periodic restart
        if it % 10 == 9:
            d = -grad_tan.copy()
        else:
            curr_grad_norm_sq = float(np.dot(grad_tan, grad_tan))
            beta = curr_grad_norm_sq / max(prev_grad_norm_sq, 1e-30)
            if beta < 0 or beta > 10.0:
                d = -grad_tan.copy()
            else:
                d = -grad_tan + beta * d

        prev_grad_tan = grad_tan.copy()
        prev_grad_norm_sq = curr_grad_norm_sq

    return best_c, best_val


def _polish_gradient_descent(c, max_iter=10000, tol=1e-18, deadline=None):
    """
    High-precision polish using projected gradient descent on the sphere.
    Uses adaptive step size and backtracking line search.
    """
    c = np.asarray(c, dtype=np.float64)
    c = c / np.linalg.norm(c)
    best_val = _objective_L4(c)
    best_c = c.copy()
    step_size = 1.0

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 1.0:
            break

        f = _Yflat @ c
        grad = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
        grad_tan = grad - np.dot(grad, c) * c
        gnorm = np.linalg.norm(grad_tan)
        if gnorm < tol:
            break

        # Backtracking line search
        step = step_size
        f_curr = best_val
        success = False
        for _ in range(40):
            c_trial = c - step * grad_tan
            c_norm = np.linalg.norm(c_trial)
            if c_norm < 1e-12:
                step *= 0.5
                continue
            c_trial = c_trial / c_norm
            f_trial = _objective_L4(c_trial)
            if f_trial < f_curr - 1e-14 * step * gnorm * gnorm:
                success = True
                break
            step *= 0.5

        if not success:
            step_size *= 0.5
            if step_size < 1e-20:
                break
            continue

        c = c_trial
        if f_trial < best_val:
            best_val = f_trial
            best_c = c.copy()
            step_size = min(step * 1.5, 1.0)
        else:
            step_size *= 0.5

    return best_c, best_val


def _optimize_sphere(c0, deadline):
    c0 = np.asarray(c0, dtype=np.float64)
    norm = np.linalg.norm(c0)
    if norm < 1e-12:
        c0 = np.ones(_NC)
        c0 /= np.linalg.norm(c0)
    else:
        c0 = c0 / norm

    def fun(c):
        return _objective_L4(c)

    def jac(c):
        return _gradient_L4(c)

    def cons_fun(c):
        return np.sum(c * c) - 1.0

    def cons_jac(c):
        return 2.0 * c

    constraints = [{'type': 'eq', 'fun': cons_fun, 'jac': cons_jac}]

    best_c = c0.copy()
    best_val = fun(c0)

    # Stage 1: SLSQP for initial constraint satisfaction
    if time.time() < deadline - 10.0:
        try:
            res = minimize(fun, c0, jac=jac, method='SLSQP', constraints=constraints,
                           options={'maxiter': 500, 'ftol': 1e-12, 'disp': False})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
                best_c /= np.linalg.norm(best_c)
        except Exception:
            pass

    # Stage 2: Conjugate gradient on sphere (primary optimizer)
    if time.time() < deadline - 15.0:
        try:
            c_cg, val_cg = _conjugate_gradient_sphere(best_c, max_iter=5000, tol=1e-17, deadline=deadline)
            if val_cg < best_val:
                best_val = val_cg
                best_c = c_cg.copy()
        except Exception:
            pass

    # Stage 3: L-BFGS-B on unconstrained scale-invariant objective
    if time.time() < deadline - 10.0:
        try:
            res2 = minimize(_objective_unconstrained, best_c, jac=_gradient_unconstrained,
                            method='L-BFGS-B', options={'maxiter': 3000, 'ftol': 1e-16, 'disp': False})
            if res2.fun < best_val:
                nc = res2.x / np.linalg.norm(res2.x)
                val_check = _objective_L4(nc)
                if val_check < best_val:
                    best_val = val_check
                    best_c = nc
        except Exception:
            pass

    # Stage 4: Second conjugate gradient pass from improved point
    if time.time() < deadline - 10.0:
        try:
            c_cg2, val_cg2 = _conjugate_gradient_sphere(best_c, max_iter=5000, tol=1e-18, deadline=deadline)
            if val_cg2 < best_val:
                best_val = val_cg2
                best_c = c_cg2.copy()
        except Exception:
            pass

    # Stage 5: Final high-precision gradient descent polish
    if time.time() < deadline - 3.0:
        try:
            c_polished, val_polished = _polish_gradient_descent(best_c, max_iter=10000, tol=1e-18, deadline=deadline)
            if val_polished < best_val:
                best_val = val_polished
                best_c = c_polished.copy()
        except Exception:
            pass

    return best_c, best_val


def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin before 630s kill

    # Generate diverse starting points
    rng = np.random.default_rng(2024)

    starts = []

    # All ones (symmetric)
    c1 = np.ones(_NC)
    c1 /= np.linalg.norm(c1)
    starts.append(c1)

    # Even m only
    c2 = np.array([1.0 if (i - _K) % 2 == 0 else 0.0 for i in range(_NC)])
    n2 = np.linalg.norm(c2)
    if n2 > 1e-12:
        c2 /= n2
    starts.append(c2)

    # Odd m only
    c3 = np.array([1.0 if (i - _K) % 2 != 0 else 0.0 for i in range(_NC)])
    n3 = np.linalg.norm(c3)
    if n3 > 1e-12:
        c3 /= n3
    starts.append(c3)

    # m >= 0 positive, m < 0 negative
    c4 = np.array([1.0 if m >= 0 else -1.0 for m in range(-_K, _K + 1)])
    c4 /= np.linalg.norm(c4)
    starts.append(c4)

    # m >= 0 negative, m < 0 positive
    c5 = np.array([-1.0 if m >= 0 else 1.0 for m in range(-_K, _K + 1)])
    c5 /= np.linalg.norm(c5)
    starts.append(c5)

    # Random starts with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        cr = rng.standard_normal(_NC)
        cr /= np.linalg.norm(cr)
        starts.append(cr)

    # Structured: cosine-weighted
    c6 = np.array([np.cos(np.pi * (m + _K) / (2 * _K)) for m in range(-_K, _K + 1)])
    n6 = np.linalg.norm(c6)
    if n6 > 1e-12:
        c6 /= n6
    starts.append(c6)

    # Structured: m^2 weighted
    c7 = np.array([float(m * m) for m in range(-_K, _K + 1)])
    n7 = np.linalg.norm(c7)
    if n7 > 1e-12:
        c7 /= n7
    starts.append(c7)

    # Structured: alternating with decay
    c8 = np.array([(-1) ** m * np.exp(-0.1 * abs(m)) for m in range(-_K, _K + 1)])
    n8 = np.linalg.norm(c8)
    if n8 > 1e-12:
        c8 /= n8
    starts.append(c8)

    best_c = starts[0].copy()
    best_val = _objective_L4(best_c)

    # Optimize from each starting point
    for i, c0 in enumerate(starts):
        if time.time() > deadline - 20.0:
            break
        try:
            c_opt, val_opt = _optimize_sphere(c0, deadline)
            if val_opt < best_val:
                best_val = val_opt
                best_c = c_opt.copy()
        except Exception:
            continue

    # Final global polish from the best found
    if time.time() < deadline - 5.0:
        try:
            c_final, val_final = _conjugate_gradient_sphere(best_c, max_iter=10000, tol=1e-18, deadline=deadline)
            if val_final < best_val:
                best_val = val_final
                best_c = c_final.copy()
        except Exception:
            pass

    if time.time() < deadline - 2.0:
        try:
            c_polished, val_polished = _polish_gradient_descent(best_c, max_iter=20000, tol=1e-19, deadline=deadline)
            if val_polished < best_val:
                best_val = val_polished
                best_c = c_polished.copy()
        except Exception:
            pass

    # Final safety check
    best_c = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(best_c)):
        best_c = starts[0]
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)

    return best_c

# EVOLVE-BLOCK-END
