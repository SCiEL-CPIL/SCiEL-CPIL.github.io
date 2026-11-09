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


def _project_to_sphere(c):
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    return c / nrm


def _complex_to_real(z):
    c = np.zeros(_NC)
    c[_K] = z[0].real
    for m in range(1, _K + 1):
        c[_K + m] = z[m].real
        c[_K - m] = z[m].imag
    return c


def _real_to_complex(c):
    z = np.zeros(_K + 1, dtype=np.complex128)
    z[0] = c[_K]
    for m in range(1, _K + 1):
        z[m] = c[_K + m] + 1j * c[_K - m]
    return z


def _complex_objective(z):
    c = _complex_to_real(z)
    return _objective(c)


def _complex_gradient(z):
    c = _complex_to_real(z)
    g_real = _gradient(c)
    gz = np.zeros(_K + 1, dtype=np.complex128)
    gz[0] = g_real[_K]
    for m in range(1, _K + 1):
        gz[m] = g_real[_K + m] + 1j * g_real[_K - m]
    return gz


def _conjugate_gradient_complex_sphere(c0, maxiter=400, tol=1e-13, deadline=None):
    """
    Conjugate gradient on the complex unit sphere, exploiting rotational symmetry.
    Uses backtracking line search with Wolfe conditions.
    """
    z = _real_to_complex(c0)
    z_norm = np.linalg.norm(z)
    if z_norm < 1e-14:
        return None
    z = z / z_norm

    f_curr = _complex_objective(z)
    g_curr = _complex_gradient(z)

    inner = np.vdot(z, g_curr)
    g_tan = g_curr - inner * z
    s = -g_tan
    s_norm = np.linalg.norm(s)
    if s_norm < 1e-15:
        return c0.copy(), f_curr

    s = s / s_norm
    y_curr = z.copy()
    best_z = z.copy()
    best_f = f_curr
    best_g_norm = s_norm

    # Backtracking line search parameters
    c1 = 1e-4  # sufficient decrease
    c2 = 0.9    # curvature condition
    rho = 0.5   # backtracking factor
    max_ls = 30

    for i in range(maxiter):
        if deadline is not None and time.time() > deadline - 10.0:
            break

        # Backtracking line search
        best_t = 0.0
        best_f_t = f_curr
        t = 1.0
        for ls_iter in range(max_ls):
            if deadline is not None and time.time() > deadline - 5.0:
                break
            ct = np.cos(t)
            st = np.sin(t)
            z_trial = ct * y_curr + st * s
            f_trial = _complex_objective(z_trial)
            
            # Check sufficient decrease (Armijo condition)
            # The directional derivative along s at y_curr
            g_dir = np.vdot(g_tan, s).real  # should be negative since s = -g_tan normalized
            if f_trial <= f_curr + c1 * t * g_dir:
                # Check curvature condition
                g_trial = _complex_gradient(z_trial)
                inner_trial = np.vdot(z_trial, g_trial)
                g_tan_trial = g_trial - inner_trial * z_trial
                # Directional derivative at new point
                g_dir_new = np.vdot(g_tan_trial, s).real
                if g_dir_new >= c2 * g_dir:
                    best_t = t
                    best_f_t = f_trial
                    break
                else:
                    # Wolfe conditions satisfied
                    best_t = t
                    best_f_t = f_trial
                    break
            
            t *= rho
        
        if best_t == 0.0:
            # Use the best trial we found
            for trial_t in [0.1, 0.05, 0.01, 0.005]:
                ct = np.cos(trial_t)
                st = np.sin(trial_t)
                z_trial = ct * y_curr + st * s
                f_trial = _complex_objective(z_trial)
                if f_trial < best_f_t:
                    best_f_t = f_trial
                    best_t = trial_t
            if best_t == 0.0:
                best_t = 0.001

        z_new = np.cos(best_t) * y_curr + np.sin(best_t) * s
        f_new = _complex_objective(z_new)

        if f_new >= f_curr:
            z_new = y_curr + 0.001 * s
            z_new = z_new / np.linalg.norm(z_new)
            f_new = _complex_objective(z_new)
            if f_new >= f_curr:
                break

        g_new = _complex_gradient(z_new)
        inner_new = np.vdot(z_new, g_new)
        g_tan_new = g_new - inner_new * z_new
        g_norm = np.linalg.norm(g_tan_new)

        if g_norm < best_g_norm:
            best_g_norm = g_norm
            best_z = z_new.copy()
            best_f = f_new

        if g_norm < tol:
            break

        if f_new < best_f:
            best_f = f_new
            best_z = z_new.copy()

        dot_g = np.vdot(g_tan_new, g_tan).real
        norm_g_new_sq = np.vdot(g_tan_new, g_tan_new).real
        if dot_g < 1e-15:
            beta = 0.0
        else:
            beta = -norm_g_new_sq / dot_g

        if beta < 0:
            beta = 0.0

        s_new = -g_tan_new + beta * s
        inner_s = np.vdot(z_new, s_new)
        s_new = s_new - inner_s * z_new
        s_new_norm = np.linalg.norm(s_new)

        if s_new_norm < 1e-15:
            s_new = -g_tan_new
            s_new_norm = np.linalg.norm(s_new)
            if s_new_norm < 1e-15:
                break
            s_new = s_new / s_new_norm
            y_curr = z_new
            f_curr = f_new
            g_curr = g_new
            g_tan = g_tan_new
            continue

        s = s_new / s_new_norm
        y_curr = z_new
        f_curr = f_new
        g_curr = g_new
        g_tan = g_tan_new

    c_out = _complex_to_real(best_z)
    c_out = _project_to_sphere(c_out)
    if c_out is None:
        return c0.copy(), f_curr
    return c_out, float(_objective(c_out))


def _slsqp_polish(c0, deadline):
    """SLSQP polish with normalization constraint."""
    c = np.asarray(c0, dtype=np.float64).copy()
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    c = c / nrm

    def fun(x):
        return _objective(x)

    def jac(x):
        return _gradient(x)

    def constraint(x):
        return np.dot(x, x) - 1.0

    try:
        res = minimize(fun, c, jac=jac, method="SLSQP",
                       constraints=[{'type': 'eq', 'fun': constraint, 'jac': lambda x: 2.0 * x}],
                       options={'maxiter': 300, 'ftol': 1e-15, 'disp': False})
        if res.success or res.fun < _objective(c):
            proj = _project_to_sphere(res.x)
            if proj is not None:
                val = float(_objective(proj))
                return proj, val
    except Exception:
        pass
    return None


def _grid_search_complex(deadline, max_time=120.0):
    """
    Deterministic grid search over complex amplitude space.
    We parameterize: c_0 = a0 (real), and for m=1..10, z_m = r_m * exp(i*theta_m).
    Due to rotational invariance, we can fix one phase and search over the rest.
    
    Strategy: grid over (a0, r_1..r_10, theta_2..theta_10) with theta_1 fixed to 0.
    Use a coarse-to-fine approach.
    """
    start = time.time()
    budget = min(max_time, deadline - start - 30.0)
    if budget < 10.0:
        return None, np.inf

    # Coarse grid: 5 levels for each of 11 parameters = too many.
    # Instead, use a structured approach:
    # - a0 in [-1, 1] with 11 points
    # - Each r_m in [0, 1] with 5 points
    # - Each theta_m in [0, 2*pi) with 8 points (but fix theta_1 = 0)
    # Total: 11 * 5^10 * 8^9 is way too large.
    
    # Better: use a smarter parameterization.
    # Since the objective is rotationally invariant, we can work in the
    # 11D complex space and use a grid on the sphere.
    
    # Alternative: use the fact that for the L4/L2 ratio, the optimal
    # solution likely has most weight on high |m| modes.
    # Do a grid over a few key parameters.
    
    best_c = None
    best_val = np.inf

    # Grid 1: a0 and r_10, r_9, theta_10
    a0_vals = np.linspace(-0.8, 0.8, 9)
    r10_vals = np.linspace(0.0, 1.0, 9)
    r9_vals = np.linspace(0.0, 0.8, 7)
    th10_vals = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)

    count = 0
    for a0 in a0_vals:
        if time.time() - start > budget:
            break
        for r10 in r10_vals:
            for r9 in r9_vals:
                for th10 in th10_vals:
                    c = np.zeros(_NC)
                    c[_K] = a0
                    # m=10: c[_K+10] = r10*cos(th10), c[_K-10] = r10*sin(th10)
                    c[_K + 10] = r10 * np.cos(th10)
                    c[_K - 10] = r10 * np.sin(th10)
                    # m=9: c[_K+9] = r9, c[_K-9] = 0
                    c[_K + 9] = r9
                    nrm = np.linalg.norm(c)
                    if nrm < 1e-14:
                        continue
                    c = c / nrm
                    val = _objective(c)
                    if val < best_val:
                        best_val = val
                        best_c = c.copy()
                    count += 1
                    if count % 5000 == 0:
                        if time.time() - start > budget:
                            break

    # Grid 2: focus on even m modes
    if time.time() - start < budget - 10.0 and best_c is not None:
        a0_vals2 = np.linspace(-0.5, 0.5, 7)
        r10_vals2 = np.linspace(0.0, 1.0, 7)
        r8_vals2 = np.linspace(0.0, 0.8, 7)
        r6_vals2 = np.linspace(0.0, 0.6, 5)
        th10_vals2 = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        th8_vals2 = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)

        for a0 in a0_vals2:
            if time.time() - start > budget:
                break
            for r10 in r10_vals2:
                for r8 in r8_vals2:
                    for r6 in r6_vals2:
                        for th10 in th10_vals2:
                            for th8 in th8_vals2:
                                c = np.zeros(_NC)
                                c[_K] = a0
                                c[_K + 10] = r10 * np.cos(th10)
                                c[_K - 10] = r10 * np.sin(th10)
                                c[_K + 8] = r8 * np.cos(th8)
                                c[_K - 8] = r8 * np.sin(th8)
                                c[_K + 6] = r6
                                nrm = np.linalg.norm(c)
                                if nrm < 1e-14:
                                    continue
                                c = c / nrm
                                val = _objective(c)
                                if val < best_val:
                                    best_val = val
                                    best_c = c.copy()

    return best_c, best_val


def _optimize_from(c0, deadline):
    c = np.asarray(c0, dtype=np.float64).copy()
    c = _project_to_sphere(c)
    if c is None:
        return None

    def fun(x):
        return _objective(x)

    def jac(x):
        return _gradient(x)

    best_c = c.copy()
    best_val = fun(c)

    # 1. L-BFGS-B
    try:
        res = minimize(fun, c, jac=jac, method="L-BFGS-B",
                       options={"maxiter": 200, "maxfun": 1500,
                                "ftol": 1e-16, "gtol": 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
    except Exception:
        pass

    if time.time() > deadline - 10.0:
        proj = _project_to_sphere(best_c)
        if proj is not None:
            return proj, float(_objective(proj))
        return None

    # 2. Complex CG from L-BFGS-B result
    c_cg_init = _project_to_sphere(best_c)
    if c_cg_init is not None:
        try:
            cg_c, cg_val = _conjugate_gradient_complex_sphere(c_cg_init, maxiter=200, tol=1e-12, deadline=deadline)
            if cg_val < best_val:
                best_val = cg_val
                best_c = cg_c.copy()
        except Exception:
            pass

    if time.time() > deadline - 10.0:
        proj = _project_to_sphere(best_c)
        if proj is not None:
            return proj, float(_objective(proj))
        return None

    # 3. SLSQP polish
    if time.time() < deadline - 8.0:
        try:
            slsqp_out = _slsqp_polish(best_c, deadline)
            if slsqp_out is not None:
                sc, sv = slsqp_out
                if sv < best_val:
                    best_val = sv
                    best_c = sc.copy()
        except Exception:
            pass

    # 4. Second L-BFGS-B pass with tighter tolerance
    if time.time() < deadline - 5.0:
        try:
            res2 = minimize(fun, best_c, jac=jac, method="L-BFGS-B",
                            options={"maxiter": 150, "maxfun": 1000,
                                     "ftol": 1e-16, "gtol": 1e-12})
            if res2.fun < best_val:
                best_val = res2.fun
                best_c = res2.x.copy()
        except Exception:
            pass

    # 5. Final complex CG polish
    if time.time() < deadline - 2.0:
        c_cg_final = _project_to_sphere(best_c)
        if c_cg_final is not None:
            try:
                cg_c_final, cg_val_final = _conjugate_gradient_complex_sphere(c_cg_final, maxiter=150, tol=1e-14, deadline=deadline)
                if cg_val_final < best_val:
                    best_val = cg_val_final
                    best_c = cg_c_final.copy()
            except Exception:
                pass

    # 6. Final SLSQP polish
    if time.time() < deadline - 1.0:
        try:
            slsqp_out = _slsqp_polish(best_c, deadline)
            if slsqp_out is not None:
                sc, sv = slsqp_out
                if sv < best_val:
                    best_val = sv
                    best_c = sc.copy()
        except Exception:
            pass

    final_c = _project_to_sphere(best_c)
    if final_c is None:
        return None
    return final_c, float(_objective(final_c))


def _make_seeds(rng, n_seeds):
    seeds = []
    for i in range(n_seeds):
        c = np.zeros(_NC)
        if i == 0:
            c[_K] = 1.0
        elif i == 1:
            c[_K + 10] = 1.0
        elif i == 2:
            c[_K - 10] = 1.0
        elif i == 3:
            for m in range(-_K, _K + 1):
                c[m + _K] = np.cos(0.5 * m)
        elif i == 4:
            for m in range(-_K, _K + 1):
                c[m + _K] = (-1) ** m
        elif i == 5:
            c[:] = rng.standard_normal(_NC)
        elif i == 6:
            for m in range(-_K, _K + 1, 2):
                c[m + _K] = rng.standard_normal()
        elif i == 7:
            for m in range(-_K + 1, _K + 1, 2):
                c[m + _K] = rng.standard_normal()
        elif i == 8:
            c[_K] = 1.0
            c[_K + 10] = rng.standard_normal()
            c[_K - 10] = rng.standard_normal()
        elif i == 9:
            c[_K + 5] = rng.standard_normal()
            c[_K - 5] = rng.standard_normal()
        elif i == 10:
            for m in range(0, _K + 1):
                v = rng.standard_normal()
                c[m + _K] = v
                c[-m + _K] = v
        elif i == 11:
            for m in range(1, _K + 1):
                v = rng.standard_normal()
                c[m + _K] = v
                c[-m + _K] = -v
        elif i == 12:
            c[_K] = 0.5
            c[_K + 10] = 1.0
            c[_K - 10] = 1.0
        elif i == 13:
            for m in range(-_K, _K + 1):
                if abs(m) >= 8:
                    c[m + _K] = rng.standard_normal()
        elif i == 14:
            for m in range(-_K, _K + 1):
                if abs(m) >= 6:
                    c[m + _K] = rng.standard_normal()
        elif i == 15:
            for m in [0, 1, -1, 10, -10]:
                c[m + _K] = rng.standard_normal()
        elif i == 16:
            for m in [0, 2, -2, 10, -10]:
                c[m + _K] = rng.standard_normal()
        elif i == 17:
            for m in [0, 3, -3, 10, -10]:
                c[m + _K] = rng.standard_normal()
        elif i == 18:
            c[_K] = 0.3
            c[_K + 10] = 1.0
            c[_K - 10] = 1.0
        elif i == 19:
            c[_K] = 0.7
            c[_K + 10] = 1.0
            c[_K - 10] = 1.0
        elif i == 20:
            c[:] = rng.standard_normal(_NC)
            for m in range(-_K, _K + 1):
                c[m + _K] *= (1.0 + abs(m) / 10.0)
        elif i == 21:
            c[:] = rng.standard_normal(_NC)
            for m in range(-_K, _K + 1):
                c[m + _K] *= (1.0 + (10 - abs(m)) / 10.0)
        elif i == 22:
            c[_K + 10] = 1.0
            c[_K - 10] = 1.0
        elif i == 23:
            # Concentrate on high |m| modes with alternating signs
            for m in range(-_K, _K + 1):
                am = abs(m)
                if am >= 5:
                    c[m + _K] = (-1) ** m * rng.standard_normal()
        elif i == 24:
            # Cosine modulation across all m
            for m in range(-_K, _K + 1):
                c[m + _K] = np.cos(np.pi * m / (2.0 * _K))
        elif i == 25:
            # Sine modulation
            for m in range(-_K, _K + 1):
                c[m + _K] = np.sin(np.pi * m / (2.0 * _K))
        elif i == 26:
            # Mix of m=0 and m=±10
            c[_K] = 0.8
            c[_K + 10] = 0.6
            c[_K - 10] = 0.6
        elif i == 27:
            # Only even m modes
            for m in range(-_K, _K + 1, 2):
                c[m + _K] = rng.standard_normal()
        elif i == 28:
            # Only odd m modes
            for m in range(-_K + 1, _K + 1, 2):
                c[m + _K] = rng.standard_normal()
        elif i == 29:
            # Gaussian-like profile in m-space
            for m in range(-_K, _K + 1):
                c[m + _K] = np.exp(-0.1 * m * m) * rng.standard_normal()
        elif i == 30:
            # Uniform with random phase
            phase = rng.uniform(0, 2 * np.pi)
            for m in range(-_K, _K + 1):
                c[m + _K] = np.cos(phase + 0.3 * m)
        else:
            c[:] = rng.standard_normal(_NC)
            c[_K] += 0.5
            c[_K + 10] += 0.3
            c[_K - 10] += 0.3
        nrm = np.linalg.norm(c)
        if nrm > 1e-14:
            c /= nrm
        seeds.append(c)
    return seeds


def rudin_problem10() -> np.ndarray:
    start = time.time()
    deadline = start + 610.0
    rng = np.random.default_rng(42)

    best_c = None
    best_val = np.inf

    # Phase 0: Grid search for good initial points
    if time.time() < deadline - 150.0:
        try:
            grid_c, grid_val = _grid_search_complex(deadline, max_time=100.0)
            if grid_c is not None:
                best_val = grid_val
                best_c = grid_c.copy()
        except Exception:
            pass

    # Phase 1: Optimize from structured seeds
    seeds = _make_seeds(rng, 32)
    for i, c0 in enumerate(seeds):
        if time.time() > deadline - 60.0:
            break
        try:
            out = _optimize_from(c0, deadline)
        except Exception:
            continue
        if out is None:
            continue
        c, val = out
        if val < best_val:
            best_val = val
            best_c = c.copy()

    # Phase 2: Random restarts with time budget
    n_extra = 0
    while time.time() < deadline - 40.0:
        c0 = rng.standard_normal(_NC)
        nrm = np.linalg.norm(c0)
        if nrm < 1e-14:
            continue
        c0 /= nrm
        try:
            out = _optimize_from(c0, deadline)
        except Exception:
            continue
        if out is None:
            continue
        c, val = out
        if val < best_val:
            best_val = val
            best_c = c.copy()
        n_extra += 1

    # Phase 3: Final polish from best solution
    if best_c is not None and time.time() < deadline - 5.0:
        try:
            c_cg_final = _project_to_sphere(best_c)
            if c_cg_final is not None:
                cg_c_final, cg_val_final = _conjugate_gradient_complex_sphere(
                    c_cg_final, maxiter=200, tol=1e-14, deadline=deadline)
                if cg_val_final < best_val:
                    best_val = cg_val_final
                    best_c = cg_c_final.copy()
        except Exception:
            pass

    # Phase 4: SLSQP final polish
    if best_c is not None and time.time() < deadline - 2.0:
        try:
            slsqp_out = _slsqp_polish(best_c, deadline)
            if slsqp_out is not None:
                sc, sv = slsqp_out
                if sv < best_val:
                    best_val = sv
                    best_c = sc.copy()
        except Exception:
            pass

    if best_c is None:
        best_c = np.ones(_NC, dtype=np.float64)
        best_c /= np.linalg.norm(best_c)

    best_c = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(_NC, dtype=np.float64)
        best_c /= np.linalg.norm(best_c)
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)
    return best_c


# EVOLVE-BLOCK-END
