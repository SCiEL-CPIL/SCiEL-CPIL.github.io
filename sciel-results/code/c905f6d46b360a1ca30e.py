# EVOLVE-BLOCK-START
import time
import numpy as np

_K = 10
_N = 2 * _K + 1  # 21

# Precompute quadrature and the 4th-moment tensor once at import time.
def _build_tensors():
    n_t, n_p = 70, 130
    # Gauss-Legendre in cos(theta) on [-1,1]
    x, w = np.polynomial.legendre.leggauss(n_t)
    # phi equally spaced
    phi = np.arange(n_p) * (2.0 * np.pi / n_p)
    dphi = 2.0 * np.pi / n_p
    # weights: dOmega = dphi * d(cos theta)
    W = np.outer(w, np.full(n_p, dphi))  # shape (n_t, n_p)
    pts = W.ravel()

    # Build Y_m for m=-10..10 at all points.
    # P_10^m(cos theta) with Condon-Shortley phase, via scipy.
    from scipy.special import lpmv
    ct = np.repeat(x, n_p)          # cos theta for each point
    st = np.sqrt(np.maximum(0.0, 1.0 - ct * ct))
    ph = np.tile(phi, n_t)

    Y = np.empty((_N, pts.size), dtype=np.float64)
    for m in range(-_K, _K + 1):
        am = abs(m)
        # N_m = sqrt(21/(4 pi) * (10-am)!/(10+am)!)
        import math
        N = np.sqrt(21.0 / (4.0 * np.pi) *
                    math.factorial(_K - am) / math.factorial(_K + am))
        P = lpmv(am, _K, ct)  # includes Condon-Shortley phase
        if m > 0:
            Y[m + _K] = np.sqrt(2.0) * N * P * np.cos(m * ph)
        elif m < 0:
            Y[m + _K] = np.sqrt(2.0) * N * P * np.sin(am * ph)
        else:
            Y[m + _K] = N * P
    # Verify orthonormality quickly (sanity, cheap)
    # G = Y @ diag(W) @ Y.T  should be ~ I
    Yw = Y * W.ravel()[None, :]
    G = Yw @ Y.T
    # T_ijkl = sum_p W_p Y_pi Y_pj Y_pk Y_pl
    # Build via outer products to keep memory reasonable: 21^4 = 194481 floats ~1.5MB
    T = np.einsum('pi,pj,pk,pl,p->ijkl', Y, Y, Y, Y, W.ravel(),
                  optimize=True)
    return T, Y, W.ravel()

_T, _Y, _W = _build_tensors()
_T = np.ascontiguousarray(_T)


def _ratio_and_grad(c):
    c = np.asarray(c, dtype=np.float64)
    s2 = float(c @ c)
    if s2 <= 1e-14:
        return np.inf, np.zeros_like(c)
    # f4 = c^T T c
    f4 = float(np.einsum('i,ijkl,j,k,l->', c, _T, c, c, c, optimize=True))
    # R = (f4/s2)^{1/4} / (s2/(4 pi))^{1/2}
    # log R = 0.25 (ln f4 - ln s2) - 0.5 (ln s2 - ln(4 pi))
    #        = 0.25 ln f4 - 0.75 ln s2 + const
    R = (f4 / s2) ** 0.25 / (s2 / (4.0 * np.pi)) ** 0.5
    # d f4 / dc_i = 4 * T_{i j k l} c_j c_k c_l  (T symmetric in all indices)
    dT = 4.0 * np.einsum('ijkl,j,k,l->i', _T, c, c, c, optimize=True)
    # d R / dc_i = R * ( (0.25 * dT_i / f4) - (0.75 * 2 c_i / s2) )
    grad = R * (0.25 * dT / f4 - 1.5 * c / s2)
    return R, grad


def _slsqp_solve(c0, deadline, max_iter=200):
    from scipy.optimize import minimize
    c0 = np.asarray(c0, dtype=np.float64)
    nrm = np.linalg.norm(c0)
    c0 = c0 / nrm if nrm > 0 else np.ones(_N) / np.sqrt(_N)

    def obj(c):
        return _ratio_and_grad(c)[0]

    def jac(c):
        return _ratio_and_grad(c)[1]

    cons = [{'type': 'eq',
             'fun': lambda c: float(c @ c) - 1.0,
             'jac': lambda c: 2.0 * c}]
    best = (obj(c0), c0.copy())
    try:
        res = minimize(obj, c0, jac=jac, method='SLSQP',
                       constraints=cons,
                       options={'maxiter': max_iter, 'ftol': 1e-14,
                                'disp': False})
        if res.success or (res.fun < best[0]):
            cc = np.asarray(res.x, dtype=np.float64)
            if np.isfinite(cc).all() and np.linalg.norm(cc) > 1e-12:
                r = obj(cc)
                if np.isfinite(r) and r < best[0]:
                    best = (r, cc)
    except Exception:
        pass
    return best


def _newton_polish(c, deadline, iters=60):
    """Minimize R on the sphere via projected Newton on log R with exact grad.
    Use a trust-region-ish damped step along -grad projected, then a 2nd-order
    correction using the Hessian of log R restricted to the tangent space."""
    c = np.asarray(c, dtype=np.float64).copy()
    c = c / np.linalg.norm(c)
    best = (float(np.inf), c.copy())

    def logR(c):
        c = np.asarray(c)
        s2 = float(c @ c)
        f4 = float(np.einsum('i,ijkl,j,k,l->', c, _T, c, c, c, optimize=True))
        return 0.25 * np.log(f4) - 0.75 * np.log(s2)

    def grad_logR(c):
        s2 = float(c @ c)
        f4 = float(np.einsum('i,ijkl,j,k,l->', c, _T, c, c, c, optimize=True))
        dT = 4.0 * np.einsum('ijkl,j,k,l->i', _T, c, c, c, optimize=True)
        return 0.25 * dT / f4 - 1.5 * c / s2

    for _ in range(iters):
        if time.time() > deadline:
            break
        g = grad_logR(c)
        # project gradient onto tangent space of sphere
        gt = g - (g @ c) * c
        if np.linalg.norm(gt) < 1e-13:
            break
        # Hessian of log R: H_ij = 0.25*(dT_i dT_j / f4^2 - d2T_ij / f4) - 1.5*(delta_ij/s2 - 2 c_i c_j / s2^2)
        dT = 4.0 * np.einsum('ijkl,j,k,l->i', _T, c, c, c, optimize=True)
        s2 = float(c @ c)
        f4 = float(np.einsum('i,ijkl,j,k,l->', c, _T, c, c, c, optimize=True))
        # d2T_ij = 12 * T_{i j k l} c_k c_l
        d2T = 12.0 * np.einsum('ijkl,k,l->ij', _T, c, c, optimize=True)
        H = 0.25 * (np.outer(dT, dT) / (f4 * f4) - d2T / f4) \
            - 1.5 * (np.eye(_N) / s2 - 2.0 * np.outer(c, c) / (s2 * s2))
        # restrict to tangent space
        P = np.eye(_N) - np.outer(c, c)
        Ht = P @ H @ P
        # solve Ht v = -gt
        try:
            # add small ridge for stability
            v = np.linalg.solve(Ht + 1e-9 * np.eye(_N), -gt)
        except np.linalg.LinAlgError:
            v = -gt
        # ensure step reduces logR via line search
        step_dir = v
        if np.linalg.norm(step_dir) < 1e-14:
            step_dir = -gt
        step_dir = step_dir - (step_dir @ c) * c  # reproject
        step_dir = step_dir / (np.linalg.norm(step_dir) + 1e-30)
        base_lr = 0.1
        accepted = False
        cur_logR = logR(c)
        for lr in [base_lr, base_lr*0.5, base_lr*0.25, base_lr*0.125,
                   base_lr*0.0625, base_lr*0.03125]:
            c_new = c + lr * step_dir
            c_new = c_new / np.linalg.norm(c_new)
            lr_logR = logR(c_new)
            if lr_logR < cur_logR:
                c = c_new
                cur_logR = lr_logR
                accepted = True
                break
        if not accepted:
            # fallback to steepest descent
            c_new = c - 0.05 * gt
            c_new = c_new / np.linalg.norm(c_new)
            if logR(c_new) < cur_logR:
                c = c_new
                cur_logR = logR(c_new)
            else:
                break
        r = (float(np.einsum('i,ijkl,j,k,l->', c, _T, c, c, c, optimize=True))
             / float(c @ c)) ** 0.25 / (float(c @ c) / (4.0 * np.pi)) ** 0.5
        if r < best[0]:
            best = (r, c.copy())
    return best


def rudin_problem10() -> np.ndarray:
    deadline = time.time() + 560.0  # safety margin before 630s kill
    rng = np.random.default_rng(2024)

    # Initial guesses: all-ones (baseline), plus random and structured seeds.
    seeds = [np.ones(_N, dtype=np.float64)]
    # structured: emphasize cos modes, sin modes, alternating
    seeds.append(np.array([(-1.0) ** (m) for m in range(-_K, _K + 1)],
                          dtype=np.float64))
    seeds.append(np.array([1.0 if m % 2 == 0 else 0.3 for m in range(-_K, _K + 1)],
                          dtype=np.float64))
    for _ in range(6):
        seeds.append(rng.standard_normal(_N))

    best = (np.inf, np.ones(_N, dtype=np.float64))
    for s in seeds:
        if time.time() > deadline - 60.0:
            break
        r0, c0 = _slsqp_solve(s, deadline)
        if r0 < best[0]:
            best = (r0, c0.copy())

    # Polish the best with Newton on the sphere
    if time.time() < deadline - 20.0:
        r1, c1 = _newton_polish(best[1], deadline)
        if r1 < best[0]:
            best = (r1, c1.copy())

    # Final: ensure validity
    c = np.asarray(best[1], dtype=np.float64)
    if not np.isfinite(c).all():
        c = np.ones(_N, dtype=np.float64)
    # scale to be safely within [1e-3, 1e5] on max|c|
    mx = float(np.max(np.abs(c)))
    if mx < 1e-3 or mx > 1e5 or mx == 0.0:
        target = 1.0
        if mx > 0:
            c = c * (target / mx)
        else:
            c = np.ones(_N, dtype=np.float64)
    return c


# EVOLVE-BLOCK-END
