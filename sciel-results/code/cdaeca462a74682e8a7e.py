"""Analytic uncertainty bound with exact gradient + multi-phase optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195  # Safety margin from 210s limit

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n-1)!! * sqrt(pi) / 2^n
def _precompute_I():
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    I[0] = sp
    for n in range(1, 14):
        # I_n = (2n-1)/2 * I_{n-1}
        I[n] = (2.0*n - 1.0) / 2.0 * I[n-1]
    return I

_I = _precompute_I()

# Constraint: P(0) = a0 + 3*a4 + 105*a8 + 945*a12 = 0
# So a12 = -(a0 + 3*a4 + 105*a8) / 945

# Coefficient matrix: p_k in terms of a0, a4, a8
# p0 = a0 + 3a4 + 105a8 + 945a12 = 0 (by constraint)
# p1 = -6a4 - 420a8 - 7560a12
# p2 = a4 + 210a8 + 13860a12
# p3 = -28a8 - 6930a12
# p4 = a8 + 1155a12
# p5 = -66a12
# p6 = a12

_da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

# dp_k/da_c for c=0,1,2 (a0, a4, a8), k=0..6
_dp_da = np.zeros((7, 3))
# p0: 0 by constraint, but derivative matters for gradient
_dp_da[0, 0] = 1.0 + 945.0 * _da12[0]
_dp_da[0, 1] = 3.0 + 945.0 * _da12[1]
_dp_da[0, 2] = 105.0 + 945.0 * _da12[2]
# p1
_dp_da[1, 0] = -7560.0 * _da12[0]
_dp_da[1, 1] = -6.0 - 7560.0 * _da12[1]
_dp_da[1, 2] = -420.0 - 7560.0 * _da12[2]
# p2
_dp_da[2, 0] = 13860.0 * _da12[0]
_dp_da[2, 1] = 1.0 + 13860.0 * _da12[1]
_dp_da[2, 2] = 210.0 + 13860.0 * _da12[2]
# p3
_dp_da[3, 0] = -6930.0 * _da12[0]
_dp_da[3, 1] = -6930.0 * _da12[1]
_dp_da[3, 2] = -28.0 - 6930.0 * _da12[2]
# p4
_dp_da[4, 0] = 1155.0 * _da12[0]
_dp_da[4, 1] = 1155.0 * _da12[1]
_dp_da[4, 2] = 1.0 + 1155.0 * _da12[2]
# p5
_dp_da[5, 0] = -66.0 * _da12[0]
_dp_da[5, 1] = -66.0 * _da12[1]
_dp_da[5, 2] = -66.0 * _da12[2]
# p6
_dp_da[6, 0] = 1.0 * _da12[0]
_dp_da[6, 1] = 1.0 * _da12[1]
_dp_da[6, 2] = 1.0 * _da12[2]

# Signs for q_k = (-1)^k p_k
_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

# Index matrix for i+j
_idx = np.arange(7)
_ij = _idx[:, None] + _idx[None, :]

# Precompute I matrices
_I0 = _I[_ij]        # I[i+j]
_I1 = _I[_ij + 1]    # I[i+j+1]
_I2 = _I[_ij + 2]    # I[i+j+2]

_tp = 2.0 * np.pi


def _compute_bound_and_grad(params):
    """
    Compute uncertainty product bound and its analytic gradient.
    Returns (bound, grad) where grad is 3-vector.
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0

    # Compute p vector
    p = np.array([
        a0 + 3.0*a4 + 105.0*a8 + 945.0*a12,  # = 0 by constraint
        -6.0*a4 - 420.0*a8 - 7560.0*a12,
        a4 + 210.0*a8 + 13860.0*a12,
        -28.0*a8 - 6930.0*a12,
        a8 + 1155.0*a12,
        -66.0*a12,
        a12
    ])

    # Moments for f: m_k = sum_{i,j} p_i p_j I[k+i+j]
    m0 = p @ _I0 @ p
    m1 = p @ _I1 @ p
    m2 = p @ _I2 @ p

    # Moments for F (Fourier transform): q_k = (-1)^k p_k
    q = p * _signs
    n0 = _tp * (q @ _I0 @ q)
    n1 = _tp * (q @ _I1 @ q)
    n2 = _tp * (q @ _I2 @ q)

    # Check validity
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])

    # Variance-like quantities
    vf = m1 / m0
    vF = n1 / n0

    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])

    bound = np.sqrt(vf * vF)

    # Compute gradient
    # Precompute Ip and Iq for efficiency
    Ip0 = _I0 @ p
    Ip1 = _I1 @ p
    Ip2 = _I2 @ p
    Iq0 = _I0 @ q
    Iq1 = _I1 @ q
    Iq2 = _I2 @ q

    grad = np.zeros(3)
    for c in range(3):
        dp = _dp_da[:, c]
        dq = dp * _signs

        dm0 = 2.0 * dp @ Ip0
        dm1 = 2.0 * dp @ Ip1

        dn0 = 2.0 * _tp * dq @ Iq0
        dn1 = 2.0 * _tp * dq @ Iq1

        # d(vf)/dc = (dm1*m0 - m1*dm0) / m0^2
        dvm2 = (dm1 * m0 - m1 * dm0) / (m0 * m0)
        dvn2 = (dn1 * n0 - n1 * dn0) / (n0 * n0)

        # d(bound)/dc = (vF*d(vf) + vf*d(vF)) / (2*bound)
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)

    return bound, grad


def _objective(params):
    val, _ = _compute_bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _compute_bound_and_grad(params)
    return val, grad


def _newton_polish(params, max_iter=100, tol=1e-14):
    """Newton's method with exact Hessian for final polish."""
    x = np.array(params, dtype=float)
    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
        f, g = _compute_bound_and_grad(x)
        if not np.isfinite(f):
            break
        if np.linalg.norm(g) < tol:
            break

        # Compute Hessian via finite differences of gradient
        H = np.zeros((3, 3))
        h = 1e-7
        for i in range(3):
            for j in range(i, 3):
                if i == j:
                    xp = x.copy(); xp[i] += h
                    xm = x.copy(); xm[i] -= h
                    _, gp = _compute_bound_and_grad(xp)
                    _, gm = _compute_bound_and_grad(xm)
                    H[i, j] = (gp[i] - gm[i]) / (2.0 * h)
                else:
                    xpp = x.copy(); xpp[i] += h; xpp[j] += h
                    xpm = x.copy(); xpm[i] += h; xpm[j] -= h
                    xmp = x.copy(); xmp[i] -= h; xmp[j] += h
                    xmm = x.copy(); xmm[i] -= h; xmm[j] -= h
                    _, gpp = _compute_bound_and_grad(xpp)
                    _, gpm = _compute_bound_and_grad(xpm)
                    _, gmp = _compute_bound_and_grad(xmp)
                    _, gmm = _compute_bound_and_grad(xmm)
                    H[i, j] = (gpp[i] - gpm[i] - gmp[i] + gmm[i]) / (4.0 * h * h)
                    H[j, i] = H[i, j]

        # Solve H dx = -g
        try:
            # Add small regularization for stability
            H_reg = H + 1e-10 * np.eye(3)
            dx = np.linalg.solve(H_reg, -g)
        except np.linalg.LinAlgError:
            break

        # Line search
        alpha = 1.0
        for _ in range(30):
            x_new = x + alpha * dx
            f_new, _ = _compute_bound_and_grad(x_new)
            if np.isfinite(f_new) and f_new < f:
                x = x_new
                break
            alpha *= 0.5
        else:
            break

    return x


def _line_search_polish(params, max_iter=10000, tol=1e-15):
    """Goldstein line search along gradient direction."""
    x = np.array(params, dtype=float)
    f, g = _compute_bound_and_grad(x)
    if not np.isfinite(f):
        return x

    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        if np.linalg.norm(g) < tol:
            break

        # Gradient direction
        direction = -g
        step = 1.0

        # Backtracking line search
        c1 = 1e-4
        for _ in range(50):
            x_new = x + step * direction
            f_new, g_new = _compute_bound_and_grad(x_new)
            if np.isfinite(f_new) and f_new <= f + c1 * step * (g @ direction):
                x = x_new
                f = f_new
                g = g_new
                break
            step *= 0.5
        else:
            break

    return x


def construct():
    """Main optimization routine."""
    # Diverse starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-1.2, 0.03, 0.0002]),
        np.array([-0.8, 0.05, 0.0005]),
        np.array([-2.5, 0.07, 0.0007]),
        np.array([-1.8, 0.06, 0.0006]),
        np.array([-1.13, 0.03, 0.0002]),
        np.array([-1.13, 0.05, 0.0004]),
        np.array([-1.0, 0.03, 0.0002]),
        np.array([-1.3, 0.05, 0.0004]),
        np.array([-0.7, 0.04, 0.0003]),
        np.array([-1.5, 0.03, 0.0002]),
        np.array([-0.3, 0.01, 0.0001]),
        np.array([-2.2, 0.09, 0.0009]),
        np.array([-1.13, 0.04, 0.0003]),
        np.array([-1.13, 0.04, 0.00035]),
    ]

    # Add random starts
    rng = np.random.default_rng(42)
    for i in range(30):
        scale = 0.1 + 0.5 * i / 30
        starts.append(rng.normal(0, scale, 3))

    best_val = np.inf
    best_params = None

    # Phase 1: L-BFGS-B from all starts
    for start in starts:
        if time.time() > _DEADLINE - 60:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])

    # Phase 2: Tighter L-BFGS-B on best
    if time.time() < _DEADLINE - 40:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-17, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 3: SLSQP polish
    if time.time() < _DEADLINE - 30:
        try:
            res = minimize(_objective_grad, best_params, method='SLSQP',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 4: Nelder-Mead for robustness
    if time.time() < _DEADLINE - 20:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 30000, 'xatol': 1e-14, 'fatol': 1e-17,
                                   'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 5: Newton polish
    if time.time() < _DEADLINE - 15:
        try:
            x_newton = _newton_polish(best_params, max_iter=50)
            f_newton, _ = _compute_bound_and_grad(x_newton)
            if f_newton < best_val and np.isfinite(f_newton):
                best_val = f_newton
                best_params = x_newton.copy()
        except Exception:
            pass

    # Phase 6: Line search polish
    if time.time() < _DEADLINE - 10:
        try:
            x_ls = _line_search_polish(best_params, max_iter=5000)
            f_ls, _ = _compute_bound_and_grad(x_ls)
            if f_ls < best_val and np.isfinite(f_ls):
                best_val = f_ls
                best_params = x_ls.copy()
        except Exception:
            pass

    # Phase 7: Final L-BFGS-B with tightest tolerances
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-17})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 8: Final Newton
    if time.time() < _DEADLINE - 2:
        try:
            x_final = _newton_polish(best_params, max_iter=30)
            f_final, _ = _compute_bound_and_grad(x_final)
            if f_final < best_val and np.isfinite(f_final):
                best_val = f_final
                best_params = x_final.copy()
        except Exception:
            pass

    return np.array(best_params)
# EVOLVE-BLOCK-END
