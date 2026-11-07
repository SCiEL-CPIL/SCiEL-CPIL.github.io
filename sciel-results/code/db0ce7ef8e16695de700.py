"""Optimize the uncertainty-inequality constant via a smooth max-surrogate + adaptive root chase."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Physicists' Hermite polynomials (exact integer coefficients)
H0 = [1]
H4 = [16, -48, 12]                       # 16x^4 - 48x^2 + 12
H8 = [256, -3072, 11520, -13440, 1680]   # x^8..x^0
H12 = [4096, -98304, 1032192, -5242880, 12451840, -12058624, 3991680]

# g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12,  c3 = -(c0 + 12 c1 + 1680 c2)/3991680
# q(x) = g(x)/x^2 (even polynomial in x, degree 10).  q(u) = P(u) with u=x^2:
#   P(u) = a5 u^5 + a4 u^4 + a3 u^3 + a2 u^2 + a1 u + a0
_A5 = np.array([0.0, 0.0, 0.0, 4096.0])
_A4 = np.array([0.0, 0.0, 0.0, -98304.0])
_A3 = np.array([0.0, 256.0, 0.0, 1032192.0])
_A2 = np.array([0.0, -3072.0, 0.0, -5242880.0])
_A1 = np.array([16.0, 11520.0, 0.0, 12451840.0])
_A0 = np.array([12.0, 1680.0, 0.0, -12058624.0])


def _q_coefficients(c0, c1, c2):
    """Return (a5,a4,a3,a2,a1,a0) of P(u)=q(sqrt(u))."""
    c = np.array([c0, c1, c2, 0.0])
    c3 = -(c0 + 12.0 * c1 + 1680.0 * c2) / 3991680.0
    c = np.array([c0, c1, c2, c3])
    a5 = float(_A5 @ c)
    a4 = float(_A4 @ c)
    a3 = float(_A3 @ c)
    a2 = float(_A2 @ c)
    a1 = float(_A1 @ c)
    a0 = float(_A0 @ c)
    return np.array([a5, a4, a3, a2, a1, a0])


def _q_values(u, a):
    """P(u) for array u, a = [a5..a0]."""
    return np.polynomial.polynomial.polyval(u, a)


def _true_root(c0, c1, c2):
    """Largest sign-changing real root r of q(x); None if none / invalid."""
    a = _q_coefficients(c0, c1, c2)
    a5, a4, a3, a2, a1, a0 = a
    if abs(a5) < 1e-30 and abs(a4) < 1e-30:
        return None
    # q(x) = a5 x^10 + a4 x^8 + a3 x^6 + a2 x^4 + a1 x^2 + a0
    coeffs = np.array([a5, 0.0, a4, 0.0, a3, 0.0, a2, 0.0, a1, 0.0, a0])
    try:
        roots = np.roots(coeffs)
    except Exception:
        return None
    pos = []
    for z in roots:
        if abs(z.imag) < 1e-9 and z.real > 1e-9:
            pos.append(z.real)
    if not pos:
        return None
    pos.sort(reverse=True)
    for r in pos:
        eps = 1e-7 * max(1.0, abs(r))
        L = _q_values(np.array([r - eps]), a)[0]
        R = _q_values(np.array([r + eps]), a)[0]
        if L * R < 0:
            return r
    return None


def _max_q_on_interval(c, r_target, n=1600):
    """max_{x in [0,r_target]} q(x)."""
    a = _q_coefficients(*c)
    u = np.linspace(0.0, r_target * r_target, n)
    return float(np.max(_q_values(u, a)))


def _objective(c, r_target, n=1600):
    return _max_q_on_interval(c, r_target, n)


def _optimize_for_rtarget(r_target, c_init, deadline, n=1600):
    """Minimize max q on [0,r_target]; return best c and its max value."""
    from scipy.optimize import minimize
    best_c = np.array(c_init, dtype=float)
    best_val = _max_q_on_interval(best_c, r_target, n)
    bounds = [(-4.0, 4.0), (-2.0, 4.0), (-0.5, 0.5)]
    for method in ("L-BFGS-B", "Nelder-Mead"):
        if time.time() > deadline - 12.0:
            break
        try:
            res = minimize(_objective, best_c.copy(), args=(r_target, n),
                           method=method, bounds=bounds,
                           options=dict(maxiter=600, maxfev=4000,
                                        ftol=1e-12, gtol=1e-8))
            if res.fun < best_val:
                best_val = float(res.fun)
                best_c = np.array(res.x, dtype=float)
        except Exception:
            pass
    return best_c, best_val


def construct():
    start = time.time()
    deadline = start + 165.0  # safety margin before 210s kill

    # Seed
    c0s, c1s, c2s = -113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0
    seed_c = np.array([c0s, c1s, c2s])
    r_seed = _true_root(c0s, c1s, c2s)
    if r_seed is None:
        r_seed = 1.5
    best_c = seed_c.copy()
    best_r = r_seed

    # 1) Establish a valid baseline r_target near the seed root.
    r_target = max(0.3, r_seed * 0.9)

    # 2) Adaptive chase: shrink r_target while we can keep max q < 0.
    for _ in range(60):
        if time.time() > deadline - 25.0:
            break
        c, mval = _optimize_for_rtarget(r_target, best_c, deadline)
        # Evaluate the true root of this improved c.
        r = _true_root(*c)
        if r is not None and r < best_r:
            best_r = r
            best_c = c.copy()
        # If q is negative throughout [0, r_target], the true root is below it.
        if mval < -1e-9:
            # True root < r_target; tighten target and continue.
            # Estimate a tighter target from the true root if available.
            new_target = (r if (r is not None and r > 0.1) else r_target * 0.85)
            new_target = min(new_target, r_target * 0.92)
            if new_target < r_target * 0.5:
                new_target = r_target * 0.5
            if new_target >= r_target:
                break
            r_target = new_target
        else:
            # Cannot push q negative on this whole interval; stop shrinking.
            break

    # 3) Final high-precision polish on the best coefficient region.
    if time.time() < deadline - 10.0:
        # Local grid refinement around best_c to catch discrete structure.
        c0b, c1b, c2b = best_c
        for dc0 in (-0.02, 0.0, 0.02):
            for dc1 in (-0.01, 0.0, 0.01):
                for dc2 in (-0.005, 0.0, 0.005):
                    if time.time() > deadline - 6.0:
                        break
                    cand = np.array([c0b + dc0, c1b + dc1, c2b + dc2])
                    r = _true_root(*cand)
                    if r is not None and r < best_r:
                        best_r = r
                        best_c = cand.copy()

    # 4) One more local Nelder-Mead on the true-root objective for a final nudge.
    if time.time() < deadline - 6.0:
        from scipy.optimize import minimize

        def root_obj(c):
            r = _true_root(*c)
            return r if r is not None else 1e6

        try:
            res = minimize(root_obj, best_c.copy(), method="Nelder-Mead",
                           options=dict(maxiter=2000, xatol=1e-9, fatol=1e-10))
            r = _true_root(*res.x)
            if r is not None and r < best_r:
                best_r = r
                best_c = np.array(res.x, dtype=float)
        except Exception:
            pass

    return best_c
# EVOLVE-BLOCK-END
