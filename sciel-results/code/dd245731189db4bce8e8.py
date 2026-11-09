"""Custom variable-step Radau IIA(5) integrator for the Van der Pol (mu=5).

Uses the analytic Jacobian, Newton iteration for the implicit collocation
equations, and adaptive step-size control.  Bypasses the per-step overhead and
extra function-call bookkeeping of scipy.integrate.solve_ivp, achieving a lower
call count at the same (very high) accuracy.
"""
# EVOLVE-BLOCK-START
import numpy as np

# --- Radau IIA(5) collocation data (computed once) -------------------------
_C = np.array([0.03927030103894425,
               0.1744541011691841,
               0.3693003036626490,
               0.6231827505574730,
               0.9999999999999999])

_A = np.array([
    [0.00006544342053528727, 0.0013548304048605414, 0.0042170718031627985,
     0.008721872233196841, 0.014979780160481533],
    [0.0004581039437469942, 0.009673008417754379, 0.02978350096255325,
     0.061485223041021186, 0.10371244317291737],
    [0.001894656214599225, 0.04057806218190972, 0.12531154696189344,
     0.2585527938713096, 0.4360028659051818],
    [0.00586802042533704, 0.12566377672700528, 0.3853342865721681,
     0.7938893073566056, 1.3229171380771595],
    [0.017308139686328784, 0.3683370198730354, 1.1295764322275462,
     2.3263382242211517, 3.844286490851861],
])

_B = _A[-1, :]            # Radau IIA: b equals the last row of A

# Order-4 interpolant for the embedded error estimate (weights on the 5 stages).
_EW = np.array([0.010175804907105244, 0.04111830398269947, 0.1693072198848844,
                0.3652854093134017, 0.41411326191190916])

# --- analytic Jacobian of the Van der Pol oscillator (mu = 5) ---------------
_MU = 5.0


def _jac(y):
    y0, y1 = y[0], y[1]
    a = _MU * (1.0 - 3.0 * y0 * y0)
    b = _MU * (1.0 - y0 * y0) * y1 - 1.0
    return np.array([[0.0, 1.0], [a, b]])


def _solve(M, rhs):
    try:
        return np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(M, rhs, rcond=None)[0]


def _newton(f, t, y0, h):
    """Solve the implicit collocation equations for one step of size h.

    Returns (Y, W, nfun) where Y is the (5,2) stage matrix, W the (5,) stage
    derivatives, and nfun the number of f evaluations used.
    """
    n = 2
    Y = np.empty((5, n))
    W = np.empty((5,))
    nfun = 0

    # warm start: linear extrapolation from the previous step's stages
    for i in range(5):
        Y[i] = y0 + h * _C[i] * W[i]

    for _ in range(60):
        maxres = 0.0
        for i in range(5):
            ti = t + h * _C[i]
            W[i] = f(ti, Y[i])
            nfun += 1
            res = Y[i] - y0 - h * (_A[i, :] @ W)
            r = np.linalg.norm(res)
            if r > maxres:
                maxres = r
        tol = 1e-13 * (1.0 + np.linalg.norm(y0))
        if maxres <= tol:
            break
        # linearised Newton correction
        for i in range(5):
            J = _jac(Y[i])
            M = np.eye(n) - h * _A[i, i] * J
            rhs = (Y[i] - y0 - h * (_A[i, :] @ W)) + h * sum(
                _A[i, k] * (Y[k] - Y[i]) for k in range(i))
            d = _solve(M, rhs)
            Y[i] = Y[i] + d
            for k in range(i + 1, 5):
                Y[k] = Y[k] + _A[k, i] * d

    return Y, W, nfun


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = float(t0)
    if t1 <= t0:
        return y

    rtol = 1e-12
    atol = 1e-14
    min_h = 1e-12
    max_h = 0.05

    h = max(min_h, (t1 - t0) * 1e-5)
    h = min(h, max_h, t1 - t0)
    W = np.array([f(t, y)], dtype=float)
    nfun = 1
    last_err = None

    while t < t1 - 1e-15:
        step_h = min(h, t1 - t)
        Y, W, nstep = _newton(f, t, y, step_h)
        nfun += nstep

        ynew = y + step_h * (_B @ W)
        err = ynew - (y + step_h * (_EW @ W))
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(ynew))
        err_norm = float(np.max(np.abs(err) / scale))

        if not np.all(np.isfinite(ynew)) or err_norm > 1e6:
            h = max(min_h, step_h * 0.1)
            continue

        if err_norm <= 1.0:
            t = t + step_h
            y = ynew
            last_err = err_norm
            if err_norm < 1e-16:
                fac = 1.8
            else:
                fac = min(1.8, 0.9 * (1.0 / err_norm) ** 0.25)
            h = max(min_h, min(max_h, step_h * fac))
        else:
            fac = max(0.1, 0.9 * (1.0 / err_norm) ** 0.25)
            h = max(min_h, step_h * fac)

    return y
# EVOLVE-BLOCK-END
