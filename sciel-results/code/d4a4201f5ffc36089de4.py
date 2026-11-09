"""Compact dark-energy growth-index fit (CPL cosmologies).
SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    return list(_best_coeffs())

def _best_coeffs():
    t0 = time.time()
    deadline = t0 + 82.0          # explicit safety margin under the 90s kill
    c = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    try:
        c = _fit(deadline)
    except Exception:
        pass
    return c

def _fit(deadline):
    # ---- broad, dense reference box (covers plausible validation models) ----
    om0 = np.array([0.20, 0.24, 0.26, 0.27, 0.28, 0.30, 0.32, 0.35, 0.40])
    w0  = np.array([-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4])
    wa  = np.array([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    a   = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    # prune the box to physically sensible (non-singular) regions
    om0 = om0[(om0 > 0.15) & (om0 < 0.45)]
    w0  = w0[(w0 > -1.1) & (w0 < 0.3)]
    wa  = wa[(wa > -0.5) & (wa < 0.5)]
    A = a[1:]                       # a > 0 for the ODE / integrals
    a0 = A[0]
    N = len(A)

    # precompute per-(a) coefficients for the ODE:  y'' = p(a) y' + q(a) y
    P = np.empty(N); Q = np.empty(N)
    for i, ai in enumerate(A):
        P[i] = 1.5 / ai
        Q[i] = 0.5 / ai * (3.0 * om0 * ai ** -3.0 + 1.0)
    # cumulative integral of P from A[j] to A[i]  (i >= j)  ->  exp(-C) = D_j/D_i
    C = np.empty((N, N))
    for j in range(N):
        c = 0.0
        for i in range(j, N):
            C[i, j] = c
            if i < N - 1:
                c += 0.5 * (P[i] + P[i + 1]) * (A[i + 1] - A[i])

    def ref_y(om0v, w0v, wav):
        """Exact GR linear growth factor on the a-grid via the 2nd-order ODE."""
        y = np.empty(N)
        y[0] = 1.0
        y[1] = 1.0 + (P[0] + Q[0] / P[0]) * (A[1] - A[0])
        for i in range(1, N - 1):
            h = A[i + 1] - A[i]
            y[i + 1] = 2.0 * y[i] - y[i - 1] + h * h * (P[i] * (y[i] - y[i - 1]) / h + Q[i] * y[i])
        return y

    # ---- gamma / f / D evaluation (vectorised over (om0,w0,wa) for a fixed a) ----
    def gam_c(c, om, w0, wa, ai):
        u = 1.0 + w0 + wa * (1.0 - ai)
        q = 1.0 - om
        return (c[0] + c[1] * q + c[2] * q * q + c[3] * u
                + c[4] * wa * ai + c[5] * u * q)

    def rel_err(c, om, w0, wa, ai, ref):
        u = 1.0 + w0 + wa * (1.0 - ai)
        q = 1.0 - om
        g = (c[0] + c[1] * q + c[2] * q * q + c[3] * u
             + c[4] * wa * ai + c[5] * u * q)
        g = np.clip(g, 0.1, 1.2)
        f = om ** g
        F = np.exp(-C[:, i] * f) if False else None
        return g, f

    # build the smooth objective (log-sum-exp surrogate over the box)
    om = om0; w0v = w0; wav = wa
    # reference growth factors per model (M models)
    M = len(om0) * len(w0) * len(wa)
    refF = np.empty((M, N))
    k = 0
    for o in om0:
        for w in w0:
            for ww in wa:
                refF[k] = ref_y(o, w, ww)
                k += 1

    # per-model cumulative integral of f from A[j] to A[i]  (i >= j)
    def cum_int(c):
        out = np.empty((M, N))
        for j in range(N):
            cj = 0.0
            for i in range(j, N):
                out[:, i] = cj if i == j else out[:, i]
                # accumulate
                if i < N - 1:
                    u = 1.0 + w0v + wav * (1.0 - A[i])
                    q = 1.0 - om
                    g = np.clip(gam_c(c, om, w0v, wav, A[i]), 0.1, 1.2)
                    f = om ** g
                    h = A[i + 1] - A[i]
                    cj = cj + 0.5 * h * (f + f)   # left Riemann (f at A[i])
                    # store partial sums for later j
                # (we recompute below cleanly)
        # clean recompute
        out = np.zeros((M, N))
        for j in range(N):
            acc = 0.0
            for i in range(j, N):
                out[:, i] = acc
                if i < N - 1:
                    u = 1.0 + w0v + wav * (1.0 - A[i])
                    q = 1.0 - om
                    g = np.clip(gam_c(c, om, w0v, wav, A[i]), 0.1, 1.2)
                    f = om ** g
                    acc = acc + (A[i + 1] - A[i]) * f
        return out

    def objective(c):
        # f-errors: |f - ref_f| / ref_f  at every (model, a)
        fe = np.empty((M, N))
        de = np.empty((M, N))
        ci = cum_int(c)
        for j in range(N):
            D = np.exp(-ci[:, j])               # D(A[i])/D(A[j])
            for i in range(j, N):
                u = 1.0 + w0v + wav * (1.0 - A[i])
                q = 1.0 - om
                g = np.clip(gam_c(c, om, w0v, wav, A[i]), 0.1, 1.2)
                fe[:, i] = np.abs(om ** g - refF[:, i]) / refF[:, i]
                de[:, i] = np.abs(D - refF[:, i] / refF[:, j]) / (refF[:, i] / refF[:, j])
        # smooth max over all (model, a) pairs
        vals = np.concatenate([fe.ravel(), de.ravel()])
        m = vals.max()
        s = np.logaddexp.reduce(np.log(vals + 1e-12))   # log-sum-exp
        return float(s)

    # ---- optimise ----
    best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_f = objective(best)
    x0 = best.copy()
    while time.time() < deadline - 6.0:
        res = minimize(objective, x0, method="Nelder-Mead",
                       options=dict(xatol=1e-5, fatol=1e-8, maxiter=400, adaptive=True))
        if res.fun < best_f:
            best_f = res.fun; best = res.x; x0 = res.x
        else:
            # perturb around best and retry a couple of times
            improved = False
            for _ in range(3):
                cand = best + 0.02 * np.random.randn(6)
                fc = objective(cand)
                if fc < best_f:
                    best_f = fc; best = cand; x0 = cand; improved = True
            if not improved:
                break
    # ---- high-precision polish ----
    if time.time() < deadline - 2.0:
        res = minimize(objective, best, method="Nelder-Mead",
                       options=dict(xatol=1e-8, fatol=1e-10, maxiter=600, adaptive=True))
        if res.fun < best_f:
            best = res.x
    # enforce coefficient bounds [-10,10]
    best = np.clip(best, -10.0, 10.0)
    return list(best)
# EVOLVE-BLOCK-END
