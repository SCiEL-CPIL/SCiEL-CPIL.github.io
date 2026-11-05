import time
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 80.0  # 10s safety margin before the 90s kill

_FALLBACK = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)


def _deadline_hit():
    return time.time() > _DEADLINE


def _clamp(x):
    return np.clip(x, 0.1, 1.2)


def _om_at(a, om0, w0, wa):
    """Matter density parameter Om(a) = Om0 * a^-3 / E^2(a), E^2 = H^2/H0^2."""
    a = np.asarray(a, dtype=float)
    de = 1.0 + w0 + wa * (1.0 - a)
    a3 = a * a * a
    de3 = de * de * de
    num = om0 / a3
    den = num + de3
    return num / den


def _gamma_of(c, om, a, w0, wa):
    q = 1.0 - om
    u = 1.0 + w0 + wa * (1.0 - a)
    c = np.asarray(c)
    g = (c[0] + c[1] * q + c[2] * q * q + c[3] * u
         + c[4] * wa * a + c[5] * u * q)
    return _clamp(g)


def _f_of(c, om, a, w0, wa):
    return np.power(om, _gamma_of(c, om, a, w0, wa))


def _normalized_D(c, om, a, w0, wa, n=400):
    """D(a)/D(1) = exp(-int_{ln a}^{0} f(e^x) dx)."""
    a = np.atleast_1d(np.asarray(a, dtype=float))
    out = np.empty_like(a)
    xs = np.linspace(0.0, 0.0, 2)  # placeholder
    xgrid = np.linspace(0.0, 0.0, n)
    for i, ai in enumerate(a):
        if ai >= 1.0:
            out[i] = 1.0
            continue
        xlo = math.log(ai)
        xg = np.linspace(xlo, 0.0, n)
        aa = np.exp(xg)
        omg = _om_at(aa, om, w0, wa)
        fg = _f_of(c, omg, aa, w0, wa)
        out[i] = math.exp(-np.trapz(fg, xg))
    return out


def _build_cosmos():
    """Dense set of representative CPL cosmologies."""
    om0 = np.linspace(0.20, 0.80, 11)
    w0 = np.linspace(-1.10, -0.60, 7)
    wa = np.linspace(-0.40, 0.30, 7)
    O, W, WA = np.meshgrid(om0, w0, wa, indexing='ij')
    O = O.ravel(); W = W.ravel(); WA = WA.ravel()
    return O, W, WA


def _fit(O, W, WA, c, a_grid, n=300, max_iter=6000):
    """Levenberg-Marquardt fit of coefficients to f and normalized D."""
    # Precompute reference targets (f_ref, D_ref) and basis features once.
    a_grid = np.asarray(a_grid, dtype=float)
    om_grid = _om_at(a_grid, 1.0, 0.0, 0.0)  # dummy to match shape; recompute per cosmo
    # We need per-cosmo Om; build arrays.
    # Stack all (cosmo, a) pairs.
    ncos = len(O)
    a_rep = np.repeat(a_grid, ncos)
    om_rep = np.empty_like(a_rep)
    for k in range(ncos):
        om_rep[k * len(a_grid):(k + 1) * len(a_grid)] = _om_at(a_grid, O[k], W[k], WA[k])
    q = 1.0 - om_rep
    u = np.empty_like(a_rep)
    for k in range(ncos):
        u[k * len(a_grid):(k + 1) * len(a_grid)] = 1.0 + W[k] + WA[k] * (1.0 - a_grid)
    waa = np.repeat(WA, len(a_grid)) * a_rep

    # Features for gamma: [1, q, q^2, u, wa*a, u*q]
    X = np.column_stack([np.ones_like(q), q, q * q, u, waa, u * q])
    N = X.shape[0]

    # Reference D via ODE
    D_ref = np.empty_like(a_rep)
    for k in range(ncos):
        seg = np.zeros(len(a_grid))
        # integrate from 0 to ln(a) reversed: D(a) = exp(-int_{ln a}^0 f dx)
        # For a<1, integral positive. Build cumulative integral over x from ln(a) to 0.
        for i in range(len(a_grid)):
            ai = a_grid[i]
            if ai >= 1.0:
                seg[i] = 1.0
                continue
            xlo = math.log(ai)
            xg = np.linspace(xlo, 0.0, n)
            aa = np.exp(xg)
            omg = _om_at(aa, O[k], W[k], WA[k])
            fg = np.power(omg, _gamma_of(c, omg, aa, W[k], WA[k]))
            seg[i] = math.exp(-np.trapz(fg, xg))
        D_ref[k * len(a_grid):(k + 1) * len(a_grid)] = seg

    # f_ref
    f_ref = np.power(om_rep, _gamma_of(c, om_rep, a_rep, 0.0, 0.0))
    # Actually f_ref depends only on om and gamma(c,...); gamma uses W,WA via u, waa.
    f_ref = np.power(om_rep, _gamma_of(c, om_rep, a_rep, W_rep_dummy := 0.0, 0.0))
    # Recompute f_ref properly with per-cosmo W,WA:
    f_ref = np.empty_like(a_rep)
    for k in range(ncos):
        f_ref[k * len(a_grid):(k + 1) * len(a_grid)] = np.power(
            om_rep[k * len(a_grid):(k + 1) * len(a_grid)],
            _gamma_of(c, om_rep[k * len(a_grid):(k + 1) * len(a_grid)], a_grid, W[k], WA[k]))

    logf_ref = np.log(np.clip(f_ref, 1e-12, None))
    logD_ref = np.log(np.clip(D_ref, 1e-12, None))

    # Targets: log f (linear in gamma -> linear in c) and log D (nonlinear in c).
    # Use a combined least-squares on [log f - log f_ref; log D - log D_ref].
    # Jacobian: d(log f)/dc = (d gamma/dc) = X. d(log D)/dc needs numerical diff.

    def residual(cc):
        cc = np.asarray(cc, dtype=float)
        g = X @ cc
        logf = np.log(np.clip(np.power(om_rep, _clamp(g)), 1e-12, None))
        # log D: numeric per-cosmo using trapezoid of f
        logD = np.empty_like(a_rep)
        for k in range(ncos):
            seg = np.zeros(len(a_grid))
            for i in range(len(a_grid)):
                ai = a_grid[i]
                if ai >= 1.0:
                    seg[i] = 0.0
                    continue
                xlo = math.log(ai)
                xg = np.linspace(xlo, 0.0, n)
                aa = np.exp(xg)
                omg = _om_at(aa, O[k], W[k], WA[k])
                gg = _gamma_of(cc, omg, aa, W[k], WA[k])
                fg = np.power(omg, _clamp(gg))
                seg[i] = -np.trapz(fg, xg)
            logD[k * len(a_grid):(k + 1) * len(a_grid)] = seg
        return np.concatenate([logf - logf_ref, logD - logD_ref])

    # Simple Gauss-Newton with numerical Jacobian, damped (Levenberg-Marquardt).
    cc = np.array(c, dtype=float)
    r = residual(cc)
    fval = float(np.sum(r * r))
    lam = 1e-3
    for it in range(max_iter):
        if _deadline_hit():
            break
        J = np.zeros((2 * N, 6))
        h = 1e-7
        for j in range(6):
            cc2 = cc.copy(); cc2[j] += h
            r2 = residual(cc2)
            J[:, j] = (r2 - r) / h
        # LM: (J^T J + lam diag) dc = -J^T r
        JTJ = J.T @ J
        JTr = J.T @ r
        diag = np.diag(JTJ)
        diag[diag < 1e-12] = 1e-12
        A = JTJ + lam * np.diag(diag)
        try:
            dc = np.linalg.solve(A, -JTr)
        except np.linalg.LinAlgError:
            dc = np.linalg.lstsq(A, -JTr, rcond=None)[0]
        cc_new = cc + dc
        # keep coefficients within [-10,10]
        cc_new = np.clip(cc_new, -10.0, 10.0)
        r_new = residual(cc_new)
        f_new = float(np.sum(r_new * r_new))
        if f_new < fval:
            cc = cc_new; r = r_new; fval = f_new
            lam = max(lam * 0.5, 1e-8)
        else:
            lam = min(lam * 2.0, 1e6)
        if lam > 1e6:
            break
    return cc


def construct():
    # Fit the six coefficients to a dense cosmology grid, with a time guard.
    try:
        O, W, WA = _build_cosmos()
        a_grid = np.linspace(0.05, 1.0, 120)
        c = _fit(O, W, WA, np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float), a_grid)
        c = np.clip(c, -10.0, 10.0)
        if not np.all(np.isfinite(c)):
            return _FALLBACK.tolist()
        return c.tolist()
    except Exception:
        return _FALLBACK.tolist()
