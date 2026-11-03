"""Uncertainty inequality: tune H0,H4,H8 (H12 derived so g(0)=0) to minimize C4."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_T0 = time.time()
_DEADLINE = 170.0  # safety margin vs 180s eval / 210s hard kill
_FALLBACK = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_C4_FLOOR = 1.0  # seed value; never return a worse C4 than this


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _phys_hermite(n, x):
    """Physicists' (probabilists') Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h = np.ones_like(x)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - (k - 1.0) * h
        h, h1 = h1, h2
    return h1


def _build_g(c0, c4, c8, c12, x):
    g = (c0 * _phys_hermite(0, x)
         + c4 * _phys_hermite(4, x)
         + c8 * _phys_hermite(8, x)
         + c12 * _phys_hermite(12, x))
    return g * np.exp(-0.5 * x * x)


def _c12_for_zero(c0, c4, c8):
    """Choose c12 so g(0)=0 (H_n(0) nonzero for even n)."""
    s = c0 * _phys_hermite(0, 0.0) + c4 * _phys_hermite(4, 0.0) + c8 * _phys_hermite(8, 0.0)
    h12_0 = _phys_hermite(12, 0.0)
    if abs(h12_0) < 1e-14:
        return 0.0
    return -s / h12_0


def _c4_value(c0, c4, c8):
    c12 = _c12_for_zero(c0, c4, c8)
    x = np.linspace(-6.0, 6.0, 40001)
    g = _build_g(c0, c4, c8, c12, x)
    dx = x[1] - x[0]
    norm = np.trapz(g * g, x)
    if not np.isfinite(norm) or norm < 1e-18:
        return _C4_FLOOR + 1e6
    g = g / np.sqrt(norm)
    fg = np.fft.fft(g)
    fn = np.fft.fftshift(fg)
    df = 2.0 * np.pi / ((x[-1] - x[0]) * len(x))
    kf = np.fft.fftshift(np.fft.fftfreq(len(x), d=dx)) * 2.0 * np.pi
    f = fn / np.sqrt(len(x) * df)
    nrm = np.trapz(np.abs(f) ** 2, kf)
    if not np.isfinite(nrm) or nrm < 1e-18:
        return _C4_FLOOR + 1e6
    f = f / np.sqrt(nrm)
    A = np.sqrt(np.trapz(x * x * np.abs(g) ** 2, x))
    B = np.sqrt(np.trapz(kf * kf * np.abs(f) ** 2, kf))
    return A * B


def _objective_sphere(theta, phi):
    """Map 2 spherical coords -> 3 coefficients (unit norm), return C4."""
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    c0 = st * sp
    c4 = st * cp
    c8 = ct
    return _c4_value(c0, c4, c8)


def _refine(c0, c4, c8):
    """Local Nelder-Mead polish in the 3-coefficient space."""
    def obj(v):
        return _c4_value(v[0], v[1], v[2])

    x0 = np.array([c0, c4, c8], dtype=float)
    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-9, "fatol": 1e-10,
                            "adaptive": True})
    if res.success or (res.fun is not None and np.isfinite(res.fun)):
        return np.asarray(res.x, dtype=float), (res.fun if np.isfinite(res.fun) else _C4_FLOOR)
    return x0, obj(x0)


def construct():
    best_c = _FALLBACK.copy()
    best_c4 = _c4_value(*_FALLBACK)
    if not np.isfinite(best_c4):
        best_c4 = _C4_FLOOR

    def _update(v, val):
        nonlocal best_c, best_c4
        if np.isfinite(val) and val < best_c4:
            best_c4 = val
            best_c = np.asarray(v, dtype=float)

    # 1) Global search: Differential Evolution in spherical (scale-free) coords.
    if _time_left() > 30.0:
        try:
            de = differential_evolution(
                _objective_sphere,
                bounds=[(0.0, np.pi), (0.0, 2.0 * np.pi)],
                seed=12345,
                maxiter=80,
                popsize=20,
                tol=1e-7,
                init="sobol",
                polish=False,
                workers=1,
            )
            if de.x is not None:
                st, ct = np.sin(de.x[0]), np.cos(de.x[0])
                sp, cp = np.sin(de.x[1]), np.cos(de.x[1])
                _update([st * sp, st * cp, ct], de.fun)
        except Exception:
            pass

    # 2) Deterministic multi-start refinement from seed + sphere grid + DE.
    starts = [list(_FALLBACK)]
    if _time_left() > 25.0:
        for i in range(9):
            th = (i + 0.5) * np.pi / 9.0
            for j in range(9):
                ph = (j + 0.5) * 2.0 * np.pi / 9.0
                st, ct = np.sin(th), np.cos(th)
                sp, cp = np.sin(ph), np.cos(ph)
                starts.append([st * sp, st * cp, ct])
    if _time_left() > 20.0:
        rng = np.random.default_rng(7)
        for _ in range(12):
            v = rng.normal(0.0, 1.0, size=3)
            v = v / (np.linalg.norm(v) + 1e-12)
            starts.append(list(v))

    for v in starts:
        if _time_left() < 8.0:
            break
        try:
            xc, val = _refine(v[0], v[1], v[2])
            _update(xc, val)
        except Exception:
            pass

    # 3) Final polish of the best point.
    if _time_left() > 5.0:
        try:
            xc, val = _refine(*best_c)
            _update(xc, val)
        except Exception:
            pass

    return best_c
# EVOLVE-BLOCK-END
