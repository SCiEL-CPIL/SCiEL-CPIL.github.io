import time
import numpy as np
from scipy.optimize import minimize

_SIZE = 32
_NPW = 13
_G = _NPW
_N = 2 * _G + 1


def _kpoints():
    pts = []
    N = 15
    for t in np.linspace(0, 1, N):
        pts.append((0.5 * t, 0.0))
    for t in np.linspace(0, 1, N):
        pts.append((0.5, 0.5 * t))
    return pts


_KPTS = _kpoints()


def _fourier_coeffs(eps):
    """Exact Fourier coefficients C_G of a pixel dielectric map."""
    n = _SIZE
    kx = np.arange(-_G, _G + 1, dtype=float)
    kk = (kx[np.newaxis, :] * kx[:, np.newaxis]).reshape(-1)
    order = np.argsort(np.argsort(kk))
    kk = kk[order]
    kx1 = kx[:, np.newaxis]
    kx2 = kx[np.newaxis, :]
    phase = np.exp(1j * np.pi * (kx1 + kx2) / n).reshape(-1)[order]
    c = np.fft.ifft(phase * eps.reshape(-1).astype(float)).real
    return c, kk


def _circulant(c):
    """Circulant matrix with first row c (length N*N)."""
    cc = c.reshape(-1)
    ii = np.arange(_N * _N)[:, None]
    jj = np.arange(_N * _N)[None, :]
    return cc[(jj - ii) % (_N * _N)]


def _band_energies(kx, ky, c, kk):
    """Sorted TM band energies omega^2 at (kx,ky)."""
    gx = np.arange(-_G, _G + 1, dtype=float)
    kx1 = (gx[:, None] + kx)
    kx2 = (gx[None, :] + ky)
    A = (kx1 ** 2 + kx2 ** 2).reshape(-1)
    n = _SIZE
    kx1 = np.arange(-_G, _G + 1, dtype=float)[:, None]
    kx2 = np.arange(-_G, _G + 1, dtype=float)[None, :]
    phase = np.exp(1j * np.pi * (kx1 + kx2) / n).reshape(-1)
    eps = np.fft.ifft(phase * c).real
    inv_eps = 1.0 / eps
    c_inv = np.fft.ifft(phase * inv_eps).real
    D = _circulant(c_inv)
    M = D * A[None, :]
    return np.linalg.eigvalsh(M)


def _gap_ratio(eps):
    c, kk = _fourier_coeffs(eps)
    max4 = 0.0
    min5 = np.inf
    for (kx, ky) in _KPTS:
        w = _band_energies(kx, ky, c, kk)
        max4 = max(max4, w[3])
        min5 = min(min5, w[4])
    if min5 <= 0:
        return -1.0
    return 2.0 * (min5 - max4) / (min5 + max4)


def _smooth_gap(x):
    """Differentiable gap surrogate for continuous field x in [0,1]."""
    n = _SIZE
    eps = 1.0 + 11.0 * np.clip(x, 0.0, 1.0).reshape(n, n)
    c, kk = _fourier_coeffs(eps)
    beta = 40.0
    l4 = -np.inf
    l5 = -np.inf
    for (kx, ky) in _KPTS:
        w = _band_energies(kx, ky, c, kk)
        l4 = np.logaddexp(l4, beta * w[3])
        l5 = np.logaddexp(l5, -beta * w[4])
    max4 = l4 / beta
    min5 = -l5 / beta
    if min5 <= 0:
        return -1.0
    return 2.0 * (min5 - max4) / (min5 + max4)


def _symmetrize(x):
    y = np.asarray(x, dtype=float)
    return (y + y[::-1, ::-1] + y[::-1, :] + y[:, ::-1]) / 4.0


def _binarize(x):
    return (np.round(x) > 0.5).astype(int)


def _make_eps(x):
    return 1.0 + 11.0 * np.clip(x, 0.0, 1.0)


def design(problem):
    start = time.time()
    deadline = start + 190.0
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)

    # Centered air hole in dielectric background (classic high TM-gap design).
    i = np.arange(size)
    xx = (i + 0.5) / size - 0.5
    X, Y = np.meshgrid(xx, xx)
    r = np.sqrt(X * X + Y * Y)
    x0 = (r >= 0.25).astype(float)   # 1 dielectric, 0 air hole

    best_x = _binarize(x0)
    best_score = _gap_ratio(_make_eps(best_x))

    x = x0.reshape(-1)
    bounds = [(0.0, 1.0)] * len(x)

    def neg_obj(xv):
        return -_smooth_gap(xv)

    for it in range(6):
        if time.time() > deadline - 40.0:
            break
        try:
            res = minimize(neg_obj, x, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 30, "maxfun": 300, "ftol": 1e-9})
            x = np.clip(res.x, 0.0, 1.0)
        except Exception:
            break
        x = _symmetrize(x.reshape(size, size)).reshape(-1)
        xb = _binarize(x.reshape(size, size))
        s = _gap_ratio(_make_eps(xb))
        if s > best_score:
            best_score = s
            best_x = xb
        x = best_x.astype(float).reshape(-1)

    cell = best_x.astype(int)
    cell = ((cell + cell[::-1, ::-1] + cell[::-1, :] + cell[:, ::-1]) >= 2).astype(int)
    out = [[int(cell[i, j]) for j in range(size)] for i in range(size)]
    return out
