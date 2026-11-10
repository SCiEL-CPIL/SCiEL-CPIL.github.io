"""Direct geometric construction for logarithmic energy on sphere."""
import numpy as np
from math import fsum, log
import time
from scipy.optimize import minimize

N = 150
DEADLINE = 150.0
SAFETY = 0.85


def _energy(pts):
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return 1e18
            terms.append(-log(d))
    return fsum(terms)


def _cartesian_energy_and_grad(X):
    """Energy and gradient for points on sphere (Cartesian coords, projected)."""
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = len(X)
    diff = X[:, None, :] - X[None, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1e-30)
    r = np.sqrt(r2)
    np.fill_diagonal(r, 1.0)
    E = -np.sum(np.log(r)) * 0.5
    # dE/dX_i = -sum_{j!=i} (X_i - X_j) / r_ij^2
    inv_r2 = 1.0 / r2
    np.fill_diagonal(inv_r2, 0.0)
    dE_dX = -np.sum(diff * inv_r2[:, :, None], axis=1)
    # Project gradient to tangent space: g_tan = g - (g·x)x
    g_tan = dE_dX - np.sum(dE_dX * X, axis=1, keepdims=True) * X
    return E, g_tan


def _opt_cartesian(X0, maxiter=1000):
    """Optimize in Cartesian coords with projection."""
    try:
        res = minimize(_cartesian_energy_and_grad, X0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
        return res.x
    except Exception:
        return X0


def construct():
    start = time.time()
    dl = start + DEADLINE * SAFETY

    def _fibonacci_spiral():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1 - z ** 2))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts / np.linalg.norm(pts, axis=1, keepdims=True)

    best = None
    best_E = 1e18

    # Try multiple initializations
    np.random.seed(42)
    initializations = [_fibonacci_spiral()]
    for _ in range(8):
        X = np.random.randn(N, 3)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        initializations.append(X)

    for X0 in initializations:
        if time.time() > dl:
            break
        Xopt = _opt_cartesian(X0, maxiter=1000)
        Xopt /= np.linalg.norm(Xopt, axis=1, keepdims=True)
        E = _energy(Xopt)
        if E < best_E:
            best_E = E
            best = Xopt.copy()

    if best is None:
        best = _fibonacci_spiral()

    # Final refinement with tighter tolerances if time permits
    if time.time() < dl - 5.0:
        Xref = _opt_cartesian(best, maxiter=1000)
        Xref /= np.linalg.norm(Xref, axis=1, keepdims=True)
        Eref = _energy(Xref)
        if Eref < best_E:
            best = Xref
            best_E = Eref
        # Second polish pass if still time
        if time.time() < dl - 3.0:
            Xref2 = _opt_cartesian(best, maxiter=500)
            Xref2 /= np.linalg.norm(Xref2, axis=1, keepdims=True)
            Eref2 = _energy(Xref2)
            if Eref2 < best_E:
                best = Xref2
                best_E = Eref2

    return best
