"""Rewrite for the 150-point logarithmic-energy minimisation on the sphere."""
import numpy as np
from scipy.optimize import minimize
import math
import time

N = 150
_DEADLINE = 150.0 - 5.0  # safety margin before the 150 s kill

# ---------- energy helpers (vectorised) ----------

def _energy_and_grad(X):
    """Return (energy, grad) for X of shape (N,3) on the sphere.

    energy = sum_{i<j} log(1/|xi - xj|)
    grad   = d(energy)/dX  (Euclidean, not yet projected)
    """
    N = X.shape[0]
    # Pairwise differences
    diff = X[:, None, :] - X[None, :, :]          # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)           # (N, N)
    # Avoid log(0) on the diagonal
    np.fill_diagonal(dist, 1.0)
    log_inv = -np.log(dist)                       # log(1/d)
    np.fill_diagonal(log_inv, 0.0)
    E = 0.5 * np.sum(log_inv)                     # each pair counted twice

    # Gradient: d/dx_i sum_{j!=i} log(1/|x_i - x_j|)
    #           = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # Vectorised: grad[i] = sum_j (x_i - x_j)/dist[i,j]^2  (j != i)
    # diff[i,j] = x_i - x_j
    inv_d2 = 1.0 / (dist ** 2)                    # (N,N)
    np.fill_diagonal(inv_d2, 0.0)
    G = np.einsum('ij,ijk->ik', inv_d2, diff)     # (N,3)
    return E, G


def _projected_grad_desc_step(X, lr=0.01, max_iter=200):
    """Simple projected gradient descent on the sphere."""
    for _ in range(max_iter):
        _, G = _energy_and_grad(X)
        X_new = X - lr * G
        X_new = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)
        if np.allclose(X_new, X, atol=1e-12):
            break
        X = X_new
    return X


def _lbfgs_optimize(X0, time_left):
    """Run L-BFGS-B with a sphere-constraint penalty, returning best X."""
    def _penalty_energy_and_grad(X, mu=1e6):
        E, G = _energy_and_grad(X)
        r = X - X0 / np.linalg.norm(X0, axis=1, keepdims=True)  # not used; see below
        # Actually we optimise on R^3 with penalty for |x|^2 = 1
        # Use a different formulation: penalise deviation from unit norm
        norms2 = np.sum(X * X, axis=1)
        pen = mu * np.sum((norms2 - 1.0) ** 2)
        G_pen = G + 4.0 * mu * (norms2 - 1.0)[:, None] * X
        return E + pen, G_pen

    # Better: optimise on the sphere directly via reparametrisation is complex.
    # Use penalty method with large mu.
    mu = 1e5
    best_X = X0.copy()
    best_E = _energy_and_grad(X0)[0]

    def _obj(X):
        E, _ = _energy_and_grad(X)
        norms2 = np.sum(X * X, axis=1)
        pen = mu * np.sum((norms2 - 1.0) ** 2)
        return E + pen

    def _grad(X):
        E, G = _energy_and_grad(X)
        norms2 = np.sum(X * X, axis=1)
        G_pen = G + 4.0 * mu * (norms2 - 1.0)[:, None] * X
        return G_pen

    while time.time() < time_left:
        res = minimize(_obj, X0, jac=_grad, method='L-BFGS-B',
                       options={'maxiter': 2000, 'gtol': 1e-10, 'ftol': 1e-14})
        X_new = res.x
        # Project back to sphere
        X_new = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)
        E_new = _energy_and_grad(X_new)[0]
        if np.isfinite(E_new) and E_new < best_E:
            best_E = E_new
            best_X = X_new.copy()
            X0 = X_new.copy()
        if res.success or res.status in (0, 1):
            break
        if time.time() > time_left:
            break
    return best_X, best_E


# ---------- Fibonacci spiral initialisation ----------

def _fibonacci_sphere():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------- main construct ----------

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE

    best_X = None
    best_E = np.inf
    rng = np.random.default_rng(42)

    # --- Phase 1: multiple random restarts with L-BFGS penalty ---
    n_restarts = 8
    for k in range(n_restarts):
        if time.time() > deadline:
            break
        if k == 0:
            X0 = _fibonacci_sphere()
        else:
            X0 = _random_sphere(rng)
        X_opt, E_opt = _lbfgs_optimize(X0, deadline - 1.0)
        if np.isfinite(E_opt) and E_opt < best_E:
            best_E = E_opt
            best_X = X_opt.copy()

    # --- Phase 2: local polish via projected gradient descent with adaptive lr ---
    if best_X is not None and time.time() < deadline - 2.0:
        X = best_X.copy()
        lr = 0.005
        for _ in range(5000):
            if time.time() > deadline - 1.0:
                break
            _, G = _energy_and_grad(X)
            X_new = X - lr * G
            X_new = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)
            if np.allclose(X_new, X, atol=1e-13):
                break
            X = X_new
        E_polish = _energy_and_grad(X)[0]
        if np.isfinite(E_polish) and E_polish < best_E:
            best_E = E_polish
            best_X = X.copy()

    # --- Phase 3: final high-precision polish with very small steps ---
    if best_X is not None and time.time() < deadline - 0.5:
        X = best_X.copy()
        for _ in range(2000):
            if time.time() > deadline - 0.5:
                break
            _, G = _energy_and_grad(X)
            X_new = X - 1e-5 * G
            X_new = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)
            if np.allclose(X_new, X, atol=1e-15):
                break
            X = X_new
        E_final = _energy_and_grad(X)[0]
        if np.isfinite(E_final) and E_final < best_E:
            best_E = E_final
            best_X = X.copy()

    if best_X is None:
        # Fallback: Fibonacci spiral
        best_X = _fibonacci_sphere()

    # Ensure validity: normalise, check finite, check no two points too close
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    if not np.all(np.isfinite(best_X)):
        best_X = _fibonacci_sphere()

    # Check minimum distance
    diff = best_X[:, None, :] - best_X[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    if d.min() < 1e-9:
        best_X = _fibonacci_sphere()

    return best_X
