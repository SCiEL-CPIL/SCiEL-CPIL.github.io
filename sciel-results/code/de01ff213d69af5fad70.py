"""Minimize logarithmic energy of 150 points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 135.0          # safety margin vs 150 s kill
START = time.time()

def _remaining():
    return DEADLINE - (time.time() - START)


def _log_energy(x):
    """x: (N,3) unit vectors. Returns sum_{i<j} log(1/|x_i-x_j|)."""
    d = np.sqrt(np.maximum(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1), 1e-30))
    iu = np.triu_indices(N, k=1)
    return math.fsum(np.log(1.0 / d[iu]))


def _energy_and_grad_flat(xf):
    x = xf.reshape(N, 3)
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    u = x / nrm
    diff = u[:, None, :] - u[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    d = np.sqrt(np.maximum(d2, 1e-30))
    iu = np.triu_indices(N, k=1)
    inv = 1.0 / d
    energy = math.fsum(np.log(inv[iu]))
    # d(energy)/d u_k = - sum_{j != k} (u_k - u_j)/|u_k-u_j|
    G = np.sum(diff * inv[:, :, None], axis=1)
    # project gradient onto tangent space
    G = G - (np.sum(G * u, axis=1, keepdims=True)) * u
    gf = (G * (nrm / nrm)).reshape(-1)  # u==x/nrm; tangent proj already in u-space
    # map back to x-space: for x = nrm*u, gradient wrt x projected: use same tangent dir scaled
    gf = G.reshape(-1)
    return energy, gf


def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _project_sphere(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return x / nrm


def _lbfgs(x0, maxiter=600):
    x0 = _project_sphere(x0).reshape(-1)
    res = minimize(_energy_and_grad_flat, x0, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'maxls': 30, 'ftol': 1e-16, 'gtol': 1e-14,
                            'maxfun': 4000})
    return _project_sphere(res.x.reshape(N, 3)), res.fun


def _simulated_annealing(x0, budget_s):
    x = _project_sphere(x0).astype(float).copy()
    best = x.copy()
    best_e = _log_energy(x)
    e = best_e
    t_end = time.time() + budget_s
    T = 0.02
    cool = 0.999
    rng = np.random.default_rng(1234)
    while time.time() < t_end and _remaining() > 2.0:
        # random move: pick k points, nudge
        k = rng.integers(1, 4)
        idx = rng.choice(N, size=k, replace=False)
        step = T * (0.5 + 0.5 * rng.random())
        x2 = x.copy()
        for j in idx:
            x2[j] = x2[j] + step * rng.standard_normal(3)
        x2 = _project_sphere(x2)
        e2 = _log_energy(x2)
        if e2 < e or rng.random() < math.exp(-(e2 - e) / max(T, 1e-9)):
            x = x2
            e = e2
            if e < best_e:
                best_e = e
                best = x.copy()
        T *= cool
    return best, best_e


def construct():
    best_x = None
    best_e = math.inf
    rng = np.random.default_rng(7)

    starts = []
    starts.append(_fibonacci_sphere(N))
    # perturbed fibonacci
    for amp in (0.05, 0.15, 0.3):
        starts.append(_project_sphere(_fibonacci_sphere(N) + amp * rng.standard_normal((N, 3))))
    for _ in range(3):
        starts.append(_random_sphere(N, rng))

    for s in starts:
        if _remaining() < 8.0:
            break
        x, e = _lbfgs(s)
        if math.isfinite(e) and e < best_e:
            best_e = e
            best_x = x.copy()

    # simulated annealing polish from the best
    if best_x is not None and _remaining() > 12.0:
        x, e = _simulated_annealing(best_x, min(_remaining() - 4.0, 40.0))
        if math.isfinite(e) and e < best_e:
            best_e = e
            best_x = x.copy()
        # short L-BFGS re-polish after SA
        if _remaining() > 8.0:
            x, e = _lbfgs(best_x, maxiter=300)
            if math.isfinite(e) and e < best_e:
                best_e = e
                best_x = x.copy()

    if best_x is None:
        best_x = _fibonacci_sphere(N)

    # validity guard: ensure distinct points
    d = np.sqrt(np.sum((best_x[:, None, :] - best_x[None, :, :]) ** 2, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.any(d[iu] < 1e-9):
        # jitter colliding points
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(best_x[i] - best_x[j]) < 1e-9:
                    best_x[j] = best_x[j] + 1e-6 * rng.standard_normal(3)
        best_x = _project_sphere(best_x)

    return best_x
