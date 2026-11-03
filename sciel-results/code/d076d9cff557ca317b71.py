"""Minimal logarithmic energy of 100 points on the sphere.

Strategy:
  * Several initializations (Fibonacci spiral, perturbed, random).
  * Riemannian gradient descent on the sphere (projected + re-normalised)
    with a line-search / backtracking step-size rule.
  * A final high-precision polish pass (tighter tolerance, more iterations).
  * Explicit wall-clock deadline guard so we always return the best found.
"""
import time
import math
import numpy as np

N = 100
_DEADLINE = time.time() + 140.0  # safety margin vs 150s kill

_RNG = np.random.default_rng(12345)


def _log_energy(P):
    # P: (N,3) unit vectors. Returns sum_{i<j} log(1/|Pi-Pj|).
    # Use chordal distances.
    D2 = np.einsum('ij,ij->i', P, P)[:, None] + np.einsum('ij,ij->i', P, P)[None, :] \
        - 2.0 * (P @ P.T)
    np.fill_diagonal(D2, 0.0)
    D2 = np.clip(D2, 1e-18, None)
    D = np.sqrt(D2)
    # sum over i<j of log(1/D) = - sum_{i<j} log(D)
    iu = np.triu_indices(N, k=1)
    vals = np.log(1.0 / D[iu])
    return math.fsum(vals.tolist())


def _log_energy_and_grad(P):
    # Gradient of E = sum_{i<j} log(1/|Pi-Pj|) = -sum_{i<j} log|Pi-Pj|
    # dE/dPi = sum_{j != i} (Pi - Pj) / |Pi - Pj|^2
    D2 = np.einsum('ij,ij->i', P, P)[:, None] + np.einsum('ij,ij->i', P, P)[None, :] \
        - 2.0 * (P @ P.T)
    np.fill_diagonal(D2, 1.0)
    D2 = np.clip(D2, 1e-18, None)
    D = np.sqrt(D2)
    # grad_i = sum_j (Pi - Pj)/Dij^2
    # (Pi - Pj)/Dij^2 ; note 1/Dij^2 = 1/D2
    inv = 1.0 / D2
    inv[np.diag_indices(N)] = 0.0
    # grad = P @ inv - (P.T @ inv)  ... let's do it clearly
    # term_ij = (Pi - Pj) * inv_ij
    # grad_i = sum_j term_ij
    # = P_i * sum_j inv_ij - sum_j P_j inv_ij
    row_sum = inv.sum(axis=1)
    grad = P * row_sum[:, None] - (inv @ P)
    # Energy
    iu = np.triu_indices(N, k=1)
    vals = np.log(1.0 / D[iu])
    return math.fsum(vals.tolist()), grad


def _project(P):
    nrm = np.linalg.norm(P, axis=1, keepdims=True)
    nrm = np.where(nrm < 1e-15, 1.0, nrm)
    return P / nrm


def _golden_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _rand_sphere(n, rng):
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _optimize(P, max_iter, tol, deadline):
    P = _project(P.astype(np.float64))
    E, G = _log_energy_and_grad(P)
    best_E, best_P = E, P.copy()
    step = 1e-3
    for it in range(max_iter):
        if time.time() > deadline:
            break
        # Riemannian gradient step with backtracking
        Gn = np.linalg.norm(G)
        if Gn < tol:
            break
        d = -G / (Gn + 1e-300)
        lr = step
        for _ in range(30):
            cand = P + lr * d
            cand = _project(cand)
            E2, _ = _log_energy_and_grad(cand)
            if E2 < E:
                break
            lr *= 0.5
        else:
            # accept tiny step
            cand = _project(P + 1e-6 * d)
            E2 = _log_energy(cand)
        if E2 < E:
            P = cand
            E = E2
            step = min(step * 1.05, 5e-2)
        else:
            step *= 0.5
        if E < best_E:
            best_E, best_P = E, P.copy()
    return best_E, best_P


def construct():
    deadline = time.time() + 140.0
    inits = []
    inits.append(_golden_spiral())
    # perturbed fibonacci
    for _ in range(3):
        base = _golden_spiral()
        noise = _RNG.standard_normal((N, 3)) * 0.05
        inits.append(_project(base + noise))
    # random
    inits.append(_rand_sphere(N, _RNG))

    best_E = None
    best_P = None
    for P0 in inits:
        if time.time() > deadline - 5:
            break
        E, P = _optimize(P0, max_iter=400, tol=1e-10, deadline=deadline - 5)
        if best_E is None or E < best_E:
            best_E, best_P = E, P
    # final high-precision polish
    if time.time() < deadline - 2:
        E, P = _optimize(best_P, max_iter=400, tol=1e-13, deadline=deadline - 2)
        if E < best_E:
            best_E, best_P = E, P

    best_P = _project(best_P)
    # validate
    if not np.all(np.isfinite(best_P)):
        return _golden_spiral()
    return best_P
