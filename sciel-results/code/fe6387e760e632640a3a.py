"""Rewritten program: replica-exchange simulated annealing + gradient polish
for minimal logarithmic energy of 150 points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 140.0  # safety margin before 150s kill
START_TIME = time.time()


def _remaining():
    return DEADLINE - (time.time() - START_TIME)


def _energy_and_grad(pts):
    """Compute log-energy sum_{i<j} log(1/|xi-xj|) and its gradient w.r.t. pts.
    pts: (N,3) array of unit vectors. Gradient is in ambient R^3; we project
    onto tangent space externally or let L-BFGS handle via re-normalisation."""
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]  # (N,N,3)
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)  # avoid log(0)
    # Energy: sum over i<j of log(1/d) = - sum_{i<j} log(d)
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    energy = -np.sum(np.log(d))
    # Gradient: d/dx_i sum_{j!=i} -log|xi-xj| = sum_{j!=i} (xi-xj)/|xi-xj|^2
    # Because d/dx log(1/|xi-xj|) = d/dx (-log|xi-xj|) = -(xi-xj)/|xi-xj|^2
    # Wait: energy = sum log(1/d) = -sum log(d). d/dx_i log(d_ij) = (xi-xj)/d_ij^2 * ... 
    # Let's derive: d_ij = |xi - xj|. d/dxi log(d_ij) = (xi-xj)/|xi-xj|^2.
    # So d/dxi (-log d_ij) = -(xi-xj)/d_ij^2.
    # grad_i = sum_{j != i} -(xi - xj)/d_ij^2 = sum_{j != i} (xj - xi)/d_ij^2
    grad = np.zeros_like(pts)
    # diff[i,j] = xi - xj. We want (xj - xi)/d^2 = -diff/d^2
    inv_d2 = 1.0 / (dist ** 2)
    np.fill_diagonal(inv_d2, 0.0)
    # grad_i = sum_j (xj - xi)/d_ij^2 = - sum_j diff[i,j] / d_ij^2
    # = - (diff * inv_d2[:, :, None]).sum(axis=1)
    grad = -(diff * inv_d2[:, :, None]).sum(axis=1)
    return energy, grad


def _renorm(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n = np.where(n < 1e-15, 1.0, n)
    return pts / n


def _fibonacci_start(seed=0):
    rng = np.random.RandomState(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # small random perturbation to break symmetry
    pts = pts + rng.randn(N, 3) * 1e-4
    return _renorm(pts)


def _random_start(seed):
    rng = np.random.RandomState(seed)
    v = rng.randn(N, 3)
    return _renorm(v)


def _anneal_replica(pts, T, steps, rng, max_time):
    """Run gradient + random moves at temperature T. Returns (pts, best_pts, best_E)."""
    best_E, best_pts = _energy_and_grad(pts)[0], pts.copy()
    cur_E = best_E
    for s in range(steps):
        if _remaining() < 2.0:
            break
        # Gradient step (simple projected gradient descent with adaptive step)
        E, g = _energy_and_grad(pts)
        cur_E = E
        # Project gradient to tangent space: g_t = g - (g . x) x
        g_dot_x = np.sum(g * pts, axis=1, keepdims=True)
        g_t = g - g_dot_x * pts
        # Adaptive step: try to reduce energy
        step = 0.01 / (1.0 + T * 0.1)
        cand = _renorm(pts - step * g_t)
        E_cand, _ = _energy_and_grad(cand)
        # Metropolis accept
        dE = E_cand - cur_E
        if dE < 0 or rng.rand() < math.exp(-dE / max(T, 1e-12)):
            pts = cand
            cur_E = E_cand
        if cur_E < best_E:
            best_E = cur_E
            best_pts = pts.copy()
        # Occasionally add a random kick (more at high T)
        if rng.rand() < 0.1 * min(1.0, T):
            kick = rng.randn(N, 3) * 0.02 * (1.0 + T)
            pts = _renorm(pts + kick)
            cur_E, _ = _energy_and_grad(pts)
    return pts, best_pts, best_E


def construct():
    global START_TIME
    START_TIME = time.time()
    rng = np.random.RandomState(42)

    # Initial diverse starts
    starts = []
    starts.append(_fibonacci_start(0))
    starts.append(_random_start(1))
    starts.append(_random_start(2))
    # Anisotropic scaling start
    p = _fibonacci_start(3)
    p[:, 0] *= 1.1
    p[:, 1] *= 0.9
    starts.append(_renorm(p))

    # Replica-exchange simulated annealing
    # Temperatures: cold to hot
    T_list = [0.01, 0.1, 1.0, 5.0, 20.0]
    n_replicas = len(T_list)
    replicas = [s.copy() for s in starts[:n_replicas]]
    best_overall = min(replicas, key=lambda p: _energy_and_grad(p)[0])
    best_E = _energy_and_grad(best_overall)[0]

    # Schedule: cool down over time
    total_steps = 2000
    for step in range(total_steps):
        if _remaining() < 10.0:
            break
        # Cool temperatures
        cool_factor = 1.0 - 0.3 * (step / total_steps)
        T_now = [max(0.005, T * cool_factor) for T in T_list]

        # Update each replica
        for r in range(n_replicas):
            if _remaining() < 10.0:
                break
            replicas[r], bp, be = _anneal_replica(replicas[r], T_now[r], 2, rng, _remaining())
            if be < best_E:
                best_E = be
                best_overall = bp.copy()

        # Replica exchange: try to swap adjacent replicas
        for r in range(n_replicas - 1):
            if _remaining() < 10.0:
                break
            E_cold = _energy_and_grad(replicas[r])[0]
            E_hot = _energy_and_grad(replicas[r + 1])[0]
            # Swap if hot replica has lower energy than cold (standard RE criterion)
            # Acceptance: P = 1 if E_hot < E_cold, else exp(-(E_cold - E_hot)/(1/T_cold - 1/T_hot))
            dE = E_hot - E_cold
            dInvT = 1.0 / max(T_now[r], 1e-9) - 1.0 / max(T_now[r + 1], 1e-9)
            if dE < 0:
                replicas[r], replicas[r + 1] = replicas[r + 1].copy(), replicas[r].copy()
            else:
                # P = exp(-dE / dInvT) but dInvT could be negative; use standard formula
                # Actually: P = exp(-(E_cold/T_cold - E_hot/T_hot))
                logP = -(E_cold / max(T_now[r], 1e-9) - E_hot / max(T_now[r + 1], 1e-9))
                if rng.rand() < math.exp(min(0.0, logP)):
                    replicas[r], replicas[r + 1] = replicas[r + 1].copy(), replicas[r].copy()

    # Final high-precision polish on best configuration
    if _remaining() > 5.0:
        def obj(x):
            pts = _renorm(x)
            E, _ = _energy_and_grad(pts)
            return E

        def obj_grad(x):
            pts = _renorm(x)
            E, g = _energy_and_grad(pts)
            # Project gradient to tangent
            g_dot_x = np.sum(g * pts, axis=1, keepdims=True)
            g_t = g - g_dot_x * pts
            return g_t.ravel()

        x0 = best_overall.ravel()
        res = minimize(obj, x0, jac=obj_grad, method='L-BFGS-B',
                       options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-15})
        if _remaining() > 2.0:
            cand = _renorm(res.x.reshape(N, 3))
            E_cand, _ = _energy_and_grad(cand)
            if E_cand < best_E:
                best_overall = cand
                best_E = E_cand

    # Final verification
    final = _renorm(best_overall)
    # Ensure no points too close
    diff = final[:, None, :] - final[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if np.min(dist[iu]) < 1e-9:
        # Fallback to fibonacci
        final = _fibonacci_start(0)
    return final
