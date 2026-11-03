"""Rewritten: optimize log-energy on the sphere via projected gradient descent with restarts."""
import time
import numpy as np
from math import fsum

N = 150
DEADLINE = 150.0
SAFETY = 5.0  # seconds before the hard limit


def _energy(pts):
    # pts: (N,3). Returns sum_{i<j} log(1/|x_i-x_j|) = -sum log|diff|.
    N = pts.shape[0]
    total = 0.0
    for i in range(N):
        d = np.linalg.norm(pts[i] - pts[i + 1:], axis=1)
        total += -np.sum(np.log(d))
    return total


def _energy_grad(pts):
    # grad of E = -sum_{i<j} log|x_i-x_j| w.r.t. x_i
    # dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    N = pts.shape[0]
    grad = np.zeros_like(pts)
    for i in range(N):
        diff = pts[i] - pts
        d2 = np.einsum('ij,ij->i', diff, diff)
        d2[i] = np.inf  # avoid self
        contrib = diff / d2[:, None]
        grad[i] = np.sum(contrib, axis=0)
    return grad


def _project_tangent(pts, grad):
    # project grad onto tangent plane at each point: g - (g.x_i) x_i
    dots = np.einsum('ij,ij->i', grad, pts)
    return grad - dots[:, None] * pts


def _fibonacci_sphere():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _grad_descent(pts, rng, budget):
    pts = pts.copy()
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    best = pts.copy()
    best_e = _energy(best)
    lr = 1e-3
    n = 0
    while time.time() < budget:
        g = _energy_grad(pts)
        gt = _project_tangent(pts, g)
        gnorm = np.linalg.norm(gt)
        if gnorm < 1e-12:
            break
        step = gt * lr
        pts_new = pts + step
        pts_new /= np.linalg.norm(pts_new, axis=1, keepdims=True)
        e_new = _energy(pts_new)
        if e_new < best_e:
            best = pts_new.copy()
            best_e = e_new
            lr = min(lr * 1.15, 5e-2)
        else:
            lr *= 0.7
        pts = pts_new
        n += 1
    return best, best_e


def construct():
    t0 = time.time()
    budget = t0 + DEADLINE - SAFETY
    rng = np.random.default_rng(12345)
    best_pts = None
    best_e = np.inf

    # Start from a few different initializations.
    inits = [_fibonacci_sphere()]
    for _ in range(3):
        inits.append(_random_sphere(rng))

    for init in inits:
        if time.time() > budget:
            break
        pts, e = _grad_descent(init, rng, budget)
        if e < best_e:
            best_e = e
            best_pts = pts

    if best_pts is None:
        best_pts = _fibonacci_sphere()

    # Final high-precision polish with smaller steps.
    if time.time() < budget - 2.0:
        best_pts, best_e = _grad_descent(best_pts, rng, budget)

    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    return best_pts
