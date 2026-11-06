"""Rewritten program: explicit direct approach for 100-point log-energy on S^2."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-15):
        return -1e18
    return math.fsum(np.log(1.0 / d_upper).tolist())


def _log_energy_grad(pts):
    """Compute logarithmic energy and gradient on sphere (vectorized)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)

    # Energy: sum_{i<j} -log(d_ij)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-15):
        return -1e18, np.zeros_like(pts)
    energy = -np.sum(np.log(d_upper))

    # Gradient computation
    inv_d2 = np.zeros_like(d2)
    mask_non_diag = ~np.eye(n, dtype=bool)
    inv_d2[mask_non_diag] = 1.0 / d2[mask_non_diag]

    grad = -np.sum(inv_d2[:, :, np.newaxis] * diff, axis=1)

    # Project gradient onto tangent space
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= radial * pts

    return energy, grad


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _icosahedron_sphere(n):
    """Generate points based on icosahedron symmetry."""
    t = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    pts = []
    for i in range(n):
        v = verts[i % len(verts)]
        noise = np.random.RandomState(i).randn(3) * 0.1
        v = v + noise
        v /= np.linalg.norm(v)
        pts.append(v)
    return np.array(pts)


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _optimize_lbfgs(pts0, deadline):
    """Optimize using L-BFGS-B with sphere projection."""
    x0 = pts0.flatten()

    def objective(x):
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        e, _ = _log_energy_grad(pts)
        return e

    def grad(x):
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        _, g = _log_energy_grad(pts)
        return g.flatten()

    result = minimize(objective, x0, jac=grad, method='L-BFGS-B',
                      options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8})

    if result.success or result.fun < objective(x0):
        best_pts = _project_to_sphere(result.x.reshape(-1, 3))
        best_energy = _log_energy(best_pts)
        return best_pts, best_energy
    else:
        return pts0, _log_energy(pts0)


def _optimize_momentum(pts0, deadline, max_iter=3000):
    """Optimize using gradient descent with momentum and adaptive step size."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)

    v = np.zeros_like(pts)
    lr = 0.005
    momentum = 0.9
    no_improve_count = 0
    best_lr = lr

    for it in range(max_iter):
        if time.time() > deadline:
            break

        e, grad = _log_energy_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
            no_improve_count = 0
            best_lr = lr
        else:
            no_improve_count += 1

        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-13:
            break

        # Adaptive momentum: increase as we converge
        current_momentum = min(0.95, 0.9 + 0.05 * (it / max_iter))
        
        v = current_momentum * v - lr * grad
        pts_new = pts + v
        pts_new = _project_to_sphere(pts_new)

        e_new = _log_energy(pts_new)
        if e_new < e:
            pts = pts_new
            # More aggressive learning rate increase when improving
            lr = min(lr * 1.03, 0.05)
        else:
            lr *= 0.65
            v *= 0.4
            # Reset to best known learning rate if stuck
            if no_improve_count > 50:
                lr = best_lr * 0.5
                v = np.zeros_like(pts)
                no_improve_count = 0
            if lr < 1e-12:
                break

    return best_pts, best_energy


def _optimize_l_bfgs_precise(pts0, deadline, maxiter=2000):
    """High-precision L-BFGS-B optimization with tighter tolerances."""
    x0 = pts0.flatten()

    def objective(x):
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        e, _ = _log_energy_grad(pts)
        return e

    def grad(x):
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        _, g = _log_energy_grad(pts)
        return g.flatten()

    result = minimize(objective, x0, jac=grad, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'maxfun': maxiter * 5,
                               'ftol': 1e-15, 'gtol': 1e-10, 'm': 50})

    if result.success or result.fun < objective(x0):
        best_pts = _project_to_sphere(result.x.reshape(-1, 3))
        best_energy = _log_energy(best_pts)
        return best_pts, best_energy
    else:
        return pts0, _log_energy(pts0)


def _polish_high_precision(pts0, deadline, max_iter=5000):
    """Very fine-tuned momentum optimization for final polishing."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)

    v = np.zeros_like(pts)
    lr = 0.001
    momentum = 0.97
    no_improve_count = 0
    best_lr = lr

    for it in range(max_iter):
        if time.time() > deadline:
            break

        e, grad = _log_energy_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
            no_improve_count = 0
            best_lr = lr
        else:
            no_improve_count += 1

        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-14:
            break

        # Use higher momentum for smoother convergence in polish phase
        v = momentum * v - lr * grad
        pts_new = pts + v
        pts_new = _project_to_sphere(pts_new)

        e_new = _log_energy(pts_new)
        if e_new < e:
            pts = pts_new
            # Gradual learning rate increase
            lr = min(lr * 1.01, 0.005)
        else:
            lr *= 0.7
            v *= 0.3
            # If stuck for too long, reset with best known lr
            if no_improve_count > 80:
                lr = best_lr * 0.5
                v = np.zeros_like(pts)
                no_improve_count = 0
            if lr < 1e-15:
                break

    return best_pts, best_energy


def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    best_pts = None
    best_energy = np.inf
    # Track top candidates for diverse refinement
    candidates = []  # list of (energy, pts)

    # Multiple diverse starting configurations
    starts = [
        ("fibonacci", _fibonacci_sphere(N)),
        ("random_42", _random_sphere(N, seed=42)),
        ("random_123", _random_sphere(N, seed=123)),
        ("random_999", _random_sphere(N, seed=999)),
        ("random_777", _random_sphere(N, seed=777)),
        ("random_555", _random_sphere(N, seed=555)),
        ("random_314", _random_sphere(N, seed=314)),
        ("icosahedron", _icosahedron_sphere(N)),
    ]

    # Phase 1: Quick L-BFGS-B on all starts to find promising basins
    for name, pts0 in starts:
        if time.time() > deadline - 55:
            break

        x0 = pts0.flatten()

        def objective_quick(x):
            pts = x.reshape(-1, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            pts = pts / norms
            e, _ = _log_energy_grad(pts)
            return e

        def grad_quick(x):
            pts = x.reshape(-1, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            pts = pts / norms
            _, g = _log_energy_grad(pts)
            return g.flatten()

        result = minimize(objective_quick, x0, jac=grad_quick, method='L-BFGS-B',
                          options={'maxiter': 200, 'maxfun': 800, 'ftol': 1e-10, 'gtol': 1e-6})

        if result.success or result.fun < objective_quick(x0):
            pts_opt = _project_to_sphere(result.x.reshape(-1, 3))
            e_opt = _log_energy(pts_opt)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt
            # Keep top 3 diverse candidates
            candidates.append((e_opt, pts_opt))
            candidates.sort(key=lambda x: x[0])
            if len(candidates) > 3:
                candidates.pop()

    # Phase 2: Momentum refinement on top candidates (diverse search)
    for i, (e_cand, pts_cand) in enumerate(candidates):
        if time.time() > deadline - 35:
            break
        time_per_candidate = (deadline - 35 - time.time()) / max(1, len(candidates) - i)
        pts_mom, e_mom = _optimize_momentum(pts_cand, time.time() + time_per_candidate, max_iter=4000)
        if e_mom < best_energy:
            best_energy = e_mom
            best_pts = pts_mom

    # Phase 3: High-precision L-BFGS-B on best
    if best_pts is not None and time.time() < deadline - 20:
        pts_precise, e_precise = _optimize_l_bfgs_precise(best_pts, deadline, maxiter=2000)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = pts_precise

    # Phase 4: Final high-precision polishing
    if best_pts is not None and time.time() < deadline - 8:
        pts_polish, e_polish = _polish_high_precision(best_pts, deadline, max_iter=10000)
        if e_polish < best_energy:
            best_energy = e_polish
            best_pts = pts_polish

    # Phase 5: One more L-BFGS-B pass if time allows
    if best_pts is not None and time.time() < deadline - 3:
        pts_final, e_final = _optimize_l_bfgs_precise(best_pts, deadline, maxiter=1500)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = pts_final

    return best_pts
# EVOLVE-BLOCK-END
