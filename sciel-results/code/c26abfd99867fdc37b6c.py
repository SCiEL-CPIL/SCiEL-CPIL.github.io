"""
Logarithmic energy minimization of 150 points on the unit sphere.

Strategy:
  - Multi-start local optimization (Fibonacci spiral, geodesic dome, random).
  - Basin hopping with Riemannian L-BFGS as the primary global search mechanism.
  - Riemannian L-BFGS polish for high-precision local optimization.
  - Final high-precision polish pass with tight tolerances.

All time is bounded by an explicit wall-clock deadline with a safety margin.
"""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# Energy and gradient
# ---------------------------------------------------------------------------
def _energy_and_grad(pts):
    """
    Logarithmic energy and tangent-space gradient.
    pts: (N, 3) unit vectors.
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    grad_i (Cartesian) = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2, projected to tangent.
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)

    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    energy = -math.fsum(np.log(dists).tolist())

    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    row_sums = np.sum(inv_dist2, axis=1)
    grad = inv_dist2 @ pts - pts * row_sums[:, np.newaxis]

    # Project to tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    return energy, grad


def _energy_only(pts):
    """Compute only the energy (fsum for precision)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    return -math.fsum(np.log(dists).tolist())


def _project_to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


# ---------------------------------------------------------------------------
# Riemannian (exponential map) utilities
# ---------------------------------------------------------------------------
def _exp_map_tangent(pts, tangent):
    """
    Exponential map on the sphere: move pts along tangent vectors.
    For a unit vector x and tangent v, the exponential map is
        exp_x(v) = cos(|v|) x + sin(|v|) (v/|v|)
    """
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    safe_norm = np.where(norm > 1e-15, norm, 1.0)
    u = tangent / safe_norm
    c = np.cos(norm)
    s = np.sin(norm)
    out = c * pts + s * u
    return _project_to_sphere(out)


# ---------------------------------------------------------------------------
# Riemannian L-BFGS polish (optimizes in tangent space)
# ---------------------------------------------------------------------------
def _riemannian_lbfgs_polish(pts, deadline, maxiter=400, ftol=1e-16, gtol=1e-13):
    """
    Polish a configuration using L-BFGS-B on tangent-space coordinates.
    """
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    def fun(t):
        tv = t.reshape(N, 3)
        dot = np.sum(tv * pts, axis=1, keepdims=True)
        tv = tv - dot * pts
        new_pts = _exp_map_tangent(pts, tv)
        e, g = _energy_and_grad(new_pts)
        return e, g.flatten()

    t0 = np.zeros(3 * N)
    try:
        res = minimize(fun, t0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': ftol,
                                'gtol': gtol, 'maxls': 30})
        if time.time() < deadline:
            tv = res.x.reshape(N, 3)
            dot = np.sum(tv * pts, axis=1, keepdims=True)
            tv = tv - dot * pts
            new_pts = _exp_map_tangent(pts, tv)
            new_energy = _energy_only(new_pts)
            if new_energy < best_energy and np.all(np.isfinite(new_pts)):
                best_energy = new_energy
                best_pts = new_pts.copy()
    except Exception:
        pass

    return best_pts, best_energy


# ---------------------------------------------------------------------------
# Cartesian L-BFGS polish (fallback / final)
# ---------------------------------------------------------------------------
def _lbfgs_polish(pts, deadline, maxiter=500, ftol=1e-16, gtol=1e-14):
    """L-BFGS-B polish using flat Cartesian coordinates with projection."""
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': ftol,
                                'gtol': gtol, 'maxls': 30})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_res = _energy_only(p_res)
        if e_res < best_energy and np.all(np.isfinite(p_res)):
            best_energy = e_res
            best_pts = p_res.copy()
    except Exception:
        pass
    return best_pts, best_energy


# ---------------------------------------------------------------------------
# Initial configurations
# ---------------------------------------------------------------------------
def _fibonacci_spiral(n, offset=0.0):
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _geodesic_dome(n, seed=0):
    """
    Generate points via recursive subdivision of an icosahedron.
    """
    rng = np.random.RandomState(seed)
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi_g, 0], [1, phi_g, 0], [-1, -phi_g, 0], [1, -phi_g, 0],
        [0, -1, phi_g], [0, 1, phi_g], [0, -1, -phi_g], [0, 1, -phi_g],
        [phi_g, 0, -1], [phi_g, 0, 1], [-phi_g, 0, -1], [-phi_g, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    edges = [(0, 11), (0, 5), (0, 1), (0, 7), (0, 10), (1, 5), (5, 11),
             (11, 10), (10, 7), (7, 1), (1, 6), (6, 10), (2, 11), (2, 5),
             (2, 9), (2, 4), (2, 8), (3, 5), (3, 11), (3, 4), (3, 9),
             (3, 6), (4, 9), (5, 4), (11, 4), (10, 7), (6, 1), (8, 4),
             (9, 8), (8, 5), (6, 9), (7, 2), (8, 3)]

    while len(verts) < n:
        new_verts = []
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            mid /= np.linalg.norm(mid)
            new_verts.append(mid)
        verts = np.vstack([verts, np.array(new_verts)])
        new_edges = []
        base = len(verts) - len(new_verts)
        for k, (i, j) in enumerate(edges):
            m = base + k
            new_edges.append((i, m))
            new_edges.append((m, j))
        edges = new_edges

    if len(verts) > n:
        idx = rng.choice(len(verts), size=n, replace=False)
        verts = verts[idx]
    return _project_to_sphere(verts)


def _random_points(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)


# ---------------------------------------------------------------------------
# Basin Hopping with Riemannian L-BFGS
# ---------------------------------------------------------------------------
def _basin_hopping(pts0, deadline, rng, max_iter=200, T_start=1.0, T_end=0.01,
                   step_scale_start=0.5, step_scale_end=0.05,
                   polish_maxiter=150):
    """
    Basin hopping: perturb current best configuration with random tangent-space
    noise, polish with Riemannian L-BFGS, accept/reject via Metropolis criterion.
    """
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    current_pts = pts.copy()
    current_energy = best_energy

    n_iter = 0
    cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_iter))
    step_cooling_rate = (step_scale_end / step_scale_start) ** (1.0 / max(1, max_iter))
    T = T_start
    step_scale = step_scale_start

    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        n_iter += 1

        # Generate random perturbation in tangent space
        noise = rng.randn(N, 3) * step_scale
        # Project to tangent space
        dot_n = np.sum(noise * current_pts, axis=1, keepdims=True)
        noise = noise - dot_n * current_pts

        # Move via exponential map
        perturbed = _exp_map_tangent(current_pts, noise)
        perturbed = _project_to_sphere(perturbed)

        # Polish
        polished, pol_energy = _riemannian_lbfgs_polish(perturbed, deadline - 1.0,
                                                        maxiter=polish_maxiter)

        # Metropolis acceptance
        dE = pol_energy - current_energy
        if dE < 0 or rng.rand() < math.exp(-dE / max(T, 1e-10)):
            current_pts = polished.copy()
            current_energy = pol_energy

        if pol_energy < best_energy:
            best_energy = pol_energy
            best_pts = polished.copy()

        # Anneal
        T *= cooling_rate
        step_scale *= step_cooling_rate

    return best_pts, best_energy


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()
    total_time_budget = 145.0  # 5s safety margin before 150s kill
    deadline_global = start_time + total_time_budget

    rng = np.random.RandomState(777)

    # Initial configurations
    starts = [
        _fibonacci_spiral(N),
        _geodesic_dome(N, seed=0),
        _random_points(N, seed=1),
        _random_points(N, seed=2),
        _random_points(N, seed=3),
    ]

    best_pts = None
    best_energy = float('inf')

    # Phase 1: Local optimization from multiple starts (~10% of time)
    phase1_deadline = start_time + 0.10 * total_time_budget
    for pts0 in starts:
        if time.time() > phase1_deadline:
            break
        pts_opt, e = _riemannian_lbfgs_polish(pts0, phase1_deadline, maxiter=150)
        if e < best_energy:
            best_energy = e
            best_pts = pts_opt.copy()

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        best_energy = _energy_only(best_pts)

    # Phase 2: Basin hopping (~80% of time)
    phase2_deadline = start_time + 0.90 * total_time_budget
    bh_pts, bh_energy = _basin_hopping(best_pts, phase2_deadline, rng,
                                       max_iter=300, T_start=0.8, T_end=0.005,
                                       step_scale_start=0.4, step_scale_end=0.03,
                                       polish_maxiter=120)
    if bh_energy < best_energy:
        best_energy = bh_energy
        best_pts = bh_pts.copy()

    # Phase 3: Final high-precision polish (~10% of time)
    if time.time() < deadline_global - 3.0:
        best_pts, e = _riemannian_lbfgs_polish(best_pts, deadline_global - 1.0,
                                               maxiter=1000, ftol=1e-18, gtol=1e-15)
        if e < best_energy:
            best_energy = e
            best_pts = best_pts.copy()
        # One more Cartesian polish for final squeeze
        if time.time() < deadline_global - 1.0:
            best_pts, e2 = _lbfgs_polish(best_pts, deadline_global - 0.5,
                                         maxiter=500, ftol=1e-18, gtol=1e-15)
            if e2 < best_energy:
                best_energy = e2
                best_pts = best_pts.copy()

    # Safety check
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)

    best_pts = _project_to_sphere(best_pts)
    return best_pts

# EVOLVE-BLOCK-END
