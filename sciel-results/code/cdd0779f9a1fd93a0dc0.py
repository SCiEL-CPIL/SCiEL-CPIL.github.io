"""
Logarithmic energy minimization of 150 points on the unit sphere.

Strategy:
  - Multi-start local optimization (Fibonacci spiral, geodesic dome, random).
  - Basin hopping with Riemannian (exponential-map / rotation) perturbations
    that respect the sphere geometry.
  - Riemannian L-BFGS polish with Barzilai-Borwein adaptive step sizes
    for accelerated local convergence.
  - Final high-precision polish pass.

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
    which is equivalent to rotating x by angle |v| about the axis v/|v|.
    This respects the sphere geometry exactly.
    """
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    # Avoid division by zero
    safe_norm = np.where(norm > 1e-15, norm, 1.0)
    u = tangent / safe_norm
    c = np.cos(norm)
    s = np.sin(norm)
    # Where norm ~ 0, c ~ 1, s ~ 0, so result ~ x
    out = c * pts + s * u
    return _project_to_sphere(out)


def _log_map_tangent(pts, target):
    """
    Logarithmic map: tangent vector at pts that maps to target on the sphere.
    v = theta * (target - cos(theta) x) / sin(theta), theta = arccos(x . target).
    """
    cos_theta = np.sum(pts * target, axis=1, keepdims=True)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)
    # Avoid division by zero
    safe_sin = np.where(sin_theta > 1e-12, sin_theta, 1.0)
    v = (theta / safe_sin) * (target - cos_theta * pts)
    # Zero out where theta ~ 0
    v = np.where((theta > 1e-12)[:, np.newaxis], v, 0.0)
    return v


# ---------------------------------------------------------------------------
# Barzilai-Borwein step size computation
# ---------------------------------------------------------------------------
def _bb_step_size(prev_grad, curr_grad, prev_pts, curr_pts, prev_step):
    """
    Compute Barzilai-Borwein step size from gradient and position changes.
    Uses the BB+ variant (min of the two BB step sizes) for robustness.
    """
    # s = x_k - x_{k-1} (flattened)
    # y = g_k - g_{k-1} (flattened)
    s = (curr_pts - prev_pts).flatten()
    y = (curr_grad - prev_grad).flatten()
    
    s_norm_sq = np.dot(s, s)
    y_norm_sq = np.dot(y, y)
    
    if s_norm_sq < 1e-30 or y_norm_sq < 1e-30:
        return prev_step
    
    s_dot_y = np.dot(s, y)
    
    # BB- (s'y / s's) and BB+ (s'y / y'y)
    bb_minus = s_dot_y / s_norm_sq
    bb_plus = s_dot_y / y_norm_sq
    
    # Use the one that gives a smaller step (more conservative)
    step = min(abs(bb_minus), abs(bb_plus))
    
    # Clamp to reasonable range
    step = np.clip(step, 1e-10, 1.0)
    
    return step


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
# Barzilai-Borwein gradient descent polish (tangent space)
# ---------------------------------------------------------------------------
def _bb_gradient_polish(pts, deadline, maxiter=500, ftol=1e-15, gtol=1e-12):
    """
    Polish using Barzilai-Borwein adaptive step sizes in tangent space.
    This can be faster than L-BFGS near convergence due to adaptive step sizing.
    """
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    # Initial step size
    step = 0.01
    
    # Track previous state for BB computation
    prev_pts = pts.copy()
    prev_grad = None
    
    for it in range(maxiter):
        if time.time() > deadline:
            break
        
        e, g = _energy_and_grad(pts)
        
        # Check convergence
        g_norm = np.linalg.norm(g)
        if g_norm < gtol or abs(e - best_energy) < ftol:
            break
        
        # Compute BB step size if we have previous gradient
        if prev_grad is not None:
            step = _bb_step_size(prev_grad, g, prev_pts, pts, step)
        
        # Line search: try decreasing step until we get sufficient decrease
        alpha = step
        best_alpha = step
        best_e = e
        
        for ls in range(20):
            # Take step in tangent space
            tangent = -alpha * g
            new_pts = _exp_map_tangent(pts, tangent)
            new_pts = _project_to_sphere(new_pts)
            
            new_e = _energy_only(new_pts)
            
            if new_e < best_e:
                best_e = new_e
                best_alpha = alpha
                # Check for sufficient decrease (Armijo-like)
                if new_e < e - 1e-4 * alpha * np.dot(g.flatten(), g.flatten()):
                    break
            alpha *= 0.5
        
        # Update
        if best_e < e:
            tangent = -best_alpha * g
            prev_pts = pts.copy()
            prev_grad = g.copy()
            pts = _exp_map_tangent(pts, tangent)
            pts = _project_to_sphere(pts)
            
            if best_e < best_energy:
                best_energy = best_e
                best_pts = pts.copy()
        else:
            # No improvement, reduce step
            step *= 0.5
    
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

    # Subdivide until we have enough points
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
# Basin hopping with Riemannian perturbations
# ---------------------------------------------------------------------------
def _basin_hopping(pts0, deadline, rng, max_iter=200):
    """
    Basin Hopping with Riemannian (exponential-map) perturbations.
    """
    current = _project_to_sphere(pts0.copy())
    current_energy = _energy_only(current)
    best = current.copy()
    best_energy = current_energy

    T = 0.4
    T_min = 1e-7
    cooling = 0.96

    for it in range(max_iter):
        if time.time() > deadline - 1.0:
            break

        # Perturb a random subset using the exponential map
        n_perturb = rng.randint(5, 18)
        idxs = rng.choice(N, n_perturb, replace=False)

        tangent = np.zeros_like(current)
        for idx in idxs:
            rand_vec = rng.randn(3)
            tv = rand_vec - np.dot(rand_vec, current[idx]) * current[idx]
            norm = np.linalg.norm(tv)
            if norm < 1e-15:
                continue
            tv /= norm
            angle = T * abs(rng.randn())
            tangent[idx] = angle * tv

        new_pts = _exp_map_tangent(current, tangent)
        new_pts = _project_to_sphere(new_pts)

        # Polish with Riemannian L-BFGS
        new_pts, new_energy = _riemannian_lbfgs_polish(new_pts, deadline, maxiter=80)

        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > T_min and rng.rand() < math.exp(-delta_e / T)):
            current = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best = current.copy()

        T *= cooling
        if T < T_min:
            break

    return best, best_energy


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

    # Phase 1: Local optimization from multiple starts (~15% of time)
    phase1_deadline = start_time + 0.15 * total_time_budget
    for pts0 in starts:
        if time.time() > phase1_deadline:
            break
        pts_opt, e = _riemannian_lbfgs_polish(pts0, phase1_deadline, maxiter=100)
        if e < best_energy:
            best_energy = e
            best_pts = pts_opt.copy()

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        best_energy = _energy_only(best_pts)

    # Phase 2: Basin hopping (~55% of time)
    phase2_deadline = start_time + 0.70 * total_time_budget
    bh_pts, bh_energy = _basin_hopping(best_pts, phase2_deadline, rng, max_iter=200)
    if bh_energy < best_energy:
        best_energy = bh_energy
        best_pts = bh_pts.copy()

    # Phase 3: Barzilai-Borwein gradient polish (~15% of time)
    phase3_deadline = start_time + 0.85 * total_time_budget
    bb_pts, bb_energy = _bb_gradient_polish(best_pts, phase3_deadline, maxiter=300,
                                             ftol=1e-16, gtol=1e-13)
    if bb_energy < best_energy:
        best_energy = bb_energy
        best_pts = bb_pts.copy()

    # Phase 4: Final high-precision polish (~15% of time)
    if time.time() < deadline_global - 2.0:
        # Riemannian L-BFGS with tight tolerances
        best_pts, e = _riemannian_lbfgs_polish(best_pts, deadline_global - 1.0,
                                               maxiter=500, ftol=1e-18, gtol=1e-15)
        if e < best_energy:
            best_energy = e
            best_pts = best_pts.copy()
        
        # One more Cartesian polish for final squeeze
        if time.time() < deadline_global - 1.0:
            best_pts, e2 = _lbfgs_polish(best_pts, deadline_global - 0.5,
                                         maxiter=300, ftol=1e-18, gtol=1e-15)
            if e2 < best_energy:
                best_energy = e2
                best_pts = best_pts.copy()
        
        # Final BB polish
        if time.time() < deadline_global - 0.5:
            best_pts, e3 = _bb_gradient_polish(best_pts, deadline_global - 0.2,
                                               maxiter=200, ftol=1e-18, gtol=1e-15)
            if e3 < best_energy:
                best_energy = e3
                best_pts = best_pts.copy()

    # Safety check
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)

    best_pts = _project_to_sphere(best_pts)
    return best_pts

# EVOLVE-BLOCK-END
