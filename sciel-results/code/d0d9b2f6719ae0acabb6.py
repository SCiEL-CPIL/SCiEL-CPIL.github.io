"""
Logarithmic energy minimization of 150 points on the unit sphere.

Strategy:
  - Multi-start local optimization (Fibonacci spiral, geodesic dome, random).
  - Molecular dynamics (Velocity-Verlet) with temperature annealing as the
    primary global search mechanism, exploiting the physical analogy to
    repulsive charges.
  - Riemannian L-BFGS polish for high-precision local optimization.
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


# ---------------------------------------------------------------------------
# Riemannian L-BFGS polish (optimizes in tangent space)
# ---------------------------------------------------------------------------
def _riemannian_lbfgs_polish(pts, deadline, maxiter=400, ftol=1e-16, gtol=1e-13):
    """
    Polish a configuration using L-BFGS-B on tangent-space coordinates.
    We represent the configuration as a flat tangent vector t (size 3N) and
    define f(t) = energy(exp(pts, t)), grad_f(t) = grad of energy at exp(pts,t)
    pulled back (which for the sphere exponential map with the metric is just
    the tangent gradient, since the exponential map is an isometry in the
    infinitesimal sense for our purposes here).

    We use a trust-region-like approach: start from t=0 (current config),
    optimize t, and if it improves, set pts = exp(pts, t_best) and repeat.
    """
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    def fun(t):
        # t is flat tangent vector (3N,). Interpret as (N,3) tangent.
        tv = t.reshape(N, 3)
        # Make sure tangent is actually tangent (subtract radial part)
        dot = np.sum(tv * pts, axis=1, keepdims=True)
        tv = tv - dot * pts
        new_pts = _exp_map_tangent(pts, tv)
        e, g = _energy_and_grad(new_pts)
        # Gradient of f w.r.t. t: since exp is local isometry, g (tangent) is
        # the gradient in t-space directly.
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
    Provides a different structure than the Fibonacci spiral.
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
        # Rebuild edges: each old edge (i,j) becomes (i, mid) and (mid, j)
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
# Molecular Dynamics (Velocity-Verlet) with temperature annealing
# ---------------------------------------------------------------------------
def _molecular_dynamics(pts0, deadline, rng, max_steps=10000, 
                        dt=0.002, T_start=0.5, T_end=0.001,
                        damping=0.995, polish_interval=500):
    """
    Molecular dynamics using Velocity-Verlet integration on the sphere
    with temperature annealing. This is structurally different from basin
    hopping: it uses physical dynamics where points repel each other, and
    temperature controls the exploration.
    
    The key insight: logarithmic repulsion is like a physical system of
    charges, so MD is a natural search method. Momentum helps escape
    local minima, and temperature annealing gradually cools the system.
    """
    pts = _project_to_sphere(pts0.copy())
    vel = np.zeros_like(pts)
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    # Initial force
    _, grad0 = _energy_and_grad(pts)
    force = -grad0  # Negative gradient = force
    
    # Project force to tangent space
    dot_f = np.sum(force * pts, axis=1, keepdims=True)
    force = force - dot_f * pts
    
    T = T_start
    cooling_rate = (T_end / T_start) ** (1.0 / max_steps)
    
    for step in range(max_steps):
        if time.time() > deadline - 1.0:
            break
        
        # Velocity-Verlet half step for velocity
        vel_half = vel + 0.5 * dt * force
        
        # Project velocity to tangent space
        dot_v = np.sum(vel_half * pts, axis=1, keepdims=True)
        vel_half = vel_half - dot_v * pts
        
        # Position update
        pts_new = pts + dt * vel_half
        pts_new = _project_to_sphere(pts_new)
        
        # New force
        _, grad_new = _energy_and_grad(pts_new)
        force_new = -grad_new
        
        # Project force to tangent space
        dot_f_new = np.sum(force_new * pts_new, axis=1, keepdims=True)
        force_new = force_new - dot_f_new * pts_new
        
        # Velocity update with damping
        vel = (vel_half + 0.5 * dt * force_new) * damping
        
        # Project velocity to tangent space
        dot_v2 = np.sum(vel * pts_new, axis=1, keepdims=True)
        vel = vel - dot_v2 * pts_new
        
        pts = pts_new
        force = force_new
        
        # Add thermal noise (Langevin-like)
        if T > T_end * 10:
            noise_scale = math.sqrt(2 * T * dt)
            noise = rng.randn(N, 3) * noise_scale
            # Project noise to tangent space
            dot_n = np.sum(noise * pts, axis=1, keepdims=True)
            noise = noise - dot_n * pts
            vel += noise
        
        # Track best
        energy = _energy_only(pts)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Periodic polish with L-BFGS to remove drift
        if step > 0 and step % polish_interval == 0 and time.time() < deadline - 5.0:
            polished, pol_energy = _riemannian_lbfgs_polish(pts, deadline - 5.0, maxiter=50)
            if pol_energy < best_energy:
                best_energy = pol_energy
                best_pts = polished.copy()
                # Reset dynamics from polished state
                pts = polished.copy()
                vel = np.zeros_like(pts)
                _, grad_pol = _energy_and_grad(pts)
                force = -grad_pol
                dot_f_pol = np.sum(force * pts, axis=1, keepdims=True)
                force = force - dot_f_pol * pts
        
        # Cool temperature
        T *= cooling_rate
    
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

    # Phase 1: Local optimization from multiple starts (~15% of time)
    phase1_deadline = start_time + 0.15 * total_time_budget
    for pts0 in starts:
        if time.time() > phase1_deadline:
            break
        pts_opt, e = _riemannian_lbfgs_polish(pts0, phase1_deadline, maxiter=120)
        if e < best_energy:
            best_energy = e
            best_pts = pts_opt.copy()

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        best_energy = _energy_only(best_pts)

    # Phase 2: Molecular dynamics with annealing (~70% of time)
    phase2_deadline = start_time + 0.85 * total_time_budget
    md_pts, md_energy = _molecular_dynamics(best_pts, phase2_deadline, rng,
                                            max_steps=15000, dt=0.002,
                                            T_start=0.4, T_end=0.001,
                                            damping=0.995, polish_interval=400)
    if md_energy < best_energy:
        best_energy = md_energy
        best_pts = md_pts.copy()

    # Phase 3: Final high-precision polish (~15% of time)
    if time.time() < deadline_global - 2.0:
        best_pts, e = _riemannian_lbfgs_polish(best_pts, deadline_global - 1.0,
                                               maxiter=800, ftol=1e-18, gtol=1e-15)
        if e < best_energy:
            best_energy = e
            best_pts = best_pts.copy()
        # One more Cartesian polish for final squeeze
        if time.time() < deadline_global - 1.0:
            best_pts, e2 = _lbfgs_polish(best_pts, deadline_global - 0.5,
                                         maxiter=400, ftol=1e-18, gtol=1e-15)
            if e2 < best_energy:
                best_energy = e2
                best_pts = best_pts.copy()

    # Safety check
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)

    best_pts = _project_to_sphere(best_pts)
    return best_pts

# EVOLVE-BLOCK-END
