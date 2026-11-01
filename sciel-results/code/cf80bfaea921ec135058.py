"""Packing 50 equal circles in a unit circle - momentum dynamics + soft-min L-BFGS refinement."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START = time.time()
_DEADLINE = _START + 130.0  # 20s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    # Pairwise distances
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    return min(r_cont, r_pair)


def _soft_min(values, beta):
    """Compute the soft minimum of an array of values."""
    v = np.asarray(values, dtype=np.float64)
    m = np.min(v)
    exp_terms = np.exp(-beta * (v - m))
    return m - (1.0 / beta) * np.log(np.sum(exp_terms))


def _objective_and_gradient(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints)
    params: flat array of length 2*n
    Returns (objective_value, gradient)
    """
    centers = params.reshape(n, 2)
    
    # Pairwise distances
    i_idx, j_idx = np.triu_indices(n, k=1)
    diff = centers[i_idx, :] - centers[j_idx, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    boundary_dists = 1.0 - norms
    
    # Combine all constraints
    vals = np.concatenate([dists / 2.0, boundary_dists])
    
    obj = -_soft_min(vals, beta)
    
    # Compute gradient
    m = np.min(vals)
    exp_vals = np.exp(-beta * (vals - m))
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    grad = np.zeros_like(centers)
    
    # Pairwise part
    pair_weights = weights[:len(i_idx)]
    safe_dists = np.where(dists > 1e-8, dists, 1.0)
    unit_vecs = diff / safe_dists[:, np.newaxis]
    weighted_vecs = pair_weights[:, np.newaxis] * 0.5 * unit_vecs
    
    np.add.at(grad, i_idx, weighted_vecs)
    np.add.at(grad, j_idx, -weighted_vecs)
    
    # Boundary part
    bound_weights = weights[len(i_idx):]
    safe_norms = np.where(norms > 1e-8, norms, 1.0)
    grad -= bound_weights[:, np.newaxis] * centers / safe_norms[:, np.newaxis]
    
    return obj, -grad


def _optimize_lbfgs(centers, beta, maxiter, gtol):
    """Optimize using L-BFGS-B."""
    try:
        res = minimize(
            _objective_and_gradient,
            centers.flatten(),
            args=(N, beta),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol, 'disp': False}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return centers


def _get_hex_lattice(n, seed_offset=0):
    """Generate a hexagonal lattice initialization."""
    rng = np.random.default_rng(42 + seed_offset)
    s_est = np.sqrt(2 * np.pi / (n * np.sqrt(3)))
    
    points = []
    max_range = int(2.0 / s_est) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s_est + (j % 2) * s_est / 2.0
            y = j * s_est * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
                
    if len(points) < n:
        points = []
        for _ in range(n * 10):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points = np.vstack([points, [[x, y]]])
    
    points = points + rng.normal(0, 0.01, points.shape)
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]


def _get_ring_init(n, seed=42):
    """
    Generate a ring-based initialization matching known optimal structure.
    The best known packing has rings: 1, 6, 12, 18, 13 = 50
    """
    rng = np.random.default_rng(seed)
    
    rings = [1, 6, 12, 18, 13]
    while sum(rings) > n:
        rings[-1] -= 1
        if rings[-1] <= 0:
            rings.pop()
    while sum(rings) < n:
        rings.append(1)
    
    points = []
    ring_radii = [0.0, 0.25, 0.50, 0.75, 0.95]
    
    for ring_idx, count in enumerate(rings):
        if count == 0:
            continue
        if ring_idx == 0:
            points.append([0.0, 0.0])
        else:
            radius = ring_radii[min(ring_idx, len(ring_radii) - 1)]
            angle_offset = rng.uniform(0, 2 * np.pi)
            for k in range(count):
                angle = angle_offset + 2 * np.pi * k / count
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                r_jitter = rng.normal(0, 0.005)
                x += r_jitter * np.cos(angle)
                y += r_jitter * np.sin(angle)
                points.append([x, y])
    
    points = np.array(points[:n])
    
    norms = np.linalg.norm(points, axis=1)
    mask = norms > 0.95
    if np.any(mask):
        points[mask] *= (0.95 / norms[mask])[:, np.newaxis]
    
    return points


def _momentum_dynamics(centers, start_time, max_steps=5000):
    """
    Rigid-body momentum dynamics with impulse-based collision resolution.
    Uses velocity Verlet integration for stability.
    """
    current = centers.copy().astype(np.float64)
    velocities = np.zeros_like(current)
    n = len(current)
    
    best_score = _score(current)
    best_positions = current.copy()
    
    dt = 0.002
    damping = 0.995
    repulsion_strength = 5.0
    boundary_strength = 2.0
    
    for step in range(max_steps):
        if time.time() - start_time > _DEADLINE - 20:
            break
        
        # Compute forces
        forces = np.zeros_like(current)
        
        # Pairwise repulsion
        i_idx, j_idx = np.triu_indices(n, k=1)
        diff = current[i_idx, :] - current[j_idx, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        safe_dists = np.where(dists > 1e-8, dists, 1.0)
        
        # Strong repulsion: 1/d^2
        repulsion_mag = repulsion_strength / (safe_dists ** 2 + 1e-4)
        unit_vecs = diff / safe_dists[:, np.newaxis]
        weighted_forces = repulsion_mag[:, np.newaxis] * unit_vecs
        
        np.add.at(forces, i_idx, weighted_forces)
        np.add.at(forces, j_idx, -weighted_forces)
        
        # Boundary force: push inward when close to edge
        norms = np.linalg.norm(current, axis=1)
        boundary_violation = np.maximum(0, norms - 0.85)
        safe_norms = np.where(norms > 1e-8, norms, 1.0)
        boundary_force = boundary_strength * boundary_violation[:, np.newaxis] * current / safe_norms[:, np.newaxis]
        forces -= boundary_force
        
        # Velocity Verlet: update velocity
        velocities = damping * velocities + dt * forces
        
        # Update position
        new_positions = current + dt * velocities
        
        # Hard constraint: keep inside unit disk
        new_norms = np.linalg.norm(new_positions, axis=1)
        mask = new_norms > 1.0
        if np.any(mask):
            new_positions[mask] *= (1.0 / new_norms[mask])[:, np.newaxis]
            # Kill outward velocity component
            for idx in np.where(mask)[0]:
                n_hat = new_positions[idx] / new_norms[idx]
                v_out = np.dot(velocities[idx], n_hat)
                if v_out > 0:
                    velocities[idx] -= v_out * n_hat
        
        current = new_positions
        
        # Track best
        sc = _score(current)
        if sc > best_score:
            best_score = sc
            best_positions = current.copy()
        
        # Adaptive step: reduce if forces are large
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 10.0:
            dt *= 0.5
        elif max_force < 0.1:
            dt = min(dt * 1.05, 0.005)
        
        # Convergence check
        if max_force < 1e-6:
            break
    
    return best_positions, best_score


def construct():
    """Construct 50 circle centers in unit circle maximizing common radius."""
    best_pts = None
    best_score = 0.0
    
    # Multiple initialization strategies
    initializations = [
        _get_hex_lattice(N, seed_offset=0),
        _get_ring_init(N, seed=42),
        _get_hex_lattice(N, seed_offset=1),
        _get_ring_init(N, seed=123),
        _get_hex_lattice(N, seed_offset=2),
    ]
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() - _START > _DEADLINE - 25:
            break
        
        # Phase 1: Momentum dynamics to escape local minima
        dyn_pts, dyn_score = _momentum_dynamics(init_pts, _START, max_steps=3000)
        
        # Phase 2: L-BFGS refinement with increasing beta
        current_pts = dyn_pts
        current_score = dyn_score
        
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if time.time() - _START > _DEADLINE - 15:
                break
            refined = _optimize_lbfgs(current_pts, beta, maxiter=200, gtol=1e-6)
            ref_score = _score(refined)
            if ref_score > current_score:
                current_pts = refined
                current_score = ref_score
        
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()
    
    if best_pts is None:
        best_pts = _get_hex_lattice(N, seed_offset=0)
        best_score = _score(best_pts)
    
    # Final L-BFGS polish
    if time.time() - _START < _DEADLINE - 10:
        final_pts = _optimize_lbfgs(best_pts, 300.0, maxiter=300, gtol=1e-7)
        final_score = _score(final_pts)
        if final_score > best_score:
            best_pts = final_pts
            best_score = final_score
    
    # Final validation
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        best_pts[mask] /= r[mask, None] * (1.0 + 1e-6)
    
    return best_pts
