"""Logarithmic energy minimization of 150 points on the unit sphere.
Multi-start with Adam, Velocity-Verlet, basin hopping, and multi-pass L-BFGS polish."""

N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Log energy and tangent-space gradient for points on unit sphere."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)

    iu = np.triu_indices(n, k=1)
    energy = -math.fsum(np.log(np.sqrt(dist2[iu])).tolist())

    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)

    # grad[i] = sum_{j != i} (x_j - x_i) / dist_ij^2
    grad = inv_dist2 @ pts
    row_sums = np.sum(inv_dist2, axis=1)
    grad -= pts * row_sums[:, np.newaxis]

    # Project to tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    return energy, grad


def _energy_only(pts):
    """Compute just the energy."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(n, k=1)
    return -math.fsum(np.log(np.sqrt(dist2[iu])).tolist())


def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def _fibonacci_spiral(n, offset=0.0, twist=1.0):
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i * twist
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _geodesic_dome(n, subdivisions=4):
    """Generate points from a geodesic dome (icosahedron-based)."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Faces of icosahedron
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    all_pts = verts.copy()
    face_list = faces.copy()
    
    for _ in range(subdivisions):
        new_faces = []
        mid_cache = {}
        
        def midpoint(p1_idx, p2_idx):
            key = (min(p1_idx, p2_idx), max(p1_idx, p2_idx))
            if key in mid_cache:
                return mid_cache[key]
            p1 = all_pts[p1_idx]
            p2 = all_pts[p2_idx]
            mid = (p1 + p2) / 2.0
            mid = mid / np.linalg.norm(mid)
            idx = len(all_pts)
            all_pts = np.vstack([all_pts, mid])
            mid_cache[key] = idx
            return idx
        
        for f in face_list:
            a, b, c = f
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])
        
        face_list = new_faces
    
    # Subsample if we have more than n points
    if len(all_pts) > n:
        rng = np.random.RandomState(123)
        idxs = rng.choice(len(all_pts), n, replace=False)
        return all_pts[idxs]
    else:
        # Pad with random points if needed
        while len(all_pts) < n:
            p = rng.randn(3)
            p = p / np.linalg.norm(p)
            all_pts = np.vstack([all_pts, p])
        return all_pts[:n]


def _random_points(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _adam_optimize(pts0, max_iter=4000, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
                   deadline=None, check_interval=25):
    """Adam optimization on the sphere using tangent-space gradient."""
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)

    for t in range(1, max_iter + 1):
        if deadline is not None and (t % check_interval == 0 or t == 1):
            if time.time() > deadline:
                break

        energy, grad = _energy_and_grad(pts)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        update = m_hat / (np.sqrt(v_hat) + eps)

        pts_new = pts - lr * update
        pts_new = _project_to_sphere(pts_new)

        if not np.all(np.isfinite(pts_new)):
            break

        energy_new = _energy_only(pts_new)
        if energy_new < best_energy:
            best_energy = energy_new
            best_pts = pts_new.copy()
        pts = pts_new

    return best_pts, best_energy


def _lbfgs_polish(pts, deadline, maxiter=400):
    """L-BFGS-B polish using flat Cartesian coords with projection inside."""
    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': maxiter * 30,
                                'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _energy_only(p_res)
        if e_check < _energy_only(pts):
            return p_res
    except Exception:
        pass
    return pts


def _lbfgs_tight_polish(pts, deadline, maxiter=800):
    """Ultra-tight L-BFGS-B polish for final high precision."""
    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': maxiter * 40,
                                'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 50})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _energy_only(p_res)
        if e_check < _energy_only(pts):
            return p_res
    except Exception:
        pass
    return pts


def _tangent_vector(point, rng):
    rand_vec = rng.randn(3)
    tangent = rand_vec - np.dot(rand_vec, point) * point
    norm = np.linalg.norm(tangent)
    if norm < 1e-15:
        rand_vec = rng.randn(3)
        tangent = rand_vec - np.dot(rand_vec, point) * point
        norm = np.linalg.norm(tangent)
    return tangent / norm


def _basin_hopping(pts0, deadline, rng, max_iter=80):
    """Basin hopping: perturb a subset, Adam-polish, Metropolis accept."""
    current = _project_to_sphere(pts0.copy())
    current_energy = _energy_only(current)
    best = current.copy()
    best_energy = current_energy

    T = 0.15
    T_min = 1e-7
    cooling = 0.97

    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        if T < T_min:
            break

        n_perturb = rng.randint(3, 10)
        idxs = rng.choice(N, n_perturb, replace=False)
        new_pts = current.copy()
        for idx in idxs:
            tangent = _tangent_vector(new_pts[idx], rng)
            new_pts[idx] = new_pts[idx] + T * tangent
        new_pts = _project_to_sphere(new_pts)

        new_pts, new_energy = _adam_optimize(new_pts, max_iter=800, lr=0.005,
                                             deadline=deadline - 2.0, check_interval=10)

        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best = current.copy()
        T *= cooling

    return best, best_energy


def _velocity_verlet_sphere(points, deadline, dt=0.01, n_steps=5000, damping=0.98):
    """Velocity-Verlet integration on the sphere with adaptive timestep and damping."""
    pts = _project_to_sphere(points.copy())
    vel = np.zeros_like(pts)
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    _, grad0 = _energy_and_grad(pts)
    force = -grad0
    
    for step in range(n_steps):
        if time.time() > deadline - 1.0:
            break
        
        vel_half = vel + 0.5 * dt * force
        dot_v = np.sum(vel_half * pts, axis=1, keepdims=True)
        vel_half = vel_half - dot_v * pts
        
        pts_new = pts + dt * vel_half
        pts_new = _project_to_sphere(pts_new)
        
        _, grad_new = _energy_and_grad(pts_new)
        force_new = -grad_new
        dot_f = np.sum(force_new * pts_new, axis=1, keepdims=True)
        force_new = force_new - dot_f * pts_new
        
        vel = (vel_half + 0.5 * dt * force_new) * damping
        dot_v2 = np.sum(vel * pts_new, axis=1, keepdims=True)
        vel = vel - dot_v2 * pts_new
        
        pts = pts_new
        force = force_new
        
        energy = _energy_only(pts)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        if step > 0 and energy > best_energy * 1.01:
            dt *= 0.95
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin

    rng = np.random.RandomState(777)

    # Diverse initializations
    starts = [_fibonacci_spiral(N)]
    for off in [0.1, 0.3, 0.5, 0.7]:
        starts.append(_fibonacci_spiral(N, offset=off))
    for tw in [0.8, 1.3, 2.0]:
        starts.append(_fibonacci_spiral(N, twist=tw))
    starts.append(_geodesic_dome(N, subdivisions=3))
    for seed in [1, 2, 3, 7, 13, 42, 100, 200]:
        starts.append(_random_points(N, seed=seed))
    for k in range(3):
        p = _fibonacci_spiral(N) + rng.randn(N, 3) * 0.15
        starts.append(_project_to_sphere(p))

    best_pts = None
    best_energy = float('inf')

    # Phase 1: Adam optimization from each start
    for idx, pts0 in enumerate(starts):
        if time.time() > deadline - 50.0:
            break
        pts_opt, energy_opt = _adam_optimize(pts0, max_iter=3000, lr=0.005,
                                             deadline=deadline - 50.0,
                                             check_interval=25)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt.copy()

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        best_energy = _energy_only(best_pts)

    # Phase 2: Velocity-Verlet molecular dynamics
    if time.time() < deadline - 42.0:
        vv_pts, vv_energy = _velocity_verlet_sphere(best_pts, deadline - 42.0,
                                                     dt=0.005, n_steps=8000, damping=0.97)
        if vv_energy < best_energy:
            best_energy = vv_energy
            best_pts = vv_pts.copy()

    # Phase 2b: Basin hopping from the best configuration
    if time.time() < deadline - 28.0:
        bh_pts, bh_energy = _basin_hopping(best_pts, deadline - 28.0, rng,
                                           max_iter=50)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pts = bh_pts.copy()

    # Phase 3: L-BFGS-B polish
    if time.time() < deadline - 14.0:
        best_pts = _lbfgs_polish(best_pts, deadline - 14.0, maxiter=350)
        best_energy = _energy_only(best_pts)

    # Phase 3b: Second basin hopping round from polished config
    if time.time() < deadline - 10.0:
        bh2_pts, bh2_energy = _basin_hopping(best_pts, deadline - 10.0, rng,
                                             max_iter=30)
        if bh2_energy < best_energy:
            best_energy = bh2_energy
            best_pts = bh2_pts.copy()

    # Phase 4: Ultra-tight L-BFGS-B polish
    if time.time() < deadline - 3.0:
        best_pts = _lbfgs_tight_polish(best_pts, deadline - 3.0, maxiter=600)
        best_energy = _energy_only(best_pts)

    # Phase 5: Final L-BFGS pass
    if time.time() < deadline - 0.5:
        best_pts = _lbfgs_polish(best_pts, deadline - 0.5, maxiter=200)
        best_energy = _energy_only(best_pts)

    best_pts = _project_to_sphere(best_pts)

    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_spiral(N)

    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_spiral(N)

    return best_pts
# EVOLVE-BLOCK-END
