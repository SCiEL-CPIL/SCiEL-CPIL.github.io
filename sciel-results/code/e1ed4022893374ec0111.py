"""Optimize logarithmic energy of 150 points on the unit sphere.

Strategy:
  - Multi-start local optimization (Fibonacci spiral, geodesic dome, random, clustered).
  - Basin hopping with Riemannian (exponential-map) perturbations that respect sphere geometry.
  - Riemannian L-BFGS polish: optimize in tangent space, then map back via exponential map.
  - Final high-precision Riemannian polish pass.
"""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150
DEADLINE = 148.0  # seconds, with 2s safety margin before 150s limit


def _energy_and_grad(pts):
    """Compute logarithmic energy and tangent-space gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    where d_ij is chordal distance.
    
    Tangent gradient: grad_i = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2, projected to tangent.
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


def _exp_map_tangent(pts, tangent):
    """
    Exponential map on the sphere: move pts along tangent vectors.
    For a unit vector x and tangent v, the exponential map is
        exp_x(v) = cos(|v|) x + sin(|v|) (v/|v|)
    which is equivalent to rotating x by angle |v| about the axis v/|v|.
    This respects the sphere geometry exactly.
    """
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    safe_norm = np.where(norm > 1e-15, norm, 1.0)
    u = tangent / safe_norm
    c = np.cos(norm)
    s = np.sin(norm)
    out = c * pts + s * u
    return _project_to_sphere(out)


def _fibonacci_sphere(n, offset=0.0):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _geodesic_dome(n, seed=0):
    """
    Generate points via recursive subdivision of an icosahedron.
    Provides a different structure than the Fibonacci spiral.
    """
    rng = np.random.RandomState(seed)
    
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Icosahedron faces
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    points = list(verts)
    
    # Subdivide until we have enough points
    while len(points) < n:
        new_faces = []
        mid_cache = {}
        
        def midpoint(p1, p2):
            key = (min(p1, p2), max(p1, p2))
            if key not in mid_cache:
                mid = (points[p1] + points[p2]) / 2.0
                mid = mid / np.linalg.norm(mid)
                mid_cache[key] = len(points)
                points.append(mid)
            return mid_cache[key]
        
        for face in faces:
            a, b, c = face
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])
        
        faces = new_faces
    
    return np.array(points[:n])


def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _riemannian_lbfgs_polish(pts, deadline, maxiter=2000, ftol=1e-16, gtol=1e-13):
    """
    Polish a configuration using L-BFGS-B on tangent-space coordinates.
    We represent the configuration as a flat tangent vector t (size 3N) and
    define f(t) = energy(exp(pts, t)), grad_f(t) = tangent gradient at exp(pts,t).
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
                                'gtol': gtol, 'maxls': 50})
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


def _optimize_single(pts0, max_time):
    """Optimize a single starting configuration using Riemannian L-BFGS-B.
    
    Uses exponential map to respect sphere geometry exactly.
    """
    start_time = time.time()
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    # Phase 1: Initial Riemannian L-BFGS polish
    phase1_deadline = start_time + max_time * 0.7
    if time.time() < phase1_deadline:
        try:
            pts_opt, e_opt = _riemannian_lbfgs_polish(pts, phase1_deadline, maxiter=2000)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass
    
    # Phase 2: Basin hopping with Riemannian perturbations
    if time.time() < start_time + max_time * 0.9:
        T = 0.1
        T_min = 1e-6
        cooling = 0.99
        
        while time.time() < start_time + max_time * 0.9 and T > T_min:
            # Pick 1-3 random points to perturb simultaneously
            n_move = np.random.randint(1, 4)
            indices = np.random.choice(N, size=n_move, replace=False)
            
            new_pts = best_pts.copy()
            for idx in indices:
                rand_dir = np.random.standard_normal(3)
                rand_dir -= np.dot(rand_dir, best_pts[idx]) * best_pts[idx]
                norm = np.linalg.norm(rand_dir)
                if norm > 1e-15:
                    rand_dir /= norm
                step_size = T * 0.2 * np.sqrt(n_move)
                tangent = step_size * rand_dir
                new_pts[idx] = _exp_map_tangent(best_pts[idx:idx+1], tangent.reshape(1,3))[0]
            
            new_e = _energy_only(new_pts)
            delta_e = new_e - best_energy
            
            if delta_e < 0 or np.random.random() < math.exp(-delta_e / T):
                best_pts = new_pts
                best_energy = new_e
                # Quick polish after accepting
                if time.time() < start_time + max_time * 0.9:
                    try:
                        pts_polish, e_polish = _riemannian_lbfgs_polish(best_pts, 
                                                                        time.time() + 1.0, 
                                                                        maxiter=200)
                        if e_polish < best_energy:
                            best_energy = e_polish
                            best_pts = pts_polish.copy()
                    except Exception:
                        pass
            
            T *= cooling
    
    # Phase 3: Final high-precision polish
    if time.time() < start_time + max_time - 2:
        try:
            pts_final, e_final = _riemannian_lbfgs_polish(best_pts, 
                                                          start_time + max_time - 1, 
                                                          maxiter=5000, gtol=1e-14)
            if e_final < best_energy:
                best_energy = e_final
                best_pts = pts_final.copy()
        except Exception:
            pass
    
    return _project_to_sphere(best_pts), best_energy


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere(N))
    
    # 2. Geodesic dome
    starts.append(_geodesic_dome(N, seed=42))
    
    # 3. Perturbed Fibonacci (different noise levels)
    for noise_scale in [0.03, 0.1, 0.2]:
        fib = _fibonacci_sphere(N)
        noise = noise_scale * rng.standard_normal((N, 3))
        perturbed = fib + noise
        starts.append(_project_to_sphere(perturbed))
    
    # 4. Random points
    for _ in range(4):
        starts.append(_random_sphere(N, rng))
    
    # 5. Clustered starts (to explore different basins)
    for _ in range(2):
        centers = _random_sphere(10, rng)
        pts = []
        for c in centers:
            local = c + 0.1 * rng.standard_normal((15, 3))
            pts.append(_project_to_sphere(local))
        starts.append(np.vstack(pts))
    
    # Optimize each start
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Optimize all starts with equal time allocation
    n_starts = len(starts)
    phase1_deadline = start_time + (DEADLINE - start_time) * 0.65
    time_per_opt = (phase1_deadline - start_time) / n_starts
    time_per_opt = max(4.0, min(15.0, time_per_opt))
    
    for i, s in enumerate(starts):
        if time.time() > phase1_deadline - 5:
            break
        
        remaining = phase1_deadline - time.time()
        if remaining < 4:
            break
        
        opt_time = min(time_per_opt, remaining - 2)
        if opt_time < 3:
            break
        
        try:
            pts_opt, e_opt = _optimize_single(s, opt_time)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            continue
    
    # Phase 2: Final high-precision polish on the best result
    if best_pts is not None and time.time() < DEADLINE - 5:
        remaining = DEADLINE - time.time() - 3
        if remaining > 5:
            try:
                pts_final, e_final = _riemannian_lbfgs_polish(best_pts, 
                                                              time.time() + remaining, 
                                                              maxiter=5000, gtol=1e-14)
                if e_final < best_energy:
                    best_energy = e_final
                    best_pts = pts_final.copy()
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    assert best_pts.shape == (N, 3)
    assert np.all(np.isfinite(best_pts))
    assert np.all(np.linalg.norm(best_pts, axis=1) > 1e-15)
    
    # Check no two points are too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    iu = np.triu_indices(N, k=1)
    min_dists = dists[iu]
    for idx in np.where(min_dists < 1e-9)[0]:
        i, j = iu[0][idx], iu[1][idx]
        best_pts[j] += 1e-8 * rng.standard_normal(3)
        best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts
