"""Sphere covering with 50 points: multi-start + smooth LSE+Riesz combined objective +
multi-temperature cascade + Lloyd + SA + high-precision polish.

Key improvements over prior version:
1. Combined LSE+Riesz objective balances smooth covering-radius minimization with
   global spread (Riesz s=2), preventing premature collapse of local structure.
2. Multi-temperature cascade: run LSE at several temperatures from high to low,
   allowing the optimizer to first explore the landscape broadly then refine.
3. Tighter time management with explicit deadline guards at every stage.
4. Final high-precision polish with very low temperature and tight tolerances.
"""

import numpy as np
import time
from scipy.spatial import ConvexHull, cKDTree
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 50

def _normalize(pts):
    """Normalize points to unit sphere."""
    pts = np.asarray(pts, dtype=float)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius(points):
    """Compute the covering radius in radians using the convex hull."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        tree = cKDTree(points)
        rng = np.random.RandomState(42)
        n_samples = 20000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return np.max(2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1)))
    
    max_r = 0.0
    for simplex in hull.simplices:
        pts = points[simplex]
        if len(pts) != 3:
            continue
        a, b, c = pts[0], pts[1], pts[2]
        v = np.cross(b - a, c - a)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            continue
        v = v / v_norm
        dot_a = np.dot(v, a)
        if dot_a < 0:
            v = -v
            dot_a = -dot_a
        cos_r = np.clip(dot_a, -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_r:
            max_r = r
    return max_r

def _facet_circumcenters(points):
    """Compute circumcenters and radii for all triangular facets of the convex hull."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None
    
    centers = []
    radii = []
    simplices_valid = []
    for simplex in hull.simplices:
        pts = points[simplex]
        if len(pts) != 3:
            continue
        a, b, c = pts[0], pts[1], pts[2]
        v = np.cross(b - a, c - a)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            continue
        v = v / v_norm
        dot_a = np.dot(v, a)
        if dot_a < 0:
            v = -v
            dot_a = -dot_a
        cos_r = np.clip(dot_a, -1.0, 1.0)
        r = np.arccos(cos_r)
        centers.append(v)
        radii.append(r)
        simplices_valid.append(simplex)
    
    if not centers:
        return None, None, None
    
    return np.array(centers), np.array(radii), simplices_valid

def _smooth_covering_radius_grad(x, temperature=0.05):
    """
    Compute smooth log-sum-exp approximation of covering radius and its gradient.
    x is a flat array of shape (N*3,).
    """
    pts = x.reshape(N, 3)
    pts = _normalize(pts)
    
    centers, radii, simplices_valid = _facet_circumcenters(pts)
    if centers is None:
        tree = cKDTree(pts)
        rng = np.random.RandomState(42)
        n_samples = 5000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        max_d = np.max(dists)
        max_r = 2.0 * np.arcsin(np.clip(max_d / 2.0, 0, 1))
        return max_r, np.zeros_like(x)
    
    T = max(temperature, 1e-6)
    scaled = radii / T
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    obj = log_sum_exp * T
    
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    
    grad_pts = np.zeros_like(pts)
    for v, simplex, w in zip(centers, simplices_valid, weights):
        if w < 1e-10:
            continue
        for idx in simplex:
            p = pts[idx]
            dot_vp = np.dot(v, p)
            sin_r = np.sqrt(np.maximum(0.0, 1.0 - dot_vp * dot_vp))
            if sin_r < 1e-12:
                continue
            dr_dp = -v / sin_r
            dr_dp_tangent = dr_dp - np.dot(dr_dp, p) * p
            grad_pts[idx] += w * dr_dp_tangent
    
    return obj, (grad_pts * T).flatten()

def _combined_lse_riesz_grad(x, temperature=0.05, riesz_weight=0.1, s=2.0):
    """
    Combined objective: LSE covering radius + weighted Riesz s-energy.
    The Riesz term provides a smooth global-spreading force that prevents
    premature local collapse while the LSE term drives the covering radius down.
    """
    pts = x.reshape(N, 3)
    pts = _normalize(pts)
    
    # LSE part
    centers, radii, simplices_valid = _facet_circumcenters(pts)
    if centers is None:
        tree = cKDTree(pts)
        rng = np.random.RandomState(42)
        n_samples = 5000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        max_d = np.max(dists)
        max_r = 2.0 * np.arcsin(np.clip(max_d / 2.0, 0, 1))
        return max_r, np.zeros_like(x)
    
    T = max(temperature, 1e-6)
    scaled = radii / T
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    obj_lse = log_sum_exp * T
    
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    
    grad_lse = np.zeros_like(pts)
    for v, simplex, w in zip(centers, simplices_valid, weights):
        if w < 1e-10:
            continue
        for idx in simplex:
            p = pts[idx]
            dot_vp = np.dot(v, p)
            sin_r = np.sqrt(np.maximum(0.0, 1.0 - dot_vp * dot_vp))
            if sin_r < 1e-12:
                continue
            dr_dp = -v / sin_r
            dr_dp_tangent = dr_dp - np.dot(dr_dp, p) * p
            grad_lse[idx] += w * dr_dp_tangent
    
    # Riesz part
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(np.maximum(dist2, 1e-15))
    
    obj_riesz = np.sum(dist ** (-s))
    
    grad_riesz = np.zeros_like(pts)
    for i in range(N):
        d = dist[i, :].copy()
        d[i] = np.inf
        contrib = -s * (pts[i] - pts) / (d ** (s + 2))[:, np.newaxis]
        contrib[i, :] = 0
        g = np.sum(contrib, axis=0)
        g -= np.dot(g, pts[i]) * pts[i]
        grad_riesz[i] = g
    
    obj = obj_lse + riesz_weight * obj_riesz
    grad = (grad_lse * T + riesz_weight * grad_riesz).flatten()
    
    return obj, grad

def _riesz_energy_grad(x, s=2.0):
    """Compute Riesz s-energy and its gradient. Lower energy = more uniform."""
    pts = x.reshape(N, 3)
    pts = _normalize(pts)
    
    # Vectorized pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(np.maximum(dist2, 1e-15))
    
    energy = np.sum(dist ** (-s))
    
    # Gradient
    grad = np.zeros_like(pts)
    for i in range(N):
        d = dist[i, :]
        d[i] = np.inf
        # d/dp_i sum_{j!=i} |p_i - p_j|^(-s) = -s * sum_{j!=i} (p_i - p_j) / |p_i - p_j|^(s+2)
        contrib = -s * (pts[i] - pts) / (d ** (s + 2))[:, np.newaxis]
        contrib[i, :] = 0
        grad[i] = np.sum(contrib, axis=0)
        # Project onto tangent plane
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    
    return energy, grad.flatten()

def _lbfgs_smooth(points, time_limit=None, start_time=None, temperature=0.05, maxiter=300):
    """Refine points using L-BFGS-B with smooth covering radius objective."""
    points = _normalize(points)
    x0 = points.flatten()
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    try:
        result = minimize(
            lambda x: _smooth_covering_radius_grad(x, temperature)[0],
            x0,
            jac=lambda x: _smooth_covering_radius_grad(x, temperature)[1],
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000}
        )
        if result.fun < best_cr:
            new_pts = _normalize(result.x.reshape(N, 3))
            cr = _covering_radius(new_pts)
            if cr < best_cr:
                best_cr = cr
                best_points = new_pts.copy()
    except Exception:
        pass
    
    if time_limit is not None and start_time is not None:
        if time.time() - start_time > time_limit:
            return best_points, best_cr
    
    return best_points, best_cr

def _lbfgs_combined(points, time_limit=None, start_time=None, temperature=0.05,
                    riesz_weight=0.1, s=2.0, maxiter=300):
    """Refine points using L-BFGS-B with combined LSE+Riesz objective."""
    points = _normalize(points)
    x0 = points.flatten()
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    try:
        result = minimize(
            lambda x: _combined_lse_riesz_grad(x, temperature, riesz_weight, s)[0],
            x0,
            jac=lambda x: _combined_lse_riesz_grad(x, temperature, riesz_weight, s)[1],
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000}
        )
        new_pts = _normalize(result.x.reshape(N, 3))
        cr = _covering_radius(new_pts)
        if cr < best_cr:
            best_cr = cr
            best_points = new_pts.copy()
    except Exception:
        pass
    
    if time_limit is not None and start_time is not None:
        if time.time() - start_time > time_limit:
            return best_points, best_cr
    
    return best_points, best_cr

def _lbfgs_riesz(points, time_limit=None, start_time=None, s=2.0, maxiter=200):
    """Refine points using L-BFGS-B with Riesz energy objective."""
    points = _normalize(points)
    x0 = points.flatten()
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    try:
        result = minimize(
            lambda x: _riesz_energy_grad(x, s)[0],
            x0,
            jac=lambda x: _riesz_energy_grad(x, s)[1],
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 3000}
        )
        if result.success:
            new_pts = _normalize(result.x.reshape(N, 3))
            cr = _covering_radius(new_pts)
            if cr < best_cr:
                best_cr = cr
                best_points = new_pts.copy()
    except Exception:
        pass
    
    return best_points, best_cr

def _spherical_gd(points, obj_func, grad_func, lr=0.01, max_iters=500, time_limit=None, start_time=None):
    """Spherical gradient descent with adaptive learning rate."""
    points = _normalize(points)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        x = points.flatten()
        obj, grad = grad_func(x)
        
        # Adaptive learning rate
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        step = lr * grad / grad_norm
        
        new_points = _normalize(points + step.reshape(N, 3))
        new_cr = _covering_radius(new_points)
        
        if new_cr < best_cr:
            best_cr = new_cr
            best_points = new_points.copy()
            points = new_points
        else:
            lr *= 0.9
    
    return best_points, best_cr

def _spherical_lloyd(points, max_iters=1000, tol=1e-12, time_limit=None, start_time=None):
    """Spherical Lloyd's algorithm using Voronoi cell centroids."""
    points = _normalize(points)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        new_points = np.zeros_like(points)
        
        # For each point, find all facets it belongs to, compute circumcenters, average
        point_to_facets = [[] for _ in range(len(points))]
        for fi, simplex in enumerate(hull.simplices):
            for idx in simplex:
                point_to_facets[idx].append(fi)
        
        for i in range(len(points)):
            if not point_to_facets[i]:
                new_points[i] = points[i]
                continue
            
            circumcenters = []
            for fi in point_to_facets[i]:
                simplex = hull.simplices[fi]
                pts = points[simplex]
                a, b, c = pts[0], pts[1], pts[2]
                v = np.cross(b - a, c - a)
                v_norm = np.linalg.norm(v)
                if v_norm < 1e-12:
                    continue
                v = v / v_norm
                dot_a = np.dot(v, a)
                if dot_a < 0:
                    v = -v
                circumcenters.append(v)
            
            if not circumcenters:
                new_points[i] = points[i]
            else:
                centroid = np.mean(circumcenters, axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm < 1e-12:
                    new_points[i] = points[i]
                else:
                    new_points[i] = centroid / centroid_norm
        
        diff = np.max(np.linalg.norm(new_points - points, axis=1))
        if diff < tol:
            points = new_points
            break
        
        points = new_points
        cr = _covering_radius(points)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    return best_points, best_cr

def _simulated_annealing(points, time_limit=None, start_time=None, T0=0.12, T1=0.005, n_iter=3000):
    """Simulated annealing to escape local minima."""
    points = _normalize(points)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    current_cr = best_cr
    
    rng = np.random.RandomState(42)
    
    for it in range(n_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        alpha = it / max(n_iter - 1, 1)
        T = T0 * (T1 / T0) ** alpha
        
        # Perturb random points
        n_pert = rng.randint(1, 5)
        indices = rng.choice(N, size=n_pert, replace=False)
        perturbed = points.copy()
        for idx in indices:
            noise = rng.randn(3) * T
            perturbed[idx] = points[idx] + noise
            perturbed[idx] /= np.linalg.norm(perturbed[idx])
        
        cr_new = _covering_radius(perturbed)
        
        if cr_new < current_cr:
            points = perturbed
            current_cr = cr_new
            if cr_new < best_cr:
                best_cr = cr_new
                best_points = points.copy()
        else:
            accept_prob = np.exp(-(cr_new - current_cr) / max(T, 1e-10))
            if rng.random() < accept_prob:
                points = perturbed
                current_cr = cr_new
    
    return best_points, best_cr

def _fibonacci_points(n, offset=0.5):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _jittered_fibonacci_points(n, jitter=0.05, seed=42):
    """Generate jittered Fibonacci spiral points."""
    rng = np.random.RandomState(seed)
    pts = _fibonacci_points(n)
    for i in range(n):
        noise = rng.randn(3) * jitter
        pts[i] = pts[i] + noise
        pts[i] /= np.linalg.norm(pts[i])
    return pts

def _random_points(n, seed=42):
    """Generate random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _icosahedron_subdivided_points(n):
    """Generate points by subdividing an icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Subdivide each triangle
    def subdivide_triangle(a, b, c, depth):
        if depth == 0:
            return [a, b, c]
        ab = (a + b) / 2
        ab /= np.linalg.norm(ab)
        bc = (b + c) / 2
        bc /= np.linalg.norm(bc)
        ca = (c + a) / 2
        ca /= np.linalg.norm(ca)
        return (subdivide_triangle(a, ab, ca, depth-1) +
                subdivide_triangle(ab, b, bc, depth-1) +
                subdivide_triangle(ca, bc, c, depth-1) +
                subdivide_triangle(ab, bc, ca, depth-1))
    
    # Get icosahedron faces
    hull = ConvexHull(ico)
    points = []
    for simplex in hull.simplices:
        a, b, c = ico[simplex[0]], ico[simplex[1]], ico[simplex[2]]
        points.extend(subdivide_triangle(a, b, c, 2))
    
    points = np.array(points)
    # Remove duplicates approximately
    unique = [points[0]]
    for p in points[1:]:
        if np.min(np.linalg.norm(p - np.array(unique), axis=1)) > 1e-6:
            unique.append(p)
    points = np.array(unique)
    
    # If we have more than n, sample; if less, add random
    if len(points) >= n:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(points), size=n, replace=False)
        points = points[indices]
    else:
        rng = np.random.RandomState(42)
        extra = rng.randn(n - len(points), 3)
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        points = np.vstack([points, extra])
    
    return points

def _multi_stage_optimize(points, time_limit=None, start_time=None):
    """Multi-stage optimization with combined LSE+Riesz and multi-temperature cascade."""
    points = _normalize(points)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    def _remaining():
        if time_limit is not None and start_time is not None:
            return max(0.0, time_limit - (time.time() - start_time))
        return deadline - time.time()
    
    # Stage 1: Riesz energy for initial spread
    rem = _remaining()
    if rem > 15:
        points, cr = _lbfgs_riesz(points, time_limit=min(rem * 0.2, 12), start_time=start_time, s=2.0, maxiter=120)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Stage 2: Multi-temperature LSE+Riesz cascade
    # Start with higher temperature for exploration, then decrease for exploitation
    temperatures = [0.15, 0.08, 0.04, 0.02]
    for T in temperatures:
        rem = _remaining()
        if rem < 8:
            break
        # Use combined objective with decreasing Riesz weight as T decreases
        rw = 0.15 if T > 0.1 else 0.08 if T > 0.05 else 0.03
        points, cr = _lbfgs_combined(points, time_limit=min(rem * 0.25, 15), start_time=start_time,
                                     temperature=T, riesz_weight=rw, s=2.0, maxiter=250)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Stage 3: Lloyd relaxation
    rem = _remaining()
    if rem > 8:
        points, cr = _spherical_lloyd(points, max_iters=400, time_limit=min(rem * 0.2, 10), start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Stage 4: Simulated annealing
    rem = _remaining()
    if rem > 12:
        points, cr = _simulated_annealing(points, time_limit=min(rem * 0.25, 12), start_time=start_time,
                                          T0=0.08, T1=0.003, n_iter=1500)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Stage 5: Final high-precision LSE polish
    rem = _remaining()
    if rem > 8:
        points, cr = _lbfgs_smooth(points, time_limit=min(rem * 0.3, 12), start_time=start_time,
                                   temperature=0.01, maxiter=300)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Stage 6: Final Lloyd with tight tolerance
    rem = _remaining()
    if rem > 5:
        points, cr = _spherical_lloyd(points, max_iters=800, tol=1e-14,
                                      time_limit=min(rem * 0.5, 10), start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    return best_points, best_cr

# Global deadline
deadline = None

def construct():
    global deadline
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Generate diverse starting points
    starts = [
        _fibonacci_points(N, offset=0.5),
        _fibonacci_points(N, offset=0.0),
        _fibonacci_points(N, offset=1.0),
        _jittered_fibonacci_points(N, jitter=0.05, seed=42),
        _jittered_fibonacci_points(N, jitter=0.1, seed=123),
        _random_points(N, seed=42),
        _random_points(N, seed=123),
        _random_points(N, seed=999),
        _icosahedron_subdivided_points(N),
    ]
    
    best_points = None
    best_cr = np.inf
    
    # Phase 1: Multi-start with full multi-stage optimization
    # Allocate ~55% of total time to multi-start
    total_budget = deadline - start_time
    multi_start_budget = total_budget * 0.55
    time_per_start = max(12, multi_start_budget / len(starts))
    
    for i, sp in enumerate(starts):
        remaining = deadline - time.time()
        if remaining < 15:
            break
        
        t_limit = min(time_per_start, remaining * 0.85)
        points, cr = _multi_stage_optimize(sp, time_limit=t_limit, start_time=start_time)
        
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
        
        # Update time budget adaptively
        elapsed = time.time() - start_time
        remaining_starts = len(starts) - i - 1
        if remaining_starts > 0:
            time_per_start = max(10, (deadline - time.time()) * 0.55 / remaining_starts)
    
    if best_points is None:
        best_points = _fibonacci_points(N)
        best_cr = _covering_radius(best_points)
    
    # Phase 2: Additional refinement on best solution
    remaining = deadline - time.time()
    if remaining > 20:
        points, cr = _multi_stage_optimize(best_points, time_limit=remaining * 0.35, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Phase 3: Perturbation-based local search
    remaining = deadline - time.time()
    if remaining > 15:
        rng = np.random.RandomState(42)
        max_perts = 8
        for p in range(max_perts):
            remaining = deadline - time.time()
            if remaining < 10:
                break
            
            n_pert = rng.randint(2, 6)
            indices = rng.choice(N, size=n_pert, replace=False)
            perturbed = best_points.copy()
            for idx in indices:
                noise = rng.randn(3) * 0.05
                perturbed[idx] = best_points[idx] + noise
                perturbed[idx] /= np.linalg.norm(perturbed[idx])
            
            t_limit = min(10, remaining * 0.4)
            points, cr = _multi_stage_optimize(perturbed, time_limit=t_limit, start_time=start_time)
            if cr < best_cr:
                best_cr = cr
                best_points = points.copy()
    
    # Phase 4: Final high-precision polish
    remaining = deadline - time.time()
    if remaining > 10:
        # Very low temperature LSE for precise local refinement
        points, cr = _lbfgs_smooth(best_points, time_limit=min(remaining * 0.25, 12),
                                   start_time=start_time, temperature=0.008, maxiter=400)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
        
        remaining = deadline - time.time()
        if remaining > 5:
            points, cr = _spherical_lloyd(best_points, max_iters=1500, tol=1e-15,
                                          time_limit=min(remaining * 0.4, 10), start_time=start_time)
            if cr < best_cr:
                best_cr = cr
                best_points = points.copy()
    
    # Normalize final result
    best_points = _normalize(best_points)
    
    return best_points
