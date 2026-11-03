N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _compute_facet_properties(points, hull):
    simplices = hull.simplices
    n_facets = len(simplices)
    centers = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    for i, simplex in enumerate(simplices):
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        v1 = b - a
        v2 = c - a
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-15:
            centers[i] = (a + b + c) / 3
            norms = np.linalg.norm(centers[i])
            if norms > 1e-15:
                centers[i] /= norms
            else:
                centers[i] = a
            radii[i] = np.arccos(np.clip(np.dot(centers[i], a), -1, 1))
            continue
            
        n = cross / norm_cross
        if np.dot(n, a) < 0:
            n = -n
        centers[i] = n
        cos_r = np.clip(np.dot(n, a), -1.0, 1.0)
        radii[i] = np.arccos(cos_r)
        
    return centers, radii


def _optimize_covering(points, max_iters=5000, base_step=0.03):
    points = points.copy()
    best_points = points.copy()
    best_radius = np.inf
    
    try:
        hull = ConvexHull(points)
        centers, radii = _compute_facet_properties(points, hull)
        best_radius = np.max(radii)
    except Exception:
        return best_points, best_radius
    
    for it in range(max_iters):
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        centers, radii = _compute_facet_properties(points, hull)
        current_max_radius = np.max(radii)
        
        if current_max_radius < best_radius:
            best_radius = current_max_radius
            best_points = points.copy()
        
        step = base_step * (0.3 + 0.7 * (1 - it / max_iters))
        
        # Compute facet-based forces
        forces = np.zeros_like(points)
        for f_idx in range(len(hull.simplices)):
            r = radii[f_idx]
            weight = max(0, r - best_radius * 0.95) ** 2
            if weight < 1e-10:
                continue
            simplex = hull.simplices[f_idx]
            target = centers[f_idx]
            for v_idx in simplex:
                p = points[v_idx]
                v = target - p
                v_norm = np.linalg.norm(v)
                if v_norm > 1e-15:
                    forces[v_idx] += weight * (v / v_norm)
        
        # Add pairwise repulsion for close points (vectorized)
        # Compute all pairwise distances
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, 1.0)  # Avoid division by zero
        
        # Only consider close pairs
        close_mask = dists < 0.3
        close_mask = close_mask & (np.arange(N)[:, None] < np.arange(N)[None, :])  # Upper triangle
        
        # Compute repulsion forces
        repulse_forces = np.zeros_like(points)
        for i in range(N):
            for j in np.where(close_mask[i])[0]:
                d = points[i] - points[j]
                dist = dists[i, j]
                repulse = 0.01 * (0.3 - dist) * d / max(dist, 1e-15)
                repulse_forces[i] += repulse
                repulse_forces[j] -= repulse
        
        forces += repulse_forces
        
        # Apply forces
        for i in range(N):
            f_norm = np.linalg.norm(forces[i])
            if f_norm > 1e-15:
                points[i] = points[i] + step * (forces[i] / f_norm)
        
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
    
    return best_points, best_radius


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def construct():
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.inf
    
    initializations = []
    
    for offset in [0.5, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9]:
        pts = _fibonacci_spiral(N, offset=offset)
        jitter = rng.standard_normal((N, 3)) * 0.02
        pts = pts + jitter
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        initializations.append(pts)
    
    for _ in range(5):
        pts = _random_sphere(N, rng)
        initializations.append(pts)
    
    for init_pts in initializations:
        refined, radius = _optimize_covering(init_pts, max_iters=3000, base_step=0.03)
        if radius < best_radius:
            best_radius = radius
            best_points = refined.copy()
    
    if best_points is not None:
        final_points, final_radius = _optimize_covering(best_points, max_iters=2000, base_step=0.01)
        if final_radius < best_radius:
            best_radius = final_radius
            best_points = final_points.copy()
    
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
