"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _compute_circumcenters(hull, pts):
    """Compute circumcenters and radii for all hull facets."""
    centers = []
    radii = []
    for simplex in hull.simplices:
        a, b, c = pts[simplex]
        # Normal to the plane
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        n = n / n_norm
        
        # Circumcenter of triangle abc on the unit sphere
        # u is parallel to the normal n
        # We choose the sign such that u is on the same side as the facet normal (outward)
        # The outward normal of the convex hull facet is n_out.
        # u should be in the direction of n_out.
        
        # Check distance
        dot_val = np.clip(np.dot(n, a), -1.0, 1.0)
        r = np.arccos(dot_val)
        
        centers.append(n)
        radii.append(r)
        
    return np.array(centers), np.array(radii)

def _get_covering_radius(pts):
    """Compute the covering radius (in radians) of a set of points on the unit sphere."""
    try:
        hull = ConvexHull(pts)
        centers, radii = _compute_circumcenters(hull, pts)
        if len(radii) == 0:
            return np.pi
        return np.max(radii)
    except Exception:
        return np.pi

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    best_cr = np.inf
    best_pts = None
    
    # Start with a good initial guess
    # Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimize using gradient descent with pressure term
    lr = 0.01
    pressure_strength = 0.5
    repulsion_strength = 0.1
    n_iter = 2000
    momentum = 0.9
    v = np.zeros_like(pts)
    
    start_time = time.time()
    
    for it in range(n_iter):
        if time.time() - start_time > 100:
            break
            
        # 1. Compute Repulsive Gradient
        dots = np.dot(pts, pts.T)
        np.fill_diagonal(dots, 2.0) # Avoid zero
        dots = np.clip(dots, -1.0 + 1e-15, 1.0 - 1e-15)
        dists = np.arccos(dots)
        sins = np.sin(dists)
        
        # Repulsive force: sum_j x_j / (d^3 * sin(d)) ? 
        # Let's use the previous formula which was stable.
        weights = np.zeros_like(dists)
        mask = ~np.eye(N, dtype=bool)
        safe_dists = np.where(dists < 1e-6, 1e-6, dists)
        safe_sins = np.where(sins < 1e-6, 1e-6, sins)
        weights[mask] = 1.0 / (safe_dists[mask] ** 3 * safe_sins[mask])
        rep_grad = repulsion_strength * (weights @ pts)
        
        # Project rep_grad onto tangent space
        for i in range(N):
            radial_comp = np.dot(rep_grad[i], pts[i])
            rep_grad[i] = rep_grad[i] - radial_comp * pts[i]
            
        # 2. Compute Pressure Gradient
        press_grad = np.zeros_like(pts)
        try:
            hull = ConvexHull(pts)
            centers, radii = _compute_circumcenters(hull, pts)
            
            # Find worst facets
            # We push vertices of facets with radius > current best or a threshold
            threshold = max(best_cr * 0.95, 0.35) # ~20 deg
            
            for idx, r_val in enumerate(radii):
                if r_val > threshold:
                    u = centers[idx]
                    simplex = hull.simplices[idx]
                    for v_idx in simplex:
                        p = pts[v_idx]
                        # Force direction: from p towards u
                        # Tangent vector: u - (u.p)p
                        tang = u - np.dot(u, p) * p
                        # Add to gradient
                        press_grad[v_idx] += pressure_strength * tang
        except Exception:
            pass
            
        # 3. Combine Gradients
        total_grad = rep_grad + press_grad
        
        # 4. Update
        v = momentum * v - lr * total_grad
        pts_new = pts + v
        norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
        pts_new = pts_new / norms
        pts = pts_new
        
        # 5. Evaluate
        if it % 10 == 0:
            cr = _get_covering_radius(pts)
            if cr < best_cr:
                best_cr = cr
                best_pts = pts.copy()

    return best_pts
# EVOLVE-BLOCK-END
