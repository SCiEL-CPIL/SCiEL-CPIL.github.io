"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, LinearConstraint

N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _covering_radius(points):
    """Compute covering radius in radians using convex hull facets."""
    points = np.asarray(points, dtype=np.float64)
    # Normalize
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, None]
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: brute force on a fine grid (expensive)
        # Use a moderate grid
        lat = np.linspace(-np.pi/2, np.pi/2, 80)
        lon = np.linspace(0, 2*np.pi, 80, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        Y = np.column_stack([
            np.cos(LA) * np.cos(LO).ravel(),
            np.cos(LA) * np.sin(LO).ravel(),
            np.sin(LA).ravel()
        ])
        # Compute min distance from each Y to nearest point
        # Using cdist-like via dot product
        # d^2 = 2 - 2*dot
        dots = Y @ points.T
        min_dot = dots.max(axis=1)
        max_ang = np.arccos(np.clip(min_dot, -1, 1))
        return float(max_ang.max())
    
    # For each facet, compute the circumcenter and check if it's in the cap
    # The covering radius is the max over facets of the angular distance from facet center to farthest vertex
    # Actually, for a sphere, the covering radius is determined by the facets of the convex hull.
    # For each facet, the farthest point on the sphere from all points in that facet's cap is the facet's circumcenter (if it exists on the sphere side).
    
    # A more robust approach: for each facet, compute the point on the sphere farthest from all points.
    # This is the circumcenter of the facet's vertices, projected to the sphere.
    
    max_cov = 0.0
    for simplex in hull.simplices:
        verts = points[simplex]
        # The covering radius for this facet is the angular distance from the facet's "center" to the farthest vertex
        # But actually, we need the point on the sphere that is farthest from ALL points, which for a facet is its circumcenter
        # Compute circumcenter of the facet vertices (in 3D)
        # For a triangle (2D simplex in 3D), the circumcenter can be computed
        
        if len(verts) == 3:
            # Triangle circumcenter
            a, b, c = verts[0], verts[1], verts[2]
            # Solve for circumcenter: |x-a|^2 = |x-b|^2 = |x-c|^2
            # 2*x.(b-a) = |b|^2 - |a|^2
            # 2*x.(c-a) = |c|^2 - |a|^2
            ab = b - a
            ac = c - a
            ab2 = np.dot(ab, ab)
            ac2 = np.dot(ac, ac)
            bc = c - b
            bc2 = np.dot(bc, bc)
            
            # Using the formula for circumcenter of a triangle
            # x = a + (ac2 * (b-a) - ab2 * (c-a)) / (2 * cross(ab, ac) · n) ... this is getting complex
            # Simpler: solve linear system
            # [2*ab, 2*ac] * [x1, x2, x3]^T = [|b|^2 - |a|^2, |c|^2 - |a|^2]
            # But we need 3 equations. The third is that x is equidistant, so we can use the plane equation.
            
            # Alternative: use the fact that for points on the unit sphere, the circumcenter of a spherical triangle
            # is related to the normal of the plane through the three points.
            
            # Let's just compute the circumcenter in 3D space
            # For three points a, b, c, the circumcenter x satisfies:
            # (x-a)·(b-a) = |b-a|^2 / 2
            # (x-a)·(c-a) = |c-a|^2 / 2
            # (x-a)·(c-b) = |c-b|^2 / 2  (redundant)
            
            # Actually, let's use a different approach: the point on the sphere farthest from all points
            # is found by checking the circumcenters of all facets.
            
            # For a facet with vertices v1, ..., vk, the farthest point is the circumcenter of the facet
            # if it lies on the sphere side.
            
            # Let me use a simpler heuristic: for each facet, compute the centroid and then
            # the angular distance to the farthest vertex. This is an approximation.
            
            centroid = verts.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            dots = centroid @ verts.T
            max_ang = np.arccos(np.clip(dots.max(), -1, 1))
            max_cov = max(max_cov, max_ang)
            
        elif len(verts) == 4:
            # Tetrahedron - use centroid approximation
            centroid = verts.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            dots = centroid @ verts.T
            max_ang = np.arccos(np.clip(dots.max(), -1, 1))
            max_cov = max(max_cov, max_ang)
    
    return float(max_cov)


def construct():
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, None]
    
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    # Iterative refinement: move each point to reduce the local covering radius
    # Use a smooth approximation: minimize the sum of exp(alpha * dist^2) for nearby points
    
    alpha = 20.0  # Temperature parameter for soft-max
    step_size = 0.1
    
    for iteration in range(200):
        if time.time() > deadline:
            break
            
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                    break
                    
            p = pts[idx]
            
            # Find the "worst" direction: where the covering radius is largest near this point
            # Compute the gradient of the soft covering radius with respect to p
            
            # For each other point q, compute the contribution to the soft max
            dots = pts @ p
            # The soft-max of angular distances: we want to maximize the minimum distance
            # So we minimize the soft-max of (angular distance)^2, which is equivalent to minimizing
            # the soft-max of (1 - dot)^2
            
            # Actually, let's use a different approach:
            # The covering radius is determined by the facets. For each facet containing p,
            # moving p affects the circumcenter.
            
            # Simpler: use a gradient-free local search
            # Try small perturbations in random directions
            
            # Compute current local covering radius contribution
            # Move p slightly in the direction that reduces the worst-case distance
            
            # Find the point on the sphere farthest from all points (approximation)
            # Use the facet containing p if it exists
            
            # For now, use a simple heuristic: move p away from the cluster
            # Compute the mean of nearby points
            dots = pts @ p
            # Weight by proximity
            weights = np.exp(-alpha * (1 - dots)**2)
            weights[idx] = 0
            if weights.sum() > 0:
                center = (weights[:, None] * pts).sum(axis=0)
                center = center / np.linalg.norm(center)
                # Move p away from center
                new_p = p - step_size * center
                new_p = new_p / np.linalg.norm(new_p)
                
                # Check if this improves the covering radius
                pts[idx] = new_p
                new_radius = _covering_radius(pts)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_pts = pts.copy()
                    improved = True
                else:
                    pts[idx] = p  # Revert
        
        if not improved:
            # Try smaller step sizes
            step_size *= 0.7
            if step_size < 0.001:
                break
    
    # Final polish: use scipy minimize on the soft-max objective
    if time.time() < deadline - 10:
        def objective(x):
            pts_x = x.reshape(N, 3)
            norms_x = np.linalg.norm(pts_x, axis=1)
            pts_x = pts_x / norms_x[:, None]
            # Soft-max covering radius
            # For each point on a fine grid, compute min distance
            # Use a moderate grid for speed
            lat = np.linspace(-np.pi/2, np.pi/2, 40)
            lon = np.linspace(0, 2*np.pi, 40, endpoint=False)
            LA, LO = np.meshgrid(lat, lon, indexing='ij')
            Y = np.column_stack([
                np.cos(LA) * np.cos(LO).ravel(),
                np.cos(LA) * np.sin(LO).ravel(),
                np.sin(LA).ravel()
            ])
            dots = Y @ pts_x.T
            min_dot = dots.max(axis=1)
            # Soft-max of (1 - min_dot)
            # Use log-sum-exp for stability
            vals = 1 - min_dot
            vals = np.clip(vals, 1e-10, None)
            log_sum = np.log(np.sum(np.exp(vals * alpha)))
            return log_sum / alpha
        
        x0 = best_pts.ravel()
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'gtol': 1e-6})
        if res.success or res.fun < best_radius:
            pts_res = res.x.reshape(N, 3)
            norms_res = np.linalg.norm(pts_res, axis=1)
            pts_res = pts_res / norms_res[:, None]
            res_radius = _covering_radius(pts_res)
            if res_radius < best_radius:
                best_radius = res_radius
                best_pts = pts_res
    
    # Normalize final result
    norms = np.linalg.norm(best_pts, axis=1)
    best_pts = best_pts / norms[:, None]
    
    return best_pts
# EVOLVE-BLOCK-END
