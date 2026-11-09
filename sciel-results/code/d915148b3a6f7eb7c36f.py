"""
Strategy: Direct Geometric Facet Repair for Spherical Covering.
Instead of global Lloyd's iterations which minimize average error, this strategy
directly targets the covering radius by identifying the "worst" facet (the empty
spherical triangle with the largest circumradius) and locally adjusting its three
vertices to shrink that specific facet. This is a min-max optimization approach
tailored to the geometry of the convex hull.

The algorithm alternates between:
1. Computing the convex hull and finding the facet with the maximum circumradius.
2. Performing a local gradient descent step on the three vertices of that facet
   to minimize the circumradius of that facet, while keeping them on the sphere.
3. Repeating this process, effectively "healing" the largest holes in the covering.
"""

import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    Returns (radius_in_degrees, worst_facet_indices).
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        # Vectorized computation of circumradii
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        # Circumcenter direction is cross(p1-p2, p1-p3)
        v1 = p1 - p2
        v2 = p1 - p3
        c_dir = np.cross(v1, v2)
        
        norms = np.linalg.norm(c_dir, axis=1)
        valid = norms > 1e-10
        
        # Initialize circumcenters
        c = np.zeros_like(c_dir)
        c[valid] = c_dir[valid] / norms[valid, np.newaxis]
        
        # Orient towards centroid
        centroids = (p1 + p2 + p3) / 3.0
        c_norms = np.linalg.norm(centroids, axis=1)
        valid_centroids = c_norms > 1e-10
        centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
        
        dots = np.sum(c * centroids, axis=1)
        c[dots < 0] *= -1
        
        # Compute angular distance from circumcenter to any vertex (e.g., p1)
        cos_theta = np.sum(c * p1, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        # Invalidate degenerate facets
        theta[~valid] = 0.0
        
        max_idx = np.argmax(theta)
        max_radius_rad = theta[max_idx]
        
        return max_radius_rad * 180.0 / np.pi, facets[max_idx]
        
    except Exception:
        return 180.0, np.array([0, 1, 2])

def _shrink_facet(points, facet_indices, steps=20, lr=0.05):
    """
    Locally adjust the three points of a specific facet to minimize its circumradius.
    Uses simple projected gradient descent on the sphere.
    """
    p1_idx, p2_idx, p3_idx = facet_indices
    p1 = points[p1_idx].copy()
    p2 = points[p2_idx].copy()
    p3 = points[p3_idx].copy()
    
    for _ in range(steps):
        # Compute current circumradius
        v1 = p1 - p2
        v2 = p1 - p3
        c_dir = np.cross(v1, v2)
        n = np.linalg.norm(c_dir)
        if n < 1e-10:
            break
        c = c_dir / n
        
        centroid = (p1 + p2 + p3) / 3.0
        nc = np.linalg.norm(centroid)
        if nc > 1e-10:
            centroid /= nc
            if np.dot(c, centroid) < 0:
                c = -c
        
        cos_theta = np.clip(np.dot(c, p1), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        # Gradient of theta w.r.t p1, p2, p3
        # theta = arccos(c . p1). dc/dp1 is complex, but we can use finite differences
        # or analytical approximation.
        # Actually, for a spherical triangle, the circumcenter moves as vertices move.
        # A simpler heuristic: move each vertex slightly towards the circumcenter 
        # if the triangle is "obtuse" in a way that allows shrinking, 
        # or use the fact that the gradient of the circumradius wrt a vertex 
        # points roughly towards the circumcenter for acute triangles.
        
        # Let's use a robust finite difference gradient for the circumradius.
        eps = 1e-4
        grad_p1 = np.zeros(3)
        grad_p2 = np.zeros(3)
        grad_p3 = np.zeros(3)
        
        for k in range(3):
            # Perturb p1
            p1_p = p1.copy(); p1_p[k] += eps
            p1_p = p1_p / np.linalg.norm(p1_p)
            v1_p = p1_p - p2; v2_p = p1_p - p3
            c_p = np.cross(v1_p, v2_p)
            n_p = np.linalg.norm(c_p)
            if n_p > 1e-10:
                c_p /= n_p
                cent_p = (p1_p + p2 + p3) / 3.0
                nc_p = np.linalg.norm(cent_p)
                if nc_p > 1e-10:
                    cent_p /= nc_p
                    if np.dot(c_p, cent_p) < 0: c_p = -c_p
                    cos_p = np.clip(np.dot(c_p, p1_p), -1.0, 1.0)
                    theta_p = np.arccos(cos_p)
                else:
                    theta_p = theta
            else:
                theta_p = theta
            grad_p1[k] = (theta_p - theta) / eps
            
            # Perturb p2
            p2_p = p2.copy(); p2_p[k] += eps
            p2_p = p2_p / np.linalg.norm(p2_p)
            v1_p = p1 - p2_p; v2_p = p1 - p3
            c_p = np.cross(v1_p, v2_p)
            n_p = np.linalg.norm(c_p)
            if n_p > 1e-10:
                c_p /= n_p
                cent_p = (p1 + p2_p + p3) / 3.0
                nc_p = np.linalg.norm(cent_p)
                if nc_p > 1e-10:
                    cent_p /= nc_p
                    if np.dot(c_p, cent_p) < 0: c_p = -c_p
                    cos_p = np.clip(np.dot(c_p, p1), -1.0, 1.0)
                    theta_p = np.arccos(cos_p)
                else:
                    theta_p = theta
            else:
                theta_p = theta
            grad_p2[k] = (theta_p - theta) / eps
            
            # Perturb p3
            p3_p = p3.copy(); p3_p[k] += eps
            p3_p = p3_p / np.linalg.norm(p3_p)
            v1_p = p1 - p2; v2_p = p1 - p3_p
            c_p = np.cross(v1_p, v2_p)
            n_p = np.linalg.norm(c_p)
            if n_p > 1e-10:
                c_p /= n_p
                cent_p = (p1 + p2 + p3_p) / 3.0
                nc_p = np.linalg.norm(cent_p)
                if nc_p > 1e-10:
                    cent_p /= nc_p
                    if np.dot(c_p, cent_p) < 0: c_p = -c_p
                    cos_p = np.clip(np.dot(c_p, p1), -1.0, 1.0)
                    theta_p = np.arccos(cos_p)
                else:
                    theta_p = theta
            else:
                theta_p = theta
            grad_p3[k] = (theta_p - theta) / eps
        
        # Project gradients to tangent space
        grad_p1 = grad_p1 - np.dot(grad_p1, p1) * p1
        grad_p2 = grad_p2 - np.dot(grad_p2, p2) * p2
        grad_p3 = grad_p3 - np.dot(grad_p3, p3) * p3
        
        # Update
        p1 = p1 - lr * grad_p1
        p2 = p2 - lr * grad_p2
        p3 = p3 - lr * grad_p3
        
        # Normalize
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        p3 /= np.linalg.norm(p3)
        
        points[p1_idx] = p1
        points[p2_idx] = p2
        points[p3_idx] = p3

    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius, _ = _covering_radius(points)
    
    # Phase 1: Global Lloyd's for a few iterations to get a reasonable starting point
    # This ensures the hull is stable and points are somewhat spread out.
    for iter in range(50):
        if time.time() > deadline - 60.0:
            break
        # Simple Lloyd step
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
            vertex_to_facets = {i: [] for i in range(N)}
            for idx, fac in enumerate(facets):
                for v in fac:
                    vertex_to_facets[v].append(idx)
            
            new_points = points.copy()
            for v in range(N):
                facs = vertex_to_facets[v]
                if not facs: continue
                centers = []
                for f_idx in facs:
                    p1 = points[facets[f_idx][0]]
                    p2 = points[facets[f_idx][1]]
                    p3 = points[facets[f_idx][2]]
                    v1 = p1 - p2
                    v2 = p1 - p3
                    c_dir = np.cross(v1, v2)
                    n = np.linalg.norm(c_dir)
                    if n < 1e-10: continue
                    c = c_dir / n
                    centroid = (p1 + p2 + p3) / 3.0
                    nc = np.linalg.norm(centroid)
                    if nc > 1e-10:
                        centroid /= nc
                        if np.dot(c, centroid) < 0: c = -c
                        centers.append(c)
                if centers:
                    cell_centroid = np.mean(centers, axis=0)
                    nc = np.linalg.norm(cell_centroid)
                    if nc > 1e-10:
                        cell_centroid /= nc
                        new_points[v] = cell_centroid
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            new_points = new_points / norms
            points = new_points
        except Exception:
            pass
        
        current_radius, _ = _covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
    # Phase 2: Direct Facet Repair
    # Iterate: find worst facet, shrink it.
    no_improvement_count = 0
    current_radius = best_radius
    
    while time.time() < deadline:
        # Find worst facet
        current_radius, worst_facet = _covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        # If stuck, try a different facet or add noise
        if no_improvement_count > 100:
            # Randomly perturb one point to escape local minimum
            idx = np.random.randint(N)
            noise = np.random.normal(0, 0.01, 3)
            points[idx] += noise
            points[idx] /= np.linalg.norm(points[idx])
            no_improvement_count = 0
            continue
            
        # Shrink the worst facet
        points = _shrink_facet(points, worst_facet, steps=5, lr=0.02)
        
        # Check if we broke the hull (points coincident or inside hemisphere)
        try:
            hull = ConvexHull(points)
        except Exception:
            # Revert to best
            points = best_points.copy()
            no_improvement_count += 10
            continue

    return best_points

if __name__ == "__main__":
    pts = construct()
    rad, _ = _covering_radius(pts)
    print(f"Covering radius: {rad} degrees")
    print(pts.shape)
