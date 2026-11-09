"""
Strategy: Global Stochastic Search (Differential Evolution) + Local Geometric Repair.

The previous strategy relied on L-BFGS-B with a soft-max surrogate objective. 
While smooth, this often gets trapped in local minima where the soft-max gradient 
is weak or misleading relative to the true hard-max covering radius.

This new strategy uses a two-phase approach:
1. Global Phase: Use scipy.optimize.differential_evolution (DE) to search the 
   390-dimensional space of point coordinates. DE is derivative-free and robust 
   to non-smoothness. The objective function is the EXACT covering radius (hard max), 
   which is non-differentiable but well-defined. We use a small population and 
   limited iterations to fit within the time budget, aiming to find a configuration 
   in the basin of attraction of a high-quality local optimum.
2. Local Phase: Take the best result from DE and apply a fast, direct geometric 
   repair loop. This loop identifies the worst facet (largest circumradius) and 
   moves its vertices slightly towards the circumcenter (or away, depending on 
   convexity) to reduce the covering radius. This step exploits the local geometry 
   that DE may have roughly aligned but not precisely optimized.

This structural change moves from "continuous smooth flow" to "global jump + local snap," 
which is a fundamentally different search paradigm.
"""

import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

N = 130

def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    Returns a large value if the hull is invalid or points are degenerate.
    """
    try:
        # Ensure points are on the sphere (sanity check, though DE might drift)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        valid_norms = norms > 1e-10
        if not np.all(valid_norms):
            return 180.0
        points = points / norms
        
        hull = ConvexHull(points)
        facets = hull.simplices
        
        # Vectorized circumradius calculation for all facets
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        v1 = p1 - p2
        v2 = p1 - p3
        
        c_dir = np.cross(v1, v2)
        norms_c = np.linalg.norm(c_dir, axis=1)
        
        # Filter out degenerate facets (area ~ 0)
        valid = norms_c > 1e-10
        if not np.any(valid):
            return 180.0
            
        c = np.zeros_like(c_dir)
        c[valid] = c_dir[valid] / norms_c[valid, np.newaxis]
        
        # Ensure circumcenter points outward (same hemisphere as centroid)
        centroids = (p1 + p2 + p3) / 3.0
        c_norms = np.linalg.norm(centroids, axis=1)
        valid_centroids = c_norms > 1e-10
        centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
        
        dots = np.sum(c * centroids, axis=1)
        c[dots < 0] *= -1
        
        # Angular distance from circumcenter to any vertex (they are equal)
        cos_theta = np.sum(c * p1, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        theta[~valid] = 0.0
        max_theta = np.max(theta)
        
        return max_theta * 180.0 / np.pi
        
    except Exception:
        return 180.0

def _de_objective(x_flat):
    """
    Objective function for Differential Evolution.
    x_flat: 1D array of shape (3*N,).
    Returns: scalar covering radius in degrees.
    """
    points = x_flat.reshape(N, 3)
    
    # Project to sphere to maintain validity during search
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    valid = norms > 1e-10
    if not np.all(valid):
        return 180.0
    points = points / norms
    
    return _covering_radius(points)

def _local_repair(points, max_iterations=200, deadline=None):
    """
    Local geometric repair: iteratively fix the worst facet.
    """
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    for _ in range(max_iterations):
        if deadline and time.time() > deadline - 5.0:
            break
            
        try:
            hull = ConvexHull(best_points)
            facets = hull.simplices
            p1 = best_points[facets[:, 0]]
            p2 = best_points[facets[:, 1]]
            p3 = best_points[facets[:, 2]]
            
            v1 = p1 - p2
            v2 = p1 - p3
            
            c_dir = np.cross(v1, v2)
            norms_c = np.linalg.norm(c_dir, axis=1)
            valid = norms_c > 1e-10
            if not np.any(valid):
                break
                
            c = np.zeros_like(c_dir)
            c[valid] = c_dir[valid] / norms_c[valid, np.newaxis]
            
            centroids = (p1 + p2 + p3) / 3.0
            c_norms = np.linalg.norm(centroids, axis=1)
            valid_centroids = c_norms > 1e-10
            centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
            
            dots = np.sum(c * centroids, axis=1)
            c[dots < 0] *= -1
            
            cos_theta = np.sum(c * p1, axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            theta[~valid] = 0.0
            
            max_idx = np.argmax(theta)
            if not valid[max_idx]:
                break
                
            f_idx = facets[max_idx]
            cc = c[max_idx]
            
            # Move vertices slightly towards the circumcenter to reduce the radius
            # Note: Moving towards the circumcenter reduces the distance from cc to vertex?
            # No, the circumcenter is the center of the sphere passing through vertices.
            # The "cap" is the region outside the triangle but inside the sphere.
            # To minimize the max distance to the nearest point, we want to move points
            # AWAY from the circumcenter if the circumcenter is the "worst point" in the cap?
            # Actually, the covering radius is the distance from the circumcenter to the vertices.
            # If we move vertices TOWARDS the circumcenter, the new circumcenter will be different.
            # A standard heuristic for covering: move the worst triangle's vertices
            # slightly towards the centroid of the triangle or towards each other?
            
            # Let's try moving them towards the facet centroid (mean of vertices) to "tighten" the triangle.
            # Or, move them away from the circumcenter?
            # If the circumcenter is far, the triangle is "large".
            # Moving vertices towards the circumcenter reduces the radius of the circumscribed circle
            # IF the circumcenter is inside the triangle (acute). For obtuse, it's outside.
            # A safer heuristic: Move vertices slightly towards the centroid of the facet.
            
            centroid = (p1[max_idx] + p2[max_idx] + p3[max_idx]) / 3.0
            # Normalize centroid to sphere
            c_norm = np.linalg.norm(centroid)
            if c_norm > 1e-10:
                centroid /= c_norm
            
            # Small step size
            step = 0.02
            new_points = best_points.copy()
            for v in f_idx:
                # Move towards centroid on the sphere (slerp-like approximation)
                diff = centroid - best_points[v]
                new_points[v] = best_points[v] + step * diff
                # Re-normalize
                norm_v = np.linalg.norm(new_points[v])
                if norm_v > 1e-10:
                    new_points[v] /= norm_v
            
            new_radius = _covering_radius(new_points)
            
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points
            else:
                # If it didn't improve, try a smaller step or break?
                # For robustness, just break if no improvement in a few steps?
                # Let's keep it simple: if no improvement, try next iteration with same logic?
                # It might oscillate. Let's just break if stuck.
                # Actually, let's try a different direction: move AWAY from circumcenter?
                # For now, if no improvement, we break to save time for other restarts.
                break
                
        except Exception:
            break
            
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = init_points.copy()
    best_radius = _covering_radius(best_points)
    
    # 2. Global Phase: Differential Evolution
    # Bounds: [-1, 1] for each coordinate
    bounds = [(-1.0, 1.0)] * (N * 3)
    
    # DE parameters:
    # popsize: 5 * dim is standard, but 390*5 = 1950 evals per gen. Too slow.
    # Let's use a smaller popsize, e.g., 10.
    # maxiter: limited by time.
    # tol: default.
    
    # Estimate time per eval:
    # ConvexHull for N=130 is ~1-2 ms.
    # DE with popsize=10, maxiter=100 -> 1000 evals.
    # 1000 * 2ms = 2s. This is very fast! We can afford more.
    # Let's try popsize=20, maxiter=200. 4000 evals * 2ms = 8s.
    # We have ~140s. We can do much more.
    # Let's use popsize=30, maxiter=500. 15000 evals * 2ms = 30s.
    # Remaining time for local repair and multiple restarts.
    
    try:
        de_result = differential_evolution(
            _de_objective,
            bounds,
            maxiter=300,
            popsize=20,
            tol=1e-5,
            seed=42,
            polish=False,  # We do our own local repair
            init='sobol'  # Better initial distribution
        )
        
        if time.time() < deadline:
            x_opt = de_result.x
            points_opt = x_opt.reshape(N, 3)
            
            # Project to sphere
            norms = np.linalg.norm(points_opt, axis=1, keepdims=True)
            valid = norms > 1e-10
            points_opt[valid] /= norms[valid]
            
            current_radius = _covering_radius(points_opt)
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points_opt.copy()
                
    except Exception:
        pass
        
    # 3. Local Phase: Geometric Repair
    if time.time() < deadline - 10.0:
        repaired_points, repaired_radius = _local_repair(
            best_points, 
            max_iterations=500, 
            deadline=deadline
        )
        
        if repaired_radius < best_radius:
            best_radius = repaired_radius
            best_points = repaired_points
            
    # 4. Optional: Second round of DE with a smaller population and tighter bounds?
    # Or just return the best found.
    
    # Final sanity check
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    valid = norms > 1e-10
    best_points[valid] /= norms[valid]
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Covering radius: {_covering_radius(pts)} degrees")
    print(pts.shape)
