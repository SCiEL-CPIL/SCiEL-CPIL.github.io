import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _compute_covering_radius(pts):
    """
    Computes the covering radius in radians and returns the worst facet.
    """
    try:
        hull = ConvexHull(pts)
        facets = hull.simplices
        
        p_i = pts[facets[:, 0]]
        p_j = pts[facets[:, 1]]
        p_k = pts[facets[:, 2]]
        
        # Circumcenter direction is proportional to p_i + p_j + p_k
        c_vecs = p_i + p_j + p_k
        norms_c = np.linalg.norm(c_vecs, axis=1, keepdims=True)
        norms_c[norms_c < 1e-10] = 1.0
        c_vecs = c_vecs / norms_c
        
        # Cosine of circumradius angle is dot(c, p_i)
        dots = np.einsum('ij,ij->i', c_vecs, p_i)
        dots = np.clip(dots, -1.0, 1.0)
        thetas = np.arccos(dots)
        
        max_idx = np.argmax(thetas)
        max_theta = thetas[max_idx]
        
        return max_theta, facets[max_idx]
    except Exception:
        return np.pi, None

def _fibonacci_sphere(n):
    """Generate a Fibonacci sphere distribution."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(points)

def _gradient_descent(pts, time_limit):
    """
    Deterministic gradient descent on the tangent space.
    """
    start_time = time.time()
    best_radius, _ = _compute_covering_radius(pts)
    best_pts = pts.copy()
    
    # We will do a sequence of gradient descent passes with decreasing step sizes
    # Step sizes are chosen to be small enough for stability but large enough for progress
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for step_size in step_sizes:
        if time.time() - start_time > time_limit:
            break
            
        # Run descent for this step size until it starts increasing or time is up
        while time.time() - start_time < time_limit:
            current_radius, worst_facet = _compute_covering_radius(pts)
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
                
            if worst_facet is None:
                break
                
            i, j, k = worst_facet
            p_i = pts[i]
            p_j = pts[j]
            p_k = pts[k]
            
            # Circumcenter of the worst facet
            c = p_i + p_j + p_k
            norm_c = np.linalg.norm(c)
            if norm_c < 1e-10:
                break
            c = c / norm_c
            
            # Gradient of the circumradius with respect to vertex positions
            # We want to move vertices towards the circumcenter to shrink the facet.
            # The gradient of the angle theta = arccos(c . p) with respect to p
            # is roughly in the direction of the tangent component of c.
            
            # For each vertex in the worst facet, move it towards c
            # We can move all three, or just one. Moving all three symmetrically is often better.
            
            # Tangent projection of (c - p)
            def tangent_move(p, c, step):
                # Direction from p to c
                dir_vec = c - p
                # Project to tangent plane at p
                dot_proj = np.dot(dir_vec, p)
                dir_tangent = dir_vec - dot_proj * p
                norm_dir = np.linalg.norm(dir_tangent)
                if norm_dir < 1e-10:
                    return p
                dir_tangent = dir_tangent / norm_dir
                new_p = p + step * dir_tangent
                return _normalize(new_p)
            
            # Move all three vertices of the worst facet towards the center
            new_p_i = tangent_move(p_i, c, step_size)
            new_p_j = tangent_move(p_j, c, step_size)
            new_p_k = tangent_move(p_k, c, step_size)
            
            pts[i] = new_p_i
            pts[j] = new_p_j
            pts[k] = new_p_k
            
            # Check if this move improved the global radius
            new_radius, _ = _compute_covering_radius(pts)
            
            if new_radius > current_radius * 1.001: # If it got significantly worse
                # Revert
                pts[i] = p_i
                pts[j] = p_j
                pts[k] = p_k
                # If we can't improve with this step size, break to next smaller step
                break
                
    return best_pts, best_radius

def construct():
    """
    Constructs a set of N points on the unit sphere using deterministic gradient descent.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s kill
    
    rng = np.random.default_rng(42)
    
    # Multi-start with different initializations
    # 1. Fibonacci
    # 2. Random
    # 3. Perturbed Fibonacci
    
    inits = []
    inits.append(_fibonacci_sphere(N))
    
    for _ in range(4):
        pts = _fibonacci_sphere(N)
        noise = rng.normal(0, 0.1, (N, 3))
        pts = pts + noise
        inits.append(_normalize(pts))
        
    for _ in range(4):
        pts = rng.normal(0, 1, (N, 3))
        inits.append(_normalize(pts))
        
    best_global_radius = np.pi
    best_global_pts = inits[0].copy()
    
    # Allocate time budget to each start
    # We have 10 starts. 140s / 10 = 14s per start.
    # But we can be smarter: give more time to better starts?
    # For simplicity, equal budget, but stop early if we find a very good solution.
    
    for init_pts in inits:
        if time.time() - start_time > time_limit:
            break
            
        # Give each start a specific time slice
        # Or just run gradient descent until the global time limit is hit,
        # but that might starve later starts.
        # Let's do a fixed number of steps or a time slice.
        # Let's use a time slice of 12s per start.
        slice_limit = 12.0
        slice_start = time.time()
        
        pts, radius = _gradient_descent(init_pts, slice_limit)
        
        if radius < best_global_radius:
            best_global_radius = radius
            best_global_pts = pts.copy()
            
    return _normalize(best_global_pts)

if __name__ == "__main__":
    pts = construct()
    print(f"Shape: {pts.shape}")
    print(f"Norms: {np.linalg.norm(pts, axis=1)[:5]}")
    print(f"Covering Radius: {_compute_covering_radius(pts)[0] * 180 / np.pi:.6f} degrees")
