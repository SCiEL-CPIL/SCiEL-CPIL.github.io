import numpy as np
import time
from scipy.spatial import Delaunay
import math

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    # Helper to normalize
    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms

    points = normalize(points)

    # Helper to compute Voronoi vertices (circumcenters of Delaunay triangles)
    def get_voronoi_vertices(pts):
        try:
            tri = Delaunay(pts)
            simplices = tri.simplices
            # Get the 3 points for each triangle
            p0 = pts[simplices[:, 0]]
            p1 = pts[simplices[:, 1]]
            p2 = pts[simplices[:, 2]]
            
            # Compute circumcenter for each triangle
            # Formula: C = (||p0||^2 (p1 x p2) + ||p1||^2 (p2 x p0) + ||p2||^2 (p0 x p1)) / (2 * p0 dot (p1 x p2))
            # Since points are on unit sphere, ||p||^2 = 1
            cross_12 = np.cross(p1, p2)
            cross_20 = np.cross(p2, p0)
            cross_01 = np.cross(p0, p1)
            
            numer = cross_12 + cross_20 + cross_01
            denom = 2.0 * np.sum(p0 * cross_12, axis=1, keepdims=True)
            
            # Handle degenerate triangles (denom close to 0)
            valid = np.abs(denom.flatten()) > 1e-10
            circumcenters = np.zeros((len(simplices), 3))
            circumcenters[valid] = numer[valid] / denom[valid]
            
            # Normalize to sphere
            circumcenters = normalize(circumcenters)
            
            # Find nearest site for each vertex
            # Compute distance matrix: (num_vertices, N)
            # dists[i, j] = ||V_i - P_j||
            # Using dot product: d = sqrt(2 - 2*dot)
            dots = circumcenters @ pts.T
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.sqrt(2.0 - 2.0 * dots)
            
            nearest_indices = np.argmin(dists, axis=1)
            nearest_dists = dists[np.arange(len(circumcenters)), nearest_indices]
            
            return circumcenters, nearest_indices, nearest_dists
        except Exception:
            # Fallback if Delaunay fails (e.g., coplanar points)
            # Return empty or dummy values to allow loop to continue
            return np.zeros((0, 3)), np.array([], dtype=int), np.array([])

    # Optimization Loop
    # Parameters
    beta = 10.0  # Soft-max sharpness
    initial_step = 0.05
    cooling_rate = 0.999
    min_step = 0.0001
    
    step_size = initial_step
    best_points = points.copy()
    best_radius = np.inf

    # Pre-compute initial radius for tracking
    def compute_covering_radius(pts):
        V, idx, d = get_voronoi_vertices(pts)
        if len(V) == 0:
            return np.inf
        return np.max(d)

    current_radius = compute_covering_radius(points)
    if current_radius < best_radius:
        best_radius = current_radius
        best_points = points.copy()

    steps = 0
    while time.time() < deadline:
        V, idx, d = get_voronoi_vertices(points)
        if len(V) == 0:
            break
            
        # Soft-max weights
        # d is shape (M,)
        # We want to emphasize large d
        max_d = np.max(d)
        # Shift to avoid overflow in exp
        shifted_d = d - max_d
        weights = np.exp(beta * shifted_d)
        total_weight = np.sum(weights)
        
        if total_weight == 0:
            break
            
        # Gradients
        grads = np.zeros_like(points)
        
        # For each vertex, add contribution to its nearest site
        # grad_i += weights[j] * ( - (V_j - P_i) / ||V_j - P_i|| )
        # Note: The gradient of distance d = ||V - P|| w.r.t P is -(V-P)/d.
        # We want to MINIMIZE the max distance.
        # So we move P in the direction of -grad_d.
        # -grad_d = (V - P) / d.
        # So we move P towards V.
        
        for k in range(len(V)):
            i = idx[k]
            v = V[k]
            p = points[i]
            dist = d[k]
            if dist < 1e-8:
                continue
            # Vector from P to V
            dir_vec = (v - p) / dist
            # Weighted contribution
            grads[i] += weights[k] * dir_vec
            
        # Normalize gradients to prevent huge steps
        grad_norms = np.linalg.norm(grads, axis=1, keepdims=True)
        grad_norms[grad_norms == 0] = 1.0
        grads = grads / grad_norms
        
        # Update points
        points = points + step_size * grads
        
        # Project back to sphere
        points = normalize(points)
        
        # Cool down
        step_size *= cooling_rate
        if step_size < min_step:
            step_size = min_step
            # If step size is very small, we might be converged
            # Check if we should break early to save time for a final polish?
            # For now, just continue until deadline.
            
        # Evaluate radius
        current_radius = compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        steps += 1
        if steps % 100 == 0:
            # Optional: print progress if needed, but keep silent for evaluator
            pass

    # Final Polish: Use a smaller step size for a few more iterations if time permits
    if time.time() < deadline - 5.0:
        step_size = min_step / 10.0
        for _ in range(50):
            if time.time() > deadline - 1.0:
                break
            V, idx, d = get_voronoi_vertices(best_points)
            if len(V) == 0: break
            max_d = np.max(d)
            shifted_d = d - max_d
            weights = np.exp(beta * shifted_d)
            total_weight = np.sum(weights)
            if total_weight == 0: break
            
            grads = np.zeros_like(best_points)
            for k in range(len(V)):
                i = idx[k]
                v = V[k]
                p = best_points[i]
                dist = d[k]
                if dist < 1e-8: continue
                dir_vec = (v - p) / dist
                grads[i] += weights[k] * dir_vec
                
            grad_norms = np.linalg.norm(grads, axis=1, keepdims=True)
            grad_norms[grad_norms == 0] = 1.0
            grads = grads / grad_norms
            
            new_points = best_points + step_size * grads
            new_points = normalize(new_points)
            
            new_radius = compute_covering_radius(new_points)
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points

    return best_points
