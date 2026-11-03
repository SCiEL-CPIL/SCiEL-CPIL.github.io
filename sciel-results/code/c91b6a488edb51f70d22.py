import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest configuration point. This is equivalent to the maximum circumradius
    of the facets of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0.0
        
        for tri in facets:
            p1 = points[tri[0]]
            p2 = points[tri[1]]
            p3 = points[tri[2]]
            
            # Compute the normal to the plane defined by p1, p2, p3
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            
            # Ensure the normal points outward (away from origin)
            if np.dot(c, p1) < 0:
                c = -c
            
            # The angular distance from the circumcenter c to the vertices is the circumradius.
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            dist = np.arccos(dot_val)
            
            if dist > max_radius:
                max_radius = dist
                
    except Exception:
        return np.pi / 2
        
    return np.degrees(max_radius)

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def _gradient_descent(points, start_time, deadline, initial_lr=0.01, min_lr=1e-5):
    """
    Optimize the configuration using a differentiable surrogate objective.
    We minimize the soft-max of squared distances to the nearest point.
    """
    current_points = points.copy()
    best_points = current_points.copy()
    best_radius = _compute_covering_radius(current_points)
    lr = initial_lr
    
    # Precompute indices for efficiency? No, we need full pairwise distances.
    # N=130, so 130x130 matrix is small.
    
    while time.time() < deadline:
        # Compute pairwise distances (squared chordal distance is proportional to angular distance for small angles,
        # but let's use angular distance directly or cosine similarity)
        # Let's use the angle: d_ij = arccos(dot(p_i, p_j))
        # However, arccos is not differentiable at 1.
        # Let's use the squared distance: ||p_i - p_j||^2 = 2 - 2 cos(theta)
        # Minimizing max distance is equivalent to minimizing max (2 - 2 cos(theta))
        # Which is equivalent to maximizing min cos(theta)
        
        # Compute cosine similarity matrix
        # C[i, j] = dot(p_i, p_j)
        C = current_points @ current_points.T
        
        # Distance matrix (squared chordal)
        D2 = 2.0 - 2.0 * C
        # Clip to avoid numerical issues
        D2 = np.maximum(D2, 0.0)
        
        # For each point i, find the distance to its nearest neighbor j (j != i)
        # Actually, the covering radius is determined by the "holes", which are related to the Voronoi cells.
        # The soft-max approach:
        # Objective: sum_i soft_max_j (D2[i, j])
        # This encourages all points to have small distances to their nearest neighbors, which generally improves covering.
        # But it might cause points to clump.
        
        # A better objective for covering:
        # We want to minimize the maximum distance from any point ON THE SPHERE to the nearest config point.
        # This is hard to compute directly with gradients.
        # However, the worst-case points are the circumcenters of the facets.
        # So, let's stick to the convex hull approach for the objective but use gradients?
        # No, the prompt suggests using a differentiable soft-max of pairwise distances.
        
        # Let's try: Minimize sum_i max_j (D2[i, j]) is not covering radius.
        # Covering radius is related to the Delaunay triangulation.
        
        # Alternative: Repulsive potential.
        # Let's use the gradient of the soft-max of the distances to the nearest neighbor.
        # This tends to equalize the nearest neighbor distances.
        
        beta = 20.0 # Temperature for soft-max
        # D2 is N x N
        # For each row i, we want the soft-max of D2[i, :]
        # But we should exclude i itself? D2[i, i] = 0, so it dominates.
        # We want the distance to the NEAREST OTHER point.
        # So we should set D2[i, i] to a large value or ignore it.
        
        D2_mod = D2.copy()
        np.fill_diagonal(D2_mod, np.inf)
        
        # Soft-max: exp(-beta * D2) / sum(exp(-beta * D2))
        # This gives weights to the nearest neighbors.
        # The "soft nearest neighbor distance" is -1/beta * log(sum(exp(-beta * D2)))
        
        # Compute log-sum-exp for stability
        # max_D2 = np.max(D2_mod, axis=1, keepdims=True)
        # log_sum_exp = max_D2 + np.log(np.sum(np.exp(-beta * (D2_mod - max_D2)), axis=1, keepdims=True))
        
        # Actually, we want to minimize the distance, so we use exp(-beta * D2)
        # Let's compute the gradient of the sum of soft-min of D2.
        
        # Soft-min of a vector x is -1/beta * log(sum(exp(-beta * x)))
        # Gradient w.r.t x_k is: exp(-beta * x_k) / sum(exp(-beta * x))
        
        # Let's compute weights W[i, j] = exp(-beta * D2[i, j]) / sum_j exp(-beta * D2[i, j])
        # But D2[i, i] is inf, so exp(-inf) = 0.
        
        # To avoid underflow, subtract the max (which is inf? No, we set diagonal to inf).
        # We should only consider finite values.
        
        # Let's just compute the gradient of the sum of squared nearest neighbor distances?
        # Simpler: Use the convex hull based gradient.
        # The covering radius is the max circumradius.
        # The gradient of the circumradius w.r.t the vertices of the worst facet is known.
        
        # Let's go back to the convex hull method but with smaller steps and more iterations,
        # or use the gradient of the soft-max of the facet circumradii.
        
        # For now, let's try a simple repulsive force model which is differentiable.
        # Force on p_i is sum_j (p_i - p_j) * f(||p_i - p_j||)
        # This maximizes the minimum distance between points, which helps covering.
        
        # Compute gradients of the objective: sum_i soft_min_j D2[i, j]
        # d/dD2[i, k] soft_min = exp(-beta * D2[i, k]) / sum_j exp(-beta * D2[i, j])
        # dD2[i, k]/dp_i = -2 * (p_k - p_i)
        # dD2[i, k]/dp_k = -2 * (p_i - p_k)
        
        # So gradient w.r.t p_i is sum_k (d/dD2[i, k]) * dD2[i, k]/dp_i
        # = sum_k w[i, k] * (-2 * (p_k - p_i))
        # = 2 * sum_k w[i, k] * (p_i - p_k)
        
        # This is a repulsive force towards the centroid of the weighted neighbors.
        
        # Compute weights
        # We need to handle the inf on the diagonal.
        # Let's set diagonal to a large number instead of inf to avoid issues.
        D2_work = D2.copy()
        np.fill_diagonal(D2_work, 10.0) # Large enough to be ignored
        
        # Subtract max for stability
        max_D2 = np.max(D2_work, axis=1, keepdims=True)
        exp_terms = np.exp(-beta * (D2_work - max_D2))
        # Set diagonal to 0
        exp_terms[np.arange(N), np.arange(N)] = 0.0
        
        sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
        # Avoid division by zero
        sum_exp = np.maximum(sum_exp, 1e-12)
        
        weights = exp_terms / sum_exp
        
        # Gradient
        # grad[i] = 2 * sum_k weights[i, k] * (p_i - p_k)
        # = 2 * (p_i - sum_k weights[i, k] * p_k)
        
        weighted_sum = weights @ current_points
        grad = 2.0 * (current_points - weighted_sum)
        
        # Update
        current_points = current_points - lr * grad
        current_points /= np.linalg.norm(current_points, axis=1, keepdims=True)
        
        # Check improvement every 10 steps
        if int(time.time() * 10) % 10 == 0: # Roughly every 0.1s
            current_radius = _compute_covering_radius(current_points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
            else:
                # Decay learning rate
                lr *= 0.95
                if lr < min_lr:
                    break
        
        if lr < min_lr:
            break
            
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    
    # Normalize
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Phase 1: Gradient Descent with soft-max surrogate
    # This should quickly improve the distribution.
    t1 = start_time + 60.0
    points = _gradient_descent(points, start_time, t1, initial_lr=0.05, min_lr=1e-4)
    
    current_radius = _compute_covering_radius(points)
    if current_radius < best_radius:
        best_radius = current_radius
        best_points = points.copy()
    
    # Phase 2: Local optimization using convex hull (similar to original but faster)
    # We can use the improved points from Phase 1 as a starting point.
    
    step_size = 0.01
    min_step_size = 1e-5
    
    while time.time() < deadline:
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
            
            worst_facet_idx = -1
            worst_radius = 0.0
            worst_circumcenter = None
            worst_vertices = None
            
            for idx, tri in enumerate(facets):
                p1 = points[tri[0]]
                p2 = points[tri[1]]
                p3 = points[tri[2]]
                
                v1 = p2 - p1
                v2 = p3 - p1
                cross = np.cross(v1, v2)
                norm_cross = np.linalg.norm(cross)
                
                if norm_cross < 1e-12:
                    continue
                    
                c = cross / norm_cross
                
                if np.dot(c, p1) < 0:
                    c = -c
                
                dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
                dist = np.arccos(dot_val)
                
                if dist > worst_radius:
                    worst_radius = dist
                    worst_facet_idx = idx
                    worst_circumcenter = c
                    worst_vertices = tri
                    
        except Exception:
            break
            
        if worst_facet_idx == -1:
            break
            
        prev_v1 = points[worst_vertices[0]].copy()
        prev_v2 = points[worst_vertices[1]].copy()
        prev_v3 = points[worst_vertices[2]].copy()
        
        for v_idx in worst_vertices:
            p = points[v_idx]
            direction = worst_circumcenter - p
            norm_dir = np.linalg.norm(direction)
            if norm_dir < 1e-12:
                continue
            direction /= norm_dir
            
            points[v_idx] = points[v_idx] + step_size * direction
            points[v_idx] /= np.linalg.norm(points[v_idx])
        
        current_radius = _compute_covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            step_size *= 0.98
        else:
            points[worst_vertices[0]] = prev_v1
            points[worst_vertices[1]] = prev_v2
            points[worst_vertices[2]] = prev_v3
            step_size *= 0.9
            
            if step_size < min_step_size:
                # Try random jitter
                for v_idx in worst_vertices:
                    noise = np.random.normal(0, 0.005, 3)
                    points[v_idx] = points[v_idx] + noise
                    points[v_idx] /= np.linalg.norm(points[v_idx])
                
                step_size = 0.005
                current_radius = _compute_covering_radius(points)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
                else:
                    points[worst_vertices[0]] = prev_v1
                    points[worst_vertices[1]] = prev_v2
                    points[worst_vertices[2]] = prev_v3
                    break # If we can't improve even with jitter, we are likely at a local optimum
    
    return best_points
