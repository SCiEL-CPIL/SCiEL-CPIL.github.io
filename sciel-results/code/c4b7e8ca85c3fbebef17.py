import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_hull_radii_and_grads(points):
    """
    Computes the circumradius for each facet of the convex hull of points.
    Also computes the gradient of the circumradius with respect to the vertices.
    
    Returns:
    radii: array of shape (F,) with circumradius of each facet
    grads: list of arrays, grads[k] is (3, 3) matrix where grads[k][i] is gradient for vertex i of facet k
           Actually, it's easier to return a sparse-like structure or just accumulate gradients directly.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None

    F = len(hull.simplices)
    radii = np.zeros(F)
    # We will accumulate gradients for each point
    # grad_accum[j] is the gradient vector for point j
    grad_accum = np.zeros_like(points)
    
    # We also need to know which facets are "active" to weight them
    # But for now, let's just compute the raw gradients
    
    for k, simplex in enumerate(hull.simplices):
        verts = points[simplex]
        a, b, c = verts[0], verts[1], verts[2]
        
        # Normal to the facet
        normal = np.cross(b - a, c - a)
        norm_val = np.linalg.norm(normal)
        if norm_val < 1e-12:
            radii[k] = 0.0
            continue
        normal = normal / norm_val
        
        # Ensure normal points outward
        centroid = np.mean(verts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
            
        # Circumradius is the max angle between normal and vertices
        # For a spherical triangle, the circumcenter is the normal.
        # The radius is the angular distance from normal to any vertex.
        # Due to numerical errors, they might differ slightly. We take the max.
        dots = np.dot(verts, normal)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        
        # The circumradius is the max angle
        r_k = np.max(angles)
        radii[k] = r_k
        
        # Compute gradients for this facet
        # d(theta)/d(v) for theta = arccos(n . v)
        # grad = -(n - (n.v)v) / sqrt(1 - (n.v)^2)
        # This is the gradient of the angle for vertex v.
        # We want to MINIMIZE the radius, so we move in -grad direction.
        
        for i, v in enumerate(verts):
            dot_v = np.dot(v, normal)
            dot_v = np.clip(dot_v, -1.0, 1.0)
            denom = np.sqrt(np.maximum(1.0 - dot_v * dot_v, 1e-12))
            
            # Gradient of angle wrt v
            # g = -(normal - dot_v * v) / denom
            g = -(normal - dot_v * v) / denom
            
            # Add to accumulator
            # We will scale these by weights later
            # For now, store in a temporary list? No, we need weights.
            # Let's just store the gradients for all facets and weight them in the main loop.
            # To save memory/time, we can just accumulate if we assume uniform weight first.
            # But we want to focus on the worst facets.
            # Let's return the radii and the vertices, and compute gradients in the main loop for active facets.
            pass

    return radii, hull

def construct():
    start_time = time.time()
    total_time_limit = 145.0  # 5s margin
    
    best_points = None
    best_radius = np.pi
    
    rng = np.random.default_rng(42)
    
    # Generate multiple starting configurations
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    spiral_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Random Points
    random_points = rng.normal(size=(N, 3))
    random_points /= np.linalg.norm(random_points, axis=1, keepdims=True)
    
    starts = [spiral_points, random_points]
    
    # Add a few perturbed spirals
    for _ in range(3):
        p = spiral_points + rng.normal(scale=0.1, size=(N, 3))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        starts.append(p)
        
    for start_idx, start_pts in enumerate(starts):
        if time.time() - start_time > total_time_limit - 10.0:
            break
            
        points = start_pts.copy()
        
        # Optimization Loop
        # We use a gradient descent method on the maximum facet circumradius.
        # Step size decays over time.
        
        max_iters = 500
        step_size = 0.05
        decay = 0.999
        
        for it in range(max_iters):
            if time.time() - start_time > total_time_limit - 5.0:
                break
                
            # Compute hull and radii
            try:
                hull = ConvexHull(points)
            except Exception:
                # If hull fails, randomize or break
                points = rng.normal(size=(N, 3))
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                continue
                
            F = len(hull.simplices)
            radii = np.zeros(F)
            
            # Compute radii for all facets
            for k, simplex in enumerate(hull.simplices):
                verts = points[simplex]
                a, b, c = verts[0], verts[1], verts[2]
                normal = np.cross(b - a, c - a)
                norm_val = np.linalg.norm(normal)
                if norm_val < 1e-12:
                    radii[k] = 0.0
                    continue
                normal = normal / norm_val
                centroid = np.mean(verts, axis=0)
                if np.dot(normal, centroid) < 0:
                    normal = -normal
                dots = np.dot(verts, normal)
                dots = np.clip(dots, -1.0, 1.0)
                angles = np.arccos(dots)
                radii[k] = np.max(angles)
                
            current_max_radius = np.max(radii)
            
            if current_max_radius < best_radius:
                best_radius = current_max_radius
                best_points = points.copy()
                
            # If we are very close to the known optimum, stop early for this start
            if current_max_radius < np.deg2rad(11.32):
                break
                
            # Identify active facets (those close to the max)
            # Use a soft threshold
            # weights = exp(beta * (radii - current_max_radius))
            beta = 50.0  # Sharpness of focus on worst facets
            weights = np.exp(beta * (radii - current_max_radius))
            # Normalize weights? Not strictly necessary if step size is small,
            # but helps with stability.
            sum_weights = np.sum(weights)
            if sum_weights < 1e-12:
                break
            weights = weights / sum_weights
            
            # Compute gradient
            grad = np.zeros_like(points)
            
            for k, simplex in enumerate(hull.simplices):
                w_k = weights[k]
                if w_k < 1e-8:
                    continue
                    
                verts = points[simplex]
                a, b, c = verts[0], verts[1], verts[2]
                normal = np.cross(b - a, c - a)
                norm_val = np.linalg.norm(normal)
                if norm_val < 1e-12:
                    continue
                normal = normal / norm_val
                centroid = np.mean(verts, axis=0)
                if np.dot(normal, centroid) < 0:
                    normal = -normal
                    
                for i, v in enumerate(verts):
                    dot_v = np.dot(v, normal)
                    dot_v = np.clip(dot_v, -1.0, 1.0)
                    denom = np.sqrt(np.maximum(1.0 - dot_v * dot_v, 1e-12))
                    
                    # Gradient of angle wrt v
                    # We want to minimize the angle, so move in -grad direction
                    # grad_angle = -(normal - dot_v * v) / denom
                    # So update = -grad_angle = (normal - dot_v * v) / denom
                    g = (normal - dot_v * v) / denom
                    
                    # Accumulate weighted gradient
                    grad[simplex[i]] += w_k * g
                    
            # Normalize gradient to unit length to control step size
            grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
            grad_norm = np.where(grad_norm < 1e-12, 1.0, grad_norm)
            grad = grad / grad_norm
            
            # Update points
            # Move in direction of grad (which is direction of INCREASE in angle? No.)
            # Let's re-verify the sign.
            # theta = arccos(n . v)
            # d theta / d v = - (n - (n.v)v) / sqrt(1 - (n.v)^2)
            # We want to DECREASE theta.
            # So we should move in the direction of - (d theta / d v)
            # - (d theta / d v) = (n - (n.v)v) / sqrt(1 - (n.v)^2)
            # This is exactly what `g` is.
            # So `grad` points in the direction of DECREASE of the radius.
            # So we ADD step_size * grad.
            
            points += step_size * grad
            
            # Renormalize to sphere
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
            # Decay step size
            step_size *= decay
            
    return best_points
