import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_X = X.copy()
    best_radius = np.inf
    
    def compute_radius_and_grads(X):
        # Compute Convex Hull
        try:
            hull = ConvexHull(X)
        except Exception:
            return np.inf, np.zeros_like(X)
        
        simplices = hull.simplices
        grads = np.zeros_like(X)
        max_dist = 0.0
        facet_dists = []
        facet_nearest = []
        
        for s in simplices:
            a, b, c = X[s[0]], X[s[1]], X[s[2]]
            # Circumcenter
            v = np.cross(a - b, b - c)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-10:
                continue
            v /= norm_v
            
            # Find nearest point among a, b, c to v
            dists = np.array([np.linalg.norm(v - a), np.linalg.norm(v - b), np.linalg.norm(v - c)])
            min_idx = np.argmin(dists)
            d = dists[min_idx]
            
            facet_dists.append(d)
            facet_nearest.append(s[min_idx])
            max_dist = max(max_dist, d)
            
        if max_dist == 0.0:
            return 0.0, grads
            
        # Threshold for "active" facets
        # Using a small epsilon relative to max_dist
        eps = 1e-5 
        for d, idx in zip(facet_dists, facet_nearest):
            if d > max_dist - eps:
                # Gradient contribution
                # We need the vector from X[idx] to v
                # But we lost v. We need to recompute or store it.
                pass
                
        # Recompute to get vectors
        # This is inefficient, let's store v in the first loop
        # Let's refactor
        
        return max_dist, grads

    # Refactored compute function
    def compute_radius_and_grads(X):
        try:
            hull = ConvexHull(X)
        except Exception:
            return np.inf, np.zeros_like(X)
        
        simplices = hull.simplices
        grads = np.zeros_like(X)
        max_dist = 0.0
        facet_data = [] # list of (dist, nearest_idx, v)
        
        for s in simplices:
            a, b, c = X[s[0]], X[s[1]], X[s[2]]
            v = np.cross(a - b, b - c)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-10:
                continue
            v /= norm_v
            
            dists = np.array([np.linalg.norm(v - a), np.linalg.norm(v - b), np.linalg.norm(v - c)])
            min_idx = np.argmin(dists)
            d = dists[min_idx]
            
            facet_data.append((d, s[min_idx], v))
            max_dist = max(max_dist, d)
            
        if max_dist == 0.0:
            return 0.0, grads
            
        eps = 1e-4
        for d, idx, v in facet_data:
            if d > max_dist - eps:
                # Gradient of distance w.r.t X[idx]
                # dist = ||v - X[idx]||
                # grad_X = (X[idx] - v) / dist
                # We want to minimize dist, so move X[idx] towards v
                # Update direction: v - X[idx]
                # Gradient descent step: X_new = X - eta * grad_obj
                # grad_obj = (X - v) / d
                # So X_new = X - eta * (X - v) / d
                # This is equivalent to moving towards v.
                grads[idx] += (X[idx] - v) / d
                
        return max_dist, grads

    # Optimization Loop
    eta = 0.1
    for iter in range(10000):
        if time.time() > deadline:
            break
            
        radius, grads = compute_radius_and_grads(X)
        
        if radius < best_radius:
            best_radius = radius
            best_X = X.copy()
            
        # Normalize gradients to unit length? 
        # The magnitude of the gradient depends on the number of active facets.
        # Let's normalize the total gradient vector for each point to unit length before scaling by eta.
        # Or just scale eta.
        
        # Project gradient onto tangent plane
        # g_tan = g - (g . X) X
        g_tan = grads - np.sum(grads * X, axis=1, keepdims=True) * X
        
        # Update
        X_new = X + eta * g_tan
        
        # Normalize
        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        X_new = X_new / norms
        
        # Check for improvement? 
        # If no improvement, reduce eta
        # Simple: just update.
        
        X = X_new
        
    # Final polish?
    # Return best found
    return best_X
