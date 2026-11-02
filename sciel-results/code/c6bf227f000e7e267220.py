import numpy as np
from scipy.spatial import ConvexHull

def construct():
    N = 50
    
    def fibonacci_spiral():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def optimize_covering_radius(points, max_iter=1000):
        """
        Minimizes the covering radius using convex hull facet circumradii.
        """
        points = points.copy()
        best_radius = np.inf
        best_points = points.copy()
        
        for it in range(max_iter):
            try:
                hull = ConvexHull(points)
            except Exception:
                # If hull fails, points might be coplanar or clustered.
                # Add small noise and re-normalize.
                points += np.random.normal(0, 0.01, points.shape)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                continue
                
            facets = hull.simplices
            radii = np.zeros(len(facets))
            centers = np.zeros((len(facets), 3))
            
            for k, tri in enumerate(facets):
                a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
                n = np.cross(b - a, c - a)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    radii[k] = 0
                    centers[k] = a
                    continue
                n = n / norm_n
                # Ensure normal points outward
                if np.dot(n, a) < 0:
                    u = -n
                else:
                    u = n
                centers[k] = u
                dot_val = np.clip(np.dot(u, a), -1.0, 1.0)
                radii[k] = np.arccos(dot_val)
                
            current_max_rad = np.max(radii)
            
            if current_max_rad < best_radius:
                best_radius = current_max_rad
                best_points = points.copy()
                
            # If we are very close to the known best, we can stop early or tighten criteria
            if current_max_rad < np.deg2rad(18.3000226):
                break
                
            # Calculate push directions
            push = np.zeros((N, 3))
            weight = np.zeros(N)
            
            # Use a soft threshold to focus on the worst facets
            # The threshold should be close to the max radius to only move points 
            # that are contributing to the bottleneck.
            threshold = current_max_rad - 0.05 * np.pi 
            
            for k, tri in enumerate(facets):
                if radii[k] > threshold:
                    # Weight by the square of the radius to emphasize worst facets
                    w = radii[k] ** 2
                    for idx in tri:
                        # Move point towards the circumcenter of the facet
                        push[idx] += w * centers[k]
                        weight[idx] += w
                        
            # Apply updates
            moved = False
            # Adaptive step size
            step_size = 0.05 * (1.0 - it / max_iter) + 0.005
            
            for i in range(N):
                if weight[i] > 1e-10:
                    target = push[i] / weight[i]
                    norm_t = np.linalg.norm(target)
                    if norm_t > 1e-10:
                        target /= norm_t
                        # Move towards target
                        new_p = points[i] + step_size * (target - points[i])
                        new_p /= np.linalg.norm(new_p)
                        points[i] = new_p
                        moved = True
                        
            if not moved:
                break
                
        return best_points, best_radius

    def optimize_min_distance(points, max_iter=200):
        """
        Maximizes the minimum pairwise dot product (minimizes max angular distance).
        This helps to enforce a uniform combinatorial structure.
        """
        points = points.copy()
        for it in range(max_iter):
            # Compute pairwise dot products
            dots = np.dot(points, points.T)
            # Set diagonal to 1 to ignore self
            np.fill_diagonal(dots, 1.0)
            
            min_dot = np.min(dots)
            
            # Find pairs with minimum dot product
            min_pairs = np.argwhere(dots == min_dot)
            
            # Calculate gradient for each point
            # To increase min_dot, we should move points apart.
            # For a pair (i, j) with low dot, move i away from j and j away from i.
            grad = np.zeros((N, 3))
            for i, j in min_pairs:
                if i == j: continue
                # Move i away from j: direction is -points[j]
                grad[i] -= points[j]
                # Move j away from i: direction is -points[i]
                grad[j] -= points[i]
                
            # Normalize gradients
            norms = np.linalg.norm(grad, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            grad /= norms
            
            # Update points
            step_size = 0.01
            points = points + step_size * grad
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
        return points

    best_points = None
    best_radius = np.inf
    
    # Try a few different initializations
    for seed in range(5):
        np.random.seed(seed)
        
        # Start with Fibonacci spiral, then add random perturbation
        points = fibonacci_spiral()
        points += np.random.normal(0, 0.1, points.shape)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Phase 1: Uniformity (Max Min Distance)
        # This ensures points are well-spread and likely form a convex hull
        points = optimize_min_distance(points, max_iter=100)
        
        # Phase 2: Covering Radius Minimization
        pts, rad = optimize_covering_radius(points, max_iter=1000)
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts

    return best_points
