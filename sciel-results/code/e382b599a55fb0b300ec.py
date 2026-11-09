import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _calc_covering_radius(pts):
    """Calculate the covering radius from the convex hull facets."""
    try:
        hull = ConvexHull(pts)
        simplices = hull.simplices
        max_dist = 0.0
        facet_dists = []
        for s in simplices:
            v1, v2, v3 = pts[s[0]], pts[s[1]], pts[s[2]]
            n = np.cross(v2 - v1, v3 - v1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n = n / n_norm
            if np.dot(n, v1) < 0:
                n = -n
            dot_val = np.dot(n, v1)
            if abs(dot_val) < 1e-10:
                continue
            k = 1.0 / (2.0 * dot_val)
            c = k * n
            c = c / np.linalg.norm(c)
            dist = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
            facet_dists.append(dist)
            if dist > max_dist:
                max_dist = dist
        return max_dist, np.array(facet_dists) if facet_dists else np.array([])
    except Exception:
        return np.inf, np.array([])

def _smooth_covering_radius(pts, beta=50.0):
    """Smooth surrogate using log-sum-exp of facet distances."""
    max_dist, facet_dists = _calc_covering_radius(pts)
    if len(facet_dists) == 0 or not np.isfinite(max_dist):
        return np.inf, np.zeros(N, 3)
    
    # Soft max via log-sum-exp
    shifted = facet_dists - np.max(facet_dists)
    weights = np.exp(beta * shifted) / np.sum(np.exp(beta * shifted))
    
    # Compute gradient contribution: weight each facet's circumcenter direction
    hull = ConvexHull(pts)
    simplices = hull.simplices
    
    # Accumulate weighted circumcenter directions per vertex
    grad = np.zeros_like(pts)
    
    for i, s in enumerate(simplices):
        v1, v2, v3 = pts[s[0]], pts[s[1]], pts[s[2]]
        n = np.cross(v2 - v1, v3 - v1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        n = n / n_norm
        if np.dot(n, v1) < 0:
            n = -n
        dot_val = np.dot(n, v1)
        if abs(dot_val) < 1e-10:
            continue
        k = 1.0 / (2.0 * dot_val)
        c = k * n
        c = c / np.linalg.norm(c)
        
        # The gradient of arccos(dot(c,v)) w.r.t. v is -c/sqrt(1-dot^2)
        # But we want to move vertices away from the circumcenter direction
        # Actually, we want to minimize the max distance, so we should move
        # vertices in the direction that reduces the worst facet's circumradius
        
        # Simpler: move each vertex towards the weighted average of adjacent circumcenters
        # but with soft-max weighting emphasizing the worst facets
        w = weights[i] if i < len(weights) else 0.0
        grad[s[0]] += w * c
        grad[s[1]] += w * c
        grad[s[2]] += w * c
    
    smooth_val = np.log(np.sum(np.exp(beta * facet_dists))) / beta
    return smooth_val, grad

def _fibonacci_sphere():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere():
    """Generate random points on the sphere."""
    theta = np.random.uniform(0, 2 * np.pi, N)
    phi = np.arccos(2 * np.random.uniform(0, 1, N) - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def _optimize(points, time_limit, use_smooth=True):
    """Optimize the configuration with momentum and optional smooth surrogate."""
    best_points = points.copy()
    best_radius = np.inf
    
    # Initial radius
    current_radius, _ = _calc_covering_radius(points)
    if current_radius < best_radius:
        best_radius = current_radius
        best_points = points.copy()
    
    momentum = np.zeros_like(points)
    lr = 0.03  # Learning rate
    iterations = 0
    max_iters = 30000
    start_time = time.time()
    
    while time.time() - start_time < time_limit and iterations < max_iters:
        iterations += 1
        
        try:
            hull = ConvexHull(points)
            simplices = hull.simplices
            
            # Calculate target for each point (average of adjacent circumcenters)
            targets = np.zeros_like(points)
            counts = np.zeros(N)
            facet_dists = []
            circumcenters = []
            
            for s in simplices:
                v1, v2, v3 = points[s[0]], points[s[1]], points[s[2]]
                n = np.cross(v2 - v1, v3 - v1)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                n = n / n_norm
                if np.dot(n, v1) < 0:
                    n = -n
                dot_val = np.dot(n, v1)
                if abs(dot_val) < 1e-10:
                    continue
                k = 1.0 / (2.0 * dot_val)
                c = k * n
                c = c / np.linalg.norm(c)
                dist = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
                facet_dists.append(dist)
                circumcenters.append((c, s, dist))
                
                # Add this circumcenter to the targets of the 3 vertices
                targets[s[0]] += c
                targets[s[1]] += c
                targets[s[2]] += c
                counts[s[0]] += 1
                counts[s[1]] += 1
                counts[s[2]] += 1
            
            if len(facet_dists) == 0:
                break
            
            # Use soft-max weighting to emphasize worst facets
            facet_dists = np.array(facet_dists)
            beta = 30.0
            shifted = facet_dists - np.max(facet_dists)
            weights = np.exp(beta * shifted) / np.sum(np.exp(beta * shifted))
            
            # Recompute weighted targets
            weighted_targets = np.zeros_like(points)
            weighted_counts = np.zeros(N)
            
            for i, (c, s, dist) in enumerate(circumcenters):
                w = weights[i]
                weighted_targets[s[0]] += w * c
                weighted_targets[s[1]] += w * c
                weighted_targets[s[2]] += w * c
                weighted_counts[s[0]] += w
                weighted_counts[s[1]] += w
                weighted_counts[s[2]] += w
            
            # Combine uniform and weighted targets
            for j in range(N):
                if counts[j] > 0:
                    uniform_target = targets[j] / counts[j]
                else:
                    uniform_target = points[j]
                
                if weighted_counts[j] > 1e-10:
                    weighted_target = weighted_targets[j] / weighted_counts[j]
                else:
                    weighted_target = points[j]
                
                # Blend: 50% uniform, 50% weighted
                targets[j] = 0.5 * uniform_target + 0.5 * weighted_target
            
            # Update points with momentum
            diff = targets - points
            momentum = 0.7 * momentum + lr * diff
            points += momentum
            
            # Normalize
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            
        except Exception:
            break

        # Evaluate
        current_radius, _ = _calc_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Add small noise to escape local minima periodically
        if iterations % 200 == 0:
            noise = np.random.normal(0, 0.005, points.shape)
            points += noise
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
    
    # Final polish: tighter tolerance, more iterations
    if time.time() - start_time < time_limit - 5:
        polish_points = best_points.copy()
        polish_momentum = np.zeros_like(polish_points)
        polish_lr = 0.01
        polish_iters = 0
        polish_max_iters = 5000
        
        while time.time() - start_time < time_limit - 5 and polish_iters < polish_max_iters:
            polish_iters += 1
            try:
                hull = ConvexHull(polish_points)
                simplices = hull.simplices
                
                targets = np.zeros_like(polish_points)
                counts = np.zeros(N)
                
                for s in simplices:
                    v1, v2, v3 = polish_points[s[0]], polish_points[s[1]], polish_points[s[2]]
                    n = np.cross(v2 - v1, v3 - v1)
                    n_norm = np.linalg.norm(n)
                    if n_norm < 1e-10:
                        continue
                    n = n / n_norm
                    if np.dot(n, v1) < 0:
                        n = -n
                    dot_val = np.dot(n, v1)
                    if abs(dot_val) < 1e-10:
                        continue
                    k = 1.0 / (2.0 * dot_val)
                    c = k * n
                    c = c / np.linalg.norm(c)
                    
                    targets[s[0]] += c
                    targets[s[1]] += c
                    targets[s[2]] += c
                    counts[s[0]] += 1
                    counts[s[1]] += 1
                    counts[s[2]] += 1
                
                for j in range(N):
                    if counts[j] > 0:
                        targets[j] /= counts[j]
                    else:
                        targets[j] = polish_points[j]
                
                diff = targets - polish_points
                polish_momentum = 0.5 * polish_momentum + polish_lr * diff
                polish_points += polish_momentum
                
                norms = np.linalg.norm(polish_points, axis=1, keepdims=True)
                polish_points = polish_points / norms
                
            except Exception:
                break
            
            current_radius, _ = _calc_covering_radius(polish_points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = polish_points.copy()
    
    return best_points

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s safety margin for 120s limit
    
    # Try multiple initializations
    initializations = [
        _fibonacci_sphere(),
        _random_sphere(),
        _random_sphere(),
        _random_sphere(),
    ]
    
    # Allocate time per initialization
    time_per_init = (time_limit - 5) / len(initializations)
    
    best_overall = None
    best_overall_radius = np.inf
    
    for init in initializations:
        if time.time() - start_time > time_limit - 5:
            break
        
        elapsed = time.time() - start_time
        remaining = time_limit - 5 - elapsed
        if remaining <= 0:
            break
        
        result = _optimize(init, remaining)
        radius, _ = _calc_covering_radius(result)
        
        if radius < best_overall_radius:
            best_overall_radius = radius
            best_overall = result.copy()
    
    if best_overall is None:
        best_overall = _fibonacci_sphere()
    
    return best_overall
