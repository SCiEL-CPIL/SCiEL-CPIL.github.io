"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Cover the unit sphere with 50 equal caps.
    Strategy:
    1. Start with multiple diverse initial configurations (Fibonacci, random, grid-like).
    2. Refine using accelerated Lloyd's algorithm (Voronoi relaxation).
    3. Use gradient-based optimization targeting worst facets.
    4. Use time budget to iterate until deadline.
    """
    start_time = time.time()
    timeout = 100.0  # Safety margin before 120s limit

    def fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_precise(points):
        """
        Compute the covering radius using the convex hull approach.
        """
        try:
            hull = ConvexHull(points)
        except Exception:
            return compute_covering_radius_sampled(points, 50000)
        
        max_circumradius = 0.0
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            e1 = v2 - v1
            e2 = v3 - v1
            
            A_mat = np.array([
                [2.0 * np.dot(e1, e1), 2.0 * np.dot(e1, e2)],
                [2.0 * np.dot(e1, e2), 2.0 * np.dot(e2, e2)]
            ])
            b_vec = np.array([np.dot(e1, e1), np.dot(e2, e2)])
            
            try:
                params = np.linalg.solve(A_mat, b_vec)
            except np.linalg.LinAlgError:
                continue
            
            circumcenter = v1 + params[0] * e1 + params[1] * e2
            norm_cc = np.linalg.norm(circumcenter)
            if norm_cc < 1e-10:
                continue
            circumcenter_norm = circumcenter / norm_cc
            
            dot_val = np.clip(np.dot(circumcenter_norm, v1), -1.0, 1.0)
            angular_radius = np.arccos(dot_val)
            
            if angular_radius > max_circumradius:
                max_circumradius = angular_radius
        
        return np.degrees(max_circumradius)

    def compute_covering_radius_sampled(points, num_test_points=100000):
        """
        Estimate covering radius by sampling many points on the sphere.
        """
        test_points = fibonacci_sphere(num_test_points)
        dot_products = test_points @ points.T
        dot_products = np.clip(dot_products, -1.0, 1.0)
        min_dot = np.min(dot_products, axis=1)
        angles = np.arccos(min_dot)
        max_angle = np.max(angles)
        return np.degrees(max_angle)

    def lloyd_iteration(points, test_points, step_size=1.0):
        """One iteration of Lloyd's algorithm (vectorized)."""
        dot_products = test_points @ points.T
        dot_products = np.clip(dot_products, -1.0, 1.0)
        nearest = np.argmax(dot_products, axis=1)

        # Vectorized centroid calculation using bincount
        # Sum of test_points for each cluster
        # We need to sum rows of test_points where nearest == i
        # This is equivalent to (test_points.T * ones(nearest)) @ ones? No.
        # Efficient way: np.add.at or bincount with weights.
        # Since test_points are (M, 3), we can do:
        # sums = np.zeros((N, 3))
        # np.add.at(sums, nearest, test_points)
        
        sums = np.zeros((len(points), 3))
        np.add.at(sums, nearest, test_points)
        
        counts = np.bincount(nearest, minlength=len(points))
        
        # Avoid division by zero
        counts_safe = np.where(counts == 0, 1, counts)
        
        centroids = sums / counts_safe[:, np.newaxis]
        
        # Normalize centroids
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        new_points = centroids / norms
        
        # Handle empty clusters (keep old point)
        empty_mask = counts == 0
        if np.any(empty_mask):
            new_points[empty_mask] = points[empty_mask]

        # Interpolate for stability
        result = (1 - step_size) * points + step_size * new_points
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return result / norms

    def gradient_descent_step(points, test_points, num_steps=5, base_lr=0.01):
        """Gradient descent on covering radius using worst facet analysis."""
        for step in range(num_steps):
            try:
                hull = ConvexHull(points)
            except Exception:
                break
            
            # Find all facets and their circumradii
            facet_data = []
            for simplex in hull.simplices:
                v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                
                e1 = v2 - v1
                e2 = v3 - v1
                
                A_mat = np.array([
                    [2.0 * np.dot(e1, e1), 2.0 * np.dot(e1, e2)],
                    [2.0 * np.dot(e1, e2), 2.0 * np.dot(e2, e2)]
                ])
                b_vec = np.array([np.dot(e1, e1), np.dot(e2, e2)])
                
                try:
                    params = np.linalg.solve(A_mat, b_vec)
                except np.linalg.LinAlgError:
                    continue
                
                cc = v1 + params[0] * e1 + params[1] * e2
                norm_cc = np.linalg.norm(cc)
                if norm_cc < 1e-10:
                    continue
                cc_norm = cc / norm_cc
                
                dot_val = np.clip(np.dot(cc_norm, v1), -1.0, 1.0)
                angular_radius = np.arccos(dot_val)
                
                facet_data.append((angular_radius, simplex, cc_norm))
            
            if not facet_data:
                break
            
            # Sort by radius, take top few
            facet_data.sort(key=lambda x: x[0], reverse=True)
            top_facets = facet_data[:min(5, len(facet_data))]
            
            # For each top facet, compute gradient direction to reduce radius
            gradient = np.zeros_like(points)
            for radius, simplex, cc_norm in top_facets:
                # Move each vertex slightly toward the circumcenter
                # This reduces the angular spread
                weight = radius / np.pi  # Normalize
                for idx in simplex:
                    current = points[idx]
                    # Direction to circumcenter, projected onto tangent plane
                    to_cc = cc_norm - current
                    # Project to tangent plane (remove radial component)
                    radial_component = np.dot(to_cc, current) * current
                    tangent_dir = to_cc - radial_component
                    norm_td = np.linalg.norm(tangent_dir)
                    if norm_td > 1e-10:
                        gradient[idx] += weight * tangent_dir / norm_td
            
            # Normalize gradient
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > 1e-10:
                gradient = gradient / grad_norm
                # Apply with learning rate
                points = points + base_lr * gradient
                # Renormalize
                norms = np.linalg.norm(points, axis=1, keepdims=True)
                norms = np.where(norms < 1e-10, 1.0, norms)
                points = points / norms
            
            # Also do a small Lloyd step to maintain good distribution
            points = lloyd_iteration(points, test_points, step_size=0.1)
        
        return points

    def separate_close_points(points, min_angle=0.1):
        """Separate points that are too close together."""
        cos_min = np.cos(min_angle)
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dot_val = np.dot(points[i], points[j])
                if dot_val > cos_min:
                    diff = points[i] - points[j]
                    norm_diff = np.linalg.norm(diff)
                    if norm_diff > 1e-10:
                        diff_norm = diff / norm_diff
                        push = 0.01 * (1 - dot_val) / (1 - cos_min)
                        points[i] = points[i] + push * diff_norm
                        points[j] = points[j] - push * diff_norm
                        points[i] = points[i] / np.linalg.norm(points[i])
                        points[j] = points[j] / np.linalg.norm(points[j])
        return points

    # Pre-generate test points
    num_test = 100000
    test_points = fibonacci_sphere(num_test)
    rng = np.random.RandomState(42)

    # Multiple initializations
    best_points = None
    best_score = np.inf

    initializations = []
    
    # 1. Fibonacci
    initializations.append(fibonacci_sphere(N))
    
    # 2. Random
    for _ in range(5):
        random_pts = rng.randn(N, 3)
        random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
        initializations.append(random_pts)
    
    # 3. Grid-like (lat/lon)
    latitudes = np.linspace(-np.pi/2, np.pi/2, 10)
    longitudes = np.linspace(0, 2*np.pi, 5, endpoint=False)
    grid_pts = []
    for lat in latitudes:
        for lon in longitudes:
            x = np.cos(lat) * np.cos(lon)
            y = np.cos(lat) * np.sin(lon)
            z = np.sin(lat)
            grid_pts.append([x, y, z])
    grid_pts = np.array(grid_pts[:N])
    grid_pts /= np.linalg.norm(grid_pts, axis=1, keepdims=True)
    initializations.append(grid_pts)
    
    # 4. Perturbed Fibonacci
    fib = fibonacci_sphere(N)
    for _ in range(3):
        perturbed = fib + rng.randn(N, 3) * 0.1
        perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
        initializations.append(perturbed)

    # Optimize each initialization with time budget
    # Reduce initializations to save time if needed, but let's keep them and just tighten loops
    time_per_init = (timeout - 10) / len(initializations)  # 10s reserved for final
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() - start_time >= timeout - 5:
            break
        
        points = init_pts.copy()
        init_deadline = start_time + (init_idx + 1) * time_per_init
        
        # Phase 1: Lloyd's algorithm (fast convergence)
        step_size = 0.5
        # Limit number of iterations to prevent hanging if time check is slow
        max_lloyd_iters = 200
        iters = 0
        while time.time() - start_time < min(init_deadline, start_time + time_per_init * 0.5) and iters < max_lloyd_iters:
            points = lloyd_iteration(points, test_points, step_size=step_size)
            step_size *= 0.95
            iters += 1
        
        # Phase 2: Gradient descent on covering radius (reduced steps)
        # Gradient descent is expensive, do it less frequently
        max_grad_iters = 20
        iters = 0
        while time.time() - start_time < min(init_deadline, start_time + time_per_init * 0.8) and iters < max_grad_iters:
            points = gradient_descent_step(points, test_points, num_steps=2, base_lr=0.005)
            points = separate_close_points(points)
            iters += 1
        
        # Phase 3: Fine Lloyd refinement
        step_size = 0.1
        max_fine_iters = 100
        iters = 0
        while time.time() - start_time < init_deadline and iters < max_fine_iters:
            points = lloyd_iteration(points, test_points, step_size=step_size)
            step_size *= 0.9
            iters += 1
            if step_size < 0.01:
                break
        
        # Evaluate using fast sampled method first, only do precise if it looks good
        fast_score = compute_covering_radius_sampled(points, num_test_points=10000)
        if fast_score < best_score + 0.1: # Threshold to trigger precise check
            score = compute_covering_radius_precise(points)
            if score < best_score:
                best_score = score
                best_points = points.copy()
    
    # Final refinement with remaining time
    if best_points is not None and time.time() - start_time < timeout - 5:
        points = best_points.copy()
        step_size = 0.05
        max_final_iters = 50
        iters = 0
        while time.time() - start_time < timeout - 5 and iters < max_final_iters:
            points = lloyd_iteration(points, test_points, step_size=step_size)
            step_size *= 0.95
            iters += 1
            
            # Occasional precise check
            if iters % 10 == 0:
                current_score = compute_covering_radius_precise(points)
                if current_score < best_score:
                    best_score = current_score
                    best_points = points.copy()
    
    # Final normalization
    if best_points is None:
        best_points = fibonacci_sphere(N)
    
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
