"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

    # 1. Initialization: Fibonacci Spiral
    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    points = fibonacci_spiral(N).astype(np.float64)

    # 2. Grid for evaluating covering radius
    # Use a reasonably fine grid to approximate the max distance
    # 100 lat x 200 lon = 20,000 points. 
    # This is a good balance between accuracy and speed.
    lat = np.linspace(-np.pi/2, np.pi/2, 100)
    lon = np.linspace(0, 2*np.pi, 200, endpoint=False)
    LA, LO = np.meshgrid(lat, lon, indexing='ij')
    z_grid = np.sin(LA)
    x_grid = np.cos(LA) * np.cos(LO)
    y_grid = np.cos(LA) * np.sin(LO)
    grid_pts = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])
    
    # Normalize grid points to be safe
    grid_norms = np.linalg.norm(grid_pts, axis=1, keepdims=True)
    grid_pts = grid_pts / grid_norms

    M = len(grid_pts)

    def compute_covering_radius(pts):
        """Compute the max angular distance from grid points to nearest data point."""
        dot = np.dot(grid_pts, pts.T) # (M, N)
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot) # (M, N)
        min_dists = np.min(ang_dists, axis=1) # (M,)
        return np.max(min_dists)

    best_points = points.copy()
    best_rad = compute_covering_radius(best_points)

    # Gradient Descent with Soft-Max Objective
    # Objective: F = (1/beta) * log(sum(exp(beta * d_k)))
    # We want to minimize F.
    
    # Parameters
    beta = 20.0  # Sharpness of soft-max
    eta = 0.01   # Learning rate
    decay_eta = 0.99
    decay_beta = 1.005
    max_iter = 5000

    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break

        # 1. Compute distances and nearest neighbors
        dot = np.dot(grid_pts, points.T) # (M, N)
        dot = np.clip(dot, -1.0, 1.0)
        
        # Distances
        ang_dists = np.arccos(dot) # (M, N)
        
        # Find nearest point index for each grid point
        nearest_idx = np.argmin(ang_dists, axis=1) # (M,)
        
        # Get the min distance for each grid point
        min_dists = np.take_along_axis(ang_dists, nearest_idx[:, np.newaxis], axis=1).ravel() # (M,)
        
        # 2. Compute Soft-Max weights
        # To avoid overflow, subtract max
        max_d = np.max(min_dists)
        exp_vals = np.exp(beta * (min_dists - max_d))
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp # (M,)
        
        # 3. Compute Gradient for each point
        # grad_i = sum_{k: nearest_idx[k]==i} weights[k] * grad_{p_i} d_k
        # d_k = arccos(y_k . p_i)
        # grad_{p_i} d_k = -(y_k - (y_k.p_i)p_i) / sqrt(1 - (y_k.p_i)^2)
        
        # We need to sum contributions for each point i.
        # We can use np.add.at for accumulation, or bincount if weights were simple, 
        # but here we have vector gradients.
        
        # For each grid point k, we have a weight w_k and a vector y_k.
        # The gradient contribution to point i (where i = nearest_idx[k]) is:
        # w_k * ( - (y_k - dot_k * p_i) / sqrt(1 - dot_k^2) )
        
        # Let's compute the vector part first.
        # We need y_k and dot_k (which is dot[k, nearest_idx[k]])
        y_nearest = grid_pts # (M, 3)
        dot_nearest = np.take_along_axis(dot, nearest_idx[:, np.newaxis], axis=1).ravel() # (M,)
        
        # Numerator: y_k - dot_k * p_i
        # We need to scatter-add these vectors to the correct point index.
        
        # Pre-compute the denominator term: 1 / sqrt(1 - dot^2)
        # Handle numerical stability
        denom = np.sqrt(np.maximum(1e-12, 1.0 - dot_nearest**2))
        inv_denom = 1.0 / denom
        
        # The gradient vector for point i from grid point k is:
        # -w_k * (y_k / denom_k - dot_k * p_i / denom_k)
        # = -w_k / denom_k * y_k + w_k * dot_k / denom_k * p_i
        
        # Let A_k = -w_k / denom_k
        # Let B_k = w_k * dot_k / denom_k
        # grad_i = sum_{k in i} ( A_k * y_k + B_k * p_i )
        
        A = -weights * inv_denom # (M,)
        B = weights * dot_nearest * inv_denom # (M,)
        
        # Sum A_k * y_k for each i
        # np.add.at is slow. Let's try bincount with weights?
        # We can't bincount vectors directly.
        # We can do:
        # sum_A_y = np.zeros((N, 3))
        # np.add.at(sum_A_y, nearest_idx, A[:, np.newaxis] * y_nearest)
        
        # Alternative: Use scatter_add if available (not in standard numpy easily for 3D without add.at)
        # Let's use np.add.at, it's O(M) which is fine for M=20k, N=130.
        
        sum_A_y = np.zeros((N, 3))
        np.add.at(sum_A_y, nearest_idx, A[:, np.newaxis] * y_nearest)
        
        # Sum B_k for each i
        sum_B = np.bincount(nearest_idx, weights=B, minlength=N)
        
        # Gradient for point i:
        # grad_i = sum_A_y[i] + sum_B[i] * p_i
        
        gradients = sum_A_y + sum_B[:, np.newaxis] * points
        
        # 4. Update
        points -= eta * gradients
        
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points /= norms
        
        # 5. Evaluate and Track Best
        if it % 10 == 0:
            rad = compute_covering_radius(points)
            if rad < best_rad:
                best_rad = rad
                best_points = points.copy()
        
        # 6. Schedule
        eta *= decay_eta
        beta *= decay_beta
        
        # Stop if eta is too small
        if eta < 1e-6:
            break

    # Final Polish: A few iterations of direct "pull towards worst point"
    # This helps if the gradient descent gets stuck in a flat region or local min.
    if time.time() < deadline - 5.0:
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            # Find worst point on grid
            dot = np.dot(grid_pts, best_points.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            worst_idx = np.argmax(min_dists)
            worst_point = grid_pts[worst_idx]
            
            # Find nearest data point
            nearest_i = np.argmin(ang_dists[worst_idx])
            
            # Move nearest point towards worst point
            # Step size proportional to distance
            dist = ang_dists[worst_idx, nearest_i]
            if dist > 1e-6:
                # Geodesic step
                # v = worst_point - (worst_point . p_i) p_i
                # p_new = p_i + step * v / ||v||
                dot_wp = np.dot(worst_point, best_points[nearest_i])
                v = worst_point - dot_wp * best_points[nearest_i]
                v_norm = np.linalg.norm(v)
                if v_norm > 1e-9:
                    step = 0.5 * dist # Move halfway
                    best_points[nearest_i] += step * (v / v_norm)
                    best_points[nearest_i] /= np.linalg.norm(best_points[nearest_i])
            
            rad = compute_covering_radius(best_points)
            if rad < best_rad:
                best_rad = rad

    return best_points

# EVOLVE-BLOCK-END
