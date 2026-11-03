import numpy as np
import time

N = 50

def construct():
    # Start time
    start_time = time.time()
    
    # Initialize points using Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimization parameters
    max_time = 140  # seconds, leaving a safety margin
    initial_lr = 0.01
    final_lr = 0.0001
    initial_beta = 0.5
    final_beta = 0.05
    n_iterations = 10000
    
    best_points = points.copy()
    best_score = np.inf
    
    for it in range(n_iterations):
        # Check time limit
        if time.time() - start_time > max_time:
            break
        
        # Compute distances between all pairs of points
        # points shape: (N, 3)
        # dists shape: (N, N)
        dists = np.sqrt(np.sum(points[:, np.newaxis, :] - points[np.newaxis, :, :]**2, axis=2))
        
        # For each point, compute the maximum distance to any other point
        # But we want the covering radius, which is the maximum over the sphere.
        # A common approximation is to use the maximum distance between points.
        # However, the true covering radius is related to the largest empty cap.
        # A better approximation is to use the log-sum-exp of the distances.
        
        # For each point i, compute the log-sum-exp of the distances to all other points
        # This approximates the maximum distance from point i to the set of points.
        # Then the covering radius is approximately the maximum over i of this value.
        
        # Actually, the covering radius is the maximum over the entire sphere, not just the points.
        # A common heuristic is to minimize the maximum distance between points, which is a lower bound on the covering radius.
        # But this is not always optimal.
        
        # A better approach is to minimize the sum of the exponentiated distances, which approximates the maximum.
        # Let's use the log-sum-exp trick on the distances from each point to all other points.
        
        # For each point i, compute the log-sum-exp of the distances to all other points
        # We can do this for all points at once.
        
        # dists[i, j] is the distance between point i and point j
        # We want to compute for each i: log(sum_j exp(dists[i, j] / beta))
        # This approximates the maximum distance from point i to the set of points.
        
        # But we want the covering radius, which is the maximum over the entire sphere.
        # A common approximation is to use the maximum distance between points, but this is not accurate.
        
        # A better approximation is to use the fact that the covering radius is related to the largest empty cap.
        # The center of the largest empty cap is often near the circumcenter of a facet of the convex hull.
        # But this is complex to compute.
        
        # For simplicity, I will use a heuristic: minimize the maximum distance between points.
        # This is a lower bound on the covering radius, and often leads to good coverings.
        
        # Compute the maximum distance for each point
        max_dists = np.max(dists, axis=1)
        
        # Use log-sum-exp to smooth the maximum
        # log_sum_exp = log(sum(exp(dists / beta)))
        # We can compute this for each point.
        
        # To avoid numerical issues, we can use the log-sum-exp trick:
        # log(sum(exp(x_i))) = max(x_i) + log(sum(exp(x_i - max(x_i))))
        
        # For each point i, compute the log-sum-exp of the distances to all other points
        # dists[i, :] is the vector of distances from point i to all points
        # We can compute the log-sum-exp for all points at once.
        
        # Let's compute the log-sum-exp for each point
        # dists shape: (N, N)
        # We want to compute for each i: log(sum_j exp(dists[i, j] / beta))
        
        # To avoid numerical issues, we can subtract the maximum for each row
        max_dists_row = np.max(dists, axis=1, keepdims=True)
        exp_dists = np.exp((dists - max_dists_row) / beta)
        log_sum_exp = max_dists_row.squeeze() + np.log(np.sum(exp_dists, axis=1))
        
        # The objective is the maximum over i of log_sum_exp[i]
        # But we want to minimize the covering radius, which is approximated by the maximum distance.
        # So we can minimize the sum of log_sum_exp[i] or the maximum of log_sum_exp[i].
        # Let's minimize the sum, which is a smooth approximation of the maximum.
        
        obj = np.sum(log_sum_exp)
        
        # Compute the gradient of obj with respect to the points
        # This is complex, so I will use a numerical gradient.
        
        # For simplicity, I will use a random perturbation to estimate the gradient.
        # This is not efficient, but it is simple.
        
        # Alternatively, I can use a fixed number of iterations and a simple update rule.
        # Let's use a simple heuristic: move each point away from its nearest neighbor.
        
        # For each point, find its nearest neighbor and move it in the opposite direction.
        # This is a simple heuristic that often works well.
        
        # Find the nearest neighbor for each point
        nearest_idx = np.argmin(dists, axis=1)
        
        # Move each point away from its nearest neighbor
        for i in range(N):
            j = nearest_idx[i]
            if i != j:
                # Move point i away from point j
                direction = points[i] - points[j]
                direction = direction / np.linalg.norm(direction)
                points[i] += initial_lr * direction
        
        # Normalize the points to the unit sphere
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Update the learning rate and beta
        progress = it / n_iterations
        current_lr = initial_lr * (final_lr / initial_lr) ** progress
        current_beta = initial_beta * (final_beta / initial_beta) ** progress
        
        # Update the best points if the objective improved
        # Compute the covering radius for the current points
        # Approximate by the maximum distance between points
        current_score = np.max(dists)
        if current_score < best_score:
            best_score = current_score
            best_points = points.copy()
    
    return best_points
