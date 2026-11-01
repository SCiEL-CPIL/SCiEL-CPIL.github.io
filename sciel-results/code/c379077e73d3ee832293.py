"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(points):
    """Compute the max-min radius for a set of points in [-0.5, 0.5]^2."""
    n = len(points)
    if n == 0:
        return 0.0
    # Distance to boundary
    boundary = np.minimum(0.5 - np.abs(points), axis=1)
    min_boundary = np.min(boundary)
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    return min(min_boundary, min_pair)


def _soft_min(values, temperature):
    """Soft-min using log-sum-exp for numerical stability."""
    # We want to approximate min(values) with a smooth function
    # -log-sum-exp(-values/T) * T approximates min(values) as T -> 0
    v = np.array(values, dtype=np.float64)
    v_min = np.min(v)
    # For stability, shift
    exp_vals = np.exp(-(v - v_min) / temperature)
    log_sum = np.log(np.sum(exp_vals))
    return v_min - temperature * log_sum


def _soft_max(values, temperature):
    """Soft-max using log-sum-exp for numerical stability."""
    v = np.array(values, dtype=np.float64)
    v_max = np.max(v)
    exp_vals = np.exp((v - v_max) / temperature)
    log_sum = np.log(np.sum(exp_vals))
    return v_max + temperature * log_sum


def _objective_soft(points, temperature):
    """
    Smooth surrogate for the max-min radius objective.
    We maximize min(min_pairwise_dist/2, min_boundary_dist).
    
    Using soft-min to smooth the min operation.
    """
    n = len(points)
    if n == 0:
        return 0.0
    
    # Boundary distances
    boundary = 0.5 - np.abs(points)
    boundary_dist = np.min(boundary, axis=1)  # shape (n,)
    
    # Pairwise distances / 2
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_dist = dist / 2.0  # shape (n, n)
    
    # Combine all constraints: boundary distances and pairwise half-distances
    # We want min of all these values
    all_vals = np.concatenate([boundary_dist, pair_dist[np.triu_indices(n, k=1)]])
    
    # Soft-min approximation
    return _soft_min(all_vals, temperature)


def _objective_grad(points, temperature):
    """
    Compute gradient of the soft-min objective with respect to points.
    """
    n = len(points)
    if n == 0:
        return np.zeros_like(points)
    
    # Boundary distances
    boundary = 0.5 - np.abs(points)
    boundary_dist = np.min(boundary, axis=1)  # shape (n,)
    
    # Pairwise distances / 2
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_dist = dist / 2.0  # shape (n, n)
    
    # All constraint values
    all_vals = np.concatenate([boundary_dist, pair_dist[np.triu_indices(n, k=1)]])
    
    # Soft-min weights: w_i = exp(-v_i / T) / sum(exp(-v_j / T))
    v_min = np.min(all_vals)
    exp_vals = np.exp(-(all_vals - v_min) / temperature)
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient with respect to points
    grad = np.zeros_like(points)
    
    # Boundary part: for each point i, boundary_dist_i = min(0.5 - |x_i|, 0.5 - |y_i|)
    # The gradient of 0.5 - |x| is -sign(x) for the active constraint
    # We need to track which constraint is active (min over x and y)
    for i in range(n):
        bx = 0.5 - abs(points[i, 0])
        by = 0.5 - abs(points[i, 1])
        w_b = weights[i]  # weight for this point's boundary constraint
        
        if bx <= by:
            # x-constraint active
            grad[i, 0] += w_b * (-np.sign(points[i, 0]))
        else:
            # y-constraint active
            grad[i, 1] += w_b * (-np.sign(points[i, 1]))
    
    # Pairwise part
    # For each pair (i, j), pair_dist[i,j] = ||p_i - p_j|| / 2
    # Gradient w.r.t. p_i: (p_i - p_j) / (2 * ||p_i - p_j||)
    # Gradient w.r.t. p_j: (p_j - p_i) / (2 * ||p_i - p_j||)
    for i in range(n):
        for j in range(i + 1, n):
            d = points[i] - points[j]
            dist_ij = np.linalg.norm(d)
            if dist_ij > 1e-12:
                w_p = weights[n + (i * (n - 1) - i * (i + 1) // 2 + j - i - 1)]
                # Actually, let's recompute the index properly
                # The upper triangle indices: for i < j, the index in the flattened upper triangle
                # Let's just use a different approach
                pass
    
    # Simpler: recompute pairwise gradients properly
    grad = np.zeros_like(points)
    
    # Boundary gradients
    for i in range(n):
        bx = 0.5 - abs(points[i, 0])
        by = 0.5 - abs(points[i, 1])
        w_b = weights[i]
        if bx <= by:
            grad[i, 0] += w_b * (-np.sign(points[i, 0]) if points[i, 0] != 0 else 1.0)
        else:
            grad[i, 1] += w_b * (-np.sign(points[i, 1]) if points[i, 1] != 0 else 1.0)
    
    # Pairwise gradients
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = points[i] - points[j]
            dist_ij = np.linalg.norm(d)
            if dist_ij > 1e-12:
                w_p = weights[n + idx]
                grad[i] += w_p * d / (2.0 * dist_ij)
                grad[j] += w_p * (-d) / (2.0 * dist_ij)
            idx += 1
    
    return grad


def _gradient_ascent(points, temperature, lr, max_iter=100):
    """Perform gradient ascent on the soft-min objective."""
    for _ in range(max_iter):
        grad = _objective_grad(points, temperature)
        points = points + lr * grad
        # Project back into the square
        points = np.clip(points, -0.5, 0.5)
    return points


def _anneal_optimize(points, temperatures, lr_schedule, max_iter_per_temp):
    """
    Multi-temperature annealing with gradient ascent.
    """
    current = points.copy()
    for temp, lr in zip(temperatures, lr_schedule):
        current = _gradient_ascent(current, temp, lr, max_iter_per_temp)
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial: square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    points = grid[:N].copy()
    
    best_points = points.copy()
    best_score = _score(points)
    
    # Temperature schedule: start coarse, end fine
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
    lr_schedule = [0.001, 0.0005, 0.0003, 0.0002, 0.0001, 0.00005, 0.00003, 0.00002, 0.00001, 0.000005]
    
    for temp, lr in zip(temperatures, lr_schedule):
        if time.time() > deadline - 10:
            break
        
        points = _gradient_ascent(points, temp, lr, 50)
        current_score = _score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
    
    # Final polish with very small temperature
    if time.time() < deadline - 5:
        points = _gradient_ascent(best_points, 0.00001, 0.000001, 200)
        final_score = _score(points)
        if final_score > best_score:
            best_score = final_score
            best_points = points.copy()
    
    return best_points
# EVOLVE-BLOCK-END
