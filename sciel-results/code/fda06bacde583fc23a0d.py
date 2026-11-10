# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) of the (x,y) of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 610.0  # Safety margin

    def compute_score(points):
        if points.shape != (16, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def objective(points_flat):
        points = points_flat.reshape(16, 2)
        # Compute squared distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=2)
        # Mask diagonal to avoid self-distance
        sq_dists = sq_dists[~np.eye(16, dtype=bool)]
        
        if len(sq_dists) == 0:
            return 1e10
            
        # Soft min and max of squared distances
        # Using log-sum-exp trick for stability
        # We want to maximize (min d / max d)^2
        # Equivalent to maximizing min(d^2) - max(d^2) approximately?
        # No, ratio is better.
        # Let's try maximizing: soft_min(d^2) - soft_max(d^2)
        # This encourages all distances to be equal.
        
        # Temperature for soft min/max
        T = 0.1 
        
        # Soft min: -logsumexp(-x/T) * T
        # Soft max: logsumexp(x/T) * T
        
        # To avoid underflow/overflow in exp, shift by min/max
        x = sq_dists
        x_max = np.max(x)
        x_min = np.min(x)
        
        # Soft max
        log_sum_exp_max = np.log(np.sum(np.exp((x - x_max) / T))) + x_max / T
        soft_max = T * log_sum_exp_max
        
        # Soft min
        # -soft_max(-x)
        log_sum_exp_min_neg = np.log(np.sum(np.exp(((-x) - (-x_max)) / T))) + (-x_max) / T
        # Wait, simpler:
        # soft_min(x) = -soft_max(-x)
        # soft_max(-x) = T * log(sum(exp((-x - min(-x))/T))) + min(-x)
        # min(-x) = -max(x)
        # So soft_max(-x) = T * log(sum(exp((-x + max(x))/T))) - max(x)
        # soft_min(x) = - [T * log(sum(exp((max(x) - x)/T))) - max(x)]
        #            = max(x) - T * log(sum(exp((max(x) - x)/T)))
        
        # Let's just use np.min and np.max for the gradient-free version? 
        # No, we need gradient.
        # L-BFGS-B needs a function that returns scalar.
        # We can use a smooth approximation.
        
        # A common smooth approximation for min(x) is -1/alpha * log(sum(exp(-alpha*x)))
        # For max(x) is 1/alpha * log(sum(exp(alpha*x)))
        
        alpha = 10.0 # Sharpness
        
        # Prevent overflow
        x_shifted = x - np.max(x)
        soft_max = np.max(x) + np.log(np.sum(np.exp(alpha * x_shifted))) / alpha
        
        x_shifted_min = x - np.min(x)
        # soft_min = min(x) - log(sum(exp(-alpha*(x-min(x))))) / alpha
        soft_min = np.min(x) - np.log(np.sum(np.exp(-alpha * x_shifted_min))) / alpha
        
        # We want to maximize soft_min / soft_max? 
        # Or maximize soft_min - soft_max?
        # Maximizing soft_min - soft_max pushes all distances to be equal.
        # This is a good proxy for maximizing the ratio.
        
        val = soft_min - soft_max
        if not np.isfinite(val):
            return 1e10
        return -val

    best_points = None
    best_score = -np.inf

    n = 16
    d = 2

    # Generate diverse initial candidates
    candidates = []

    # 1. Regular 4x4 grid
    grid = np.array([np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))]).reshape(2, -1).T
    candidates.append(grid)

    # 2. Triangular lattice subset
    s = 1.0
    h = s * np.sqrt(3) / 2
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * s + (j % 2) * s / 2
            y = j * h
            lattice_points.append([x, y])
    lattice_points = np.array(lattice_points)
    centroid = np.mean(lattice_points, axis=0)
    dists_to_centroid = np.sum((lattice_points - centroid) ** 2, axis=1)
    idx = np.argsort(dists_to_centroid)[:16]
    candidates.append(lattice_points[idx])

    # 3. Points on a circle (regular 16-gon)
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    circle_points = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(circle_points)

    # 4. Random points in a square
    for seed in range(10):
        np.random.seed(seed + 100)
        random_points = np.random.randn(n, d)
        candidates.append(random_points)

    # 5. Two concentric circles
    inner_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_angles = inner_angles + np.pi / 8
    inner_radius = 0.5
    outer_radius = 1.0
    inner_points = np.column_stack([inner_radius * np.cos(inner_angles), inner_radius * np.sin(inner_angles)])
    outer_points = np.column_stack([outer_radius * np.cos(outer_angles), outer_radius * np.sin(outer_angles)])
    two_circle = np.vstack([inner_points, outer_points])
    candidates.append(two_circle)

    # Evaluate initial scores
    for cand in candidates:
        score = compute_score(cand)
        if score > best_score:
            best_score = score
            best_points = cand.copy()

    # Sort candidates by score
    scored_candidates = [(compute_score(c), c) for c in candidates]
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    from scipy.optimize import minimize

    # Optimize the top 10 candidates using L-BFGS-B on the smooth surrogate
    for score, cand in scored_candidates[:10]:
        if time.time() > deadline:
            break
        try:
            # Use L-BFGS-B which is efficient for smooth functions
            result = minimize(objective, cand.flatten(), method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if time.time() > deadline:
                break
            opt_points = result.x.reshape(16, 2)
            
            # Local refinement using coordinate descent on the true non-smooth objective
            # This helps escape the flat regions of the surrogate near the optimum
            current = opt_points.copy()
            current_score = compute_score(current)
            
            # Simple coordinate descent
            step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(n):
                    for j in range(d):
                        for step in step_sizes:
                            for direction in [1, -1]:
                                new_point = current.copy()
                                new_point[i, j] += direction * step
                                new_score = compute_score(new_point)
                                if new_score > current_score:
                                    current = new_point
                                    current_score = new_score
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    step_sizes = [s/2 for s in step_sizes]
                    if step_sizes[0] < 1e-5:
                        break

            if current_score > best_score:
                best_score = current_score
                best_points = current.copy()
        except Exception:
            pass

    # If we still have time, try a few more random restarts
    if time.time() < deadline - 30:
        for attempt in range(20):
            if time.time() > deadline:
                break
            np.random.seed(attempt * 7 + 13)
            random_points = np.random.randn(n, d)
            random_points = random_points / np.std(random_points)
            try:
                result = minimize(objective, random_points.flatten(), method='L-BFGS-B',
                                 options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                if time.time() > deadline:
                    break
                opt_points = result.x.reshape(16, 2)
                opt_score = compute_score(opt_points)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
            except Exception:
                pass

    if best_points is None:
        best_points = candidates[0]

    return best_points

# EVOLVE-BLOCK-END
