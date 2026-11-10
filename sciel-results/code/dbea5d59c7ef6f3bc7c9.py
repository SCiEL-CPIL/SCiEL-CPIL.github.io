"""Improved packing for 50 equal spheres in a unit cube."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 150.0
SAFETY_MARGIN = 15.0


def _score(centres):
    """Compute the packing radius for a set of centres."""
    if centres.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Wall clearance
    wall = 0.5 - np.max(np.abs(centres), axis=1)
    min_wall = np.min(wall)
    if min_wall <= 0:
        return 0.0
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_wall, min_dist / 2.0)


def _initial_placement_hexagonal():
    """
    Generate initial placement using a hexagonal close-packed inspired layout.
    We place spheres in layers with hexagonal spacing within each layer,
    alternating offset between layers.
    """
    centres = []
    # Try to fit as many as possible in hexagonal layers
    # For a unit cube, we can try different layer spacings
    # Let's use a body-centered cubic like arrangement but with hex layers
    
    # Hexagonal layer: points at (i * dx, j * dy + (i%2) * dy/2, z)
    # For hexagonal packing in 2D: dx = 1, dy = sqrt(3)/2
    # But we need to fit in [-0.5, 0.5]^2
    
    # Let's try a simpler approach: use a 4x5x3 grid with some perturbation
    # Actually, let's use a more structured approach
    
    # Layer-based hexagonal packing
    # Each layer has points in hexagonal arrangement
    # We'll try to fit ~50 points in 3-4 layers
    
    # For a hexagonal layer in a square of side 1:
    # spacing s, number of points per row ~ 1/s
    # Let's try spacing = 0.25, giving 4 per row, with hex offset
    
    s = 0.24  # spacing between sphere centers
    dx = s
    dy = s * np.sqrt(3) / 2
    
    # Generate hexagonal points in one layer
    layer_points = []
    # We want to fit in [-0.5, 0.5] x [-0.5, 0.5]
    # Start from center and expand
    max_i = int(0.5 / dx) + 1
    max_j = int(0.5 / dy) + 1
    
    for j in range(-max_j, max_j + 1):
        for i in range(-max_i, max_i + 1):
            x = i * dx + (j % 2) * dx / 2
            y = j * dy
            if -0.48 <= x <= 0.48 and -0.48 <= y <= 0.48:
                layer_points.append([x, y])
    
    layer_points = np.array(layer_points)
    
    # Now create multiple layers with z spacing
    # We need about 50 points total. If each layer has ~12 points, we need ~5 layers
    # But let's be more careful
    
    # Let's just create a good initial guess using a perturbed grid
    # and let the optimizer do the work
    
    # Use a 4x4x4 grid = 64 points, take first 50
    # Actually, let's use a more spread out initial condition
    
    # Random initialization with minimum distance constraint
    rng = np.random.default_rng(42)
    centres = rng.uniform(-0.45, 0.45, size=(N, 3))
    
    # Simple local optimization to avoid overlaps initially
    for _ in range(100):
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centres[i] - centres[j])
                if d < 0.1:
                    # Push apart
                    if d > 1e-10:
                        push = (0.1 - d) / d * 0.5
                        centres[i] -= push * (centres[j] - centres[i])
                        centres[j] += push * (centres[j] - centres[i])
                    else:
                        centres[i][0] += 0.01
                        centres[j][0] -= 0.01
            # Keep in bounds
            centres[i] = np.clip(centres[i], -0.48, 0.48)
    
    return centres


def _optimize_from(start, time_budget):
    """
    Optimize the packing starting from 'start' using scipy minimize.
    Objective: minimize negative of the packing radius (i.e., maximize it).
    We use a smooth approximation: minimize sum of violations + penalty for small radius.
    """
    def objective(x):
        centres = x.reshape(N, 3)
        # Compute pairwise distances
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        wall = 0.5 - np.max(np.abs(centres), axis=1)
        
        # We want to maximize the minimum of (dists/2, wall)
        # Use a smooth minimum approximation
        min_d = np.min(dists)
        min_w = np.min(wall)
        
        # Soft minimum: -log(sum(exp(-a*x))) / a for large a
        a = 50.0
        # Combine all constraints
        all_vals = np.concatenate([dists.flatten() / 2.0, wall])
        # Remove inf
        all_vals = all_vals[all_vals < np.inf]
        
        if len(all_vals) == 0:
            return 1e6
        
        # Smooth min approximation
        smooth_min = -np.log(np.sum(np.exp(-a * all_vals))) / a
        
        return smooth_min
    
    def gradient(x):
        centres = x.reshape(N, 3)
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        wall = 0.5 - np.max(np.abs(centres), axis=1)
        
        min_d = np.min(dists)
        min_w = np.min(wall)
        
        all_vals = np.concatenate([dists.flatten() / 2.0, wall])
        all_vals = all_vals[all_vals < np.inf]
        
        if len(all_vals) == 0:
            return np.zeros_like(x)
        
        a = 50.0
        exp_vals = np.exp(-a * all_vals)
        sum_exp = np.sum(exp_vals)
        
        # Gradient of smooth min w.r.t. each value
        # d/dv_i [-log(sum(exp(-a*v)))/a] = exp(-a*v_i) / sum_exp
        grad_vals = exp_vals / sum_exp
        
        # Now back-propagate to centres
        grad = np.zeros_like(centres)
        
        # For pairwise distances
        # dists[i,j] = |c_i - c_j|
        # d/dc_i of dists[i,j] = (c_i - c_j) / dists[i,j]
        # d/dc_i of (dists[i,j]/2) = (c_i - c_j) / (2 * dists[i,j])
        
        # Flatten index mapping: dists is NxN, flattened in row-major
        # Index k corresponds to i = k // N, j = k % N
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                k = i * N + j
                # dists[i,j] is at position in flattened array
                # But we removed inf, so indices are shifted
                # This is getting complicated. Let's use numerical gradient instead.
        
        # Use numerical gradient for simplicity
        eps = 1e-6
        grad_num = np.zeros_like(x)
        for k in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[k] += eps
            x_minus[k] -= eps
            grad_num[k] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        
        return grad_num
    
    x0 = start.flatten()
    
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=[(-0.49, 0.49)] * (N * 3),
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
    )
    
    best_centres = result.x.reshape(N, 3)
    best_score = _score(best_centres)
    
    return best_centres, best_score


def construct():
    start_time = time.time()
    
    # Initial placement
    centres = _initial_placement_hexagonal()
    best_centres = centres.copy()
    best_score = _score(best_centres)
    
    # Try multiple restarts with different initializations
    rng = np.random.default_rng()
    
    while True:
        elapsed = time.time() - start_time
        remaining = DEADLINE - SAFETY_MARGIN - elapsed
        if remaining < 2.0:
            break
        
        # Optimize from current best
        centres_opt, score_opt = _optimize_from(best_centres, remaining)
        if score_opt > best_score:
            best_score = score_opt
            best_centres = centres_opt
        
        # Check if we should try a new random restart
        if best_score > 0.13:
            # Good enough, do one more polish
            break
        
        # New random initialization
        centres = rng.uniform(-0.45, 0.45, size=(N, 3))
        # Simple local push apart
        for _ in range(50):
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(centres[i] - centres[j])
                    if d < 0.08:
                        if d > 1e-10:
                            push = (0.08 - d) / d * 0.5
                            centres[i] -= push * (centres[j] - centres[i])
                            centres[j] += push * (centres[j] - centres[i])
                centres[i] = np.clip(centres[i], -0.48, 0.48)
        
        elapsed = time.time() - start_time
        remaining = DEADLINE - SAFETY_MARGIN - elapsed
        if remaining < 2.0:
            break
        
        centres_opt, score_opt = _optimize_from(centres, remaining)
        if score_opt > best_score:
            best_score = score_opt
            best_centres = centres_opt
    
    # Final check
    final_score = _score(best_centres)
    if final_score > best_score:
        best_score = final_score
    
    # Ensure valid output
    if best_score <= 0:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centres
