import numpy as np
import time
from scipy.spatial.distance import cdist

N = 200
START_TIME = time.time()
TIME_LIMIT = 130  # safety margin before 150s kill

def _compute_radius(centers):
    """Compute the achievable radius for given centers."""
    if len(centers) < 2:
        return 0.0
    # Minimum pairwise distance / 2
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    # Minimum distance to boundary
    min_boundary = np.min(np.abs(centers))  # distance to nearest wall (square is [-0.5,0.5]^2)
    # Actually: for each center, distance to boundary is min(0.5 - |x|, 0.5 - |y|)
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    min_boundary = np.min(np.minimum(dx, dy))
    return min(min_pair, min_boundary)

def _soft_min(vals, beta=100.0):
    """Smooth approximation of min using log-sum-exp."""
    return -np.log(np.sum(np.exp(-beta * vals))) / beta

def construct():
    global START_TIME
    elapsed = time.time() - START_TIME
    remaining = TIME_LIMIT - elapsed
    if remaining < 5:
        # Not enough time, return a quick grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Initial layout: hexagonal close-packed
    # For 200 circles in unit square, use hexagonal arrangement
    # Hex packing: rows offset by half spacing
    # Spacing s = 1/sqrt(N) approximately, but hex is denser
    
    # Start with a hexagonal pattern
    # Number of columns/rows for hex
    # In hex packing, density is pi/(2*sqrt(3)) ~ 0.9069 vs square 0.7854
    # So for same N, hex needs smaller area or larger radius
    
    # Let's use a hexagonal grid
    # Row i has circles at y = i * s * sqrt(3)/2
    # Even rows: x = j * s
    # Odd rows: x = (j + 0.5) * s
    
    # Estimate spacing for 200 circles in hex packing
    # Area per circle in hex: s^2 * sqrt(3)/2
    # Total area: N * s^2 * sqrt(3)/2 = 1 => s = sqrt(2/(N*sqrt(3)))
    s_hex = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    
    # Generate hex points
    points = []
    # How many rows?
    # y spacing = s_hex * sqrt(3)/2
    dy_hex = s_hex * np.sqrt(3.0) / 2.0
    n_rows = int(np.ceil(1.0 / dy_hex)) + 2
    for i in range(n_rows):
        y = i * dy_hex - 0.5
        if y < -0.5 - s_hex or y > 0.5 + s_hex:
            continue
        x_offset = 0.5 * s_hex if i % 2 == 1 else 0.0
        n_cols = int(np.ceil(1.0 / s_hex)) + 2
        for j in range(n_cols):
            x = j * s_hex - 0.5 + x_offset
            if x < -0.5 - s_hex or x > 0.5 + s_hex:
                continue
            points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than N, randomly sample N; if fewer, add random points
    if len(points) >= N:
        # Randomly select N
        idx = np.random.choice(len(points), N, replace=False)
        centers = points[idx].copy()
    else:
        # Pad with random points
        extra = N - len(points)
        extra_pts = np.random.uniform(-0.4, 0.4, (extra, 2))
        centers = np.vstack([points, extra_pts])
    
    # Force-directed relaxation
    # We want to maximize min distance, so use repulsive forces
    # Force between i,j: F = k / d^p * (direction from j to i) for d < threshold
    # Also boundary force: push inward if too close to wall
    
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    n_iter = 5000
    lr = 0.001
    decay = 0.9999
    beta_soft = 50.0
    
    for it in range(n_iter):
        # Check time
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Compute pairwise distances
        d = cdist(centers, centers)
        np.fill_diagonal(d, np.inf)
        
        # Repulsive force: for each pair, if d < 2*r_est, push apart
        # Use a smooth repulsion: F = (1/d^2) for d < some threshold
        # Gradient of soft-min of distances
        # d_soft = -log(sum(exp(-beta*d)))/beta
        # grad d_soft wrt center_i = -beta * sum_j exp(-beta*d_ij) * (c_i - c_j) / d_ij / sum_j exp(-beta*d_ij)
        
        # Simpler: use direct repulsion
        # Force on i from j: (c_i - c_j) / d_ij^3 * strength (for d_ij < 2*best_radius)
        strength = 1.0
        forces = np.zeros_like(centers)
        
        # Vectorized repulsion
        # For each pair, force magnitude = 1/d^2, direction = (c_i - c_j)/d
        # F_i = sum_j (c_i - c_j) / d_ij^3
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d_mat = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d_mat, 1.0)
        
        # Only apply force for close pairs (d < 2 * best_radius * 1.5)
        threshold = 2.0 * max(best_radius, 0.01) * 1.5
        mask = d_mat < threshold
        mask = mask & (d_mat > 1e-10)
        
        # Force = diff / d^3 for masked pairs
        inv_d3 = np.zeros_like(d_mat)
        inv_d3[mask] = 1.0 / d_mat[mask]**3
        forces = np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
        
        # Boundary forces
        # Push inward if close to boundary
        margin = 0.05
        for dim in range(2):
            # Left wall: x < -0.5 + margin
            close_left = centers[:, dim] < -0.5 + margin
            forces[close_left, dim] += (margin + 0.5 - centers[close_left, dim]) / margin * 0.1
            # Right wall: x > 0.5 - margin
            close_right = centers[:, dim] > 0.5 - margin
            forces[close_right, dim] -= (centers[close_right, dim] - (0.5 - margin)) / margin * 0.1
        
        # Update
        centers += lr * forces
        lr *= decay
        
        # Project back into square
        centers[:, 0] = np.clip(centers[:, 0], -0.49, 0.49)
        centers[:, 1] = np.clip(centers[:, 1], -0.49, 0.49)
        
        # Compute radius
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    # Final polish: use scipy minimize on a smooth surrogate
    if time.time() - START_TIME < TIME_LIMIT - 20:
        # Use a smooth objective: maximize soft-min of distances
        # Or use the exact min with a local search
        
        # Try a few iterations of gradient ascent on the soft-min
        beta = 100.0
        for polish_iter in range(200):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            d = cdist(best_centers, best_centers)
            np.fill_diagonal(d, np.inf)
            
            # Soft min of pairwise distances
            # grad of -log(sum(exp(-beta*d)))/beta wrt c_i
            # = -beta * sum_j exp(-beta*d_ij) * (c_i - c_j)/d_ij / sum_j exp(-beta*d_ij)
            # This is complex, so use a simpler approach:
            # Just do small random perturbations and keep if better
            
            # Or: use the fact that at optimum, the minimum distance pairs are "tight"
            # Move the centers of the tight pairs apart
            
            min_d = np.min(d)
            # Find pairs with distance close to min
            close_mask = d < min_d * 1.1
            close_mask = close_mask & (np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) > 0)
            
            if np.any(close_mask):
                # Get the pairs
                pairs = np.argwhere(close_mask)
                # For each pair, push apart
                for i, j in pairs:
                    if i < j:
                        diff = best_centers[i] - best_centers[j]
                        dist = np.linalg.norm(diff)
                        if dist > 1e-10:
                            push = 0.0001 * diff / dist
                            best_centers[i] += push
                            best_centers[j] -= push
            
            # Project
            best_centers[:, 0] = np.clip(best_centers[:, 0], -0.49, 0.49)
            best_centers[:, 1] = np.clip(best_centers[:, 1], -0.49, 0.49)
    
    # Final check and return
    final_radius = _compute_radius(best_centers)
    if final_radius >= best_radius - 1e-12:
        return best_centers
    else:
        return centers

# Ensure we return a valid array
def construct():
    """Main entry point."""
    global START_TIME
    if not hasattr(construct, '_initialized'):
        START_TIME = time.time()
        construct._initialized = True
    
    elapsed = time.time() - START_TIME
    remaining = TIME_LIMIT - elapsed
    if remaining < 5:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Initial layout: hexagonal close-packed
    s_hex = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    dy_hex = s_hex * np.sqrt(3.0) / 2.0
    n_rows = int(np.ceil(1.0 / dy_hex)) + 2
    
    points = []
    for i in range(n_rows):
        y = i * dy_hex - 0.5
        if y < -0.5 - s_hex or y > 0.5 + s_hex:
            continue
        x_offset = 0.5 * s_hex if i % 2 == 1 else 0.0
        n_cols = int(np.ceil(1.0 / s_hex)) + 2
        for j in range(n_cols):
            x = j * s_hex - 0.5 + x_offset
            if x < -0.5 - s_hex or x > 0.5 + s_hex:
                continue
            points.append([x, y])
    
    points = np.array(points)
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        centers = points[idx].copy()
    else:
        extra = N - len(points)
        extra_pts = np.random.uniform(-0.4, 0.4, (extra, 2))
        centers = np.vstack([points, extra_pts])
    
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    n_iter = 8000
    lr = 0.0005
    decay = 0.99995
    
    for it in range(n_iter):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        d = cdist(centers, centers)
        np.fill_diagonal(d, np.inf)
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d_mat = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d_mat, 1.0)
        
        threshold = 2.0 * max(best_radius, 0.01) * 1.5
        mask = d_mat < threshold
        mask = mask & (d_mat > 1e-10)
        
        inv_d3 = np.zeros_like(d_mat)
        inv_d3[mask] = 1.0 / d_mat[mask]**3
        forces = np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
        
        margin = 0.05
        for dim in range(2):
            close_left = centers[:, dim] < -0.5 + margin
            forces[close_left, dim] += (margin + 0.5 - centers[close_left, dim]) / margin * 0.1
            close_right = centers[:, dim] > 0.5 - margin
            forces[close_right, dim] -= (centers[close_right, dim] - (0.5 - margin)) / margin * 0.1
        
        centers += lr * forces
        lr *= decay
        
        centers[:, 0] = np.clip(centers[:, 0], -0.49, 0.49)
        centers[:, 1] = np.clip(centers[:, 1], -0.49, 0.49)
        
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    # Polish
    if time.time() - START_TIME < TIME_LIMIT - 15:
        for polish_iter in range(500):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            d = cdist(best_centers, best_centers)
            np.fill_diagonal(d, np.inf)
            min_d = np.min(d)
            close_mask = d < min_d * 1.1
            close_mask = close_mask & (np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) > 0)
            
            if np.any(close_mask):
                pairs = np.argwhere(close_mask)
                for i, j in pairs:
                    if i < j:
                        diff = best_centers[i] - best_centers[j]
                        dist = np.linalg.norm(diff)
                        if dist > 1e-10:
                            push = 0.00005 * diff / dist
                            best_centers[i] += push
                            best_centers[j] -= push
            
            best_centers[:, 0] = np.clip(best_centers[:, 0], -0.49, 0.49)
            best_centers[:, 1] = np.clip(best_centers[:, 1], -0.49, 0.49)
    
    return best_centers
