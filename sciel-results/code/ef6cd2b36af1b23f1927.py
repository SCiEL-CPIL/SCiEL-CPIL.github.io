import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Initial seed result
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    best_result = np.array(cells[:N])
    best_score = np.sum(best_result[:, 2])

    # Try different aspect ratios
    # Perimeter 4 => W + H = 2
    # Symmetric, so W in [0.5, 1.5] is sufficient (W=1, H=1 is square)
    aspect_ratios = np.linspace(0.6, 1.4, 12)
    
    # Grid shapes to try for initialization
    grid_shapes = [(5, 4), (4, 5), (7, 3), (3, 7), (6, 4)] # 24, 24, 21, 21, 24 cells. We take first 21.

    for W in aspect_ratios:
        if time.time() > deadline:
            break
        H = 2.0 - W
        
        for shape in grid_shapes:
            if time.time() > deadline:
                break
            
            # Initialize positions in a grid
            rows, cols = shape
            total_cells = rows * cols
            if total_cells < N:
                continue
                
            xs = np.linspace(0.1 * W, 0.9 * W, cols)
            ys = np.linspace(0.1 * H, 0.9 * H, rows)
            
            grid_x, grid_y = np.meshgrid(xs, ys)
            pos = np.column_stack((grid_x.flatten(), grid_y.flatten()))[:N]
            
            # Run optimization
            result = optimize_packing(pos.copy(), W, H, deadline)
            if result is not None:
                score = np.sum(result[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = result

    return best_result

def optimize_packing(pos, W, H, deadline):
    """
    Attempts to maximize the sum of radii by growing them uniformly and relaxing positions.
    Returns array (N, 3) or None if failed.
    """
    N = len(pos)
    # Start with small radii
    r = np.ones(N) * 0.05
    step_r = 0.005
    max_steps = 2000
    
    current_pos = pos.copy()
    
    # Pre-allocate for vectorization
    # We will use a simple iterative relaxation
    
    for step in range(max_steps):
        if time.time() > deadline:
            return None
            
        # Grow radii
        r += step_r
        
        # Relax positions to accommodate new radii
        # We do a few sub-steps of relaxation per radius increase
        sub_steps = 50
        for _ in range(sub_steps):
            # Compute pairwise differences
            # d[i,j] = pos[i] - pos[j]
            diff = current_pos[:, np.newaxis, :] - current_pos[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            
            # Avoid division by zero
            dist[dist == 0] = 1e-6
            
            # Overlap amount: r_i + r_j - dist
            overlap = r[:, np.newaxis] + r[np.newaxis, :] - dist
            # Only consider positive overlaps
            overlap = np.maximum(overlap, 0)
            
            # Force: direction is diff/dist, magnitude is overlap
            # F[i] = sum_j overlap[i,j] * diff[i,j] / dist[i,j]
            # We want to push i away from j if overlapping.
            # Force on i from j: (pos[i] - pos[j]) / dist * overlap
            # Vectorized:
            force = np.sum(overlap[:, :, np.newaxis] * diff / dist[:, :, np.newaxis], axis=1)
            
            # Wall forces
            # Left wall: if x < r, push right (positive x)
            # Right wall: if x > W - r, push left (negative x)
            # Bottom wall: if y < r, push up (positive y)
            # Top wall: if y > H - r, push down (negative y)
            
            # We can add these to the force vector
            force[:, 0] += np.where(current_pos[:, 0] < r, 1.0, 0.0) - np.where(current_pos[:, 0] > W - r, 1.0, 0.0)
            force[:, 1] += np.where(current_pos[:, 1] < r, 1.0, 0.0) - np.where(current_pos[:, 1] > H - r, 1.0, 0.0)
            
            # Update positions with a small step size (learning rate)
            # We need a decaying learning rate or a fixed small one
            lr = 0.05
            current_pos += lr * force
            
            # Keep positions within bounds (hard constraint for stability)
            current_pos[:, 0] = np.clip(current_pos[:, 0], r, W - r)
            current_pos[:, 1] = np.clip(current_pos[:, 1], r, H - r)
            
        # Check if the packing is valid (no overlaps)
        # If there are significant overlaps, we might have grown too fast
        # Calculate max overlap
        diff = current_pos[:, np.newaxis, :] - current_pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        dist[dist == 0] = 1e-6
        overlap = r[:, np.newaxis] + r[np.newaxis, :] - dist
        max_overlap = np.max(overlap)
        
        if max_overlap > 0.1: # Too much overlap, shrink back?
            # If we fail to relax, we can try smaller step_r or break
            # For simplicity, if overlap is huge, we stop growing
            break
            
        # If we get stuck in a local minimum with small overlap, we can continue
        # But if max_overlap is close to 0, we are good.
        
    # Final check for validity
    diff = current_pos[:, np.newaxis, :] - current_pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist[dist == 0] = 1e-6
    overlap = r[:, np.newaxis] + r[np.newaxis, :] - dist
    if np.max(overlap) > 1e-4:
        return None
        
    # Check wall constraints
    if np.any(current_pos[:, 0] < -1e-4) or np.any(current_pos[:, 0] > W + 1e-4) or \
       np.any(current_pos[:, 1] < -1e-4) or np.any(current_pos[:, 1] > H + 1e-4):
        return None
        
    result = np.column_stack((current_pos, r))
    return result
