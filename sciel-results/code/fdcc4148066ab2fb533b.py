"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE_SECONDS = 115.0  # Safety margin before 150s limit

def compute_radius(pts):
    """
    Computes the maximum radius for the given centers.
    pts: (N, 2) array of centers.
    Returns r such that circles of radius r are disjoint and inside [-0.5, 0.5]^2.
    """
    # 1. Distance to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    min_bound = np.min(np.minimum(d_x, d_y))
    
    # 2. Distance between centers
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_inter = np.sqrt(np.min(dist_sq))
    
    r_inter = min_dist_inter / 2.0
    r = np.minimum(r_inter, min_bound)
    
    return r

def negative_radius(x):
    """Objective to minimize: negative of the achievable radius."""
    pts = x.reshape(-1, 2)
    # Penalize points outside the square heavily
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    min_bound = np.min(np.minimum(d_x, d_y))
    
    if min_bound < 0:
        # Strong penalty for being outside
        return -min_bound * 10 + 100.0
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_inter = np.sqrt(np.min(dist_sq))
    
    r_inter = min_dist_inter / 2.0
    r = np.minimum(r_inter, min_bound)
    
    return -r

def force_directed_optimize(pts, start_time, deadline, max_iters=2000):
    """
    Force-directed optimization: push overlapping circles apart,
    pull them toward the center of available space.
    """
    pts = pts.copy()
    best_pts = pts.copy()
    best_r = compute_radius(pts)
    
    for it in range(max_iters):
        if time.time() > deadline - 2:
            break
        
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Repulsive forces between overlapping circles
        # For pairs closer than some threshold, push apart
        threshold = 0.15
        mask = dist < threshold
        mask[np.diag_indices_from(mask)] = False
        
        for i in range(N):
            for j in range(i+1, N):
                if mask[i, j]:
                    d = dist[i, j]
                    if d < 1e-9:
                        # Random small push
                        force = np.random.randn(2) * 0.001
                    else:
                        force = (pts[i] - pts[j]) / d * (threshold - d) * 0.1
                    forces[i] += force
                    forces[j] -= force
        
        # Boundary forces: push points away from edges if too close
        for i in range(N):
            for axis in range(2):
                if pts[i, axis] > 0.45:
                    forces[i, axis] -= (pts[i, axis] - 0.45) * 0.5
                elif pts[i, axis] < -0.45:
                    forces[i, axis] -= (pts[i, axis] + 0.45) * 0.5
        
        # Apply forces with adaptive step size
        step = 0.005
        new_pts = pts + forces * step
        
        # Clip to square
        new_pts = np.clip(new_pts, -0.499, 0.499)
        
        new_r = compute_radius(new_pts)
        if new_r > best_r:
            best_r = new_r
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # Reduce step size if no improvement
            step *= 0.95
            if step < 1e-6:
                break
    
    return best_pts, best_r

def gradient_optimize(pts, start_time, deadline):
    """
    Use scipy's minimize with BFGS for gradient-based optimization.
    """
    x0 = pts.flatten()
    
    try:
        result = minimize(
            negative_radius,
            x0,
            method='BFGS',
            options={'maxiter': 500, 'gtol': 1e-8}
        )
        best_pts = result.x.reshape(-1, 2)
        best_pts = np.clip(best_pts, -0.499, 0.499)
        return best_pts, compute_radius(best_pts)
    except Exception:
        return pts, compute_radius(pts)

def local_search(pts, start_time, deadline, num_restarts=20):
    """
    Local search: perturb points randomly and try to improve.
    """
    best_pts = pts.copy()
    best_r = compute_radius(pts)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 3:
            break
        
        # Perturb
        perturbation_scale = 0.01 * (1 + restart * 0.3)
        perturbed = best_pts + np.random.randn(N, 2) * perturbation_scale
        perturbed = np.clip(perturbed, -0.499, 0.499)
        
        # Quick force-directed refinement
        refined, r = force_directed_optimize(perturbed, start_time, deadline, max_iters=200)
        
        if r > best_r:
            best_r = r
            best_pts = refined
    
    return best_pts, best_r

def generate_hex_init():
    """
    Generate a hexagonal-like initialization based on known good packings.
    """
    # Use a structure inspired by BDGL packing
    # 50 circles: try rows of varying lengths
    # Common patterns: 7-6-7-6-7-6-7-4 or similar
    
    row_lengths = [7, 6, 7, 6, 7, 6, 7, 4]
    points = []
    num_rows = len(row_lengths)
    
    # Vertical spacing for hexagonal packing
    # If horizontal spacing is s, vertical is s*sqrt(3)/2
    s = 0.135  # Approximate diameter
    y_spacing = s * np.sqrt(3) / 2.0
    
    total_height = (num_rows - 1) * y_spacing
    y_start = -total_height / 2.0
    
    for i, length in enumerate(row_lengths):
        y = y_start + i * y_spacing
        x_spacing = s
        total_width = (length - 1) * x_spacing
        x_start = -total_width / 2.0
        
        # Offset odd rows
        offset = 0.0
        if i % 2 != 0:
            offset = x_spacing / 2.0
        
        for j in range(length):
            x = x_start + j * x_spacing + offset
            points.append([x, y])
    
    points = np.array(points)
    
    if len(points) > N:
        points = points[:N]
    elif len(points) < N:
        num_pad = N - len(points)
        random_pts = np.random.uniform(-0.4, 0.4, size=(num_pad, 2))
        points = np.vstack([points, random_pts])
    
    points = np.clip(points, -0.49, 0.49)
    return points

def generate_grid_init():
    """Generate a grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    if len(grid) > N:
        grid = grid[:N]
    elif len(grid) < N:
        num_pad = N - len(grid)
        random_pts = np.random.uniform(-0.4, 0.4, size=(num_pad, 2))
        grid = np.vstack([grid, random_pts])
    
    return grid

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_pts = None
    best_r = -1.0
    
    # Multiple initialization strategies
    inits = [
        generate_hex_init(),
        generate_grid_init(),
        np.random.uniform(-0.4, 0.4, size=(N, 2)),
        generate_hex_init() + np.random.normal(0, 0.005, size=(N, 2)),
    ]
    
    for init in inits:
        if time.time() > deadline - 15:
            break
        
        init = np.clip(init, -0.499, 0.499)
        
        # Step 1: Force-directed optimization
        pts_fd, r_fd = force_directed_optimize(init, start_time, deadline, max_iters=500)
        if r_fd > best_r:
            best_r = r_fd
            best_pts = pts_fd
        
        if time.time() > deadline - 10:
            break
        
        # Step 2: Gradient-based refinement
        pts_grad, r_grad = gradient_optimize(pts_fd, start_time, deadline)
        if r_grad > best_r:
            best_r = r_grad
            best_pts = pts_grad
        
        if time.time() > deadline - 8:
            break
        
        # Step 3: Local search with perturbations
        pts_ls, r_ls = local_search(pts_grad, start_time, deadline, num_restarts=10)
        if r_ls > best_r:
            best_r = r_ls
            best_pts = pts_ls
    
    # Final safety clip
    if best_pts is not None:
        best_pts = np.clip(best_pts, -0.5, 0.5)
        # Verify
        final_r = compute_radius(best_pts)
        if final_r < 0:
            # Fallback to hex init
            best_pts = generate_hex_init()
            best_pts = np.clip(best_pts, -0.5, 0.5)
    
    if best_pts is None:
        best_pts = generate_hex_init()
        best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts

# EVOLVE-BLOCK-END
