"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist

def _generate_hex_warm_start(n):
    """Generate a hexagonal lattice and select n points with good boundary margin."""
    best_points = None
    best_score = -1
    
    spacings = np.linspace(0.072, 0.076, 8)
    
    for d in spacings:
        row_height = d * np.sqrt(3) / 2
        col_width = d
        
        y_vals = np.arange(-0.5, 0.5 + row_height, row_height)
        points = []
        
        for j, y in enumerate(y_vals):
            if y < -0.5 - d or y > 0.5 + d:
                continue
            offset = 0.0
            if j % 2 != 0:
                offset = col_width / 2.0
            
            x_start = -0.5 + offset
            x_vals = np.arange(x_start, 0.5 + col_width, col_width)
            for x in x_vals:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
        
        if len(points) < n:
            continue
            
        points = np.array(points)
        
        dists_to_boundary = np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
        
        corner_dists = (np.sqrt((points[:, 0] - 0.5)**2 + (points[:, 1] - 0.5)**2) +
                       np.sqrt((points[:, 0] + 0.5)**2 + (points[:, 1] - 0.5)**2) +
                       np.sqrt((points[:, 0] - 0.5)**2 + (points[:, 1] + 0.5)**2) +
                       np.sqrt((points[:, 0] + 0.5)**2 + (points[:, 1] + 0.5)**2))
        
        idx = np.argsort(-dists_to_boundary + 0.001 * corner_dists)
        selected = points[idx[:n]]
        
        dists = cdist(selected, selected)
        min_pair = np.min(dists[np.triu_indices(n, k=1)])
        boundary_dists = np.minimum(0.5 - np.abs(selected[:, 0]), 0.5 - np.abs(selected[:, 1]))
        min_boundary = np.min(boundary_dists)
        
        score = min(min_pair / 2, min_boundary)
        
        if score > best_score:
            best_score = score
            best_points = selected.copy()
            
    if best_points is None:
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]
        
    return best_points

def _compute_radius(pts):
    """Compute the maximum valid radius for given points."""
    n = len(pts)
    dists = cdist(pts, pts)
    min_pair = np.min(dists[np.triu_indices(n, k=1)])
    boundary_dists = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
    min_boundary = np.min(boundary_dists)
    return min(min_pair / 2, min_boundary)

def construct():
    points = _generate_hex_warm_start(N)
    
    r = _compute_radius(points)
    n = N
    
    max_iter = 5000
    lr_base = 0.01
    r_increment = 0.0001
    min_r_increment = 0.00001
    
    for it in range(max_iter):
        lr = lr_base * (0.1) ** (it / max_iter)
        
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        
        target_dist = 2 * r
        mask = dists < target_dist
        violation = np.where(mask, target_dist - dists, 0.0)
        
        safe_dists = np.where(dists > 1e-8, dists, 1e-8)
        weights = np.where(mask, violation / safe_dists, 0.0)
        
        row_sums = np.sum(weights, axis=1)
        weighted_points = weights @ points
        pair_grad = row_sums[:, np.newaxis] * points - weighted_points
        
        bx = 0.5 - np.abs(points[:, 0])
        by = 0.5 - np.abs(points[:, 1])
        
        bound_grad = np.zeros_like(points)
        
        mask_x_active = bx <= by
        viol_x = np.where(mask_x_active, np.maximum(0, r - bx), 0.0)
        grad_x = 2 * viol_x * np.sign(points[:, 0])
        bound_grad[:, 0] += grad_x
        
        mask_y_active = by < bx
        viol_y = np.where(mask_y_active, np.maximum(0, r - by), 0.0)
        grad_y = 2 * viol_y * np.sign(points[:, 1])
        bound_grad[:, 1] += grad_y
        
        total_grad = pair_grad + bound_grad
        
        points -= lr * total_grad
        
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        current_r = _compute_radius(points)
        
        if current_r >= r:
            new_r = min(current_r, r + r_increment)
            if new_r > r:
                r = new_r
                r_increment = max(min_r_increment, r_increment * 0.99)
        else:
            r_increment = max(min_r_increment, r_increment * 0.95)
            
    return points
# EVOLVE-BLOCK-END
