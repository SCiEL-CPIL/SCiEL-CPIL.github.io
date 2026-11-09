"""Bootstrap direct construction for 100 equal spheres in a unit cube."""
N = 100
CUBE_HALF = 0.5
DEADLINE_S = 115.0  # safety margin before 120 s evaluator kill

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import sys

def _grid_score(points):
    """Compute the largest radius that fits for the given points."""
    if len(points) < 2:
        return 0.5
    # min pairwise distance
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # distance to boundary
    margins = CUBE_HALF - np.abs(points)
    min_boundary = np.min(margins)
    return min(min_pair, min_boundary) / 2.0

def _push_apart_refine(points, max_iter=20000, step=0.001):
    """
    Push-apart refinement: repeatedly find the closest pair and push them apart.
    This is more targeted than random jitter.
    """
    best_pts = points.copy()
    best_score = _grid_score(points)
    step_size = step
    it = 0
    
    while it < max_iter:
        # Find the closest pair
        d = best_pts[:, None, :] - best_pts[None, :, :]
        dists = np.sqrt(np.sum(d**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Find indices of closest pair
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        min_dist = dists[i, j]
        
        # Also check boundary constraints
        margins = CUBE_HALF - np.abs(best_pts)
        min_margin_idx = np.argmin(np.min(margins, axis=1))
        min_margin = np.min(margins[min_margin_idx])
        
        # Determine the most violated constraint
        if min_margin < min_dist:
            # Boundary is the bottleneck - push the point with smallest margin inward
            idx = min_margin_idx
            # Push toward center
            direction = -best_pts[idx] / (np.linalg.norm(best_pts[idx]) + 1e-10)
            new_pt = best_pts[idx].copy() + direction * step_size
            new_pt = np.clip(new_pt, -CUBE_HALF, CUBE_HALF)
            candidate = best_pts.copy()
            candidate[idx] = new_pt
        else:
            # Pairwise distance is the bottleneck - push i and j apart
            direction = best_pts[i] - best_pts[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction = direction / norm
            
            new_i = best_pts[i].copy() + direction * step_size / 2
            new_j = best_pts[j].copy() - direction * step_size / 2
            new_i = np.clip(new_i, -CUBE_HALF, CUBE_HALF)
            new_j = np.clip(new_j, -CUBE_HALF, CUBE_HALF)
            
            candidate = best_pts.copy()
            candidate[i] = new_i
            candidate[j] = new_j
        
        cand_score = _grid_score(candidate)
        if cand_score > best_score + 1e-14:
            best_pts = candidate
            best_score = cand_score
            step_size = max(step_size * 0.999, 1e-6)
        else:
            step_size = max(step_size * 0.98, 1e-6)
            if step_size < 1e-6:
                break
        it += 1
    
    return best_pts, best_score

def _soft_min_gradient_refine(points, max_iter=1000, lr=0.001, beta=100.0):
    """
    Gradient ascent using soft-min surrogate for the minimum distance objective.
    Uses numerical gradient computation.
    """
    best_pts = points.copy()
    best_score = _grid_score(points)
    
    def _soft_min_value(pts):
        """Compute soft-min of all pairwise distances and boundary margins."""
        n = len(pts)
        # Pairwise distances
        d = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(d**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Boundary margins
        margins = CUBE_HALF - np.abs(pts)
        min_margins = np.min(margins, axis=1)
        
        all_vals = np.concatenate([dists.flatten(), min_margins])
        
        # Soft-min approximation
        min_val = np.min(all_vals)
        log_sum = np.log(np.sum(np.exp(-beta * (all_vals - min_val))))
        soft_min = min_val - log_sum / beta
        
        return soft_min / 2.0
    
    for it in range(max_iter):
        # Numerical gradient
        eps = 1e-5
        grad = np.zeros_like(best_pts)
        for i in range(len(best_pts)):
            for j in range(3):
                perturbed = best_pts.copy()
                perturbed[i, j] += eps
                perturbed = np.clip(perturbed, -CUBE_HALF, CUBE_HALF)
                f_plus = _soft_min_value(perturbed)
                
                perturbed2 = best_pts.copy()
                perturbed2[i, j] -= eps
                perturbed2 = np.clip(perturbed2, -CUBE_HALF, CUBE_HALF)
                f_minus = _soft_min_value(perturbed2)
                
                grad[i, j] = (f_plus - f_minus) / (2 * eps)
        
        # Gradient ascent
        new_pts = best_pts + lr * grad
        new_pts = np.clip(new_pts, -CUBE_HALF, CUBE_HALF)
        
        new_score = _grid_score(new_pts)
        if new_score > best_score + 1e-15:
            best_pts = new_pts
            best_score = new_score
            lr *= 0.995
        else:
            lr *= 0.95
            if lr < 1e-8:
                break
    
    return best_pts, best_score

def _generate_fcc_points():
    """
    Generate FCC lattice points in a region and select 100 with best properties.
    FCC lattice: points at (i, j, k) where i+j+k is even, plus face centers.
    """
    points = []
    # Generate FCC points in a larger region, then scale and select
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # Corner point
                points.append([i, j, k])
                # Face center points
                points.append([i + 0.5, j + 0.5, k])
                points.append([i + 0.5, j, k + 0.5])
                points.append([i, j + 0.5, k + 0.5])
    
    points = np.array(points, dtype=float)
    
    # Scale to fit in [-0.5, 0.5]
    # The FCC lattice has nearest neighbor distance of 1/sqrt(2) in unit spacing
    # We want the packing to fit, so we need to scale appropriately
    # Let's try different scales and select the one that gives 100 points with best margin
    
    best_pts = None
    best_score = -1.0
    
    for scale in np.linspace(0.1, 0.4, 10):
        scaled = points * scale
        # Filter points in [-0.5, 0.5]
        mask = np.all(np.abs(scaled) <= 0.5, axis=1)
        inner = scaled[mask]
        
        if len(inner) < 100:
            continue
        
        # Compute score for all points, then select top 100 by local density
        # For each point, compute min distance to other points
        d = inner[:, None, :] - inner[None, :, :]
        dists = np.sqrt(np.sum(d**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        
        # Boundary margins
        margins = CUBE_HALF - np.abs(inner)
        min_margins = np.min(margins, axis=1)
        
        # Combined score: want large min_dist and large min_margin
        # Sort by min of the two
        combined = np.minimum(min_dists, min_margins)
        
        # Select top 100
        indices = np.argsort(-combined)[:100]
        selected = inner[indices]
        
        score = _grid_score(selected)
        if score > best_score:
            best_score = score
            best_pts = selected.copy()
    
    if best_pts is not None:
        return best_pts
    return None

def _initial_cubic_grid():
    """Simple cubic grid initialization."""
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid

def construct():
    start_time = time.time()
    
    # Generate initial configurations
    configs = []
    
    # 1. Simple cubic grid
    configs.append(_initial_cubic_grid())
    
    # 2. FCC lattice
    fcc_pts = _generate_fcc_points()
    if fcc_pts is not None:
        configs.append(fcc_pts)
    
    # 3. BCC-like: offset alternate layers
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    bcc_grid = []
    for z_idx, z in enumerate(zs):
        offset = 0.5/nx if z_idx % 2 == 0 else 0
        for y_idx, y in enumerate(ys):
            offset_y = 0.5/ny if (z_idx + y_idx) % 2 == 0 else 0
            for x_idx, x in enumerate(xs):
                bcc_grid.append([x + offset, y + offset_y, z])
    bcc_grid = np.array(bcc_grid)
    bcc_grid = np.clip(bcc_grid, -0.5, 0.5)
    configs.append(bcc_grid)
    
    # Select best initial configuration
    best_pts = None
    best_score = -1.0
    
    for pts in configs:
        if len(pts) != 100:
            continue
        pts = np.clip(pts, -0.5, 0.5)
        score = _grid_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _initial_cubic_grid()
        best_score = _grid_score(best_pts)
    
    # Refinement phase 1: Push-apart
    elapsed = time.time() - start_time
    remaining = DEADLINE_S - 10 - elapsed
    if remaining > 5:
        max_iter = min(20000, int(remaining * 100))
        refined_pts, refined_score = _push_apart_refine(best_pts, max_iter=max_iter, step=0.001)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts.copy()
    
    # Refinement phase 2: Soft-min gradient
    elapsed = time.time() - start_time
    remaining = DEADLINE_S - 5 - elapsed
    if remaining > 3:
        max_iter = min(500, int(remaining * 20))
        refined_pts, refined_score = _soft_min_gradient_refine(best_pts, max_iter=max_iter, lr=0.001, beta=100.0)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts.copy()
    
    # Final validation
    best_pts = np.clip(best_pts, -0.5, 0.5)
    best_score = _grid_score(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
