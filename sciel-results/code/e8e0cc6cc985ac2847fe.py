"""Combined program: Contact graph + smooth penalty optimization with high-precision polish."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

def _validate_solution(pos_r, w, h, n_circles, tol=1e-7):
    """
    Check if circles are valid: disjoint interiors, within bounds [0, w] x [0, h].
    """
    if pos_r.shape[0] != n_circles:
        return False
    
    xs = pos_r[:, 0]
    ys = pos_r[:, 1]
    rs = pos_r[:, 2]
    
    # Check radii are positive
    if np.any(rs < 1e-9):
        return False
        
    # Check bounds: circles must be inside [0, w] x [0, h]
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
        
    # Check non-overlap (vectorized)
    i_idx, j_idx = np.triu_indices(n_circles, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dists = np.sqrt(dx*dx + dy*dy)
    min_dists = rs[i_idx] + rs[j_idx]
    if np.any(dists < min_dists - tol):
        return False
    return True

def _validate_global(circles):
    """
    Validate circles in global coordinates. 
    The evaluator checks: 
    1. n == 21
    2. rs > 0
    3. Perimeter of bounding box <= 4
    4. No overlaps
    """
    n = len(circles)
    if n != N:
        return False
    
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    if np.any(rs <= 0):
        return False
        
    # Perimeter check
    left = np.min(xs - rs)
    right = np.max(xs + rs)
    bottom = np.min(ys - rs)
    top = np.max(ys + rs)
    width = right - left
    height = top - bottom
    perimeter = 2 * (width + height)
    
    if perimeter > 4 + 1e-9:
        return False
        
    # Disjoint check
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-9:
                return False
    return True

def _score(circles):
    if not _validate_global(circles):
        return 0.0
    return np.sum(circles[:, 2])

def _resolve_collisions(circles, w, h, max_iters=500, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    circles: (n, 3) array [x, y, r]
    Returns modified circles array.
    """
    n = circles.shape[0]
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    for _ in range(max_iters):
        moved = False
        
        # 1. Boundary constraints: push circles inside [0, w] x [0, h]
        for i in range(n):
            x, y, r = pos[i, 0], pos[i, 1], rad[i]
            if x - r < 0:
                pos[i, 0] = r
                moved = True
            if x + r > w:
                pos[i, 0] = w - r
                moved = True
            if y - r < 0:
                pos[i, 1] = r
                moved = True
            if y + r > h:
                pos[i, 1] = h - r
                moved = True
        
        # 2. Pairwise collisions: push apart
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[j]
                
                if dist < min_dist:
                    if dist < 1e-9:
                        # Random perturbation if exactly overlapping
                        angle = np.random.uniform(0, 2*np.pi)
                        push = (min_dist - dist) * 0.5
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[j, 0] += np.cos(angle) * push
                        pos[j, 1] += np.sin(angle) * push
                        moved = True
                    else:
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        # Push apart by half the overlap each
                        pos[i, 0] -= nx * overlap * 0.5
                        pos[i, 1] -= ny * overlap * 0.5
                        pos[j, 0] += nx * overlap * 0.5
                        pos[j, 1] += ny * overlap * 0.5
                        moved = True
        
        if not moved:
            break
            
    # Final clamp to ensure boundary validity
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])
        
    return np.column_stack((pos, rad))

def _generate_hex_init(w, h, n_circles):
    """
    Generate a hexagonal cluster initialization scaled to fit w x h.
    Structure: 5 rows with counts [4, 5, 4, 5, 3] (Total 21)
    """
    rows = [4, 5, 4, 5, 3]
    centers = []
    h_unit = np.sqrt(3) / 2
    y_coords = [0, h_unit, 2*h_unit, 3*h_unit, 4*h_unit]
    
    for i, count in enumerate(rows):
        y = y_coords[i]
        center_x = 1.5  # Center of the widest row (width 4)
        start_x = center_x - (count - 1) / 2.0
        for k in range(count):
            x = start_x + k
            centers.append([x, y])
            
    centers = np.array(centers)
    
    # Normalize to [0, 1] x [0, 1]
    min_x, max_x = np.min(centers[:, 0]), np.max(centers[:, 0])
    min_y, max_y = np.min(centers[:, 1]), np.max(centers[:, 1])
    w_c = max_x - min_x
    h_c = max_y - min_y
    
    # Scale to fit inside [r, w-r] x [r, h-r] approximately
    # We don't know r yet, so assume a small r initially or just fit in [0, w]x[0,h] with padding
    pad = 0.1 # 10% padding
    scale_x = (1 - 2*pad) / w_c
    scale_y = (1 - 2*pad) / h_c
    
    # Maintain aspect ratio of the cluster if possible, or stretch
    # Stretching might distort contacts, but relaxation will fix positions.
    # Let's scale uniformly to fit, then center.
    scale = min(scale_x, scale_y)
    
    scaled_centers = (centers - [min_x, min_y]) * scale
    # Center in box
    offset_x = (w - scale * w_c) / 2
    offset_y = (h - scale * h_c) / 2
    scaled_centers += [offset_x, offset_y]
    
    # Estimate initial radius
    # Min distance in scaled space
    min_dist = np.inf
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            d = np.linalg.norm(scaled_centers[i] - scaled_centers[j])
            if d < min_dist:
                min_dist = d
                
    r_init = min_dist * 0.49 # Slightly less than touching
    
    circles = np.zeros((n_circles, 3))
    circles[:, 0] = scaled_centers[:, 0]
    circles[:, 1] = scaled_centers[:, 1]
    circles[:, 2] = r_init
    
    return circles

def _smooth_max(values, eps):
    """Smooth approximation of max using log-sum-exp."""
    if len(values) == 0:
        return 0.0
    m = np.max(values)
    if m < -50 / eps:
        return 0.0
    return (1.0 / eps) * np.log(np.sum(np.exp(eps * (values - m)))) + m

def _lbfgsb_optimize(circles, w, h, deadline, smooth_eps=1.0):
    """
    Optimize radii and positions using L-BFGS-B with smooth penalty method.
    Uses log-sum-exp smooth max for nonsmooth penalties.
    """
    if time.time() > deadline - 2.0:
        return None

    n = len(circles)
    x0 = circles.flatten()
    
    # Pre-compute indices for vectorization
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        pos = x.reshape(-1, 3)
        xs = pos[:, 0]
        ys = pos[:, 1]
        rs = pos[:, 2]
        
        # 1. Boundary Penalty (smooth max of violations)
        bound_viol = np.concatenate([
            rs - xs,          # left boundary
            xs + rs - w,      # right boundary
            rs - ys,          # bottom boundary
            ys + rs - h       # top boundary
        ])
        smooth_bound = _smooth_max(bound_viol, smooth_eps)
        
        # 2. Overlap Penalty (smooth max of violations)
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = rs[i_idx] + rs[j_idx]
        overlap_viol = min_dists - dists
        smooth_overlap = _smooth_max(overlap_viol, smooth_eps)
        
        # Penalty weights
        w_bound = 50.0
        w_overlap = 50.0
        
        penalty = w_bound * smooth_bound**2 + w_overlap * smooth_overlap**2
        
        # Objective: Minimize -sum(r) + penalty
        return -np.sum(rs) + penalty

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-4, w - 1e-4)) # x
        bounds.append((1e-4, h - 1e-4)) # y
        bounds.append((1e-4, min(w, h)/2)) # r

    try:
        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'maxfun': 30000}
        )

        new_circles = res.x.reshape(-1, 3)
        
        # Post-processing: Ensure validity
        new_circles[:, 0] = np.clip(new_circles[:, 0], new_circles[:, 2], w - new_circles[:, 2])
        new_circles[:, 1] = np.clip(new_circles[:, 1], new_circles[:, 2], h - new_circles[:, 2])
        
        # Check validity
        if _validate_solution(new_circles, w, h, N, tol=1e-7):
            return new_circles
        
        # Try relaxation
        fixed = _resolve_collisions(new_circles, w, h, max_iters=200)
        if _validate_solution(fixed, w, h, N, tol=1e-7):
            return fixed
            
        # If still invalid, shrink radii
        scale = 0.95
        shrunk = new_circles.copy()
        shrunk[:, 2] *= scale
        if _validate_solution(shrunk, w, h, N, tol=1e-7):
            return shrunk
            
        return None
        
    except Exception:
        return None

def _high_precision_polish(circles, w, h, deadline):
    """
    High-precision polish pass with very tight tolerances and small smooth_eps.
    This refines a near-optimal solution to machine precision.
    """
    if time.time() > deadline - 3.0:
        return None
    
    n = len(circles)
    x0 = circles.flatten()
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        pos = x.reshape(-1, 3)
        xs = pos[:, 0]
        ys = pos[:, 1]
        rs = pos[:, 2]
        
        # Very tight smooth max
        bound_viol = np.concatenate([
            rs - xs,
            xs + rs - w,
            rs - ys,
            ys + rs - h
        ])
        smooth_bound = _smooth_max(bound_viol, 10.0)
        
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = rs[i_idx] + rs[j_idx]
        overlap_viol = min_dists - dists
        smooth_overlap = _smooth_max(overlap_viol, 10.0)
        
        penalty = 100.0 * smooth_bound**2 + 100.0 * smooth_overlap**2
        return -np.sum(rs) + penalty

    bounds = []
    for i in range(n):
        bounds.append((1e-6, w - 1e-6))
        bounds.append((1e-6, h - 1e-6))
        bounds.append((1e-6, min(w, h)/2))

    try:
        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-16, 'maxfun': 50000}
        )

        new_circles = res.x.reshape(-1, 3)
        new_circles[:, 0] = np.clip(new_circles[:, 0], new_circles[:, 2], w - new_circles[:, 2])
        new_circles[:, 1] = np.clip(new_circles[:, 1], new_circles[:, 2], h - new_circles[:, 2])
        
        if _validate_solution(new_circles, w, h, N, tol=1e-9):
            return new_circles
            
        fixed = _resolve_collisions(new_circles, w, h, max_iters=500)
        if _validate_solution(fixed, w, h, N, tol=1e-9):
            return fixed
            
        return None
        
    except Exception:
        return None

def _solve_aspect(w, h, deadline):
    """
    Solve for a specific aspect ratio (box size w x h).
    Progressive smooth-max epsilon: start coarse, refine fine.
    """
    # 1. Generate hex init
    init_circles = _generate_hex_init(w, h, N)
    
    # 2. Relax to ensure validity
    relaxed = _resolve_collisions(init_circles, w, h, max_iters=500)
    
    if not _validate_solution(relaxed, w, h, N):
        # If relaxation failed, try a simpler grid init
        cols, rows = 5, 5
        grid = np.zeros((N, 3))
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N: break
                grid[count, 0] = (i + 0.5) * w / cols
                grid[count, 1] = (j + 0.5) * h / rows
                grid[count, 2] = 0.45 * min(w/cols, h/rows)
                count += 1
        relaxed = _resolve_collisions(grid, w, h, max_iters=500)
        if not _validate_solution(relaxed, w, h, N):
            return None

    # 3. Progressive optimization: coarse to fine smooth_eps
    current = relaxed
    
    # Stage 1: Coarse optimization (eps=1.0)
    if time.time() < deadline - 15:
        opt1 = _lbfgsb_optimize(current, w, h, deadline, smooth_eps=1.0)
        if opt1 is not None:
            current = opt1
    
    # Stage 2: Medium optimization (eps=5.0)
    if time.time() < deadline - 10:
        opt2 = _lbfgsb_optimize(current, w, h, deadline, smooth_eps=5.0)
        if opt2 is not None:
            current = opt2
    
    # Stage 3: High-precision polish (eps=10.0, tighter tol)
    if time.time() < deadline - 5:
        polished = _high_precision_polish(current, w, h, deadline)
        if polished is not None:
            current = polished
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_circles = None
    best_score = 0.0
    
    # Try different aspect ratios
    # Perimeter 2(W+H)=4 => W+H=2
    # W = 2 * ar / (1 + ar), H = 2 / (1 + ar)
    # Focus on aspect ratios near 1.0 which are typically optimal for square-ish packings
    aspect_ratios = [0.9, 1.0, 1.1, 0.85, 1.15]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 20:
            break
            
        W = 2 * ar / (1 + ar)
        H = 2 / (1 + ar)
        
        try:
            result = _solve_aspect(W, H, deadline)
            
            if result is not None:
                # Ensure the result is strictly valid before scoring
                if _validate_global(result):
                    score = _score(result)
                    if score > best_score:
                        best_score = score
                        best_circles = result.copy()
                    
        except Exception:
            continue
            
    # If no valid solution found, fallback to a simple grid in a square
    if best_circles is None:
        # Fallback: 5x5 grid in 1x1 box (Perimeter 4)
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-5
                cells.append([x, y, r])
        best_circles = np.array(cells[:N])
        
    # Final safety check
    if not _validate_global(best_circles):
        # Minimal valid: 21 tiny circles in a row
        r = 0.01
        best_circles = np.zeros((N, 3))
        for i in range(N):
            best_circles[i, 0] = r + i * (2 * r + 1e-6)
            best_circles[i, 1] = r
            best_circles[i, 2] = r
        
    return best_circles

# EVOLVE-BLOCK-END
