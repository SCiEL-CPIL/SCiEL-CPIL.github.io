"""Combined program: Topological contact graph + SLSQP refinement with relaxation."""
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
        
    # Check non-overlap
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < (rs[i] + rs[j]) - tol:
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

def _slsqp_optimize(circles, w, h, deadline):
    """
    Optimize radii and positions using SLSQP with hard constraints.
    """
    if time.time() > deadline - 5.0:
        return None

    n = len(circles)
    x0 = circles.flatten()

    # Objective: maximize sum of radii => minimize negative sum
    def objective(x):
        if time.time() > deadline:
            return 0.0
        r = x[2::3]
        return -np.sum(r)

    # Constraints
    constraints = []

    # Boundary constraints
    for i in range(n):
        # x >= r
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        # x + r <= w
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        # y >= r
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        # y + r <= h
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        # r > 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-6})

    # Overlap constraints: dist >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
            })

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w)) # x
        bounds.append((1e-6, h)) # y
        bounds.append((1e-6, min(w, h)/2)) # r

    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-9}
        )

        if res.success:
            new_circles = res.x.reshape(-1, 3)
            # Validate and fix any tiny numerical issues
            if _validate_solution(new_circles, w, h, N):
                return new_circles
            else:
                # Try to fix with relaxation
                fixed = _resolve_collisions(new_circles, w, h, max_iters=100)
                if _validate_solution(fixed, w, h, N):
                    return fixed
    except Exception:
        pass

    return None

def _solve_aspect(w, h, deadline):
    """
    Solve for a specific aspect ratio (box size w x h).
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

    # 3. SLSQP Optimize
    optimized = _slsqp_optimize(relaxed, w, h, deadline)
    
    if optimized is not None:
        return optimized
    else:
        return relaxed # Return the relaxed version if optimization fails

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin for 120s limit
    
    best_circles = None
    best_score = 0.0
    
    # Try different aspect ratios
    # Perimeter 2(W+H)=4 => W+H=2
    # W = 2 * ar / (1 + ar), H = 2 / (1 + ar)
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
            
        W = 2 * ar / (1 + ar)
        H = 2 / (1 + ar)
        
        try:
            result = _solve_aspect(W, H, deadline)
            
            if result is not None:
                # Translate to global coords (currently centered in [0,W]x[0,H])
                # The evaluator calculates bounding box, so position doesn't matter for perimeter,
                # but we must ensure circles are disjoint.
                # The local optimization ensures they are inside [0,W]x[0,H] and disjoint.
                # So we can just return them as is.
                
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
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
        best_circles = np.array(cells[:N])
        
    return best_circles

# EVOLVE-BLOCK-END
