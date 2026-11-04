import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _validate_solution(pos_r, w, h, n_circles, tol=1e-7):
    """
    Check if circles are valid: disjoint interiors, within bounds.
    pos_r: (n, 3) array of [x, y, r]
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

def _resolve_collisions(circles, w, h, max_iters=2000, tol=1e-8):
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

def _generate_grid_init(w, h, n_circles, cols, rows, jitter=0.0):
    """
    Generate initial circle positions in a grid pattern.
    """
    circles = np.zeros((n_circles, 3))
    count = 0
    for i in range(cols):
        for j in range(rows):
            if count >= n_circles:
                break
            x = (i + 0.5) * w / cols
            y = (j + 0.5) * h / rows
            
            if jitter > 0:
                x += np.random.uniform(-jitter, jitter)
                y += np.random.uniform(-jitter, jitter)
            
            # Initial radius: half the min cell size, slightly smaller
            cell_w = w / cols
            cell_h = h / rows
            r = 0.4 * min(cell_w, cell_h)
            
            circles[count, 0] = x
            circles[count, 1] = y
            circles[count, 2] = r
            count += 1
            
    # If we didn't fill all circles (should not happen if cols*rows >= N), 
    # but just in case, fill remaining with small circles in center
    while count < n_circles:
        circles[count, 0] = w / 2
        circles[count, 1] = h / 2
        circles[count, 2] = 0.01
        count += 1
        
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
        # x >= r  => x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        # x + r <= w => w - x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        # y >= r  => y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        # y + r <= h => h - y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        # r > 0 => r >= 1e-6
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
            options={'maxiter': 200, 'ftol': 1e-8}
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
    Solve for a specific aspect ratio.
    1. Generate grid init.
    2. Relax.
    3. SLSQP optimize.
    """
    best_circles = None
    best_score = 0.0

    # Try a few grid configurations to find a good starting point
    # Based on aspect ratio, choose likely grid dimensions
    ar = w / h
    if ar > 1.5:
        grid_configs = [(4, 5), (3, 7), (4, 6), (5, 5)]
    elif ar < 0.67:
        grid_configs = [(5, 4), (7, 3), (6, 4), (5, 5)]
    else:
        grid_configs = [(4, 5), (5, 4), (5, 5), (4, 6), (6, 4), (3, 7), (7, 3)]
        
    # Add some random perturbation to grid inits to escape local minima
    num_inits = 3
    
    for cols, rows in grid_configs:
        if cols * rows < N:
            continue
            
        for k in range(num_inits):
            if time.time() > deadline - 10:
                break
                
            jitter = 0.05 * min(w, h) if k > 0 else 0.0
            init_circles = _generate_grid_init(w, h, N, cols, rows, jitter=jitter)
            
            # Quick relax to ensure valid start
            relaxed = _resolve_collisions(init_circles, w, h, max_iters=100)
            if not _validate_solution(relaxed, w, h, N):
                continue
            
            score = np.sum(relaxed[:, 2])
            if score > best_score:
                best_score = score
                best_circles = relaxed.copy()

            # Run SLSQP on the relaxed init
            if time.time() < deadline - 5:
                opt_circles = _slsqp_optimize(relaxed, w, h, deadline)
                if opt_circles is not None:
                    opt_score = np.sum(opt_circles[:, 2])
                    if opt_score > best_score:
                        best_score = opt_score
                        best_circles = opt_circles.copy()

    return best_circles, best_score

def construct():
    """Construct 21 circles in a perimeter-4 rectangle, maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_sum = 0.0
    best_result = None
    
    # Try different aspect ratios
    # Perimeter = 2(W + H) = 4, so W + H = 2
    # W = ar * H => H(1+ar) = 2 => H = 2/(1+ar), W = 2*ar/(1+ar)
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.43, 0.6, 1.67, 0.5, 2.0, 0.4, 1.1, 0.9]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        H = 2.0 / (1.0 + ar)
        W = 2.0 * ar / (1.0 + ar)
        
        centers, s = _solve_aspect(W, H, deadline)
        
        if centers is not None and s > best_sum:
            best_sum = s
            best_result = (centers, W, H)
    
    if best_result is None:
        # Fallback: simple grid if everything failed
        # This should rarely happen
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.4 / side) 
                 for i in range(side) for j in range(side)]
        # Ensure we have exactly N
        cells = cells[:N]
        # Relax to ensure validity
        W, H = 1.0, 1.0
        init = np.array(cells)
        relaxed = _resolve_collisions(init, W, H, max_iters=1000)
        return relaxed
    
    centers, W, H = best_result
    # Final validation and cleanup
    if not _validate_solution(centers, W, H, N):
        relaxed = _resolve_collisions(centers, W, H, max_iters=500)
        if _validate_solution(relaxed, W, H, N):
            centers = relaxed
    
    return centers

