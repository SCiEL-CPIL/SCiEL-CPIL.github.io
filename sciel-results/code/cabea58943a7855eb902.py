import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _generate_hex_init(w, h, n_circles):
    """
    Generate initial circle positions in a hexagonal pattern.
    Hexagonal packing is denser than square grid.
    """
    circles = np.zeros((n_circles, 3))
    # Estimate radius from area: n*pi*r^2 <= w*h => r ~ sqrt(w*h/(n*pi))
    r_est = np.sqrt(w * h / (n_circles * np.pi)) * 0.95
    # Hex spacing
    dx = 2.0 * r_est
    dy = np.sqrt(3) * r_est
    
    count = 0
    row = 0
    while count < n_circles:
        offset = (dx / 2) if (row % 2 == 1) else 0
        col = 0
        while count < n_circles:
            x = r_est + offset + col * dx
            y = r_est + row * dy
            if x + r_est <= w and y + r_est <= h:
                circles[count, 0] = x
                circles[count, 1] = y
                circles[count, 2] = r_est
                count += 1
            else:
                break
            col += 1
        row += 1
        # Stop if we're clearly out of bounds
        if r_est + row * dy > h + r_est:
            break
    
    # If we didn't fill all circles (shouldn't happen with reasonable r_est), 
    # place remaining in available spots
    if count < n_circles:
        # Fallback: fill remaining with slightly smaller circles
        for i in range(count, n_circles):
            # Place in a spiral or just shrink
            circles[i, 0] = w / 2
            circles[i, 1] = h / 2
            circles[i, 2] = r_est * 0.8
    
    return circles

def _tighten_positions(circles, w, h, deadline):
    """
    Keep radii fixed, optimize positions only to maximize minimum clearance.
    This finds the densest arrangement for given radii.
    """
    if time.time() > deadline - 3.0:
        return circles
    
    n = len(circles)
    r = circles[:, 2].copy()
    x0 = circles[:, :2].flatten()
    
    def objective(x):
        if time.time() > deadline:
            return 0.0
        pos = x.reshape(-1, 2)
        # Minimize negative of minimum clearance
        # Clearance for each circle = min over other circles of (dist - r_i - r_j)
        # and min over boundaries of (distance to boundary)
        clearances = []
        for i in range(n):
            # Boundary clearances
            clearances.append(pos[i, 0] - r[i])
            clearances.append(w - pos[i, 0] - r[i])
            clearances.append(pos[i, 1] - r[i])
            clearances.append(h - pos[i, 1] - r[i])
            # Pairwise clearances
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                clearances.append(dist - r[i] - r[j])
        
        clearances = np.array(clearances)
        # Use soft-min to make it differentiable
        # soft_min = -log(sum(exp(-clearances/beta))) * beta
        beta = 0.05
        # Clip to avoid overflow
        clearances_clipped = np.clip(clearances, -100, 100)
        soft_min = -beta * np.log(np.sum(np.exp(-clearances_clipped / beta)))
        return -soft_min  # We want to maximize minimum clearance, so minimize negative
    
    # Constraints: circles must be inside and non-overlapping
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*2] - r[i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*2] - r[i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*2+1] - r[i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*2+1] - r[i]})
    
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[i*2]-x[j*2])**2 + (x[i*2+1]-x[j*2+1])**2) - r[i] - r[j]
            })
    
    bounds = []
    for i in range(n):
        bounds.append((r[i], w - r[i]))
        bounds.append((r[i], h - r[i]))
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-10}
        )
        
        if res.success:
            new_pos = res.x.reshape(-1, 2)
            new_circles = np.column_stack((new_pos, r))
            if _validate_solution(new_circles, w, h, N):
                return new_circles
            else:
                fixed = _resolve_collisions(new_circles, w, h, max_iters=100)
                if _validate_solution(fixed, w, h, N):
                    return fixed
    except Exception:
        pass
    
    return circles

def _polish(circles, w, h, deadline):
    """
    High-precision final polish: tighter tolerance, more iterations.
    """
    if time.time() > deadline - 2.0:
        return circles
    
    n = len(circles)
    x0 = circles.flatten()
    
    def objective(x):
        if time.time() > deadline:
            return 0.0
        r = x[2::3]
        return -np.sum(r)
    
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-8})
    
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
            })
    
    bounds = []
    for i in range(n):
        bounds.append((1e-8, w))
        bounds.append((1e-8, h))
        bounds.append((1e-8, min(w, h)/2))
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        
        if res.success:
            new_circles = res.x.reshape(-1, 3)
            if _validate_solution(new_circles, w, h, N):
                return new_circles
            else:
                fixed = _resolve_collisions(new_circles, w, h, max_iters=500)
                if _validate_solution(fixed, w, h, N):
                    return fixed
    except Exception:
        pass
    
    return circles

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
            options={'maxiter': 500, 'ftol': 1e-9}
        )

        if res.success:
            new_circles = res.x.reshape(-1, 3)
            # Validate and fix any tiny numerical issues
            if _validate_solution(new_circles, w, h, N):
                return new_circles
            else:
                # Try to fix with relaxation
                fixed = _resolve_collisions(new_circles, w, h, max_iters=200)
                if _validate_solution(fixed, w, h, N):
                    return fixed
    except Exception:
        pass

    return None

def _generate_grid_init(w, h, n_circles, cols, rows):
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
            # Initial radius: half the min cell size, slightly smaller
            cell_w = w / cols
            cell_h = h / rows
            r = 0.45 * min(cell_w, cell_h)
            
            circles[count, 0] = x
            circles[count, 1] = y
            circles[count, 2] = r
            count += 1
            
    return circles

def _solve_aspect(w, h, deadline):
    """
    Solve for a specific aspect ratio.
    1. Generate grid init.
    2. Relax.
    3. SLSQP optimize.
    """
    # Try a few grid configurations to find a good starting point
    best_init = None
    best_init_score = 0.0
    
    # Simple heuristic for grid size based on aspect ratio
    ar = w / h
    if ar > 1.5:
        grid_configs = [(4, 5), (3, 7), (4, 6), (5, 4)]
    elif ar < 0.67:
        grid_configs = [(5, 4), (7, 3), (6, 4), (4, 5)]
    else:
        grid_configs = [(4, 5), (5, 4), (5, 5), (4, 6), (6, 4), (3, 7)]
        
    # Try grid inits
    for cols, rows in grid_configs:
        if cols * rows < N:
            continue
        init_circles = _generate_grid_init(w, h, N, cols, rows)
        # Quick relax to see if it's a valid start
        relaxed = _resolve_collisions(init_circles, w, h, max_iters=100)
        if _validate_solution(relaxed, w, h, N):
            score = np.sum(relaxed[:, 2])
            if score > best_init_score:
                best_init_score = score
                best_init = relaxed
    
    # Also try hexagonal init
    if time.time() < deadline - 10:
        hex_circles = _generate_hex_init(w, h, N)
        hex_relaxed = _resolve_collisions(hex_circles, w, h, max_iters=200)
        if _validate_solution(hex_relaxed, w, h, N):
            score = np.sum(hex_relaxed[:, 2])
            if score > best_init_score:
                best_init_score = score
                best_init = hex_relaxed

    if best_init is None:
        # Fallback to a simple square grid
        best_init = _generate_grid_init(w, h, N, 5, 5)
        best_init = _resolve_collisions(best_init, w, h, max_iters=100)
        if not _validate_solution(best_init, w, h, N):
            return None

    # Refine with SLSQP
    optimized = _slsqp_optimize(best_init, w, h, deadline)
    
    if optimized is not None:
        # Tighten positions with fixed radii
        if time.time() < deadline - 5:
            tightened = _tighten_positions(optimized, w, h, deadline)
            if np.sum(tightened[:, 2]) >= np.sum(optimized[:, 2]) - 1e-10:
                optimized = tightened
        
        # Final high-precision polish
        if time.time() < deadline - 2:
            polished = _polish(optimized, w, h, deadline)
            if _validate_solution(polished, w, h, N):
                if np.sum(polished[:, 2]) >= np.sum(optimized[:, 2]) - 1e-10:
                    optimized = polished
        
        return optimized
    
    # If SLSQP failed, try tightening the relaxed init
    if _validate_solution(best_init, w, h, N):
        if time.time() < deadline - 5:
            tightened = _tighten_positions(best_init, w, h, deadline)
            if _validate_solution(tightened, w, h, N):
                return tightened
        return best_init
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140 # Safety margin for 150s limit
    
    best_result = None
    best_score = 0.0
    
    # Aspect ratios to try
    # Perimeter 4 => w + h = 2 => h = 2 - w
    # We sweep w. Note that w and 2-w are symmetric in terms of packing difficulty 
    # if we rotate, but the coordinate system is fixed. 
    # However, the "width" in the code is the x-dimension. 
    # We should test a range of widths.
    
    # Common optimal aspects for 21 circles might be near 1.0 or specific ratios.
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.85, 1.15, 0.75, 1.25]
    
    for w in aspect_ratios:
        if time.time() > deadline:
            break
            
        h = 2.0 - w
        
        # Time remaining
        time_remaining = deadline - time.time()
        if time_remaining < 10:
            continue
            
        try:
            result = _solve_aspect(w, h, deadline)
            if result is not None:
                score = np.sum(result[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = result
        except Exception:
            pass

    if best_result is None:
        # Fallback: Simple grid
        w, h = 1.0, 1.0
        positions = []
        radii = []
        cols = 5
        rows = 5
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                pos = [(i + 0.5) * w / cols, (j + 0.5) * h / rows]
                r = 0.5 * min(w/cols, h/rows) * 0.9
                positions.append(pos)
                radii.append(r)
                count += 1
        positions = np.array(positions)
        radii = np.array(radii)
        best_result = np.column_stack([positions, radii])
        
    return best_result
