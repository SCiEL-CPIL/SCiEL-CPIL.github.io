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

def _jittered_grid_init(w, h, n_circles, seed=42):
    """
    Generate initial circle positions using a jittered grid.
    This is often more robust than hexagonal for varying aspect ratios.
    """
    np.random.seed(seed)
    circles = np.zeros((n_circles, 3))
    
    # Estimate grid dimensions
    # We want roughly n_circles cells.
    # Adjust aspect ratio to find best grid fit
    ar = w / h
    if ar >= 1:
        cols = int(np.sqrt(n_circles * ar))
        rows = int(np.ceil(n_circles / cols))
    else:
        rows = int(np.sqrt(n_circles / ar))
        cols = int(np.ceil(n_circles / rows))
        
    # Calculate cell size
    cell_w = w / cols
    cell_h = h / rows
    
    # Base radius is half the min cell dimension, slightly reduced for safety
    r_base = 0.45 * min(cell_w, cell_h)
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n_circles:
                break
            # Center of cell
            cx = (c + 0.5) * cell_w
            cy = (r + 0.5) * cell_h
            
            # Add jitter
            jitter_x = np.random.uniform(-0.2, 0.2) * cell_w
            jitter_y = np.random.uniform(-0.2, 0.2) * cell_h
            
            x = cx + jitter_x
            y = cy + jitter_y
            
            # Clamp position to keep circle inside bounds (roughly)
            x = np.clip(x, r_base, w - r_base)
            y = np.clip(y, r_base, h - r_base)
            
            circles[idx, 0] = x
            circles[idx, 1] = y
            circles[idx, 2] = r_base
            idx += 1
            
    # If we didn't fill all, duplicate last or just use what we have?
    # The problem requires exactly N circles.
    # If grid is smaller than N, we need to split cells or add more.
    # But usually cols*rows >= N.
    # If cols*rows < N, we increase resolution.
    if idx < n_circles:
        # Fallback: simple random scatter
        for i in range(idx, n_circles):
            circles[i, 0] = np.random.uniform(0, w)
            circles[i, 1] = np.random.uniform(0, h)
            circles[i, 2] = r_base * 0.5
            
    return circles

def _simulated_annealing(w, h, deadline, seed=42):
    """
    Simulated Annealing to maximize sum of radii.
    """
    np.random.seed(seed)
    
    # Initial state
    circles = _jittered_grid_init(w, h, N, seed=seed)
    circles = _resolve_collisions(circles, w, h, max_iters=100)
    
    if not _validate_solution(circles, w, h, N):
        return None
        
    current_circles = circles
    current_score = np.sum(circles[:, 2])
    
    best_circles = current_circles.copy()
    best_score = current_score
    
    # SA Parameters
    T_start = 0.1
    T_end = 1e-6
    alpha = 0.999 # Cooling rate
    max_iter = 5000
    
    n = N
    
    def get_energy(circs):
        """
        Energy = -sum(r) + penalty
        """
        xs = circs[:, 0]
        ys = circs[:, 1]
        rs = circs[:, 2]
        
        # Objective: maximize sum(r) -> minimize -sum(r)
        e_obj = -np.sum(rs)
        
        # Penalties
        e_pen = 0.0
        pen_weight = 10.0
        
        # Boundary
        # x - r >= 0 => violation = r - x
        # x + r <= w => violation = x + r - w
        v_x_min = np.maximum(0, rs - xs)
        v_x_max = np.maximum(0, xs + rs - w)
        v_y_min = np.maximum(0, rs - ys)
        v_y_max = np.maximum(0, ys + rs - h)
        
        e_pen += pen_weight * (np.sum(v_x_min**2) + np.sum(v_x_max**2) + np.sum(v_y_min**2) + np.sum(v_y_max**2))
        
        # Overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    e_pen += pen_weight * overlap**2
                    
        return e_obj + e_pen

    current_energy = get_energy(current_circles)
    
    T = T_start
    
    for i in range(max_iter):
        if time.time() > deadline - 5.0:
            break
            
        # Generate neighbor
        new_circles = current_circles.copy()
        
        # Perturb one or two circles
        # Randomly select 1-2 circles to move/resize
        num_perturb = np.random.choice([1, 2], p=[0.7, 0.3])
        indices = np.random.choice(n, size=num_perturb, replace=False)
        
        for idx in indices:
            # Perturb position
            step_size = T * 0.1
            new_circles[idx, 0] += np.random.uniform(-step_size, step_size)
            new_circles[idx, 1] += np.random.uniform(-step_size, step_size)
            
            # Perturb radius
            r_step = T * 0.05
            new_circles[idx, 2] += np.random.uniform(-r_step, r_step)
            
            # Clamp radius to positive
            new_circles[idx, 2] = np.maximum(new_circles[idx, 2], 1e-4)
            
            # Clamp position to bounds (hard constraint for stability)
            r = new_circles[idx, 2]
            new_circles[idx, 0] = np.clip(new_circles[idx, 0], r, w - r)
            new_circles[idx, 1] = np.clip(new_circles[idx, 1], r, h - r)
            
        new_energy = get_energy(new_circles)
        
        # Acceptance criterion
        delta_e = new_energy - current_energy
        
        if delta_e < 0:
            # Improvement
            current_circles = new_circles
            current_energy = new_energy
            
            # Check validity for best tracking
            if _validate_solution(current_circles, w, h, N):
                curr_score = np.sum(current_circles[:, 2])
                if curr_score > best_score:
                    best_score = curr_score
                    best_circles = current_circles.copy()
        else:
            # Accept with probability exp(-delta_e / T)
            if np.random.uniform() < np.exp(-delta_e / T):
                current_circles = new_circles
                current_energy = new_energy
                if _validate_solution(current_circles, w, h, N):
                    curr_score = np.sum(current_circles[:, 2])
                    if curr_score > best_score:
                        best_score = curr_score
                        best_circles = current_circles.copy()
                        
        T *= alpha
        
    # Final polish with local relaxation
    best_circles = _resolve_collisions(best_circles, w, h, max_iters=100)
    if _validate_solution(best_circles, w, h, N):
        return best_circles
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin under 150s
    
    best_result = None
    best_score = 0.0
    
    # Aspect ratios to try
    # Focus on ratios that allow better packing for 21 circles
    # Square is a good baseline. Rectangles might allow larger circles if aligned.
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 1.1, 0.9, 1.3, 0.75, 1.5]
    
    # We will allocate time per aspect ratio.
    # To be safe, we'll iterate through them, and if time is running out, we stop.
    
    for ar in aspect_ratios:
        if time.time() > deadline - 15.0:
            break
            
        w = 2.0 * ar / (1 + ar)
        h = 2.0 / (1 + ar)
        
        # Run SA for this aspect ratio
        # We don't know how long SA will take, so we pass the global deadline.
        # SA has an internal check.
        result = _simulated_annealing(w, h, deadline, seed=int(ar*100))
        
        if result is not None and _validate_solution(result, w, h, N):
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result

    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        best_result = _jittered_grid_init(w, h, N)
        best_result = _resolve_collisions(best_result, w, h, max_iters=100)
        if not _validate_solution(best_result, w, h, N):
            # Last resort: tiny circles
            best_result = np.zeros((N, 3))
            for i in range(N):
                x = (i % 5 + 0.5) * 0.2
                y = (i // 5 + 0.5) * 0.2
                best_result[i] = [x, y, 0.05]
            best_result = _resolve_collisions(best_result, w, h, max_iters=100)

    return best_result
