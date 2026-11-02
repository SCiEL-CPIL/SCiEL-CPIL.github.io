"""Improved circle packing for perimeter-4 rectangle, 21 circles."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _evaluate(x):
    """
    x layout: [w, h, r0..r20, x0..x20, y0..y20]
    Minimize -sum(r) subject to:
      2*(w+h) <= 4  (perimeter)
      ri >= 0
      xi - ri >= 0, xi + ri <= w
      yi - ri >= 0, yi + ri <= h
      (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2  for all i<j
    """
    w = x[0]
    h = x[1]
    r = x[2:2+N]
    cx = x[2+N:2+2*N]
    cy = x[2+2*N:2+3*N]
    
    # Perimeter constraint
    if 2*(w+h) > 4.0 + 1e-9:
        return None
    
    # Boundary constraints
    for i in range(N):
        if r[i] < -1e-9:
            return None
        if cx[i] - r[i] < -1e-9 or cx[i] + r[i] > w + 1e-9:
            return None
        if cy[i] - r[i] < -1e-9 or cy[i] + r[i] > h + 1e-9:
            return None
    
    # Pairwise constraints
    for i in range(N):
        for j in range(i+1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist_sq = dx*dx + dy*dy
            req = (r[i] + r[j])**2
            if dist_sq < req - 1e-9:
                return None
    
    return float(np.sum(r))

def _make_initial_grid(n=21):
    """Create an initial grid-like arrangement."""
    # Use a 5x5 grid but only take 21 cells
    # Arrange in roughly square shape
    # Let's try rows of varying sizes
    # 21 = 5+4+4+4+4 or 6+5+5+5 or 7+7+7
    
    # Try a compact arrangement
    # 5 columns, 5 rows would be 25, we take 21
    # Or 4x5 + 1 extra row
    
    # Let's use a 5-wide, with rows: 5,4,4,4,4 = 21
    rows = [5, 4, 4, 4, 4]
    ncols = max(rows)
    
    # Bounding box for the grid
    # Each cell is roughly 1 unit
    w = ncols
    h = len(rows)
    
    # Scale to fit perimeter 4: 2*(w+h) = 4 => w+h = 2
    # So scale factor
    scale = 2.0 / (w + h)
    w *= scale
    h *= scale
    
    circles = []
    y_idx = 0
    for row_count in rows:
        for i in range(row_count):
            # Center position within the row
            # Distribute evenly
            x_pos = (i + 0.5) / row_count * w
            y_pos = (y_idx + 0.5) / len(rows) * h
            r = 0.45 * scale  # Slightly less than half cell to allow gaps
            circles.append([x_pos, y_pos, r])
        y_idx += 1
    
    return w, h, circles

def _pack_with_optimization(seed_rng, time_limit):
    """Run optimization from a given seed."""
    w, h, circles = _make_initial_grid()
    
    # Build initial x vector
    x0 = np.array([w, h])
    for c in circles:
        x0 = np.append(x0, c[2])
    for c in circles:
        x0 = np.append(x0, c[0])
    for c in circles:
        x0 = np.append(x0, c[1])
    
    # Bounds
    bounds = [(1e-6, 2.0)]  # w
    bounds.append((1e-6, 2.0))  # h
    for i in range(N):
        bounds.append((1e-6, 1.0))  # radii
    for i in range(N):
        bounds.append((1e-6, 2.0))  # x centers
    for i in range(N):
        bounds.append((1e-6, 2.0))  # y centers
    
    # Objective: minimize -sum(r)
    def objective(x):
        val = _evaluate(x)
        if val is None:
            return 1e6  # Large penalty for infeasible
        return -val
    
    # Constraints (inequality: g(x) >= 0)
    def constraint_perimeter(x):
        return 4.0 - 2*(x[0] + x[1])
    
    # We'll use SLSQP with just the perimeter constraint and rely on bounds
    # plus the objective penalty for other constraints
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_perimeter}
    ]
    
    best_val = 0.0
    best_x = None
    
    start_time = time.time()
    
    # Try multiple optimization attempts with different perturbations
    n_attempts = 0
    while time.time() - start_time < time_limit - 5:
        n_attempts += 1
        
        # Perturb the initial solution
        if n_attempts == 1:
            x_start = x0.copy()
        else:
            x_start = x0.copy()
            # Add random perturbation
            noise_scale = 0.1 * (1.0 - 0.5 * n_attempts / 20.0)
            noise = seed_rng.uniform(-noise_scale, noise_scale, size=len(x_start))
            # Don't perturb w and h too much
            noise[0] *= 0.5
            noise[1] *= 0.5
            x_start = x_start + noise
            # Clamp to bounds
            for i in range(len(x_start)):
                x_start[i] = max(bounds[i][0], min(bounds[i][1], x_start[i]))
        
        try:
            result = minimize(
                objective, x_start, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
        except Exception:
            continue
        
        val = _evaluate(result.x)
        if val is not None and val > best_val:
            best_val = val
            best_x = result.x.copy()
        
        if time.time() - start_time > time_limit - 10:
            break
    
    if best_x is None:
        return None
    
    # Extract circles
    w_best = best_x[0]
    h_best = best_x[1]
    r_best = best_x[2:2+N]
    cx_best = best_x[2+N:2+2*N]
    cy_best = best_x[2+2*N:2+3*N]
    
    # Final validation
    final_val = _evaluate(best_x)
    if final_val is None:
        return None
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [cx_best[i], cy_best[i], r_best[i]]
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_result = None
    best_val = 0.0
    
    # Try multiple random seeds for different perturbations
    for seed in range(20):
        if time.time() > deadline:
            break
        
        seed_rng = np.random.RandomState(seed * 12345 + 67890)
        
        # Time per attempt
        time_per = min(5.0, (deadline - time.time()) / max(1, 20 - seed))
        
        result = _pack_with_optimization(seed_rng, time_per)
        
        if result is not None:
            # Validate
            w = np.max(result[:, 0]) + np.max(result[:, 2])
            h = np.max(result[:, 1]) + np.max(result[:, 2])
            # Actually need proper bounding box
            min_x = np.min(result[:, 0] - result[:, 2])
            max_x = np.max(result[:, 0] + result[:, 2])
            min_y = np.min(result[:, 1] - result[:, 2])
            max_y = np.max(result[:, 1] + result[:, 2])
            
            w_box = max_x - min_x
            h_box = max_y - min_y
            perimeter = 2 * (w_box + h_box)
            
            if perimeter <= 4.0 + 1e-6:
                val = np.sum(result[:, 2])
                if val > best_val:
                    best_val = val
                    best_result = result.copy()
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
