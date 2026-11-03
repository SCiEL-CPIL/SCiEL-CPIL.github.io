"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time

N = 21
DEADLINE = 115.0  # seconds, safety margin before 120s limit

def construct():
    start_time = time.time()
    
    # Try a 4-column by 5-row arrangement (20 circles) + 1 extra
    # Actually, let's use a smarter approach:
    # For perimeter 4, if width=w, height=h, then 2(w+h)=4, so w+h=2
    # For a square: w=h=1
    # 
    # Strategy: Use a grid-based initial solution, then optimize with scipy
    # 
    # Let's try multiple aspect ratios and grid layouts
    
    best_result = None
    best_sum = 0.0
    
    # Layout options: (cols, rows) that give >= 21 cells
    layouts = [(5, 5), (4, 6), (6, 4), (3, 7), (7, 3), (4, 5)]
    
    for cols, rows in layouts:
        if cols * rows < N:
            continue
        
        # For perimeter 4: w + h = 2
        # Try different aspect ratios
        for aspect in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0]:
            if time.time() - start_time > DEADLINE:
                break
            
            # w + h = 2, w/h = aspect => w = aspect*h => h(aspect+1)=2 => h=2/(aspect+1)
            h = 2.0 / (aspect + 1.0)
            w = aspect * h
            
            # Create grid positions
            # Use cols x rows grid, take first N
            xs = np.linspace(w / (2*cols), w - w / (2*cols), cols) if cols > 1 else [w/2]
            ys = np.linspace(h / (2*rows), h - h / (2*rows), rows) if rows > 1 else [h/2]
            
            # Generate all grid positions
            positions = []
            for j in range(rows):
                for i in range(cols):
                    positions.append([xs[i], ys[j]])
            
            # Take first N
            positions = positions[:N]
            positions = np.array(positions)
            
            # Initial radii: based on local spacing
            # Simple heuristic: min distance to neighbors / 2
            radii = np.ones(N) * 0.05
            
            # Better initial radii based on grid spacing
            dx = w / cols if cols > 1 else w
            dy = h / rows if rows > 1 else h
            base_r = min(dx, dy) * 0.45
            
            radii = np.ones(N) * base_r
            
            # Optimize
            result = _optimize_packing(positions.copy(), radii.copy(), w, h, start_time, DEADLINE)
            if result is not None:
                s = np.sum(result[:, 2])
                if s > best_sum:
                    best_sum = s
                    best_result = result.copy()
    
    if best_result is None:
        # Fallback: simple grid
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3])
        return np.array(cells)
    
    return best_result


def _optimize_packing(x0, r0, w, h, start_time, deadline):
    """Optimize circle positions and radii to maximize sum of radii."""
    from scipy.optimize import minimize
    
    N = len(x0)
    
    def flatten(x, r):
        return np.concatenate([x.flatten(), r.flatten()])
    
    def unflatten(v):
        x = v[:2*N].reshape(N, 2)
        r = v[2*N:]
        return x, r
    
    def objective(v):
        x, r = unflatten(v)
        # Penalty for going out of bounds
        penalty = 0.0
        for i in range(N):
            if x[i, 0] < r[i] or x[i, 0] > w - r[i] or x[i, 1] < r[i] or x[i, 1] > h - r[i]:
                penalty += 100.0 * (abs(max(0, r[i] - x[i, 0])) + 
                                   abs(max(0, x[i, 0] - (w - r[i]))) +
                                   abs(max(0, r[i] - x[i, 1])) +
                                   abs(max(0, x[i, 1] - (h - r[i]))))
        # Penalty for overlapping
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i,0]-x[j,0])**2 + (x[i,1]-x[j,1])**2)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += 100.0 * (min_dist - dist)**2
        # Negative sum of radii (we minimize)
        return -np.sum(r) + penalty
    
    def jac(v):
        # Numerical gradient
        x, r = unflatten(v)
        grad = np.zeros_like(v)
        eps = 1e-7
        for k in range(len(v)):
            v2 = v.copy()
            v2[k] += eps
            f2 = objective(v2)
            v2[k] -= 2*eps
            f0 = objective(v2)
            v2[k] += eps
            grad[k] = (f2 - f0) / (2*eps)
        return grad
    
    v0 = flatten(x0, r0)
    
    try:
        result = minimize(objective, v0, method='L-BFGS-B', 
                         options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8,
                                  'maxfun': 5000, 'disp': False})
        
        if time.time() - start_time > deadline - 5:
            return None
        
        x_opt, r_opt = unflatten(result.x)
        
        # Verify validity
        valid = True
        for i in range(N):
            if x_opt[i,0] < r_opt[i] - 1e-6 or x_opt[i,0] > w - r_opt[i] + 1e-6:
                valid = False
            if x_opt[i,1] < r_opt[i] - 1e-6 or x_opt[i,1] > h - r_opt[i] + 1e-6:
                valid = False
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x_opt[i,0]-x_opt[j,0])**2 + (x_opt[i,1]-x_opt[j,1])**2)
                if dist < r_opt[i] + r_opt[j] - 1e-6:
                    valid = False
        
        if not valid:
            return None
        
        # Clamp radii to be positive
        r_opt = np.maximum(r_opt, 1e-6)
        
        result_arr = np.zeros((N, 3))
        result_arr[:, 0] = x_opt[:, 0]
        result_arr[:, 1] = x_opt[:, 1]
        result_arr[:, 2] = r_opt
        
        return result_arr
    
    except Exception:
        return None


# Simpler, more robust approach using simulated annealing / local search
def _local_search(x, r, w, h, start_time, deadline):
    """Simple local search to improve packing."""
    N = len(x)
    current_sum = np.sum(r)
    
    def is_valid(x, r):
        for i in range(N):
            if x[i,0] < r[i] - 1e-9 or x[i,0] > w - r[i] + 1e-9:
                return False
            if x[i,1] < r[i] - 1e-9 or x[i,1] > h - r[i] + 1e-9:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i,0]-x[j,0])**2 + (x[i,1]-x[j,1])**2)
                if dist < r[i] + r[j] - 1e-9:
                    return False
        return True
    
    # Try to increase radii slightly
    for _ in range(100):
        if time.time() - start_time > deadline - 10:
            break
        # Try increasing a random radius
        i = np.random.randint(N)
        old_r = r[i]
        r[i] = old_r * 1.01
        if is_valid(x, r):
            current_sum = np.sum(r)
        else:
            r[i] = old_r
    
    return x, r


# Override with a cleaner implementation
def construct():
    start_time = time.time()
    
    N = 21
    
    # For perimeter 4: w + h = 2
    # Best is likely square w=h=1
    # Use a 4x5 grid (20 circles) plus one, or 5x5 grid
    
    # Let's use a well-known good configuration:
    # 5x5 grid in unit square, take 21 circles
    # Then optimize positions and radii
    
    w, h = 1.0, 1.0
    
    # Initial: 5x5 grid
    cols, rows = 5, 5
    xs = np.linspace(0.1, 0.9, cols)
    ys = np.linspace(0.1, 0.9, rows)
    
    positions = []
    for j in range(rows):
        for i in range(cols):
            if len(positions) < N:
                positions.append([xs[i], ys[j]])
    
    positions = np.array(positions)
    
    # Initial radii
    spacing = min(w/cols, h/rows)
    r_init = spacing * 0.48
    
    r = np.ones(N) * r_init
    
    # Simple iterative improvement
    # Try to increase radii and adjust positions
    
    best_r = r.copy()
    best_x = positions.copy()
    best_sum = np.sum(r)
    
    # Gradient-free optimization: try small perturbations
    np.random.seed(42)
    
    for iteration in range(500):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Pick a random circle
        i = np.random.randint(N)
        
        # Try to increase its radius
        new_r = r[i] * 1.02
        r[i] = new_r
        
        # Check validity
        if not _is_valid(positions, r, w, h):
            # Try moving the circle to make room
            # Move away from nearest neighbor
            moved = False
            for attempt in range(20):
                # Random small move
                dx = np.random.uniform(-0.02, 0.02)
                dy = np.random.uniform(-0.02, 0.02)
                old_x = positions[i].copy()
                positions[i, 0] += dx
                positions[i, 1] += dy
                
                # Keep in bounds
                positions[i, 0] = np.clip(positions[i, 0], r[i], w - r[i])
                positions[i, 1] = np.clip(positions[i, 1], r[i], h - r[i])
                
                if _is_valid(positions, r, w, h):
                    moved = True
                    break
            
            if not moved:
                r[i] = best_r[i]  # revert radius
                positions[i] = old_x  # revert position
                continue
        
        # Also try to increase other radii
        current_sum = np.sum(r)
        if current_sum > best_sum:
            best_sum = current_sum
            best_r = r.copy()
            best_x = positions.copy()
    
    result = np.zeros((N, 3))
    result[:, 0] = best_x[:, 0]
    result[:, 1] = best_x[:, 1]
    result[:, 2] = best_r
    
    return result


def _is_valid(x, r, w, h):
    N = len(x)
    for i in range(N):
        if x[i,0] < r[i] - 1e-9 or x[i,0] > w - r[i] + 1e-9:
            return False
        if x[i,1] < r[i] - 1e-9 or x[i,1] > h - r[i] + 1e-9:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((x[i,0]-x[j,0])**2 + (x[i,1]-x[j,1])**2)
            if dist < r[i] + r[j] - 1e-9:
                return False
    return True
