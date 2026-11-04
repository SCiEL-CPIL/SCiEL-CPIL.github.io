"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_value(circles):
    """Compute the sum of radii, or 0 if invalid."""
    if len(circles) != N:
        return 0.0
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # Check radii positive
    if np.any(rs <= 0):
        return 0.0
    
    # Compute bounding box
    xmin = np.min(xs - rs)
    xmax = np.max(xs + rs)
    ymin = np.min(ys - rs)
    ymax = np.max(ys + rs)
    
    w = xmax - xmin
    h = ymax - ymin
    
    # Perimeter constraint
    if 2 * (w + h) > 4.0 + 1e-6:
        return 0.0
    
    # Check disjoint interiors (distance between centers >= sum of radii)
    # Use vectorized check for efficiency
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    # For all pairs i != j, dist[i,j] >= rs[i] + rs[j]
    # We need to check off-diagonal
    mask = ~np.eye(N, dtype=bool)
    min_dist = np.full((N, N), np.inf)
    min_dist[mask] = dist[mask]
    
    # Check constraint
    if np.any(min_dist < rs[:, None] + rs[None, :] - 1e-7):
        return 0.0
    
    return np.sum(rs)

def _hexagonal_init(n_circles, w, h):
    """Generate initial positions on a hexagonal grid scaled to fit in [0,w]x[0,h]."""
    # Estimate spacing
    area = w * h
    avg_radius = np.sqrt(area / (n_circles * np.pi))
    
    # Hexagonal packing: rows offset
    # Number of rows and columns to approximate n_circles
    # For hexagonal packing, density is pi/(2*sqrt(3)) ≈ 0.9069
    # So effective area per circle is pi / 0.9069 ≈ 3.464
    # Spacing s such that s^2 * sqrt(3)/2 = area/n_circles
    s = np.sqrt(2 * area / (n_circles * np.sqrt(3)))
    
    # Number of columns and rows
    cols = int(np.ceil(w / s))
    rows = int(np.ceil(h / (s * np.sqrt(3) / 2)))
    
    if cols < 1:
        cols = 1
    if rows < 1:
        rows = 1
    
    positions = []
    for row in range(rows):
        y = row * s * np.sqrt(3) / 2
        if row % 2 == 0:
            x_start = 0
        else:
            x_start = s / 2
        
        for col in range(cols):
            x = x_start + col * s
            if 0 <= x <= w and 0 <= y <= h:
                positions.append((x, y))
    
    # If we don't have enough positions, add more randomly
    while len(positions) < n_circles:
        x = np.random.uniform(0, w)
        y = np.random.uniform(0, h)
        positions.append((x, y))
    
    # Shuffle to get a good subset
    np.random.shuffle(positions)
    positions = positions[:n_circles]
    
    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])
    
    # Initial radii: small value
    r0 = 0.01
    rs = np.full(n_circles, r0)
    
    circles = np.column_stack([xs, ys, rs])
    return circles

def _objective(params, n_circles, w, h):
    """Objective function: negative sum of radii plus penalties."""
    xs = params[:n_circles]
    ys = params[n_circles:2*n_circles]
    rs = params[2*n_circles:]
    
    # Penalty for negative radii
    penalty = 0.0
    if np.any(rs < 0):
        penalty += 1000 * np.sum(np.clip(-rs, None, None))
    
    # Check bounding box
    xmin = np.min(xs - rs)
    xmax = np.max(xs + rs)
    ymin = np.min(ys - rs)
    ymax = np.max(ys + rs)
    
    bw = xmax - xmin
    bh = ymax - ymin
    
    # Perimeter penalty
    perim = 2 * (bw + bh)
    if perim > 4.0:
        penalty += 100 * (perim - 4.0)**2
    
    # Collision penalty
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    mask = ~np.eye(n_circles, dtype=bool)
    overlap = np.clip(rs[:, None] + rs[None, :] - dist, None, None)
    overlap[~mask] = 0
    penalty += 10 * np.sum(overlap**2)
    
    # Keep circles within reasonable bounds
    out_of_bounds = 0
    if xmin < -0.1:
        out_of_bounds += (-0.1 - xmin)**2
    if xmax > w + 0.1:
        out_of_bounds += (xmax - w - 0.1)**2
    if ymin < -0.1:
        out_of_bounds += (-0.1 - ymin)**2
    if ymax > h + 0.1:
        out_of_bounds += (ymax - h - 0.1)**2
    penalty += 100 * out_of_bounds
    
    return -(np.sum(rs)) + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    best_circles = None
    best_value = 0.0
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.6, 1.6667, 0.75, 1.3333]
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
        
        # Perimeter = 2(w+h) = 4 => w+h = 2
        # h = aspect * w
        # w(1+aspect) = 2 => w = 2/(1+aspect)
        w = 2.0 / (1.0 + aspect)
        h = aspect * w
        
        # Try multiple random seeds for hexagonal init
        for seed in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(seed * 1000 + int(aspect * 100))
            circles = _hexagonal_init(N, w, h)
            
            # Flatten for optimization
            x0 = circles.flatten()
            
            # Use L-BFGS-B with analytical-like objective
            try:
                result = minimize(
                    _objective,
                    x0,
                    args=(N, w, h),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                if result.success or result.fun < _objective(x0, N, w, h):
                    x_opt = result.x
                    circles_opt = x_opt.reshape(3, N).T
                    val = _compute_value(circles_opt)
                    if val > best_value:
                        best_value = val
                        best_circles = circles_opt.copy()
                        
            except Exception:
                pass
        
        # Also try a simpler grid-based approach for this aspect ratio
        if time.time() > deadline:
            break
        
        # Grid approach: divide rectangle into cells
        # For 21 circles, try 7x3 or similar grids
        grid_configs = [
            (7, 3),  # 21 cells
            (6, 4),  # 24 cells, use 21
            (5, 5),  # 25 cells, use 21
            (9, 3),  # 27 cells, use 21
            (7, 4),  # 28 cells, use 21
        ]
        
        for rows, cols in grid_configs:
            if time.time() > deadline:
                break
            
            if rows * cols < N:
                continue
            
            # Cell dimensions
            cw = w / cols
            ch = h / rows
            
            # Initial circles at cell centers
            circles_grid = []
            for i in range(rows):
                for j in range(cols):
                    if len(circles_grid) >= N:
                        break
                    x = (j + 0.5) * cw
                    y = (i + 0.5) * ch
                    r = 0.5 * min(cw, ch) - 1e-4
                    circles_grid.append([x, y, r])
            
            circles_grid = np.array(circles_grid[:N])
            val = _compute_value(circles_grid)
            if val > best_value:
                best_value = val
                best_circles = circles_grid.copy()
            
            # Optimize from grid init
            x0 = circles_grid.flatten()
            try:
                result = minimize(
                    _objective,
                    x0,
                    args=(N, w, h),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                if result.success or result.fun < _objective(x0, N, w, h):
                    x_opt = result.x
                    circles_opt = x_opt.reshape(3, N).T
                    val = _compute_value(circles_opt)
                    if val > best_value:
                        best_value = val
                        best_circles = circles_opt.copy()
                        
            except Exception:
                pass
    
    # If we haven't found a valid solution, use the simple grid fallback
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
