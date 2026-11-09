"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 120.0
START_TIME = time.time()

def time_left():
    return DEADLINE - (time.time() - START_TIME)

def check_validity(circles):
    """Check if circles are valid: non-negative radii, disjoint interiors, within rectangle."""
    if circles is None or len(circles) != N:
        return False
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    if np.any(rs < 0):
        return False
    
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    w = max_x - min_x
    h = max_y - min_y
    
    if 2 * (w + h) > 4 + 1e-6:
        return False
    
    # Check disjoint interiors
    # Vectorized check
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-6:
                return False
    return True

def compute_penalty_and_sum(x, w):
    """
    x: [x1, y1, r1, ..., xN, yN, rN]
    w: width of rectangle (height is 2-w)
    Returns: (penalty, sum_radii)
    """
    h = 2.0 - w
    if w <= 0 or h <= 0:
        return 1e10, 0.0

    coords = x[:3 * N]
    xs = coords[0::3]
    ys = coords[1::3]
    rs = coords[2::3]

    # Clip radii to non-negative for calculation, though bounds handle it
    rs_pos = np.maximum(rs, 0)

    penalty = 0.0

    # Boundary violations
    # Left/Right
    viol_left = np.maximum(0, - (xs - rs_pos))
    viol_right = np.maximum(0, (xs + rs_pos) - w)
    # Bottom/Top
    viol_bottom = np.maximum(0, - (ys - rs_pos))
    viol_top = np.maximum(0, (ys + rs_pos) - h)
    
    penalty += np.sum(viol_left**2) + np.sum(viol_right**2) + np.sum(viol_bottom**2) + np.sum(viol_top**2)

    # Overlaps
    # Compute pairwise distances
    # dx[i,j] = xs[i] - xs[j]
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    # min_dist[i,j] = rs[i] + rs[j]
    min_dist = rs_pos[:, None] + rs_pos[None, :]
    
    # Overlap amount = max(0, min_dist - dist)
    # We only care about i < j
    tri_indices = np.triu_indices(N, k=1)
    overlaps = np.maximum(0, min_dist[tri_indices] - dist[tri_indices])
    penalty += np.sum(overlaps**2)

    sum_radii = np.sum(rs_pos)
    return penalty, sum_radii

def objective(x, w):
    """
    Minimize penalty - lambda * sum_radii.
    We want to maximize sum_radii while keeping penalty near 0.
    A large lambda encourages larger radii even if it increases penalty slightly,
    but we need penalty to be 0 for validity.
    Strategy: Minimize penalty * big_factor - sum_radii.
    """
    penalty, sum_radii = compute_penalty_and_sum(x, w)
    # If penalty is large, the configuration is invalid.
    # We use a weighted objective.
    # If we simply minimize -sum_radii, it will try to make radii huge and overlap.
    # We need to penalize overlaps heavily.
    return 100.0 * penalty - sum_radii

def hexagonal_initialization(w, h, jitter=0.0):
    """
    Generate initial positions based on a hexagonal grid.
    """
    positions = []
    # Estimate radius to fit ~21 circles
    # Area = w*h. Area per circle ~ pi*r^2.
    # 21 * pi * r^2 ~ w*h => r ~ sqrt(w*h / (21*pi))
    r_est = np.sqrt(w * h / (21 * np.pi)) * 0.9
    
    # Hex grid spacing
    # dx = 2*r, dy = sqrt(3)*r
    # Let's just place points on a hex grid with spacing slightly larger than 2*r_est
    spacing = 2.0 * r_est * 1.1
    
    # Generate grid points
    i = 0
    row = 0
    while len(positions) < N:
        y = (row + 0.5) * spacing * np.sqrt(3) / 2
        if y > h:
            break
        offset = 0 if row % 2 == 0 else spacing / 2
        col = 0
        while True:
            x = (col + 0.5) * spacing + offset
            if x > w:
                break
            # Add jitter
            jx = np.random.uniform(-jitter, jitter)
            jy = np.random.uniform(-jitter, jitter)
            px = np.clip(x + jx, r_est, w - r_est)
            py = np.clip(y + jy, r_est, h - r_est)
            positions.append([px, py, r_est])
            col += 1
            if len(positions) >= N:
                break
        row += 1
    
    # If we didn't get enough, pad with random
    while len(positions) < N:
        px = np.random.uniform(r_est, w - r_est)
        py = np.random.uniform(r_est, h - r_est)
        positions.append([px, py, r_est])
        
    positions = positions[:N]
    x = np.array(positions).flatten()
    return x

def solve_packing(w, h, max_iter=200):
    """
    Solve for a fixed aspect ratio w x h.
    """
    if time_left() < 5:
        return None, -np.inf

    # Try multiple random initializations (jittered hex)
    best_circles = None
    best_score = -np.inf
    
    # Initial radius estimate
    r_init = np.sqrt(w * h / (N * np.pi)) * 0.8
    
    # Bounds for L-BFGS-B
    # xi in [0, w], yi in [0, h], ri in [0, min(w,h)/2]
    bounds = []
    for i in range(N):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0, min(w, h) / 2.0))
        
    # Run a few restarts
    restarts = 3
    for r in range(restarts):
        if time_left() < 5:
            break
            
        jitter = 0.1 * r_init if r > 0 else 0.0
        x0 = hexagonal_initialization(w, h, jitter=jitter)
        
        # Ensure initial radii are small enough to not overlap immediately
        # Hex init uses r_est which might overlap. Let's scale down initial radii in x0
        # x0 structure: x, y, r, x, y, r...
        # We set r to a small fraction in the optimizer start to let it grow?
        # Actually, L-BFGS-B doesn't grow variables. It moves them.
        # If we start with overlapping circles, the penalty is high.
        # It's better to start with non-overlapping small circles and expand.
        # But the objective is -sum(r) + penalty.
        # If we start with small r, penalty is 0, objective is -small.
        # Gradient will push r up if it doesn't increase penalty too much.
        
        # Let's modify x0 to have smaller radii initially
        x0_flat = x0.copy()
        for i in range(N):
            x0_flat[3*i + 2] = r_init * 0.5
            
        try:
            res = minimize(objective, x0_flat, args=(w,), method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 50})
            
            if res.success or res.fun < -0.1:
                x_opt = res.x
                circles = np.zeros((N, 3))
                circles[:, 0] = x_opt[0::3]
                circles[:, 1] = x_opt[1::3]
                circles[:, 2] = x_opt[2::3]
                
                # Post-processing: ensure strict validity
                # Sometimes optimizer leaves tiny violations
                # We can do a few manual gradient steps to fix
                # Or just check validity
                if check_validity(circles):
                    score = np.sum(circles[:, 2])
                    if score > best_score:
                        best_score = score
                        best_circles = circles
        except Exception:
            continue
            
    return best_circles, best_score

def construct():
    """Main entry point."""
    best_circles = None
    best_score = -np.inf
    
    # Aspect ratios w/h. w + h = 2.
    # w ranges from 0.2 to 1.8.
    # Symmetric around 1.0.
    aspect_ratios = np.linspace(0.3, 1.7, 15)
    
    # Shuffle or sort? Start with square, then expand.
    # 1.0 is index 7 (0.3 + 7*0.1 = 1.0)
    # Let's sort by distance from 1.0 to prioritize likely good solutions early
    sorted_indices = np.argsort(np.abs(aspect_ratios - 1.0))
    aspect_ratios = aspect_ratios[sorted_indices]
    
    for w in aspect_ratios:
        if time_left() < 10:
            break
            
        h = 2.0 - w
        # Skip invalid
        if w <= 0 or h <= 0:
            continue
            
        circles, score = solve_packing(w, h)
        
        if circles is not None and score > best_score:
            best_score = score
            best_circles = circles
            
    # Fallback: Simple grid if nothing worked
    if best_circles is None:
        w = 1.0
        h = 1.0
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) * w / side
                cy = (j + 0.5) * h / side
                r = min(w, h) / (2 * side) * 0.95
                cells.append([cx, cy, r])
        best_circles = np.array(cells[:N])
        
    return best_circles
