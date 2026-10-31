"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _check_validity(pos, rad, W, H, tol=1e-6):
    """Check if configuration is valid: no overlaps, all inside rectangle."""
    for i in range(N):
        x, y = pos[i]
        r = rad[i]
        if x < r - tol or x > W - r + tol or y < r - tol or y > H - r + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist < rad[i] + rad[j] - tol:
                return False
    return True

def _continuous_refine(pos, rad, W, H, max_iter=200):
    """Refine a valid packing using continuous optimization to maximize sum of radii."""
    def objective(params):
        p = params[:2*N].reshape(N, 2)
        r = params[2*N:]
        # Penalize violations heavily
        pen = 0.0
        for i in range(N):
            x, y = p[i]
            ri = r[i]
            if x < ri:
                pen += (ri - x)**2 * 100
            if x > W - ri:
                pen += (x - (W - ri))**2 * 100
            if y < ri:
                pen += (ri - y)**2 * 100
            if y > H - ri:
                pen += (y - (H - ri))**2 * 100
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(p[i] - p[j])
                min_d = r[i] + r[j]
                if d < min_d:
                    pen += (min_d - d)**2 * 100
        # Maximize sum of radii by minimizing negative sum
        return pen - np.sum(r)
    
    x0 = np.concatenate([pos.flatten(), rad])
    bounds = []
    for _ in range(N):
        bounds.append((0, W))
        bounds.append((0, H))
    for _ in range(N):
        bounds.append((0, min(W, H)/2))
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': max_iter, 'ftol': 1e-12})
        new_pos = res.x[:2*N].reshape(N, 2)
        new_rad = res.x[2*N:]
        if _check_validity(new_pos, new_rad, W, H):
            return new_pos, new_rad
    except:
        pass
    return pos, rad

def _simulated_annealing(W, H, grid_res=20, max_iter=5000, T0=1.0, alpha=0.9995):
    """Discrete grid-based simulated annealing for circle packing."""
    # Create grid of possible positions
    xs = np.linspace(0, W, grid_res + 1)
    ys = np.linspace(0, H, grid_res + 1)
    
    # Initial placement: random grid positions
    positions = np.zeros((N, 2))
    radii = np.zeros(N)
    used_cells = set()
    
    for i in range(N):
        while True:
            gi = np.random.randint(0, grid_res + 1)
            gj = np.random.randint(0, grid_res + 1)
            cell = (gi, gj)
            if cell not in used_cells:
                used_cells.add(cell)
                positions[i] = [xs[gi], ys[gj]]
                break
    
    # Calculate initial radii based on nearest neighbors
    for i in range(N):
        min_dist = min(W - positions[i][0], positions[i][0], H - positions[i][1], positions[i][1])
        for j in range(N):
            if i != j:
                d = np.linalg.norm(positions[i] - positions[j])
                min_dist = min(min_dist, d / 2)
        radii[i] = max(0.01, min_dist * 0.95)
    
    T = T0
    best_pos = positions.copy()
    best_rad = radii.copy()
    best_value = np.sum(radii)
    
    for it in range(max_iter):
        if time.time() > 145.0:
            break
        
        # Pick a random circle to move
        i = np.random.randint(N)
        old_x, old_y = positions[i]
        old_r = radii[i]
        
        # Move to random nearby grid cell
        dx = np.random.choice([-1, 0, 1])
        dy = np.random.choice([-1, 0, 1])
        if dx == 0 and dy == 0:
            dx = np.random.choice([-1, 1])
        
        new_gi = int(round(old_x / W * grid_res)) + dx
        new_gj = int(round(old_y / H * grid_res)) + dy
        new_gi = np.clip(new_gi, 0, grid_res)
        new_gj = np.clip(new_gj, 0, grid_res)
        
        new_pos = positions.copy()
        new_pos[i] = [xs[new_gi], ys[new_gj]]
        
        # Recalculate radius for moved circle
        min_dist = min(W - new_pos[i][0], new_pos[i][0], H - new_pos[i][1], new_pos[i][1])
        for j in range(N):
            if i != j:
                d = np.linalg.norm(new_pos[i] - new_pos[j])
                min_dist = min(min_dist, d / 2)
        new_rad = radii.copy()
        new_rad[i] = max(0.01, min_dist * 0.95)
        
        # Check validity
        if _check_validity(new_pos, new_rad, W, H):
            delta = np.sum(new_rad) - np.sum(radii)
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                positions = new_pos
                radii = new_rad
                if np.sum(radii) > best_value:
                    best_value = np.sum(radii)
                    best_pos = positions.copy()
                    best_rad = radii.copy()
        else:
            # Penalty: reject if invalid
            pass
        
        T *= alpha
    
    return best_pos, best_rad

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Try multiple aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.43, 0.6, 1.67]
    
    best_result = None
    best_value = 0.0
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 15:
            break
        
        # W + H = 2 (perimeter = 4)
        W = aspect * (2.0 / (1 + aspect))
        H = 2.0 - W
        
        # Run simulated annealing
        pos, rad = _simulated_annealing(W, H, grid_res=25, max_iter=3000, T0=0.5, alpha=0.9998)
        
        # Continuous refinement
        pos, rad = _continuous_refine(pos, rad, W, H, max_iter=150)
        
        if _check_validity(pos, rad, W, H):
            value = np.sum(rad)
            if value > best_value:
                best_value = value
                best_result = np.hstack([pos, rad[:, np.newaxis]])
    
    # If no valid result found, fallback to simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 1e-3
                    cells.append([x, y, r])
        while len(cells) < N:
            cells.append([0.1, 0.1, 0.01])
        best_result = np.array(cells[:N])
    
    return best_result

# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
