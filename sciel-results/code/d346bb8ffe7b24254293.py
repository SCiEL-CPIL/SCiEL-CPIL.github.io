import numpy as np
import time
from scipy.optimize import minimize

N = 21
TIME_LIMIT = 145.0

def _check_valid(coords, w, tol=1e-5):
    """
    coords: (N, 3) array of [x, y, r]
    w: width of rectangle
    Returns True if all circles are inside [0, w] x [0, 2-w] and non-overlapping.
    """
    h = 2.0 - w
    x = coords[:, 0]
    y = coords[:, 1]
    r = coords[:, 2]
    
    # Boundary checks
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
    if np.any(r < -tol):
        return False
        
    # Overlap checks
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            if dist_sq < (min_dist - tol)**2:
                return False
    return True

def _get_radius(coords, i, w, h, others):
    """
    Calculate the maximum radius circle i can have given its position and others.
    others: list of (j, x_j, y_j, r_j) for j != i
    """
    x, y, _ = coords[i]
    max_r = min(x, w - x, y, h - y)
    for j, xj, yj, rj in others:
        d = np.sqrt((x - xj)**2 + (y - yj)**2)
        max_r = min(max_r, (d - rj) / 2.0)
    return max(0.0, max_r)

def _greedy_pack(w, seed):
    """
    Greedy packing for a fixed width w.
    Returns (coords, score) or (None, -inf) if it fails.
    """
    h = 2.0 - w
    rng = np.random.default_rng(seed)
    
    # Initial grid points for candidate centers
    # We want to try many positions.
    # Grid resolution: 50x50 is 2500 points. 100x100 is 10000.
    # Let's use a moderately dense grid, maybe 60x60.
    grid_res = 60
    gx = np.linspace(0.01, w - 0.01, grid_res)
    gy = np.linspace(0.01, h - 0.01, grid_res)
    grid_points = np.column_stack(np.meshgrid(gx, gy, indexing='ij')).reshape(-1, 2)
    
    # Add some random points to break symmetry
    n_rand = 500
    rand_points = np.column_stack([
        rng.uniform(0.01, w - 0.01, n_rand),
        rng.uniform(0.01, h - 0.01, n_rand)
    ])
    grid_points = np.vstack([grid_points, rand_points])
    
    # Shuffle to avoid bias towards top-left? Or keep sorted? 
    # Greedy "best fit" usually implies we check all and pick the best.
    # But "first fit" with randomization is faster.
    # Let's do "Best Fit": for each new circle, scan all grid points, 
    # compute max possible radius, pick the one with largest radius.
    # This is O(N * M) where M is grid size.
    # 21 * 4000 = 84,000 evaluations per pack. Very fast.
    
    placed = [] # list of (x, y, r)
    coords = np.zeros((N, 3))
    
    for i in range(N):
        best_r = -1.0
        best_pos = None
        others = [(j, placed[j][0], placed[j][1], placed[j][2]) for j in range(i)]
        
        # To speed up, we can pre-filter grid points? 
        # No, 4000 points is small.
        
        for p in grid_points:
            x, y = p
            # Quick reject if too close to boundary
            if x < 0.001 or x > w - 0.001 or y < 0.001 or y > h - 0.001:
                continue
            
            # Calculate max radius
            r_max = min(x, w - x, y, h - y)
            
            # Check against placed circles
            valid = True
            for j, xj, yj, rj in others:
                dx = x - xj
                dy = y - yj
                d = np.sqrt(dx*dx + dy*dy)
                # Max radius allowed by this neighbor
                r_allowed = (d - rj) / 2.0
                if r_allowed < r_max:
                    r_max = r_allowed
                if r_max <= 1e-6:
                    valid = False
                    break
            
            if valid and r_max > best_r:
                best_r = r_max
                best_pos = (x, y)
        
        if best_pos is None or best_r < 1e-5:
            return None, -np.inf
        
        placed.append((best_pos[0], best_pos[1], best_r))
        coords[i] = [best_pos[0], best_pos[1], best_r]
        
        # Optional: If the best radius is very small, we might be stuck.
        # But let's continue.
        
    return coords, np.sum(coords[:, 2])

def _local_search(coords, w, deadline):
    """
    Improve the packing by shifting circles to increase radii.
    This is a simple hill climbing on the sum of radii, fixing positions first,
    then adjusting radii, then positions again?
    
    Actually, a better local search for circle packing:
    1. Fix positions. Increase all radii as much as possible.
    2. If any overlap, shift the circles to resolve it.
    
    Let's try a simpler approach: 
    Iteratively:
    a) For each circle, increase its radius to the max possible given current positions of others.
    b) If this causes overlap (which it shouldn't if calculated correctly), it means the "max possible" calculation was optimistic or we need to move.
    
    Wait, the greedy step already places them with max possible radius given *previous* circles.
    But later circles might block earlier ones.
    
    So, let's do this:
    Loop:
      1. For each circle i, calculate max radius r_i given positions of all other circles (fixed).
      2. Set r_i to this max.
      3. If no changes (within tol), stop.
      4. Then, try to shift circles.
         For each pair (i, j) that is close (dist < r_i + r_j + margin),
         try to move i and j apart along their connecting vector to allow larger radii.
         
    This is getting complex. Let's use a simpler "Repulsion" based local search.
    We want to maximize sum(r_i).
    
    Alternative: Use scipy minimize to maximize sum(r) with constraints?
    Constraints:
    1. x_i - r_i >= 0
    2. x_i + r_i <= w
    3. y_i - r_i >= 0
    4. y_i + r_i <= h
    5. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    
    This is a non-convex problem. SLSQP might work for local optimization from a good start.
    
    Let's implement a local polish using SLSQP.
    """
    
    h = 2.0 - w
    x0 = coords[:, 0].copy()
    y0 = coords[:, 1].copy()
    r0 = coords[:, 2].copy()
    
    # Variables: x (N), y (N), r (N)
    # Objective: -sum(r)
    
    def obj(p):
        x = p[:N]
        y = p[N:2*N]
        r = p[2*N:]
        return -np.sum(r)
    
    constraints = []
    
    # Boundary constraints
    # x_i - r_i >= 0  =>  x_i - r_i
    # x_i + r_i <= w  =>  w - x_i - r_i
    # y_i - r_i >= 0
    # y_i + r_i <= h
    
    # Overlap constraints
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    # SLSQP handles equality and inequality constraints.
    # Inequality: fun(x) >= 0
    
    for i in range(N):
        # Left
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[i] - p[2*N+i]
        })
        # Right
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: w - p[i] - p[2*N+i]
        })
        # Bottom
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[N+i] - p[2*N+i]
        })
        # Top
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: h - p[N+i] - p[2*N+i]
        })
        
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[i]-p[j])**2 + (p[N+i]-p[N+j])**2 - (p[2*N+i] + p[2*N+j])**2
            })
            
    bounds = []
    for i in range(N):
        bounds.append((0.0, w)) # x
    for i in range(N):
        bounds.append((0.0, h)) # y
    for i in range(N):
        bounds.append((0.0, min(w, h)/2)) # r
        
    p0 = np.concatenate([x0, y0, r0])
    
    try:
        res = minimize(obj, p0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 200, 'ftol': 1e-8})
        if res.success and res.fun < -np.sum(r0):
            new_coords = np.column_stack([res.x[:N], res.x[N:2*N], res.x[2*N:]])
            return new_coords
    except Exception:
        pass
        
    return coords

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_coords = None
    best_w = None
    best_score = -np.inf
    
    # Try a range of widths
    # Perimeter 4 => 2(w+h)=4 => w+h=2.
    # w in [0.5, 1.5]
    # Square is 1.0.
    widths = np.linspace(0.6, 1.4, 17)
    
    # Shuffle widths? Or just iterate.
    # Let's iterate.
    
    for w in widths:
        if time.time() > deadline - 10.0:
            break
            
        # Run greedy pack
        # Use different seeds for diversity?
        # Greedy is deterministic for a given w and grid.
        # To get diversity, we can vary the grid or add random jitter to the grid points.
        
        # Let's run 2 variants: one standard, one with jittered grid.
        for variant in range(2):
            if time.time() > deadline - 10.0:
                break
                
            seed = int(w * 100) + variant * 1000
            coords, score = _greedy_pack(w, seed)
            
            if coords is None:
                continue
                
            # Validate
            if not _check_valid(coords, w):
                # Greedy should produce valid, but numerical errors might occur
                # Try to fix by reducing radii slightly
                coords[:, 2] *= 0.99
                if not _check_valid(coords, w):
                    continue
                score = np.sum(coords[:, 2])
                
            # Local search polish
            if time.time() < deadline - 5.0:
                polished_coords = _local_search(coords, w, deadline)
                if _check_valid(polished_coords, w):
                    p_score = np.sum(polished_coords[:, 2])
                    if p_score > score:
                        coords = polished_coords
                        score = p_score
            
            if score > best_score:
                best_score = score
                best_coords = coords.copy()
                best_w = w
                
    if best_coords is None:
        # Fallback: simple grid
        # 5x5 grid in 1x1 square
        w = 1.0
        h = 1.0
        coords = np.zeros((N, 3))
        r = 0.5 / 5 - 0.001
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx >= N: break
                coords[idx] = [(i + 0.5) * 0.2, (j + 0.5) * 0.2, r]
                idx += 1
        return coords
        
    return best_coords

