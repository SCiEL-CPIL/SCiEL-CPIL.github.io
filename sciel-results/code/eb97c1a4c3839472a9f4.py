"""Optimized packing of 50 equal circles in a unit square."""
import numpy as np
import time
import random
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _score(centres):
    """Compute the minimum radius: min of half-pair-distances and wall distances."""
    # Wall distances
    wall = 0.5 - np.abs(centres)
    min_wall = np.min(wall)
    
    # Pair distances
    # For N=50, cdist is efficient
    dist = cdist(centres, centres)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    return min(min_wall, min_pair)

def _construct_hexagonal():
    """Create a hexagonal lattice configuration."""
    # Hexagonal lattice parameters
    # Spacing s = 1 / sqrt(3) / 5 * something? 
    # For 50 circles in unit square, density is roughly 50 * pi * r^2 / 1.
    # Max density ~ 0.9. 50 * pi * r^2 ~ 0.9 => r ~ 0.075.
    # Hexagonal lattice spacing a = 2r.
    # Rows and cols.
    
    # Let's try to generate a grid that approximates hexagonal packing.
    # Approximate number of rows and cols.
    # sqrt(50) ~ 7.
    
    centres = []
    # Try to fit a 7x7 or 8x7 grid with hexagonal offset
    # We will generate a larger hexagonal grid and pick the best 50 inside the square.
    
    a = 0.15 # Initial guess for spacing, will be adjusted by optimizer
    # Actually, let's just generate a dense hexagonal grid and take the first 50 that fit, 
    # or better, generate a hexagonal grid that roughly covers the square.
    
    # Hexagonal grid generation
    # y = row * a * sqrt(3)/2
    # x = col * a + (row % 2) * a/2
    
    # We want to center it.
    # Let's generate points and then normalize/center them.
    
    pts = []
    for row in range(10):
        for col in range(10):
            x = col * 1.0 + (row % 2) * 0.5
            y = row * 0.866 # sqrt(3)/2
            pts.append([x, y])
            
    pts = np.array(pts)
    
    # Center the points
    center = pts.mean(axis=0)
    pts = pts - center
    
    # Scale to fit in [-0.45, 0.45] roughly
    max_coord = np.max(np.abs(pts))
    if max_coord > 0:
        pts = pts * (0.45 / max_coord)
        
    # Sort by distance from center to pick the "innermost" 50? 
    # Or just take the first 50? 
    # Taking the first 50 from a grid generation order might be biased.
    # Let's sort by radius from center to get a compact cluster.
    dists = np.linalg.norm(pts, axis=1)
    sorted_indices = np.argsort(dists)
    pts = pts[sorted_indices][:N]
    
    return pts.copy()

def _construct_grid():
    """Create initial grid configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _anneal(centres, max_time=80.0, seed=42):
    """Simulated annealing optimization with adaptive step size."""
    np.random.seed(seed)
    random.seed(seed)
    
    centres = centres.copy()
    current_score = _score(centres)
    best_centres = centres.copy()
    best_score = current_score
    
    start_time = time.time()
    deadline = start_time + max_time
    
    T = 0.005 # Initial temperature, needs to be small enough to allow moves but large enough to escape local minima
    T_min = 1e-8
    cooling = 0.9995
    
    n = len(centres)
    step_scale = 0.05
    
    # Pre-allocate arrays for speed?
    # We just use cdist which is fast enough.
    
    while time.time() < deadline:
        i = np.random.randint(n)
        
        dx = np.random.normal(0, step_scale)
        dy = np.random.normal(0, step_scale)
        
        new_centre = centres[i].copy()
        new_centre[0] += dx
        new_centre[1] += dy
        
        # Clip to square
        new_centre[0] = np.clip(new_centre[0], -0.499, 0.499)
        new_centre[1] = np.clip(new_centre[1], -0.499, 0.499)
        
        old_centre = centres[i].copy()
        centres[i] = new_centre
        
        new_score = _score(centres)
        delta = new_score - current_score
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_centres = centres.copy()
            # Increase step slightly if we improved
            step_scale = min(step_scale * 1.01, 0.1)
        else:
            centres[i] = old_centre
            # Decrease step if we didn't improve
            step_scale = max(step_scale * 0.99, 0.001)
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Reheat occasionally?
            if np.random.random() < 0.01:
                T = 0.001
    
    return best_centres, best_score

def _polish_nm(centres, max_time=30.0):
    """Polish using Nelder-Mead for non-smooth objective."""
    start_time = time.time()
    deadline = start_time + max_time
    
    centres = centres.copy()
    best_centres = centres.copy()
    best_score = _score(centres)
    
    def neg_min_radius(x):
        c = x.reshape(N, 2)
        wall = 0.5 - np.abs(c)
        min_wall = np.min(wall)
        dist = cdist(c, c)
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0
        return -min(min_wall, min_pair)
    
    x0 = centres.flatten()
    
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        try:
            result = minimize(
                neg_min_radius,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            x_opt = result.x.reshape(N, 2)
            x_opt = np.clip(x_opt, -0.4999, 0.4999)
            score = _score(x_opt)
            
            if score > best_score:
                best_score = score
                best_centres = x_opt.copy()
                x0 = x_opt.flatten()
        except Exception:
            break
        
        if time.time() > deadline - 10.0:
            break
        
        # Perturb for next attempt
        if attempts < 5:
            x0 = x0 + np.random.normal(0, 0.005, size=2*N)
            x0 = np.clip(x0, -0.499, 0.499)
    
    return best_centres, best_score

def construct():
    """Construct the optimal packing of 50 equal circles in a unit square."""
    start_time = time.time()
    total_deadline = start_time + 145.0
    
    np.random.seed(42)
    random.seed(42)
    
    # Generate diverse initial configurations
    initial_configs = [
        _construct_hexagonal(),
        _construct_grid(),
        np.random.uniform(-0.45, 0.45, size=(N, 2)),
        np.random.uniform(-0.45, 0.45, size=(N, 2)),
    ]
    
    best_centres = None
    best_score = -np.inf
    
    # Phase 1: Annealing from different starts
    # We have ~100s for annealing, ~45s for polishing
    anneal_configs = initial_configs[:4]
    
    # Allocate time for each anneal run
    # If we have 4 configs, give each ~20s?
    # Or run them in parallel? No, sequential.
    # Let's run 2 good configs for longer, and 2 random for shorter.
    
    runs = [
        (initial_configs[0], 40.0), # Hexagonal
        (initial_configs[1], 30.0), # Grid
        (initial_configs[2], 15.0), # Random 1
        (initial_configs[3], 15.0), # Random 2
    ]
    
    for config, alloc_time in runs:
        if time.time() > total_deadline - 50.0:
            break
            
        remaining = total_deadline - time.time() - 50.0
        anneal_time = min(alloc_time, remaining)
        if anneal_time <= 1.0:
            break
            
        centres, score = _anneal(config, max_time=anneal_time, seed=np.random.randint(100000))
        
        if score > best_score:
            best_score = score
            best_centres = centres
    
    if best_centres is None:
        best_centres = initial_configs[0]
        best_score = _score(best_centres)
    
    # Phase 2: Polish the best result
    polish_time = min(45.0, total_deadline - time.time() - 5.0)
    if polish_time > 5.0:
        polished_centres, polished_score = _polish_nm(best_centres, max_time=polish_time)
        if polished_score > best_score:
            best_score = polished_score
            best_centres = polished_centres
    
    # Final safety clip
    best_centres = np.clip(best_centres, -0.4999, 0.4999)
    
    return best_centres

# EVOLVE-BLOCK-END
