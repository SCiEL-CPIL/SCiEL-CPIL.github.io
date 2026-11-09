"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0  # seconds before hard limit

def _compute_score(circles):
    """Compute sum of radii for a valid packing, or -inf if invalid."""
    n = len(circles)
    if n == 0:
        return 0.0
    
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check all radii are positive
    if np.any(r <= 0):
        return -np.inf
    
    # Check all circles are inside the unit square
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return -np.inf
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return -np.inf
    
    # Check no overlapping circles
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - 1e-9:
                return -np.inf
    
    return np.sum(r)

def _make_initial_grid(n):
    """Create initial grid layout."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side
            cells.append([cx, cy, r])
    return np.array(cells[:n])

def _make_initial_random(n, seed=42):
    """Create random initial layout."""
    rng = np.random.RandomState(seed)
    # Place circles at random positions with small radii
    x = rng.uniform(0.1, 0.9, n)
    y = rng.uniform(0.1, 0.9, n)
    r = np.full(n, 0.02)
    return np.column_stack([x, y, r])

def _objective(params, n):
    """Objective function for optimization: minimize negative sum of radii."""
    x = params[0:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    
    # Penalty for being outside the unit square
    penalty = 0.0
    for i in range(n):
        if x[i] - r[i] < 0:
            penalty += (x[i] - r[i])**2
        if x[i] + r[i] > 1:
            penalty += (x[i] + r[i] - 1)**2
        if y[i] - r[i] < 0:
            penalty += (y[i] - r[i])**2
        if y[i] + r[i] > 1:
            penalty += (y[i] + r[i] - 1)**2
    
    # Penalty for overlapping circles
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                penalty += (min_dist - dist)**2
    
    # Objective: maximize sum of radii, so minimize negative sum + penalties
    return -np.sum(r) + 100.0 * penalty

def _optimize_from_initial(initial, n, time_budget):
    """Run optimization from a given initial configuration."""
    start_time = time.time()
    params0 = initial.flatten()
    
    # Use Nelder-Mead for local optimization (no gradient needed)
    try:
        result = minimize(
            _objective,
            params0,
            args=(n,),
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-6,
                'fatol': 1e-6,
                'adaptive': True
            }
        )
        
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            return None
        
        # Convert back to circle format
        x = result.x[0:n]
        y = result.x[n:2*n]
        r = result.x[2*n:3*n]
        
        # Clip radii to be positive
        r = np.maximum(r, 1e-6)
        
        circles = np.column_stack([x, y, r])
        
        # Validate and compute score
        score = _compute_score(circles)
        if np.isinf(score):
            return None
        
        return circles, score
    
    except Exception:
        return None

def construct():
    """Construct 26 circles in unit square maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best_circles = None
    best_score = -np.inf
    
    # Try multiple initial configurations
    initial_configs = []
    
    # Grid initial
    initial_configs.append(_make_initial_grid(N))
    
    # Random initials with different seeds
    for seed in range(10):
        initial_configs.append(_make_initial_random(N, seed=seed + 100))
    
    # Run optimization from each initial
    for i, init in enumerate(initial_configs):
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        
        if remaining < 10.0:  # Need at least 10 seconds for one optimization
            break
        
        # Allocate time budget based on remaining time
        time_budget = min(remaining * 0.8, 30.0)
        
        result = _optimize_from_initial(init, N, time_budget)
        
        if result is not None:
            circles, score = result
            if score > best_score:
                best_score = score
                best_circles = circles
        
        # Check if we should stop
        if time.time() > deadline:
            break
    
    # If no valid result, fall back to grid
    if best_circles is None:
        best_circles = _make_initial_grid(N)
    
    # Final validation
    final_score = _compute_score(best_circles)
    if np.isinf(final_score):
        best_circles = _make_initial_grid(N)
    
    return best_circles
