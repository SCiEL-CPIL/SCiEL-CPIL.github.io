"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score_from_centres(c):
    """Compute the exact radius score for given centres (N,2)."""
    n = len(c)
    # Pairwise distances
    min_pair_dist = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(c[i] - c[j])
            if d < min_pair_dist:
                min_pair_dist = d
    
    # Wall distances
    min_wall_dist = float('inf')
    for i in range(n):
        for k in range(2):
            w = 0.5 - abs(c[i, k])
            if w < min_wall_dist:
                min_wall_dist = w
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _min_distance_and_indices(c):
    """Return (min_dist, i, j) where min_dist is the minimum pairwise or wall distance."""
    n = len(c)
    min_val = float('inf')
    min_i = -1
    min_j = -1
    
    # Pairwise
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(c[i] - c[j])
            if d < min_val:
                min_val = d
                min_i = i
                min_j = j
    
    # Wall distances
    for i in range(n):
        for k in range(2):
            w = 0.5 - abs(c[i, k])
            if w < min_val:
                min_val = w
                min_i = i
                min_j = -1  # wall constraint
    
    return min_val, min_i, min_j

def _epigraph_objective(c, r):
    """
    Epigraph formulation: minimize r subject to all constraints.
    Returns violation amount (0 if feasible).
    """
    n = len(c)
    violation = 0.0
    
    # Pairwise: need dist >= 2*r
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(c[i] - c[j])
            if d < 2.0 * r:
                violation += (2.0 * r - d)
    
    # Wall: need 0.5 - |c_ik| >= r
    for i in range(n):
        for k in range(2):
            w = 0.5 - abs(c[i, k])
            if w < r:
                violation += (r - w)
    
    return violation

def _negative_smooth_score(c):
    """
    Smooth objective: negative of the minimum separation.
    Uses a smooth approximation to handle non-differentiability.
    """
    n = len(c)
    min_val = float('inf')
    
    # Pairwise
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(c[i] - c[j])
            min_val = min(min_val, d)
    
    # Wall
    for i in range(n):
        for k in range(2):
            w = 0.5 - abs(c[i, k])
            min_val = min(min_val, w)
    
    return -min_val

def _grid_init():
    """Generate a grid initial configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _hex_init():
    """Generate a hexagonal lattice initial configuration."""
    # Hexagonal packing
    dx = 1.0 / 8.0
    dy = dx * np.sqrt(3) / 2
    pts = []
    for row in range(8):
        for col in range(8):
            x = (col + 0.5) * dx - 0.5
            y = (row + 0.5) * dy - 0.5
            if row % 2 == 1:
                x += dx / 2
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([x, y])
    pts = np.array(pts[:N])
    return pts

def _random_init(rng):
    """Generate a random initial configuration."""
    return rng.uniform(-0.45, 0.45, size=(N, 2))

def _metropolis_hastings(c, r_target, rng, n_steps, T):
    """
    Metropolis-Hastings Monte Carlo to find configuration with radius >= r_target.
    Returns improved configuration if found.
    """
    c_curr = c.copy()
    r_curr = _score_from_centres(c_curr)
    
    if r_curr >= r_target:
        return c_curr
    
    best_c = c_curr.copy()
    best_r = r_curr
    
    step_size = 0.01
    
    for step in range(n_steps):
        # Choose a random particle and move it
        i = rng.integers(0, N)
        delta = rng.normal(0, step_size, size=2)
        c_prop = c_curr.copy()
        c_prop[i] += delta
        
        # Clip to square
        c_prop = np.clip(c_prop, -0.5, 0.5)
        
        r_prop = _score_from_centres(c_prop)
        
        # Accept if improvement or with probability exp(-(r_curr - r_prop)/T)
        if r_prop >= r_curr:
            c_curr = c_prop
            r_curr = r_prop
            if r_curr > best_r:
                best_r = r_curr
                best_c = c_curr.copy()
        else:
            # Metropolis acceptance
            delta_r = r_curr - r_prop
            if delta_r < T:
                accept_prob = np.exp(-delta_r / T)
                if rng.random() < accept_prob:
                    c_curr = c_prop
                    r_curr = r_prop
                    if r_curr > best_r:
                        best_r = r_curr
                        best_c = c_prop.copy()
        
        # Cool down temperature
        T *= 0.999
    
    return best_c

def _polish_powell(c, max_iter=1000):
    """
    Powell's method for local polishing.
    Uses scipy's minimize with Powell method.
    """
    from scipy.optimize import minimize
    
    def objective(x):
        pts = x.reshape(N, 2)
        pts = np.clip(pts, -0.5, 0.5)
        return -_score_from_centres(pts)
    
    try:
        result = minimize(
            objective,
            c.flatten(),
            method='Powell',
            options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        return result.x.reshape(N, 2)
    except:
        return c

def _polish_high_precision(c, max_iter=2000):
    """
    High-precision polishing using Nelder-Mead.
    """
    from scipy.optimize import minimize
    
    def objective(x):
        pts = x.reshape(N, 2)
        pts = np.clip(pts, -0.5, 0.5)
        return -_score_from_centres(pts)
    
    try:
        result = minimize(
            objective,
            c.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-14}
        )
        return result.x.reshape(N, 2)
    except:
        return c

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = 0.0
    
    # Try multiple initial configurations
    inits = [
        _grid_init(),
        _hex_init(),
        _random_init(rng),
        _random_init(rng),
        _random_init(rng)
    ]
    
    for init_idx, c_init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        
        c = c_init.copy()
        c = np.clip(c, -0.5, 0.5)
        r = _score_from_centres(c)
        
        # Phase 1: Metropolis-Hastings to explore landscape
        # Use multiple temperature schedules
        for T0 in [0.001, 0.005, 0.01]:
            if time.time() > deadline - 15.0:
                break
            
            n_steps = 5000
            c_mh = _metropolis_hastings(c, r, rng, n_steps, T0)
            r_mh = _score_from_centres(c_mh)
            
            if r_mh > r:
                c = c_mh
                r = r_mh
        
        # Phase 2: Local polishing
        if time.time() < deadline - 10.0:
            c_powell = _polish_powell(c)
            r_powell = _score_from_centres(c_powell)
            if r_powell > r:
                c = c_powell
                r = r_powell
        
        # Phase 3: High-precision polish
        if time.time() < deadline - 5.0:
            c_hp = _polish_high_precision(c)
            r_hp = _score_from_centres(c_hp)
            if r_hp > r:
                c = c_hp
                r = r_hp
        
        # Update best
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    # Final safety check
    if best_c is None:
        best_c = _grid_init()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
