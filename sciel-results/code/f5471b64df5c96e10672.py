"""Packing 200 equal circles in a unit square: L-BFGS-B + fast neighbor-based local polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 200
DEADLINE = 145.0  # seconds; leave margin before 150s kill

def _score(pts):
    """Return the common radius r for a valid configuration, or 0.0 if invalid."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Distance to boundary
    ax = np.abs(pts[:, 0])
    ay = np.abs(pts[:, 1])
    boundary = np.minimum(0.5 - ax, 0.5 - ay)
    min_boundary = np.min(boundary)
    if min_boundary < 0:
        return 0.0
    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    r = min(min_boundary, min_dist / 2.0)
    return r if r > 0 else 0.0

def _hex_init():
    """Generate a hexagonal lattice initialization with more spacing variants."""
    best_pts = None
    best_r = 0
    # Try a wider range of spacings
    s_est = 1.0 / np.sqrt(N) * 1.2
    for s in np.linspace(s_est * 0.85, s_est * 1.15, 20):
        pts_list = []
        row = 0
        while True:
            y = -0.5 + s/2 + row * s * np.sqrt(3) / 2
            if y > 0.5 - s/2:
                break
            offset = (row % 2) * s / 2
            x = -0.5 + s/2 + offset
            while x <= 0.5 - s/2:
                pts_list.append((x, y))
                x += s
            row += 1
        if len(pts_list) < N:
            continue
        pts = np.array(pts_list[:N])
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        pts = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = pts[:N].copy()
    return best_pts

def _grid_init():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in ticks for x in ticks])
    return pts[:N].copy()

def _random_init(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 2))

def _soft_min_neg(x, alpha):
    """
    Objective function for L-BFGS-B.
    x: flat array of size 2N.
    alpha: sharpness of the soft-min.
    Returns: -soft_min(distances)
    """
    pts = x.reshape(N, 2)
    
    # Distances to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y).flatten()
    
    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    pairwise_dists = dist[~np.isinf(dist)]
    
    # Combine all distances
    all_dists = np.concatenate([boundary_dists, pairwise_dists])
    
    # Soft min: -1/alpha * log(sum(exp(-alpha * d)))
    val = - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * all_dists)))
    
    # We want to maximize min distance, so minimize -min distance.
    return -val

def _exact_min_neg(x):
    """
    Exact negative minimum distance objective for final polish.
    x: flat array of size 2N.
    Returns: -min(all distances)
    """
    pts = x.reshape(N, 2)
    
    # Distances to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y)
    
    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    pairwise_dists = dist[~np.isinf(dist)]
    
    # Find the minimum
    min_boundary = np.min(boundary_dists)
    min_pairwise = np.min(pairwise_dists)
    
    # The radius is min(min_boundary, min_pairwise/2)
    # But for the objective, we want to maximize the minimum "clearance"
    # which is min(min_boundary, min_pairwise/2)
    # Actually, let's work with distances directly.
    # The constraint is that all pairwise distances >= 2r and all boundary distances >= r.
    # So the radius r = min(min_boundary, min_pairwise/2).
    # We want to maximize r, so minimize -r.
    r = min(min_boundary, min_pairwise / 2.0)
    return -r

def _optimize(pts, time_budget):
    """
    Run L-BFGS-B optimization on the initial points with adaptive alpha.
    """
    t0 = time.time()
    x0 = pts.flatten()
    
    # Phase 1: Coarse optimization with lower alpha
    # Phase 2: Fine optimization with higher alpha
    # Phase 3: Very fine optimization with even higher alpha
    
    phases = [
        {"alpha": 15.0, "max_iter": 300, "gtol": 1e-4},
        {"alpha": 50.0, "max_iter": 300, "gtol": 1e-5},
        {"alpha": 150.0, "max_iter": 300, "gtol": 1e-6},
    ]
    
    for phase in phases:
        if time.time() - t0 > time_budget:
            break
            
        alpha = phase["alpha"]
        
        def obj(x):
            return _soft_min_neg(x, alpha)
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', 
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': phase["max_iter"], 'gtol': phase["gtol"]})
            x_opt = res.x
            x0 = x_opt
        except Exception:
            pass
            
    # Final polish with exact objective
    remaining = time_budget - (time.time() - t0)
    if remaining > 2.0:
        try:
            res = minimize(_exact_min_neg, x0, method='L-BFGS-B', 
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': 200, 'gtol': 1e-8})
            x0 = res.x
        except Exception:
            pass
            
    return x0.reshape(N, 2)

def _local_polish(pts, time_budget):
    """
    Fast neighbor-based local search: perturb only points involved in tight constraints.
    Uses a KD-tree to find nearest neighbors and focuses effort on active constraints.
    """
    t0 = time.time()
    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    rng = np.random.default_rng(12345)
    step_size = 0.005
    min_step = 0.0001
    
    # Build initial tree
    tree = cKDTree(best_pts)
    
    while time.time() - t0 < time_budget:
        # Compute current constraints
        d_x = 0.5 - np.abs(best_pts[:, 0])
        d_y = 0.5 - np.abs(best_pts[:, 1])
        boundary_dists = np.minimum(d_x, d_y)
        min_boundary = np.min(boundary_dists)
        
        # Find tight pairs using KD-tree
        pairs, dists = tree.query(best_pts, k=4)
        # Filter out self-distances
        valid = np.ones((N, 3), dtype=bool)
        valid[:, 0] = False  # first neighbor is self
        pair_dists = dists[:, 1:]  # exclude self
        min_pair = np.min(pair_dists)
        
        current_r = min(min_boundary, min_pair / 2.0)
        
        # Identify points that are "active" (involved in tight constraints)
        active_idx = set()
        
        # Points near boundary
        tight_boundary = np.where(boundary_dists < current_r * 1.05)[0]
        active_idx.update(tight_boundary.tolist())
        
        # Points with tight pairs
        for i in range(N):
            for j in range(1, 4):
                if pair_dists[i, j-1] < current_r * 2.1:
                    active_idx.add(i)
                    active_idx.add(int(pairs[i, j]))
        
        if not active_idx:
            # If no active constraints, do global perturbation
            perturb = rng.normal(0, step_size, size=(N, 2))
            trial = best_pts + perturb
            trial = np.clip(trial, -0.5, 0.5)
            trial_r = _score(trial)
            if trial_r > best_r:
                best_pts = trial
                best_r = trial_r
                tree = cKDTree(best_pts)
                step_size *= 0.9
            else:
                step_size *= 1.1
                step_size = min(step_size, 0.02)
            continue
        
        active_list = list(active_idx)
        n_active = len(active_list)
        
        # Perturb only active points
        perturb = np.zeros((N, 2))
        perturb[active_list] = rng.normal(0, step_size, size=(n_active, 2))
        trial = best_pts + perturb
        trial = np.clip(trial, -0.5, 0.5)
        
        trial_r = _score(trial)
        if trial_r > best_r:
            best_pts = trial
            best_r = trial_r
            tree = cKDTree(best_pts)
            step_size *= 0.92
        else:
            step_size *= 1.08
            step_size = min(step_size, 0.02)
        
        if step_size < min_step:
            step_size = min_step
    
    return best_pts

def _jittered_hex_init(seed=0):
    """Hexagonal lattice with small random jitter to break symmetry."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0
    
    s_est = 1.0 / np.sqrt(N) * 1.2
    for s in np.linspace(s_est * 0.85, s_est * 1.15, 15):
        pts_list = []
        row = 0
        while True:
            y = -0.5 + s/2 + row * s * np.sqrt(3) / 2
            if y > 0.5 - s/2:
                break
            offset = (row % 2) * s / 2
            x = -0.5 + s/2 + offset
            while x <= 0.5 - s/2:
                pts_list.append((x, y))
                x += s
            row += 1
        if len(pts_list) < N:
            continue
        pts = np.array(pts_list[:N])
        # Add small jitter
        pts += rng.normal(0, s * 0.02, size=pts.shape)
        pts = np.clip(pts, -0.5, 0.5)
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _hex_init()
    return best_pts

def construct():
    t_start = time.time()
    
    initializers = [
        ("hex", _hex_init),
        ("hex_jit", lambda: _jittered_hex_init(0)),
        ("grid", _grid_init),
        ("rand0", lambda: _random_init(0)),
    ]
    
    best_pts = None
    best_r = 0.0
    
    for name, init_fn in initializers:
        if time.time() > DEADLINE - 20:
            break
            
        pts = init_fn()
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
        # Optimize
        remaining = DEADLINE - time.time()
        if remaining < 10:
            break
            
        # Allocate time based on initializer quality
        if name == "hex":
            time_budget = min(remaining * 0.30, 40.0)
        elif name == "hex_jit":
            time_budget = min(remaining * 0.25, 30.0)
        elif name == "grid":
            time_budget = min(remaining * 0.20, 25.0)
        else:
            time_budget = min(remaining * 0.15, 15.0)
            
        opt_pts = _optimize(pts, time_budget)
        
        # Clamp to square
        opt_pts = np.clip(opt_pts, -0.5, 0.5)
        
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
            
    # Final polish phases
    remaining = DEADLINE - time.time()
    if remaining > 15 and best_pts is not None:
        # Phase 1: More L-BFGS-B optimization
        time_budget = min(remaining * 0.4, 25.0)
        opt_pts = _optimize(best_pts, time_budget)
        opt_pts = np.clip(opt_pts, -0.5, 0.5)
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
        
        # Phase 2: Fast neighbor-based local search polish
        remaining = DEADLINE - time.time()
        if remaining > 5:
            time_budget = min(remaining * 0.85, 30.0)
            polished_pts = _local_polish(best_pts, time_budget)
            polished_r = _score(polished_pts)
            if polished_r > best_r:
                best_r = polished_r
                best_pts = polished_pts.copy()
        
        # Phase 3: Another L-BFGS-B pass after local polish
        remaining = DEADLINE - time.time()
        if remaining > 8 and best_pts is not None:
            time_budget = min(remaining * 0.5, 15.0)
            opt_pts = _optimize(best_pts, time_budget)
            opt_pts = np.clip(opt_pts, -0.5, 0.5)
            opt_r = _score(opt_pts)
            if opt_r > best_r:
                best_r = opt_r
                best_pts = opt_pts.copy()
        
        # Final local polish
        remaining = DEADLINE - time.time()
        if remaining > 3:
            time_budget = min(remaining * 0.8, 10.0)
            polished_pts = _local_polish(best_pts, time_budget)
            polished_r = _score(polished_pts)
            if polished_r > best_r:
                best_r = polished_r
                best_pts = polished_pts.copy()
    
    if best_pts is None:
        best_pts = _hex_init()
        
    return best_pts
