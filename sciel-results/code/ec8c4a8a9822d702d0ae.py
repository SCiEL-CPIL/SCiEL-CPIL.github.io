"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
NUM_TRIPLES = N * (N - 1) * (N - 2) // 6

# Precompute triple indices for vectorized area calculation
_TRIPLETS = np.array(list(combinations(range(N), 3)))

def _triangle_areas(pts):
    """
    Compute all triangle areas efficiently using precomputed indices.
    pts: (N, 2) array
    """
    i = _TRIPLETS[:, 0]
    j = _TRIPLETS[:, 1]
    k = _TRIPLETS[:, 2]
    v1 = pts[j] - pts[i]
    v2 = pts[k] - pts[i]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas) / hull_area

def _soft_min_objective(x, beta=50.0):
    """
    Objective for smooth optimization.
    Minimizes -soft_min(areas) / hull_area.
    x: (2N,) flattened point coordinates.
    """
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 1e9
    
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e9
    
    # Soft min approximation: -1/beta * log(sum(exp(-beta * x)))
    # We want to maximize min(areas)/hull_area, so minimize -min(areas)/hull_area.
    # Let f = areas / hull_area. We want to minimize -soft_min(f).
    # soft_min(f) approx min(f).
    
    # To avoid overflow in exp, shift by max
    # We are computing soft min of 'areas' then dividing by hull_area?
    # Actually, the ratio is non-linear. 
    # Let's optimize: max (min_area / hull_area).
    # Equivalent to min (-min_area / hull_area).
    # Let g = -areas. We want soft_min(g) which approximates min(g) = -max(areas)? No.
    # We want min(areas).
    # Let's use: soft_min(areas) = -1/beta * log(sum(exp(-beta * areas))).
    # This approximates min(areas).
    
    # We need to handle the division by hull_area carefully in the objective.
    # Objective: - (soft_min(areas) / hull_area)
    
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    # Clip to prevent underflow/overflow in exp, though max_shifted is 0 if we shift properly
    # exp(-beta * areas) can be very small.
    # log_sum = max_shifted + log(sum(exp(shifted - max_shifted)))
    # soft_min_val = -1/beta * log_sum
    
    # If areas are large, beta*areas is large, exp is small.
    # Let's stick to the standard log-sum-exp trick.
    
    # Note: if areas are all 0, this fails, but hull_area > 0 implies non-degenerate.
    
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_min_area = -log_sum_exp / beta
    
    if soft_min_area <= 0:
         return 1e9
         
    return -soft_min_area / hull_area

def _exact_objective(x):
    """
    Objective for Nelder-Mead polish.
    Minimizes -score(pts).
    """
    pts = x.reshape(-1, 2)
    return -_score(pts)

def _regular_polygon_init(seed=0):
    """Generate points on a regular polygon with slight noise."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles += rng.uniform(-0.05, 0.05, N)
    radii = 1.0 + rng.normal(0, 0.02, N)
    pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    return pts

def _halton_points_2d(n, seed=0):
    """Generate n 2D points using Halton sequences."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = np.zeros((n, 2))
    for i in range(n):
        pts[i, 0] = halton(i + 1, 2)
        pts[i, 1] = halton(i + 1, 3)
    
    # Center and scale
    pts -= np.mean(pts, axis=0)
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm > 0:
        pts /= max_norm
    return pts

def _concentric_ring_init(seed=0):
    """Generate points on two concentric rings."""
    rng = np.random.RandomState(seed)
    pts = np.zeros((N, 2))
    n_inner = 5
    n_outer = N - n_inner
    
    # Inner ring
    angles_in = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + rng.uniform(0, 2*np.pi)
    r_in = 0.4
    pts[:n_inner, 0] = r_in * np.cos(angles_in)
    pts[:n_inner, 1] = r_in * np.sin(angles_in)
    
    # Outer ring
    angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + rng.uniform(0, 2*np.pi)
    r_out = 1.0
    pts[n_inner:, 0] = r_out * np.cos(angles_out)
    pts[n_inner:, 1] = r_out * np.sin(angles_out)
    
    return pts

def _nelder_mead_polish(pts, max_iter=1000):
    """
    Polish using Nelder-Mead on the exact objective.
    This is crucial for resolving the non-smooth min function.
    """
    x0 = pts.flatten()
    try:
        # Nelder-Mead is derivative-free and handles non-smoothness well.
        # 'adaptive' helps with scaling issues.
        result = minimize(
            _exact_objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': max_iter, 
                'xatol': 1e-10, 
                'fatol': 1e-12, 
                'adaptive': True,
                'disp': False
            }
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_score = 0.0
    best_pts = None
    
    # Initialize with a regular polygon as a safe baseline
    best_pts = _regular_polygon_init(0)
    best_score = _score(best_pts)
    
    # Phase 1: Global/Coarse Search using L-BFGS-B on Soft-Min Objective
    # We use multiple diverse initializations.
    init_strategies = [
        lambda s: _regular_polygon_init(s),
        lambda s: _halton_points_2d(N, seed=s),
        lambda s: _concentric_ring_init(s),
    ]
    
    # We will try several starts from each strategy
    # Budget: ~80s for this phase, leaving ~60s for polish
    phase1_deadline = start_time + 80
    
    num_starts = 30
    for i in range(num_starts):
        if time.time() > phase1_deadline:
            break
            
        strategy_idx = i % len(init_strategies)
        seed = i + 100
        try:
            pts0 = init_strategies[strategy_idx](seed)
            x0 = pts0.flatten()
            
            # Use a moderate beta for smoothness. 
            # Too high -> close to hard min, hard for L-BFGS. 
            # Too low -> too smooth, might miss features.
            # 50-100 is usually good for 13 points.
            result = minimize(
                _soft_min_objective,
                x0,
                method='L-BFGS-B',
                args=(80.0,),
                options={
                    'maxiter': 300, 
                    'ftol': 1e-12, 
                    'gtol': 1e-8
                }
            )
            
            pts = result.x.reshape(-1, 2)
            score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue

    # Phase 2: Local Polish using Nelder-Mead on Exact Objective
    # This is the key improvement. L-BFGS-B stops at a point where the gradient of the
    # soft-min is zero, which is not necessarily the max of the hard-min.
    # Nelder-Mead can navigate the kinks.
    
    phase2_deadline = start_time + 145
    
    if time.time() < phase2_deadline:
        # Polish the current best
        polished_pts = _nelder_mead_polish(best_pts, max_iter=1000)
        polished_score = _score(polished_pts)
        
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts.copy()
            
        # Try polishing a few other top candidates if we have time
        # We didn't store all candidates, so we re-generate the top few diverse inits
        # and quickly polish them to see if they beat the current best.
        
        diverse_seeds = [1, 2, 3, 4, 5]
        for seed in diverse_seeds:
            if time.time() > phase2_deadline - 5:
                break
                
            # Pick a different strategy for diversity
            strat_idx = seed % len(init_strategies)
            try:
                pts0 = init_strategies[strat_idx](seed + 500)
                x0 = pts0.flatten()
                
                # Quick L-BFGS to get close
                res = minimize(
                    _soft_min_objective,
                    x0,
                    method='L-BFGS-B',
                    args=(100.0,),
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                pts = res.x.reshape(-1, 2)
                
                # Quick NM Polish
                pts = _nelder_mead_polish(pts, max_iter=500)
                score = _score(pts)
                
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
            except Exception:
                continue

    # Final sanity check and return
    if best_pts is None:
        best_pts = _regular_polygon_init(0)
        
    return best_pts

# EVOLVE-BLOCK-START
# The code above is the complete implementation.
# EVOLVE-BLOCK-END
