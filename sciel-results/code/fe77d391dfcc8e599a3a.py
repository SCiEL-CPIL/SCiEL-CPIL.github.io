"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0  # area of the unit-base equilateral triangle

# Precompute indices for all triples to speed up area calculation
_IDX_TRIPLES = None
def _get_indices():
    global _IDX_TRIPLES
    if _IDX_TRIPLES is None:
        n = N
        _IDX_TRIPLES = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    return _IDX_TRIPLES

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) with u>=0, v>=0, u+v<=1 to cartesian."""
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(points):
    """Convert cartesian points to barycentric (u, v) where point = u*V1 + v*V2."""
    u = points[:, 0] - 0.5 * points[:, 1]
    v = 2.0 * points[:, 1] / np.sqrt(3)
    return np.column_stack([u, v])

def _clamp_bary(bary):
    """Clamp barycentric coordinates to the valid region u>=0, v>=0, u+v<=1."""
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _clamp_cart(points):
    """Clamp Cartesian points to be inside the triangle."""
    bary = _cart_to_bary(points)
    bary = _clamp_bary(bary)
    return _bary_to_cart(bary)

def _objective_bary(x):
    """Objective in barycentric coordinates: negative min triangle area.
    Barycentric coords (u,v) with u>=0, v>=0, u+v<=1 naturally constrain to triangle."""
    bary = x.reshape(N, 2)
    # Clamp to valid region
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_triangle_area_vectorized(points)

def _softmin_objective(x, beta=50.0):
    """Smooth surrogate using soft-min (log-sum-exp) for the min triangle area."""
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    # Soft-min: -log(sum(exp(-beta * areas))) / beta
    # This is differentiable and approximates min(areas) as beta -> inf
    max_a = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a))))
    soft_min = max_a - log_sum / beta
    return -soft_min

def _structured_grid_init(seed=42):
    """Generate diverse structured initial configurations using grid-based approach with jitter."""
    rng = np.random.RandomState(seed)
    
    # Base hexagonal/symmetric patterns
    base_patterns = [
        # Pattern A: Vertices + edge midpoints + centroid + interior
        np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
            [1/3, 1/3],
            [0.2, 0.1], [0.8, 0.1], [0.1, 0.7], [0.5, 0.3]
        ]),
        # Pattern B: Hexagonal-ish
        np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
            [0.33, 0.33],
            [0.15, 0.15], [0.65, 0.15], [0.15, 0.65], [0.5, 0.2]
        ]),
        # Pattern C: More spread out
        np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.4, 0.0], [0.6, 0.4], [0.0, 0.4],
            [0.33, 0.33],
            [0.1, 0.05], [0.9, 0.05], [0.05, 0.9], [0.5, 0.15]
        ]),
    ]
    
    # Choose pattern based on seed
    pattern_idx = seed % len(base_patterns)
    bary = base_patterns[pattern_idx].copy()
    
    # Add structured jitter
    jitter_scale = 0.02 + 0.01 * (seed % 5)
    jitter = rng.normal(0, jitter_scale, size=bary.shape)
    bary = bary + jitter
    bary = _clamp_bary(bary)
    
    return bary

def _random_init_bary(seed=42):
    """Generate random initial configuration in barycentric coords."""
    rng = np.random.RandomState(seed)
    bary = rng.uniform(0, 1, size=(N, 2))
    bary = _clamp_bary(bary)
    return bary

def _perturb_init_bary(bary_base, seed=42, scale=0.05):
    """Generate perturbed initial configuration from a base configuration."""
    rng = np.random.RandomState(seed)
    bary = bary_base + rng.normal(0, scale, size=bary_base.shape)
    bary = _clamp_bary(bary)
    return bary

def _l_bfgs_optimize_bary(bary0, maxiter=500, use_softmin=False, beta=50.0):
    """Run L-BFGS-B optimization in barycentric coordinates."""
    try:
        x0 = bary0.flatten()
        
        # Bounds: u in [0,1], v in [0,1] (u+v<=1 handled by clamping)
        bounds = [(0, 1)] * (2 * N)
        
        if use_softmin:
            obj_func = lambda x: _softmin_objective(x, beta=beta)
        else:
            obj_func = _objective_bary
            
        res = minimize(
            obj_func, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _nelder_mead_optimize_bary(bary0, maxiter=200, use_softmin=False, beta=50.0):
    """Run Nelder-Mead optimization in barycentric coordinates."""
    try:
        x0 = bary0.flatten()
        
        if use_softmin:
            obj_func = lambda x: _softmin_objective(x, beta=beta)
        else:
            obj_func = _objective_bary
            
        res = minimize(
            obj_func, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'fatol': 1e-12, 'xatol': 1e-12, 'adaptive': True}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _multi_start_nelder_mead(bary_starts, maxiter=300, use_softmin=False, beta=50.0):
    """Run multi-start Nelder-Mead optimization from multiple starting points."""
    best_bary = None
    best_score = -np.inf
    
    for bary0 in bary_starts:
        bary_opt = _nelder_mead_optimize_bary(bary0, maxiter=maxiter, use_softmin=use_softmin, beta=beta)
        points = _bary_to_cart(bary_opt)
        score = _min_triangle_area_vectorized(points)
        
        if score > best_score:
            best_score = score
            best_bary = bary_opt.copy()
    
    return best_bary, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    best_bary = None
    best_score = -np.inf
    
    def time_left():
        return deadline - time.time()
    
    def score_bary(bary):
        points = _bary_to_cart(bary)
        return _min_triangle_area_vectorized(points)
    
    # Phase 1: Generate diverse starting points using grid + random + perturbations
    start_points = []
    
    # Structured grid starts
    grid_seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    for seed in grid_seeds:
        if time_left() < 5.0:
            break
        bary_grid = _structured_grid_init(seed=seed)
        start_points.append(bary_grid)
        
        # Also add random starts
        if len(start_points) % 3 == 0:
            bary_rand = _random_init_bary(seed=seed)
            start_points.append(bary_rand)
    
    # Phase 2: Multi-start Nelder-Mead with soft-min surrogate
    if time_left() > 10.0:
        # Use soft-min for smoother optimization
        nm_bary, nm_score = _multi_start_nelder_mead(
            start_points, maxiter=200, use_softmin=True, beta=50.0
        )
        
        if nm_bary is not None and nm_score > best_score:
            best_score = nm_score
            best_bary = nm_bary.copy()
    
    # Phase 3: Refine best solution with L-BFGS-B using exact objective
    if best_bary is not None and time_left() > 5.0:
        bary_refined = _l_bfgs_optimize_bary(best_bary, maxiter=500, use_softmin=False)
        score_refined = score_bary(bary_refined)
        if score_refined > best_score:
            best_score = score_refined
            best_bary = bary_refined.copy()
    
    # Phase 4: Multi-start with perturbations around best solution
    if best_bary is not None and time_left() > 10.0:
        rng = np.random.RandomState(123)
        perturbed_starts = []
        
        # Generate perturbed starts with varying scales
        for scale in [0.05, 0.03, 0.01]:
            for i in range(3):
                bary_pert = _perturb_init_bary(best_bary, seed=123 + i * 10 + int(scale * 100), scale=scale)
                perturbed_starts.append(bary_pert)
        
        # Run multi-start Nelder-Mead on perturbed starts
        if time_left() > 5.0:
            nm_bary2, nm_score2 = _multi_start_nelder_mead(
                perturbed_starts, maxiter=150, use_softmin=False
            )
            
            if nm_bary2 is not None and nm_score2 > best_score:
                best_score = nm_score2
                best_bary = nm_bary2.copy()
                
                # Final refinement
                if time_left() > 3.0:
                    bary_final = _l_bfgs_optimize_bary(best_bary, maxiter=300, use_softmin=False)
                    score_final = score_bary(bary_final)
                    if score_final > best_score:
                        best_score = score_final
                        best_bary = bary_final.copy()
    
    # Phase 5: Fine-tuning with very small perturbations
    if best_bary is not None and time_left() > 2.0:
        rng = np.random.RandomState(456)
        for ft in range(5):
            if time_left() < 1.0:
                break
            pert = rng.normal(0, 0.005, size=(N, 2))
            bary_ft = _clamp_bary(best_bary + pert)
            bary_opt = _l_bfgs_optimize_bary(bary_ft, maxiter=200, use_softmin=False)
            score_opt = score_bary(bary_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_bary = bary_opt.copy()
    
    # Final fallback
    if best_bary is None:
        best_bary = _structured_grid_init(seed=42)
    
    return _bary_to_cart(best_bary)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
