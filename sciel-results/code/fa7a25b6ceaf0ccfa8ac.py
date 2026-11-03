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

def _softmin_objective(x, beta=50.0):
    """Soft-min surrogate for the non-differentiable minimum triangle area.
    
    Uses log-sum-exp trick: -softmin(areas) ≈ -(1/beta) * log(sum(exp(-beta * areas)))
    This provides smooth gradients that guide optimization toward the minimum.
    
    As beta increases, this approaches the true min. For beta=50, it's a good balance
    between smoothness and accuracy.
    """
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Soft-min via log-sum-exp (numerically stable version)
    neg_areas = -areas
    max_neg = np.max(neg_areas)
    softmin = -max_neg - (1.0 / beta) * np.log(np.sum(np.exp(neg_areas - max_neg)))
    
    return -softmin

def _objective_bary(x):
    """Objective in barycentric coordinates: negative min triangle area.
    Uses soft-min surrogate for smooth gradients."""
    return _softmin_objective(x, beta=50.0)

def _structured_init_bary(seed=42):
    """Generate diverse structured initial configurations in barycentric coords."""
    rng = np.random.RandomState(seed)
    
    # Pattern 1: Vertices + edge midpoints + centroid + interior
    patterns = []
    
    # Pattern A: Symmetric grid
    pA = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [1/3, 1/3],
        [0.2, 0.1], [0.8, 0.1], [0.1, 0.7], [0.5, 0.3]
    ])
    patterns.append(pA)
    
    # Pattern B: Hexagonal-ish
    pB = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [0.33, 0.33],
        [0.15, 0.15], [0.65, 0.15], [0.15, 0.65], [0.5, 0.2]
    ])
    patterns.append(pB)
    
    # Pattern C: More spread out
    pC = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.4, 0.0], [0.6, 0.4], [0.0, 0.4],
        [0.33, 0.33],
        [0.1, 0.05], [0.9, 0.05], [0.05, 0.9], [0.5, 0.15]
    ])
    patterns.append(pC)
    
    # Pattern D: Random-ish structured
    pD = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [rng.uniform(0.3, 0.7), 0.0],
        [rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)],
        [0.0, rng.uniform(0.3, 0.7)],
        [1/3, 1/3],
        [rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3)],
        [rng.uniform(0.7, 0.9), rng.uniform(0.1, 0.3)],
        [rng.uniform(0.1, 0.3), rng.uniform(0.7, 0.9)],
        [rng.uniform(0.3, 0.7), rng.uniform(0.2, 0.5)]
    ])
    patterns.append(pD)
    
    # Choose a pattern and add small jitter
    idx = seed % len(patterns)
    bary = patterns[idx].copy()
    
    # Add small jitter
    jitter = rng.normal(0, 0.02, size=bary.shape)
    bary = bary + jitter
    bary = _clamp_bary(bary)
    
    return bary

def _l_bfgs_optimize_bary(bary0, maxiter=500, beta=50.0):
    """Run L-BFGS-B optimization in barycentric coordinates with soft-min objective."""
    try:
        x0 = bary0.flatten()
        
        # Bounds: u in [0,1], v in [0,1] (u+v<=1 handled by clamping)
        bounds = [(0, 1)] * (2 * N)
        
        def obj(x):
            return _softmin_objective(x, beta=beta)
        
        res = minimize(
            obj, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _polish_bary(bary0, maxiter=2000, beta_start=200.0, beta_end=2000.0):
    """High-precision polish pass with increasing beta for soft-min.
    
    Start with moderate beta for smooth gradients, then increase beta
    to approach the true minimum more closely.
    """
    bary = bary0.copy()
    n_stages = 5
    for stage in range(n_stages):
        # Interpolate beta from beta_start to beta_end
        t = stage / max(1, n_stages - 1)
        beta = beta_start * (beta_end / beta_start) ** t
        
        # Each stage: run L-BFGS-B with current beta
        bary = _l_bfgs_optimize_bary(bary, maxiter=maxiter // n_stages + 100, beta=beta)
    
    return bary

def _nelder_mead_optimize_bary(bary0, maxiter=200):
    """Run Nelder-Mead optimization in barycentric coordinates."""
    try:
        x0 = bary0.flatten()
        res = minimize(
            _objective_bary, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'fatol': 1e-12, 'xatol': 1e-12, 'adaptive': True}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _differential_evolution_global(seed=42, maxiter=50, popsize=15):
    """Run differential evolution for global optimization."""
    try:
        rng = np.random.RandomState(seed)
        
        # Bounds for barycentric coords
        bounds = [(0, 1)] * (2 * N)
        
        # Use a smaller population and fewer iterations for speed
        de_result = differential_evolution(
            _objective_bary,
            bounds,
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-10,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            updating='deferred'
        )
        
        bary_opt = _clamp_bary(de_result.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return None

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
    
    # Phase 1: Try multiple structured initializations with local optimization
    initial_seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for seed in initial_seeds:
        if time_left() < 3.0:
            break
        
        bary_init = _structured_init_bary(seed=seed)
        score_init = score_bary(bary_init)
        
        # Local optimization with L-BFGS-B using soft-min (faster convergence)
        if time_left() > 2.0:
            bary_opt = _l_bfgs_optimize_bary(bary_init, maxiter=400, beta=50.0)
            score_opt = score_bary(bary_opt)
            if score_opt > score_init:
                score_init = score_opt
                bary_init = bary_opt
        
        if score_init > best_score:
            best_score = score_init
            best_bary = bary_init.copy()
    
    # Phase 2: Differential evolution for global search
    if time_left() > 10.0:
        de_bary = _differential_evolution_global(seed=42, maxiter=30, popsize=12)
        if de_bary is not None:
            score_de = score_bary(de_bary)
            if score_de > best_score:
                best_score = score_de
                best_bary = de_bary.copy()
                
                # Refine with local optimization
                if time_left() > 5.0:
                    bary_refined = _l_bfgs_optimize_bary(de_bary, maxiter=400, beta=50.0)
                    score_refined = score_bary(bary_refined)
                    if score_refined > best_score:
                        best_score = score_refined
                        best_bary = bary_refined.copy()
    
    # Phase 3: Multi-start local optimization with perturbations
    rng = np.random.RandomState(123)
    restart_count = 0
    max_restarts = 50
    
    while restart_count < max_restarts and time_left() > 3.0:
        restart_count += 1
        
        # Perturb the best solution
        pert = rng.normal(0, 0.03, size=(N, 2))
        bary_start = _clamp_bary(best_bary + pert)
        
        # Local optimization with soft-min
        bary_opt = _l_bfgs_optimize_bary(bary_start, maxiter=300, beta=50.0)
        score_opt = score_bary(bary_opt)
        
        if score_opt > best_score:
            best_score = score_opt
            best_bary = bary_opt.copy()
    
    # Phase 4: High-precision polish with increasing beta
    if time_left() > 5.0:
        bary_polished = _polish_bary(best_bary, maxiter=1500, beta_start=200.0, beta_end=2000.0)
        score_polished = score_bary(bary_polished)
        if score_polished > best_score:
            best_score = score_polished
            best_bary = bary_polished.copy()
    
    # Phase 5: Fine-tuning with very small perturbations and final polish
    if time_left() > 3.0:
        for ft in range(5):
            if time_left() < 2.0:
                break
            pert = rng.normal(0, 0.005, size=(N, 2))
            bary_ft = _clamp_bary(best_bary + pert)
            bary_opt = _l_bfgs_optimize_bary(bary_ft, maxiter=200, beta=100.0)
            score_opt = score_bary(bary_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_bary = bary_opt.copy()
        
        # Final high-precision polish
        if time_left() > 2.0:
            bary_final = _polish_bary(best_bary, maxiter=1000, beta_start=500.0, beta_end=5000.0)
            score_final = score_bary(bary_final)
            if score_final > best_score:
                best_score = score_final
                best_bary = bary_final.copy()
    
    # Final fallback
    if best_bary is None:
        best_bary = _structured_init_bary(seed=42)
    
    return _bary_to_cart(best_bary)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
