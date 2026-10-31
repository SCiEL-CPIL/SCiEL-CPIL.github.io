import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
import time
from scipy.special import logsumexp

N = 16
D = 2

def _pairwise_dists_sq(pts):
    """Compute all pairwise squared distances."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    iu = np.triu_indices(n, k=1)
    return dists_sq[iu]

def _score(pts):
    """Compute (max/min)^2."""
    d2 = _pairwise_dists_sq(pts)
    dmin2 = d2.min()
    dmax2 = d2.max()
    if dmin2 < 1e-12:
        return 1e10
    return dmax2 / dmin2

def _soft_score(pts, beta=10.0):
    """Smooth surrogate using log-sum-exp for min and max squared distances."""
    d2 = _pairwise_dists_sq(pts)
    # Soft max of d2
    d_max_soft = logsumexp(beta * d2) / beta
    # Soft min of d2
    d_min_soft = -logsumexp(-beta * d2) / beta
    if d_min_soft < 1e-12:
        return 1e10
    return d_max_soft / d_min_soft

def _hex_init():
    """Hexagonal lattice initialization for 16 points."""
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # Ring 1: 6 points
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([np.cos(angle), np.sin(angle)])
    # Ring 2: 9 points (to reach 16)
    for i in range(9):
        angle = i * 2 * np.pi / 9
        pts.append([2.0*np.cos(angle), 2.0*np.sin(angle)])
    return np.array(pts[:16], dtype=float)

def _grid_init():
    """4x4 grid initialization."""
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([i, j])
    return np.array(pts, dtype=float) - 1.5

def _random_init(seed):
    rng = np.random.RandomState(seed)
    return rng.rand(N, D) * 4 - 2

def construct():
    """Main entry point."""
    deadline = time.time() + 145  # Safety margin
    
    best_score = float('inf')
    best_pts = None
    
    # Initializations
    inits = [
        _hex_init(),
        _grid_init(),
        _random_init(1),
        _random_init(2),
        _random_init(3),
    ]
    
    # 1. Global Search with Differential Evolution
    # We optimize a smooth objective to avoid non-differentiability issues
    # DE works on flattened arrays
    bounds = [(-5, 5)] * (N * D)
    
    def de_objective(x):
        pts = x.reshape(N, D)
        return _soft_score(pts, beta=5.0)
    
    try:
        # Run DE with moderate population size
        result_de = differential_evolution(
            de_objective, bounds, 
            maxiter=50, popsize=10, 
            tol=1e-4, seed=42,
            polish=False # We will polish later
        )
        if time.time() < deadline - 20:
            pts_de = result_de.x.reshape(N, D)
            score_de = _score(pts_de)
            if score_de < best_score:
                best_score = score_de
                best_pts = pts_de.copy()
    except Exception:
        pass
    
    # 2. Basin Hopping on top of best candidate
    if best_pts is None:
        best_pts = _hex_init()
        
    if time.time() < deadline - 10:
        def bh_objective(x):
            pts = x.reshape(N, D)
            return _soft_score(pts, beta=10.0)
        
        def bh_minimizer(x):
            # Use L-BFGS-B for local minimization within basin hopping
            res = minimize(bh_objective, x, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-12})
            return res.fun, res.x, {}
        
        try:
            result_bh = basinhopping(
                bh_objective, best_pts.flatten(), 
                minimizer=bh_minimizer,
                niter=20, # Number of basin hops
                T=1.0, # Temperature
                seed=42
            )
            if time.time() < deadline - 5:
                pts_bh = result_bh.x.reshape(N, D)
                score_bh = _score(pts_bh)
                if score_bh < best_score:
                    best_score = score_bh
                    best_pts = pts_bh.copy()
        except Exception:
            pass
            
    # 3. Final Polish with Exact Objective
    if time.time() < deadline - 2:
        def exact_obj(x):
            pts = x.reshape(N, D)
            return _score(pts)
        
        try:
            # Try multiple local optimizers from the best point
            for method in ['L-BFGS-B', 'Powell']:
                if time.time() >= deadline - 2:
                    break
                res = minimize(exact_obj, best_pts.flatten(), method=method,
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                pts_pol = res.x.reshape(N, D)
                score_pol = _score(pts_pol)
                if score_pol < best_score:
                    best_score = score_pol
                    best_pts = pts_pol.copy()
        except Exception:
            pass

    return best_pts
