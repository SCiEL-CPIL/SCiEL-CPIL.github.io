"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _pairwise_distances(pts):
    """Compute all pairwise distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))

def _score(pts):
    """Compute (max/min)^2 for pairwise distances."""
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    dmax = d.max()
    if dmin <= 1e-12:
        return np.inf
    return (dmax / dmin) ** 2

def _objective(pts_flat):
    """Objective function for minimization: (max/min)^2."""
    pts = pts_flat.reshape(N, 2)
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    dmax = d.max()
    if dmin <= 1e-12:
        return 1e10
    return (dmax / dmin) ** 2

def make_hexagonal_cluster(n, spread=5):
    """
    Generate a hexagonal lattice cluster of n points centered at origin.
    This provides a very good initial guess as optimal 2D packings are subsets of hex lattices.
    """
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    
    pts = []
    for i in range(-spread, spread + 1):
        for j in range(-spread, spread + 1):
            p = i * v1 + j * v2
            pts.append(p)
    pts = np.array(pts)
    
    # Center the cluster
    pts -= pts.mean(axis=0)
    
    # Sort by distance to center and pick the closest n
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    return pts[order[:n]]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_pts = None
    best_score = np.inf
    
    # 1. Generate initial candidates
    # Candidate A: Perfect hexagonal cluster
    hex_pts = make_hexagonal_cluster(N)
    
    # Candidate B: Perturbed hexagonal cluster (break symmetry)
    np.random.seed(42)
    perturbed_hex = hex_pts + np.random.randn(N, 2) * 0.1
    
    # Candidate C: Stretched hexagonal cluster
    stretched_hex = hex_pts.copy()
    stretched_hex[:, 0] *= 1.2
    stretched_hex[:, 1] *= 0.8
    
    # Candidate D: Square grid (fallback, though usually worse)
    x, y = np.meshgrid(np.arange(4), np.arange(4))
    sq_pts = np.column_stack([x.ravel(), y.ravel()]).astype(float)
    sq_pts -= sq_pts.mean(axis=0)
    
    candidates = [hex_pts, perturbed_hex, stretched_hex, sq_pts]
    
    # Evaluate initial scores
    for c in candidates:
        s = _score(c)
        if s < best_score:
            best_score = s
            best_pts = c.copy()
    
    # 2. Local Optimization
    # Powell is robust for non-smooth objectives like min/max ratios
    # We run it on the best candidates
    
    def objective_flat(flat_pts):
        return _objective(flat_pts)
    
    # Prioritize the hexagonal cluster and its perturbations
    opt_candidates = [hex_pts, perturbed_hex, stretched_hex]
    
    for init_pts in opt_candidates:
        if time.time() > deadline - 10:
            break
            
        try:
            # Powell method
            result = minimize(
                objective_flat,
                init_pts.ravel(),
                method='Powell',
                options={'maxiter': 1000, 'maxfev': 20000, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                new_pts = result.x.reshape(N, 2)
                new_score = _score(new_pts)
                if new_score < best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
        except Exception:
            pass
            
    # 3. Refinement with Nelder-Mead on the best result found
    if time.time() < deadline - 5:
        try:
            # Nelder-Mead can sometimes polish the result further
            result = minimize(
                objective_flat,
                best_pts.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_score:
                new_pts = result.x.reshape(N, 2)
                new_score = _score(new_pts)
                if new_score < best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
        except Exception:
            pass
            
    # 4. Final check and return
    if best_pts is None:
        # Fallback to hexagonal cluster if something went wrong
        return make_hexagonal_cluster(N)
        
    return best_pts
# EVOLVE-BLOCK-END
