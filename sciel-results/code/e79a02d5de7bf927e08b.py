"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Check container constraint
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists_origin > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0
        
    if N < 2:
        return r_cont
        
    # Correct pairwise distance computation
    # Using broadcasting to get full distance matrix
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    r_pair = 0.5 * np.min(dists)
    
    return min(r_cont, r_pair)


def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess."""
    best_pts = None
    best_score = 0.0
    # Try a range of spacings to find the one that fits N=50 best
    for spacing in np.linspace(0.15, 0.35, 80):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Keep points that can fit a circle of radius spacing/2 inside unit circle
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            # Select the N closest to the center to form a compact core
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind="stable")[:N]
            pts = points[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    if best_pts is not None:
        return best_pts
    # Fallback: random points in unit disk
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.81, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _smooth_objective(x, beta=0.01):
    """
    Smooth objective function for gradient-based polishing.
    Uses log-sum-exp to approximate the minimum distance smoothly.
    """
    pts = x.reshape(N, 2)
    
    # Boundary penalty
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    boundary_penalty = np.sum(np.maximum(0, dists_origin - 1.0) ** 2) * 10.0
    
    r_cont = 1.0 - np.max(dists_origin)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    
    finite_dists = dists[np.isfinite(dists)]
    if len(finite_dists) == 0:
        return -r_cont + boundary_penalty
        
    # Soft min approximation
    # We want to maximize min(dists). 
    # Minimize -soft_min(dists).
    # soft_min(d) = -beta * log(sum(exp(-d/beta)))
    val = -finite_dists / beta
    max_val = np.max(val)
    log_sum_exp = max_val + np.log(np.sum(np.exp(val - max_val)))
    soft_min_val = -beta * log_sum_exp
    
    r_pair_soft = 0.5 * soft_min_val
    r_eff = min(r_cont, r_pair_soft)
    
    return -r_eff + boundary_penalty


def _polish(pts0, time_limit):
    """
    Polish the solution using L-BFGS-B with a smooth objective.
    """
    pts = pts0.copy()
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # Try a few betas to refine
    betas = [0.05, 0.02, 0.01, 0.005]
    
    for beta in betas:
        if time.time() > time_limit:
            break
            
        def obj(x):
            return _smooth_objective(x, beta)
            
        try:
            res = minimize(
                obj,
                pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            new_pts = res.x.reshape(N, 2)
            
            # Project back to unit disk
            radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask = radii > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / radii[mask, np.newaxis]
                
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                pts = new_pts.copy()
            # If not improved, we might still want to update pts if it's closer to valid?
            # But for now, stick to best known.
        except Exception:
            pass
            
    return best_pts


def _optimize_de(pts0, time_limit):
    """
    Use Differential Evolution for global search.
    """
    from scipy.optimize import differential_evolution
    
    # Define bounds: each coordinate must be in [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    def de_objective(x):
        pts = x.reshape(N, 2)
        # Quick validity check for DE (which doesn't handle constraints natively well)
        dists_origin = np.hypot(pts[:, 0], pts[:, 1])
        if np.any(dists_origin > 1.0):
            return 10.0 + np.sum((dists_origin[dists_origin > 1.0] - 1.0) ** 2)
        
        r_cont = 1.0 - np.max(dists_origin)
        
        # Compute min pairwise distance
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        r_pair = 0.5 * np.min(dists)
        
        return -min(r_cont, r_pair)
    
    def callback(xk, convergence):
        if time.time() > time_limit:
            return True
        return False
    
    try:
        # DE is stochastic, so we set seed for reproducibility if needed, but default is fine
        # popsize=15 is a good default, maxiter=100 might be too high for time limit
        # We rely on callback to stop early
        res = differential_evolution(
            de_objective,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            callback=callback,
            seed=42
        )
        
        best_pts = res.x.reshape(N, 2)
        
        # Project back to unit disk
        radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            best_pts[mask] = best_pts[mask] / radii[mask, np.newaxis]
            
        return best_pts
    except Exception:
        return pts0


# Removed _repulsion_optimize, replaced by _optimize_de


def construct():
    start_time = time.time()
    
    # 1. Initialize with hexagonal lattice
    pts0 = _hex_lattice_init()
    best_pts = pts0
    best_score = _score(pts0)
    
    # 2. Global Optimization with Differential Evolution
    # DE is robust for this type of problem
    de_deadline = _DEADLINE - 30.0  # Leave 30s for polishing and safety
    if time.time() < de_deadline:
        try:
            de_pts = _optimize_de(best_pts, de_deadline)
            de_score = _score(de_pts)
            if de_score > best_score:
                best_pts = de_pts
                best_score = de_score
        except Exception:
            pass
    
    # 3. Local Polish
    polish_deadline = _DEADLINE - 5.0
    if time.time() < polish_deadline:
        try:
            polished_pts = _polish(best_pts, polish_deadline)
            polished_score = _score(polished_pts)
            if polished_score > best_score:
                best_pts = polished_pts
                best_score = polished_score
        except Exception:
            pass
    
    # Final safety check
    final_score = _score(best_pts)
    if final_score <= 0:
        return pts0
    
    return best_pts

# EVOLVE-BLOCK-END
