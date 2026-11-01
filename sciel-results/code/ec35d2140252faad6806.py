import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import sys

N = 14
D = 3
DEADLINE = 145.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to min pairwise distance.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _pwl_obj(pts):
    """
    Objective function for Nelder-Mead.
    Uses exact max/min to preserve the kinks of the true landscape.
    Returns the squared ratio.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return 1e10
    return (mx / mn) ** 2

def _generate_initial_points(seed):
    """
    Generate diverse initial configurations.
    1. Fibonacci sphere
    2. Icosahedron + 2 random
    3. Random Gaussian
    """
    rng = np.random.default_rng(seed)
    inits = []
    
    # 1. Fibonacci Sphere
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    pts_fib = np.column_stack([x, y, z])
    pts_fib = pts_fib - pts_fib.mean(axis=0)
    inits.append(pts_fib)
    
    # 2. Icosahedron vertices (12) + 2 points
    phi_gold = (1 + np.sqrt(5)) / 2
    ico_pts = np.array([
        [0, 1, phi_gold], [0, -1, phi_gold],
        [0, 1, -phi_gold], [0, -1, -phi_gold],
        [1, phi_gold, 0], [-1, phi_gold, 0],
        [1, -phi_gold, 0], [-1, -phi_gold, 0],
        [phi_gold, 0, 1], [-phi_gold, 0, 1],
        [phi_gold, 0, -1], [-phi_gold, 0, -1]
    ])
    ico_pts = ico_pts / np.linalg.norm(ico_pts, axis=1)[:, None]
    
    # Add 2 points at "holes" or random
    extra = rng.normal(size=(2, 3))
    extra = extra / np.linalg.norm(extra, axis=1)[:, None]
    pts_ico = np.vstack([ico_pts, extra])
    pts_ico = pts_ico - pts_ico.mean(axis=0)
    inits.append(pts_ico)
    
    # 3. Random
    pts_rand = rng.normal(size=(N, 3))
    pts_rand = pts_rand - pts_rand.mean(axis=0)
    inits.append(pts_rand)
    
    return inits

def _optimize_single(pts_init, time_limit):
    """
    Run Nelder-Mead optimization on a single initial point.
    """
    start_time = time.time()
    try:
        res = minimize(
            _pwl_obj,
            pts_init,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        return res.x
    except Exception as e:
        # print(f"Optimization failed: {e}", file=sys.stderr)
        return pts_init

def _local_refine(pts, time_limit):
    """
    Perform a few rounds of stochastic local search (perturbation + Nelder-Mead)
    to escape local minima.
    """
    best_pts = pts.copy()
    best_score = _pwl_obj(pts)
    rng = np.random.default_rng(int(time.time() * 1000) % 100000)
    
    # Number of refinement steps depends on remaining time
    # Nelder-Mead is relatively fast for N=14, D=3.
    # Let's try 5-10 perturbations if time permits.
    
    while time.time() - _t0 < time_limit - 10:
        # Perturb the best point found so far
        # Scale of perturbation: small enough to stay in basin, large enough to jump out
        # Typical distance scale is ~1.0 for normalized points.
        scale = 0.05
        perturbed = best_pts + rng.normal(scale=scale, size=best_pts.shape)
        
        # Optimize the perturbed point
        refined = _optimize_single(perturbed, time.time() - _t0)
        score = _pwl_obj(refined)
        
        if score < best_score:
            best_score = score
            best_pts = refined.copy()
        else:
            # If no improvement after a few tries, we might be stuck, 
            # but let's try a few more with different scales if time permits
            # For now, just continue the loop which uses the same best_pts
            pass
            
    return best_pts

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    best_pts = None
    best_score = np.inf
    
    # Generate initial candidates
    # Use a few different seeds for diversity
    seeds = [42, 123, 456, 789, 101]
    
    # We have ~145s. 
    # Nelder-Mead on 42 variables (14*3) is fast. 
    # Let's run multiple initializations.
    
    for seed in seeds:
        if time.time() - _t0 > DEADLINE - 30:
            break
            
        inits = _generate_initial_points(seed)
        
        for init_idx, pts_init in enumerate(inits):
            if time.time() - _t0 > DEADLINE - 20:
                break
                
            # Initial optimization
            pts_opt = _optimize_single(pts_init, time.time() - _t0)
            score_opt = _pwl_obj(pts_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_pts = pts_opt.copy()
                
            # If this is a promising start, do local refinement
            # Heuristic: if score is better than a threshold, refine more
            if score_opt < 2.5 and time.time() - _t0 < DEADLINE - 15:
                pts_refined = _local_refine(pts_opt, time.time() - _t0)
                score_refined = _pwl_obj(pts_refined)
                if score_refined < best_score:
                    best_score = score_refined
                    best_pts = pts_refined.copy()
                    
    # Fallback
    if best_pts is None:
        # Return Fibonacci sphere
        i = np.arange(N)
        phi = np.arccos(1 - 2 * (i + 0.5) / N)
        theta = np.pi * (1 + 5 ** 0.5) * i
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        best_pts = np.column_stack([x, y, z])
        
    return best_pts
