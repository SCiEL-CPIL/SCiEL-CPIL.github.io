import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats.qmc import Sobol, Halton

N = 100
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        # Allow 145s out of 150s to be safe
        _DEADLINE = time.time() + 145.0
    return _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta, iu):
    """
    Objective function for L-BFGS-B.
    Minimizes the negative soft-min approximation of the radius.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances: shape (N, 3) -> (3N,)
    wall_dists = 0.5 - np.abs(centers)
    wall_vals = wall_dists.ravel()
    
    # 2. Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    pair_dists = dists[iu] / 2.0
    
    # Combine all constraint values
    all_vals = np.concatenate([wall_vals, pair_dists])
    
    # Log-sum-exp trick for numerical stability
    min_val = np.min(all_vals)
    shifted = -beta * (all_vals - min_val)
    sum_exp = np.sum(np.exp(shifted))
    
    # soft_min = min_val - (1/beta) * log(sum_exp)
    soft_min_val = min_val - (1.0 / beta) * np.log(sum_exp)
    
    # We want to maximize soft_min_val, so we minimize -soft_min_val
    return -soft_min_val

def _init_grid():
    """
    Generate a grid-like initialization.
    """
    nx, ny, nz = 5, 5, 4
    points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cx = -0.5 + (i + 0.5) * (1.0 / nx)
                cy = -0.5 + (j + 0.5) * (1.0 / ny)
                cz = -0.5 + (k + 0.5) * (1.0 / nz)
                points.append([cx, cy, cz])
    
    centers = np.array(points[:N])
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _init_bcc():
    """
    Build a BCC (body-centered cubic) lattice initialization.
    """
    nx, ny, nz = 5, 5, 4
    points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cx = -0.5 + (i + 0.5) * (1.0 / nx)
                cy = -0.5 + (j + 0.5) * (1.0 / ny)
                cz = -0.5 + (k + 0.5) * (1.0 / nz)
                points.append([cx, cy, cz])
    
    centers = np.array(points[:N])
    rng = np.random.default_rng(123)
    centers += rng.normal(0, 0.02, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _init_sobol():
    """
    Generate N points using a Sobol sequence.
    """
    sampler = Sobol(d=3, scramble=True, seed=42)
    points = sampler.random(N)
    centers = points * 1.0 - 0.5
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _init_halton():
    """
    Generate N points using a Halton sequence.
    """
    sampler = Halton(d=3, scramble=True, seed=42)
    points = sampler.random(N)
    centers = points * 1.0 - 0.5
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _polish_l_bfgs(centers, time_budget=10.0, iu=None):
    """
    Local search using L-BFGS-B with soft-min objective and increasing beta.
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    if iu is None:
        iu = np.triu_indices(N, k=1)
    
    start_time = time.time()
    
    # Beta schedule
    betas = [50, 200, 1000, 5000]
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    for beta in betas:
        if _time_left() < 2.0 or (time.time() - start_time) > time_budget:
            break
            
        def obj(x):
            return _soft_min_objective(x, beta, iu)
        
        # Adjust maxiter based on beta
        if beta < 100:
            max_iter = 100
            ftol = 1e-10
            gtol = 1e-6
        elif beta < 1000:
            max_iter = 150
            ftol = 1e-12
            gtol = 1e-8
        else:
            max_iter = 200
            ftol = 1e-14
            gtol = 1e-10
            
        try:
            res = minimize(
                obj, 
                centers.ravel(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
            )
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            
            r_current = _compute_radius(new_centers)
            
            if r_current > best_r + 1e-12:
                best_r = r_current
                best_centers = new_centers.copy()
                centers = new_centers.copy()
                
        except Exception as e:
            #print(e)
            break
            
    return best_centers, best_r

def _simulated_annealing(centers, time_budget, iu):
    """
    Simulated Annealing to find a better configuration.
    """
    centers = centers.copy()
    current_r = _compute_radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Initial temperature
    T = 0.05
    # Cooling rate
    alpha = 0.9995
    
    # We will run until time runs out
    while _time_left() > 1.0 and (time.time() - start_time) < time_budget:
        # Pick a random sphere to move
        idx = rng.integers(0, N)
        
        # Generate a random move
        # Step size should be adaptive or fixed? Fixed is simpler.
        # Let's use a step size that is a fraction of the current radius or a fixed small value.
        # Since we don't know the optimal step size, let's try a few different magnitudes.
        # For simplicity, let's use a fixed small step size scaled by T? No, T is for acceptance.
        # Let's use a step size of 0.01 * T / T_initial? No.
        # Let's just use a random step in a box of size 0.05.
        step = rng.uniform(-0.02, 0.02, 3)
        
        new_centers = centers.copy()
        new_centers[idx] += step
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        
        new_r = _compute_radius(new_centers)
        
        # Acceptance criterion
        delta_r = new_r - current_r
        if delta_r > 0:
            # Better, accept
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            # Worse, accept with probability exp(delta_r / T)
            # If T is small, this is very unlikely.
            # If delta_r is very negative, exp is 0.
            # We need to be careful with overflow/underflow.
            # delta_r is negative.
            # prob = exp(delta_r / T)
            # If delta_r / T < -100, prob is 0.
            if delta_r / T > -100:
                prob = np.exp(delta_r / T)
            else:
                prob = 0.0
                
            if rng.random() < prob:
                centers = new_centers
                current_r = new_r
        
        # Cool down
        T *= alpha
        
        # If T is too small, we might be stuck. We could restart with a higher T?
        # For now, just keep going.
        
    return best_centers, best_r

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)
    
    best_centers = None
    best_radius = 0.0
    
    # List of initializers
    inits = [
        ("grid", _init_grid),
        ("bcc", _init_bcc),
        ("sobol", _init_sobol),
        ("halton", _init_halton),
    ]
    
    # Time allocation
    # We have ~145s.
    # We will run SA for each init, then polish the best.
    # Let's give each SA run ~20s, and then polish the best one with the remaining time.
    
    for name, init_func in inits:
        if _time_left() < 25.0:
            break
            
        centers = init_func()
        r_init = _compute_radius(centers)
        
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
            
        # Run Simulated Annealing
        sa_budget = min(20.0, _time_left() - 10.0)
        if sa_budget > 5.0:
            sa_centers, sa_r = _simulated_annealing(centers, sa_budget, iu)
            if sa_r > best_radius:
                best_radius = sa_r
                best_centers = sa_centers.copy()
                
    # Final polish with all remaining time on the best candidate found so far
    if best_centers is not None and _time_left() > 10.0:
        final_budget = _time_left() - 2.0
        polished_centers, polished_r = _polish_l_bfgs(best_centers, time_budget=final_budget, iu=iu)
        if polished_r > best_radius:
            best_radius = polished_r
            best_centers = polished_centers.copy()
            
    if best_centers is None:
        best_centers = _init_grid()
        
    # Final validation and clipping
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
