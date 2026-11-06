"""Improved packing of 50 equal spheres in a unit cube."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE_MARGIN = 5.0  # seconds before the 150s limit

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check inside cube
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    # Minimum distance to boundary
    dist_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(dist_boundary)
    # Minimum pairwise distance
    n = len(centers)
    min_pair = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair:
                min_pair = d
    if min_pair == 0.0:
        return 0.0
    return min(min_pair / 2.0, min_boundary)

def _soft_min(values, beta=100.0):
    """Soft minimum using log-sum-exp for smoothness."""
    # values is a 1D array of distances we want to minimize
    # We want a smooth approximation of min(values)
    # Use -logsumexp(-beta * values) / beta
    neg_beta_vals = -beta * values
    max_val = np.max(neg_beta_vals)
    if not np.isfinite(max_val):
        return np.min(values)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_vals - max_val)))
    return -log_sum / beta

def _objective_smooth(params, beta=100.0):
    """Smooth objective: negative soft-min of all constraints."""
    centers = params.reshape(N, 3)
    
    # Pairwise distances / 2
    n = N
    pair_dists = np.zeros((n*(n-1)//2,))
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            pair_dists[idx] = np.linalg.norm(centers[i] - centers[j]) / 2.0
            idx += 1
    
    # Boundary distances
    bound_dists = np.min(0.5 - np.abs(centers), axis=1)
    
    # Combine all constraints
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    # Soft min
    r_soft = _soft_min(all_dists, beta=beta)
    
    return -r_soft

def _construct_initial_grid():
    """Create a reasonable initial grid for 50 spheres."""
    # Try a 4x4x4 grid = 64 positions, use 50 of them
    # Or a more structured approach
    # Let's try layers: 5 layers of 10 spheres each
    # Each layer is a 2x5 or similar arrangement
    
    centers = []
    
    # 5 layers in z-direction
    z_positions = np.linspace(-0.35, 0.35, 5)
    
    for z in z_positions:
        # 10 spheres per layer in a hexagonal-like arrangement
        # 2 rows of 5
        x_positions = np.linspace(-0.4, 0.4, 5)
        y_positions = [-0.15, 0.15]
        
        for y in y_positions:
            for x in x_positions:
                centers.append([x, y, z])
    
    centers = np.array(centers[:N])
    return centers

def _random_initialization():
    """Random initialization inside the cube with some margin."""
    margin = 0.1
    centers = np.random.uniform(-0.5 + margin, 0.5 - margin, size=(N, 3))
    return centers

def _optimize_from_start(centers_start, time_limit=30.0, beta=100.0):
    """Run local optimization from a starting point using smooth objective."""
    start_time = time.time()
    params = centers_start.flatten()
    
    # Use L-BFGS-B for constrained optimization
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_params = params.copy()
    best_r = _compute_radius(centers_start)
    
    # Try optimization with smooth objective
    try:
        result = minimize(
            _objective_smooth, 
            params, 
            args=(beta,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
        )
        if result.success or -result.fun > best_r:
            new_centers = result.x.reshape(N, 3)
            new_r = _compute_radius(new_centers)
            if new_r > best_r:
                best_r = new_r
                best_params = result.x.copy()
    except Exception:
        pass
    
    # Also try with exact (nonsmooth) objective for final polish
    elapsed = time.time() - start_time
    if elapsed < time_limit:
        try:
            def _objective_exact(params):
                centers = params.reshape(N, 3)
                r = _compute_radius(centers)
                return -r
            
            result2 = minimize(
                _objective_exact, 
                best_params, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if -result2.fun > best_r:
                new_centers = result2.x.reshape(N, 3)
                new_r = _compute_radius(new_centers)
                if new_r > best_r:
                    best_r = new_r
                    best_params = result2.x.copy()
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    return best_params.reshape(N, 3), best_r, elapsed

def _simulated_annealing(centers_start, time_limit=20.0):
    """Simple simulated annealing to escape local minima."""
    start_time = time.time()
    centers = centers_start.copy()
    current_r = _compute_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    T = 0.005  # Initial temperature
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        if T < 1e-8:
            break
        
        # Random perturbation - move 1-3 spheres
        n_move = np.random.randint(1, 4)
        idxs = np.random.choice(N, size=n_move, replace=False)
        
        new_centers = centers.copy()
        for idx in idxs:
            delta = np.random.normal(0, T * 0.5, size=3)
            new_centers[idx] += delta
            new_centers[idx] = np.clip(new_centers[idx], -0.5, 0.5)
        
        new_r = _compute_radius(new_centers)
        delta_r = new_r - current_r
        
        if delta_r > 0 or np.random.random() < np.exp(delta_r / T):
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        
        T *= cooling
    
    return best_centers, best_r, time.time() - start_time

def construct():
    """Construct the best packing found within the time budget."""
    global_start = time.time()
    deadline = 150.0 - DEADLINE_MARGIN
    
    best_centers = None
    best_r = 0.0
    
    # Start with structured grid
    centers0 = _construct_initial_grid()
    r0 = _compute_radius(centers0)
    best_centers = centers0.copy()
    best_r = r0
    
    # Try optimization from grid
    if time.time() - global_start < deadline:
        time_remaining = deadline - (time.time() - global_start)
        opt_centers, opt_r, elapsed = _optimize_from_start(centers0, time_limit=min(40.0, time_remaining))
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()
    
    # Try random restarts with optimization
    n_random_tries = 5
    for i in range(n_random_tries):
        if time.time() - global_start > deadline:
            break
        
        centers_rand = _random_initialization()
        
        # Quick optimization
        time_remaining = deadline - (time.time() - global_start)
        if time_remaining < 15.0:
            break
        
        opt_centers, opt_r, elapsed = _optimize_from_start(centers_rand, time_limit=min(20.0, time_remaining))
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()
    
    # Try simulated annealing on the best found
    if time.time() - global_start < deadline - 10.0:
        time_remaining = deadline - (time.time() - global_start)
        sa_centers, sa_r, elapsed = _simulated_annealing(best_centers, time_limit=min(25.0, time_remaining))
        if sa_r > best_r:
            best_r = sa_r
            best_centers = sa_centers.copy()
    
    # Final optimization pass on the best with tighter tolerances
    if time.time() - global_start < deadline - 5.0:
        time_remaining = deadline - (time.time() - global_start)
        opt_centers, opt_r, elapsed = _optimize_from_start(best_centers, time_limit=min(15.0, time_remaining), beta=200.0)
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()
    
    # Ensure validity
    final_centers = np.clip(best_centers, -0.5, 0.5)
    final_r = _compute_radius(final_centers)
    
    if final_r <= 0.0:
        # Fallback to initial grid
        final_centers = _construct_initial_grid()
    
    return final_centers
