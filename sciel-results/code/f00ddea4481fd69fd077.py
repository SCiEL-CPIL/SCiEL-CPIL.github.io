"""Pack 50 equal circles in a unit square [-0.5, 0.5]^2 to maximize radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score(centers):
    """Compute the minimum radius for a given set of centers."""
    centers = np.asarray(centers)
    # Distance to boundary
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(dist_to_boundary)
    
    # Minimum distance between pairs (vectorized)
    n = len(centers)
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)

def _objective_smooth(flat, beta=100.0):
    """
    Smooth objective function to minimize.
    Uses a soft-min approximation to make the objective differentiable.
    We maximize the minimum clearance, so we minimize the negative of the soft-min.
    """
    centers = flat.reshape(-1, 2)
    
    # Distance to boundary for each circle
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    # Clearances
    pair_clearances = dists / 2.0
    
    # Combine all clearances into a single vector
    # We want softmin of all pair_clearances and dist_to_boundary
    # To avoid memory issues with 50*49/2 + 50 ~ 1250 values, we can compute logsumexp carefully
    # Or just concatenate if memory allows. 1250 floats is small.
    all_clearances = np.concatenate([pair_clearances.flatten(), dist_to_boundary])
    
    # Soft min: -1/beta * log(sum(exp(-beta * x)))
    # To prevent overflow, subtract max
    m = np.min(all_clearances)
    # If m is very negative, exp(-beta * (x - m)) might be safe
    # Actually, standard trick: exp(-beta * x) = exp(-beta * (x - m)) * exp(-beta * m)
    # sum(exp(-beta * x)) = exp(-beta * m) * sum(exp(-beta * (x - m)))
    # log(sum(...)) = -beta * m + log(sum(exp(-beta * (x - m))))
    # softmin = -1/beta * (-beta * m + log(sum(...))) = m - 1/beta * log(sum(...))
    
    # If beta is large, only the min matters.
    # If beta is small, it's the mean.
    
    # For stability, we can just use the min if beta is very high, but we want smoothness.
    # Let's use a moderate beta for optimization and switch to exact min for scoring.
    
    # To avoid numerical issues with exp, we shift
    shifted = all_clearances - m
    # exp(-beta * shifted) is <= 1
    sum_exp = np.sum(np.exp(-beta * shifted))
    
    soft_min_val = m - (1.0 / beta) * np.log(sum_exp)
    
    return -soft_min_val

def _hex_lattice_init(row_counts, dx, dy):
    """Create hex lattice centers with given row counts and spacings."""
    centers = []
    total_h = (len(row_counts) - 1) * dy
    y_offset = -total_h / 2.0
    
    for i, n_cols in enumerate(row_counts):
        y = y_offset + i * dy
        if n_cols == 1:
            x_offset = 0.0
        else:
            row_width = (n_cols - 1) * dx
            x_offset = -row_width / 2.0
            if i % 2 == 1:
                x_offset += dx / 2.0
        for j in range(n_cols):
            x = x_offset + j * dx
            centers.append([x, y])
    
    return np.array(centers)

def _generate_hex_configs():
    """Generate various hexagonal lattice configurations for 50 circles."""
    configs = []
    
    # Pattern: rows of varying lengths
    patterns = [
        [7, 6, 7, 6, 7, 6, 7, 4],
        [6, 7, 6, 7, 6, 7, 6, 5],
        [7, 7, 6, 7, 6, 7, 6, 4],
        [5, 7, 6, 7, 6, 7, 6, 6],
        [6, 6, 7, 6, 7, 6, 7, 5],
        [8, 6, 7, 6, 7, 6, 7, 3],
        [7, 6, 6, 7, 6, 7, 7, 4],
        [6, 7, 7, 6, 7, 6, 6, 5],
        [7, 7, 7, 6, 6, 7, 6, 4],
        [6, 6, 6, 7, 7, 6, 7, 5],
    ]
    
    for pattern in patterns:
        if sum(pattern) != N:
            continue
        configs.append(pattern)
    
    return configs

def _optimize_from_start(centers, start_time, deadline):
    """Optimize starting from given centers using L-BFGS-B with smooth objective."""
    best_centers = centers.copy()
    best_score = _score(centers)
    
    flat = centers.flatten()
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Try optimization with smooth objective
    # Start with lower beta for global-ish behavior, then increase
    for beta in [10.0, 50.0, 200.0]:
        if time.time() > deadline - 5:
            break
            
        try:
            # Define objective for this beta
            obj = lambda x: _objective_smooth(x, beta=beta)
            
            result = minimize(
                obj, 
                flat, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-7}
            )
            
            opt_centers = result.x.reshape(-1, 2)
            opt_score = _score(opt_centers)
            
            if opt_score > best_score:
                best_score = opt_score
                best_centers = opt_centers.copy()
                flat = opt_centers.flatten() # Update start point for next beta
        except Exception:
            pass
    
    return best_centers, best_score

def _local_refinement(centers, start_time, deadline):
    """Additional local refinement with small perturbations and smooth objective."""
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Try small random perturbations and re-optimize
    for attempt in range(10):
        if time.time() - start_time > deadline:
            break
        
        # Add small random noise
        noise_scale = 0.005 / (attempt + 1)
        perturbed = centers + np.random.randn(*centers.shape) * noise_scale
        perturbed = np.clip(perturbed, -0.49, 0.49)
        
        flat = perturbed.flatten()
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            # Use high beta for precise local search
            obj = lambda x: _objective_smooth(x, beta=500.0)
            
            result = minimize(
                obj,
                flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            opt_centers = result.x.reshape(-1, 2)
            opt_score = _score(opt_centers)
            
            if opt_score > best_score:
                best_score = opt_score
                best_centers = opt_centers.copy()
        except Exception:
            pass
    
    return best_centers, best_score

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_centers = None
    best_score = -np.inf
    
    # Generate hex lattice configurations
    patterns = _generate_hex_configs()
    
    # Try different spacings for each pattern
    for pattern in patterns:
        if time.time() - start_time > deadline - 20:
            break
        
        n_rows = len(pattern)
        # Estimate appropriate dx and dy
        max_cols = max(pattern)
        
        for dx in np.linspace(0.08, 0.18, 15):
            if time.time() - start_time > deadline - 20:
                break
            
            dy = dx * np.sqrt(3) / 2.0
            
            centers = _hex_lattice_init(pattern, dx, dy)
            
            if len(centers) < N:
                continue
            
            if np.any(np.abs(centers) > 0.5):
                continue
            
            s = _score(centers)
            if s > best_score:
                best_score = s
                best_centers = centers.copy()
    
    # Also try uniform grids
    for n_cols in range(5, 12):
        for n_rows in range(5, 12):
            if n_cols * n_rows < N:
                continue
            if time.time() - start_time > deadline - 20:
                break
            
            dx = 2.0 / (n_cols + 0.5)
            dy = 2.0 / (n_rows + 0.5)
            
            # Center the grid
            xs = np.linspace(-0.5 + dx/2, 0.5 - dx/2, n_cols)
            ys = np.linspace(-0.5 + dy/2, 0.5 - dy/2, n_rows)
            
            centers_list = []
            for y in ys:
                for x in xs:
                    centers_list.append([x, y])
            
            centers = np.array(centers_list[:N])
            
            if np.any(np.abs(centers) > 0.5):
                continue
            
            s = _score(centers)
            if s > best_score:
                best_score = s
                best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Optimize from best initial configuration
    best_centers, best_score = _optimize_from_start(best_centers, start_time, deadline)
    
    # Local refinement
    if time.time() - start_time < deadline - 10:
        best_centers, best_score = _local_refinement(best_centers, start_time, deadline)
        
    # Final polish: try a few more steps with very high beta if time permits
    if time.time() - start_time < deadline - 2:
        flat = best_centers.flatten()
        bounds = [(-0.5, 0.5)] * (2 * N)
        try:
            obj = lambda x: _objective_smooth(x, beta=1000.0)
            result = minimize(
                obj,
                flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'maxfun': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            opt_centers = result.x.reshape(-1, 2)
            opt_score = _score(opt_centers)
            if opt_score > best_score:
                best_centers = opt_centers
        except Exception:
            pass

    return best_centers
