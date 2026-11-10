import numpy as np
import time

N = 50

def _compute_radius(c):
    """
    Compute the maximum common radius for centers c (N, 2) in [-0.5, 0.5]^2.
    r = min( min_{i<j} |c_i - c_j| / 2, min_i min_k (0.5 - |c_ik|) )
    """
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    
    # Boundary distances
    b = 0.5 - np.abs(c)
    min_bound = np.min(b)
    
    return 0.5 * min(min_pair, 2.0 * min_bound)

def _objective(c):
    """
    Smooth surrogate objective to maximize.
    We want to maximize the minimum distance.
    Using sum of 1/d^2 penalizes close pairs heavily.
    We also need to keep them inside.
    """
    if not np.all(np.isfinite(c)):
        return 0.0
    
    # Boundary penalty: push away from edges
    # Distance to boundary is 0.5 - |c|
    # If we are outside, we get a huge penalty.
    dist_to_bound = 0.5 - np.abs(c)
    # Penalty for being close to boundary or outside
    # We want to maximize min dist to bound.
    # Let's use a large penalty for any violation.
    boundary_violation = np.sum(np.maximum(0, -dist_to_bound)**2)
    
    # Pairwise repulsion
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d_sq, 1.0) # Avoid div by zero, though we only care about off-diag
    
    # We want to maximize the sum of 1/d^2? No, that favors spreading out but is unbounded.
    # Better: Minimize a potential energy.
    # E = sum_{i<j} (1/d_ij^2) + Lambda * sum (max(0, 0.5 - dist_to_bound)^2)
    # We want to MINIMIZE E.
    # But we also want to maximize the min distance.
    # A common trick: Maximize min(d_ij, dist_to_bound).
    # Let's try minimizing: -min_dist + penalty.
    # But min_dist is non-smooth.
    
    # Let's stick to a smooth objective that correlates with packing quality.
    # Objective: Minimize sum_{i<j} (1/d_ij^2) subject to boundary constraints.
    # If circles overlap, d_ij < 2r, so 1/d^2 is large.
    # To ensure they don't overlap, we add a hard penalty if d_ij < 2*r_est?
    # We don't know r_est.
    
    # Let's use a potential that goes to infinity as d -> 0.
    # E = sum_{i<j} (1/d_ij^2)
    # This will naturally spread them out.
    # Boundary: E += M * sum (max(0, 0.5 - dist_to_bound)^2)
    
    d_sq_off_diag = d_sq - np.eye(N)
    # Avoid division by zero for diagonal (already set to 1 in d_sq, so 0 in off_diag)
    # Actually, let's just compute off-diagonal sum.
    # d_sq has 1s on diagonal.
    # We want sum over i<j.
    # Sum of 1/d_sq over all i!=j / 2.
    inv_d_sq = np.zeros_like(d_sq)
    mask = d_sq > 1e-12
    inv_d_sq[mask] = 1.0 / d_sq[mask]
    inv_d_sq[~mask] = 0.0
    # Diagonal is 0 because d_sq[diag]=1 -> inv=1, but we want 0.
    # Wait, I set diagonal of d_sq to 1. So inv_d_sq diag is 1.
    # I want to exclude diagonal.
    inv_d_sq[np.diag_indices(N)] = 0.0
    
    pair_energy = np.sum(inv_d_sq) / 2.0
    
    # Boundary penalty
    # dist_to_bound = 0.5 - |c|
    # If c is outside, dist_to_bound < 0.
    # Penalty = sum (max(0, -dist_to_bound)^2)
    # This penalizes being outside.
    # To keep them well inside, we can also penalize being close to boundary.
    # Let's just penalize violation.
    boundary_energy = 100.0 * boundary_violation
    
    return -(pair_energy + boundary_energy) # We want to maximize this (minimize energy)

def _simulated_annealing(c_init, time_limit):
    """
    Run SA on a copy of c_init until time_limit is reached.
    Returns best center configuration found.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_score = _compute_radius(c)
    
    # Temperature schedule
    T_start = 0.01
    T_end = 0.0001
    cooling_rate = 0.999
    
    T = T_start
    step_size = 0.01
    
    while T > T_end:
        if time.time() > time_limit:
            break
        
        # Propose a move: random perturbation
        # Gaussian noise
        noise = np.random.normal(0, step_size, (N, 2))
        c_new = c + noise
        
        # Keep inside square? 
        # If we clip, we lose information about "outside" moves.
        # Better to allow moves outside but penalize them in objective.
        # However, if they go far outside, objective is bad.
        # Let's clip to a slightly larger box to prevent explosion, but allow boundary crossing.
        # Actually, the objective penalizes boundary violation.
        
        score_old = _objective(c)
        score_new = _objective(c_new)
        
        delta = score_new - score_old
        
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            c = c_new
            # Check if this is a better valid packing
            # We should compute the actual radius for valid configs
            # But _objective is a proxy.
            # Let's compute _compute_radius occasionally or if score improved significantly?
            # _compute_radius is O(N^2). _objective is O(N^2).
            # Let's just track best based on _compute_radius every few steps or if objective improves.
            
            # To be safe and fast, let's evaluate _compute_radius only if _objective improved
            # or periodically.
            # Actually, _objective maximization doesn't strictly guarantee _compute_radius maximization.
            # But they are correlated.
            # Let's evaluate _compute_radius on the current best candidate every 100 steps or if T drops.
            
        T *= cooling_rate
        # Adaptive step size?
        # step_size *= 0.999
        
    # Final check
    final_score = _compute_radius(c)
    if final_score > best_score:
        best_c = c
        best_score = final_score
        
    return best_c, best_score

def _generate_grid_init():
    """
    Generate a structured grid initialization.
    50 circles.
    Try to fit them in a grid.
    7x7 = 49. Add 1 in center?
    Or 5x10 = 50.
    Let's try 5x10 grid.
    """
    c = np.zeros((N, 2))
    # 5 rows, 10 cols
    rows = 5
    cols = 10
    # Spacing
    # Square side 1.
    # If we place them on a grid, spacing is 1/cols?
    # Centers at (i+0.5)/cols - 0.5
    x_coords = np.linspace(-0.45, 0.45, cols)
    y_coords = np.linspace(-0.45, 0.45, rows)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            c[idx, 0] = x_coords[j]
            c[idx, 1] = y_coords[i]
            idx += 1
            
    # Add small random perturbation to break symmetry
    c += np.random.uniform(-0.01, 0.01, (N, 2))
    c = np.clip(c, -0.49, 0.49)
    return c

def _generate_random_init():
    """
    Random initialization inside the square.
    """
    c = np.random.uniform(-0.45, 0.45, (N, 2))
    return c

def _generate_hex_init():
    """
    Hexagonal packing initialization.
    """
    c = np.zeros((N, 2))
    # Hex grid
    # Spacing dx = 1, dy = sqrt(3)/2
    # Scale to fit.
    # Let's just use a dense random or grid for now.
    return _generate_grid_init()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    best_c = None
    best_r = 0.0
    
    # List of initializers
    inits = [
        _generate_grid_init(),
        _generate_random_init(),
        _generate_grid_init() + np.random.normal(0, 0.05, (N, 2)), # Perturbed grid
    ]
    
    # Try multiple restarts
    # How many restarts?
    # SA is fast if vectorized? 
    # _objective is O(N^2) = 2500 ops.
    # 10k steps = 25M ops. ~0.1s.
    # We can do many restarts.
    
    for i in range(50): # Max 50 restarts
        if time.time() > deadline:
            break
            
        # Pick an initializer
        if i < len(inits):
            c_init = inits[i].copy()
        else:
            # Random restarts
            c_init = _generate_random_init()
        
        # Run SA for a specific time budget
        # Give each restart ~2 seconds?
        # 50 * 2 = 100s.
        # Let's give 2s to first few, less to later ones.
        time_budget = 2.0 if i < 10 else 1.0
        end_time = time.time() + time_budget
        
        c_best, r_best = _simulated_annealing(c_init, end_time)
        
        if r_best > best_r:
            best_r = r_best
            best_c = c_best.copy()
            
    # Final polish?
    # If we have time, do a local search on the best solution.
    if best_c is not None and time.time() < deadline - 5.0:
        # Simple local search: try moving each circle slightly
        # Or just run SA again on best_c with lower T
        c_final, r_final = _simulated_annealing(best_c, deadline - 2.0)
        if r_final > best_r:
            best_c = c_final
            best_r = r_final
            
    if best_c is None:
        # Fallback
        best_c = _generate_grid_init()
        
    # Ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c

# Keep the interface
# The prompt asks to keep function names and signatures of the seed program.
# Seed program had:
# def construct():
#     ...
#     return c
# And other helpers.
# I will define construct() and necessary helpers.

# To be safe, I'll wrap the logic in construct() and define helpers above it.
# The code above defines construct().

# One check: The seed program used _BDGL. I am not using it.
# The directive says "rethinking the approach".
# Using SA is a materially different approach from "embed BDGL and polish with repulsion".

# Let's double check the _objective function.
# It returns negative energy.
# _simulated_annealing maximizes this.
# If energy is low (good packing), -energy is high.
# Correct.

# Potential issue: _objective might be flat or have local minima that don't correspond to good packings.
# But with multiple restarts and random perturbations, it should find good ones.
# The grid init is a strong start.

# Let's refine _simulated_annealing to be more efficient.
# Vectorized operations are key.

def _simulated_annealing_optimized(c_init, time_limit):
    c = c_init.copy()
    best_c = c.copy()
    best_score = _compute_radius(c)
    
    T = 0.01
    step_size = 0.01
    
    while T > 0.0001:
        if time.time() > time_limit:
            break
        
        noise = np.random.normal(0, step_size, (N, 2))
        c_new = c + noise
        
        score_old = _objective(c)
        score_new = _objective(c_new)
        
        delta = score_new - score_old
        
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            c = c_new
            # Update best if improved
            # Checking _compute_radius every step is expensive (O(N^2) with sqrt).
            # _objective is also O(N^2) but no sqrt.
            # Let's check _compute_radius every 100 steps.
            if np.random.rand() < 0.01: # 1% chance
                curr_score = _compute_radius(c)
                if curr_score > best_score:
                    best_score = curr_score
                    best_c = c.copy()
                    
        T *= 0.999
        step_size *= 0.9999
        
    # Final check
    final_score = _compute_radius(c)
    if final_score > best_score:
        best_c = c
        best_score = final_score
        
    return best_c, best_score

# Replace _simulated_annealing with _simulated_annealing_optimized in construct
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_c = None
    best_r = 0.0
    
    inits = [
        _generate_grid_init(),
        _generate_random_init(),
        _generate_grid_init() + np.random.normal(0, 0.05, (N, 2)),
    ]
    
    for i in range(50):
        if time.time() > deadline:
            break
            
        if i < len(inits):
            c_init = inits[i].copy()
        else:
            c_init = _generate_random_init()
            
        time_budget = 2.0 if i < 10 else 1.0
        end_time = time.time() + time_budget
        
        c_best, r_best = _simulated_annealing_optimized(c_init, end_time)
        
        if r_best > best_r:
            best_r = r_best
            best_c = c_best.copy()
            
    if best_c is not None and time.time() < deadline - 5.0:
        c_final, r_final = _simulated_annealing_optimized(best_c, deadline - 2.0)
        if r_final > best_r:
            best_c = c_final
            best_r = r_final
            
    if best_c is None:
        best_c = _generate_grid_init()
        
    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
