"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _compute_max_radii(centers, max_iters=100, tol=1e-8):
    """
    Compute the maximum possible radii for a given set of centers
    such that circles are disjoint and inside the unit square.
    Uses iterative relaxation.
    """
    # Wall distances
    # centers: (N, 2)
    # dist to left: x, right: 1-x, bottom: y, top: 1-y
    wall_dist = np.min(np.minimum(centers, 1.0 - centers), axis=1)
    
    # Initialize radii to wall distances
    r = wall_dist.copy()
    
    # Precompute distance matrix
    dx = centers[:, 0, np.newaxis] - centers[np.newaxis, :, 0]
    dy = centers[:, 1, np.newaxis] - centers[np.newaxis, :, 1]
    d = np.sqrt(dx * dx + dy * dy)
    
    # Iterative relaxation
    for _ in range(max_iters):
        # constraints[i, j] = d[i, j] - r[j]
        # We want r[i] <= constraints[i, j] for all j != i
        constraints = d - r[np.newaxis, :]
        
        # Set diagonal to inf
        np.fill_diagonal(constraints, np.inf)
        
        # Min constraint for each i
        min_constraint = np.min(constraints, axis=1)
        
        # New radius is min(wall_dist, min_constraint)
        new_r = np.minimum(wall_dist, min_constraint)
        
        # Check convergence
        if np.max(np.abs(new_r - r)) < tol:
            r = new_r
            break
        r = new_r
        
    # Ensure radii are non-negative
    r = np.maximum(r, 0.0)
    
    return r

def _objective(centers):
    """
    Objective function to minimize: Negative sum of max radii.
    centers: flat array of length 2*N
    """
    pos = centers.reshape(N, 2)
    # Keep centers inside [0, 1] with a small margin to avoid edge cases in sqrt
    # The SA/NM should handle bounds, but we clip here for safety in evaluation
    pos_clipped = np.clip(pos, 1e-6, 1.0 - 1e-6)
    
    radii = _compute_max_radii(pos_clipped)
    return -np.sum(radii)

def _initial_lattice():
    """Generate a hexagonal lattice-like initial state."""
    pos = np.zeros((N, 2))
    
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        if row == 5:
            x = 0.5
            y = row * dy + r
            if k < N:
                pos[k] = [x, y]
                k += 1
            continue
            
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            y = row * dy + r
            
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                k += 1
        
        if k >= N:
            break
            
    # Fill remaining if any (unlikely for N=26 with this grid)
    while k < N:
        pos[k] = [0.5, 0.5]
        k += 1
        
    return pos

def _initial_random(seed):
    """Generate a random initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    return pos

def _initial_best_fit():
    """
    Generate a tight initial packing using a greedy best-fit algorithm on centers.
    We place centers to maximize the sum of potential radii.
    """
    pos = np.zeros((N, 2))
    pos[0] = [0.5, 0.5]
    
    for i in range(1, N):
        best_x, best_y = 0.5, 0.5
        best_score = -np.inf
        
        rng = np.random.RandomState(i * 1000 + 42)
        
        # Sample random points
        for _ in range(2000):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            
            # Temporarily add this point
            temp_pos = pos[:i+1].copy()
            temp_pos[i] = [x, y]
            
            # Compute radii for this temp set
            radii = _compute_max_radii(temp_pos, max_iters=20)
            
            # Score is sum of radii
            score = np.sum(radii)
            
            if score > best_score:
                best_score = score
                best_x, best_y = x, y
                
        pos[i] = [best_x, best_y]
        
    return pos

def _simulated_annealing(pos_init, time_limit=10.0, seed=42):
    """
    Simulated Annealing to optimize centers.
    """
    rng = np.random.RandomState(seed)
    pos = pos_init.copy()
    
    # Initial radii
    radii = _compute_max_radii(pos)
    current_score = np.sum(radii)
    best_score = current_score
    best_pos = pos.copy()
    
    T = 0.05
    T_min = 0.001
    cooling = 0.9995
    step_size = 0.05
    
    start_time = time.time()
    
    while T > T_min and (time.time() - start_time) < time_limit:
        
        # Propose new position
        noise = rng.normal(0, step_size, (N, 2))
        new_pos = pos + noise
        
        # Clamp to [0, 1]
        new_pos = np.clip(new_pos, 0.0, 1.0)
        
        # Compute new radii and score
        new_radii = _compute_max_radii(new_pos)
        new_score = np.sum(new_radii)
        
        # Delta E (we want to maximize score, so minimize -score)
        # dE = new_score - current_score
        # If new_score > current_score, dE > 0 (improvement)
        delta = new_score - current_score
        
        if delta > 0 or rng.uniform() < np.exp(delta / T):
            pos = new_pos
            current_score = new_score
            radii = new_radii
            
            if current_score > best_score:
                best_score = current_score
                best_pos = pos.copy()
        
        # Cool down
        T *= cooling
        
    return best_pos, best_score

def _nelder_mead_polish(pos_init, time_limit=5.0):
    """
    Nelder-Mead polish on the centers.
    """
    x0 = pos_init.flatten()
    
    # Bounds for Nelder-Mead are not directly supported in all versions, 
    # but we can use a penalty or just rely on the objective being well-behaved 
    # and the initial guess being good. 
    # However, Nelder-Mead can wander out of bounds. 
    # We will use a wrapper that penalizes out-of-bounds heavily.
    
    def obj_wrapper(x):
        pos = x.reshape(N, 2)
        # Check bounds
        if np.any(pos < 0) or np.any(pos > 1):
            return 1e6 # Huge penalty
        return _objective(x)
        
    try:
        res = minimize(obj_wrapper, x0, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        
        # Check time
        if time.time() > time.time() + time_limit: # This check is inside the function, so it's not useful here.
             pass
             
        pos_opt = res.x.reshape(N, 2)
        # Ensure valid bounds
        pos_opt = np.clip(pos_opt, 0.0, 1.0)
        
        return pos_opt
        
    except Exception:
        return pos_init

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # Generate initial guesses
    inits = []
    inits.append(_initial_lattice())
    inits.append(_initial_best_fit())
    inits.append(_initial_random(123))
    inits.append(_initial_random(456))
    
    best_pos = inits[0]
    best_score = -np.inf
    
    # Run SA on each initial guess
    # Allocate time based on number of inits
    time_per_init = 20.0
    
    for i, pos_init in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        # Run SA
        pos_out, score = _simulated_annealing(pos_init, time_limit=time_per_init, seed=100+i)
        
        if score > best_score:
            best_score = score
            best_pos = pos_out
            
    # Polish the best result with Nelder-Mead
    if time.time() < deadline - 5.0:
        time_left = deadline - time.time() - 2.0
        if time_left > 2.0:
            polished_pos = _nelder_mead_polish(best_pos, time_limit=time_left)
            # Verify polish improved or at least didn't break
            radii = _compute_max_radii(polished_pos)
            polish_score = np.sum(radii)
            
            if polish_score > best_score:
                best_pos = polished_pos
                best_score = polish_score
                
    # Final computation of radii
    final_radii = _compute_max_radii(best_pos)
    
    # Ensure strict validity (though _compute_max_radii should guarantee it)
    # Just in case of floating point errors
    for i in range(N):
        r = final_radii[i]
        x, y = best_pos[i]
        # Clamp to walls
        max_r_wall = min(x, 1.0-x, y, 1.0-y)
        if r > max_r_wall:
            final_radii[i] = max_r_wall * 0.9999
            
    # Check overlaps one last time and shrink if necessary
    for _ in range(10):
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                dx = best_pos[i, 0] - best_pos[j, 0]
                dy = best_pos[i, 1] - best_pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = final_radii[i] + final_radii[j]
                if dist < sum_r:
                    # Shrink both slightly
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        final_radii[i] *= k
                        final_radii[j] *= k
                        valid = False
        if valid:
            break
            
    result = np.column_stack([best_pos, final_radii])
    return result

# EVOLVE-BLOCK-END
