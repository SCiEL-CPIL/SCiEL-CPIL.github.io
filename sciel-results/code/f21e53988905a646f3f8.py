"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, cdist

def _compute_radius(centres):
    """Compute the largest common radius for given centres."""
    if len(centres) == 0:
        return 0.0
    # Pairwise distances
    if len(centres) > 1:
        d = pdist(centres, 'euclidean')
        min_pair = d.min() if len(d) > 0 else np.inf
    else:
        min_pair = np.inf
    # Distance to walls
    wall_dists = np.min(np.abs(centres - 0.5), axis=1) if centres.shape[1] == 3 else np.inf
    min_wall = wall_dists.min() if len(wall_dists) > 0 else np.inf
    return min(min_pair, min_wall) / 2.0

def _compute_radius_full(centres):
    """Compute the exact radius as defined in the task."""
    if len(centres) == 0:
        return 0.0
    # Pairwise distances
    if len(centres) > 1:
        d = pdist(centres, 'euclidean')
        min_pair = d.min() if len(d) > 0 else np.inf
    else:
        min_pair = np.inf
    # Distance to walls: min over i,k of (1/2 - |c_ik|)
    # This is the minimum distance from any centre coordinate to the nearest face
    min_wall = np.min(np.abs(centres - 0.5))  # min over all coordinates
    return min(min_pair, min_wall) / 2.0

def _generate_cubic_seed():
    """Generate a simple cubic grid seed."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _generate_bcc_seed():
    """Generate a BCC (body-centered cubic) inspired seed."""
    # Use a 4x4x4 grid = 64 points, take first 50
    m = 4
    # Face-centered positions within each cell
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) / m - 0.5
                y = (j + 0.5) / m - 0.5
                z = (k + 0.5) / m - 0.5
                points.append([x, y, z])
    points = np.array(points[:N])
    return points

def _generate_random_seed(seed_val):
    """Generate a random seed with some structure."""
    rng = np.random.default_rng(seed_val)
    # Start with a grid and add jitter
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    jitter = rng.normal(0, 0.02, size=grid.shape)
    return grid + jitter

def _optimize_gradient(centres, max_iters=2000, deadline=None):
    """Optimize using gradient ascent on the radius objective."""
    centres = centres.copy().astype(np.float64)
    best_centres = centres.copy()
    best_radius = _compute_radius_full(centres)
    
    lr = 0.005
    lr_decay = 0.999
    momentum = np.zeros_like(centres)
    beta = 0.9
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute current radius
        r = _compute_radius_full(centres)
        
        # Compute gradient of the objective
        # The objective is min(min_pair_dist, min_wall_dist) / 2
        # We need to find which constraint is binding
        
        # Pairwise distances
        if len(centres) > 1:
            d = pdist(centres, 'euclidean')
            min_pair = d.min()
        else:
            min_pair = np.inf
        
        min_wall = np.min(np.abs(centres - 0.5))
        
        # Determine which is binding
        grad = np.zeros_like(centres)
        
        if min_pair <= min_wall:
            # Pairwise constraint is binding
            # Find the closest pair
            idx = np.argmin(d)
            # Convert from condensed index to pair index
            n = len(centres)
            # In scipy's pdist, the index corresponds to (i,j) where i < j
            # We need to find which pair
            count = 0
            for i in range(n):
                for j in range(i+1, n):
                    if count == idx:
                        break
                    count += 1
                if count == idx:
                    break
            
            # Direction from i to j
            diff = centres[j] - centres[i]
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                unit = diff / dist
                grad[i] -= unit
                grad[j] += unit
        else:
            # Wall constraint is binding
            # Find which point and which face
            min_val = np.inf
            min_i = 0
            min_k = 0
            for i in range(len(centres)):
                for k in range(3):
                    val = np.abs(centres[i][k] - 0.5)
                    if val < min_val:
                        min_val = val
                        min_i = i
                        min_k = k
            
            # Push away from the wall
            sign = 1.0 if centres[min_i][min_k] >= 0.0 else -1.0
            # If centre is at positive side (near +0.5), push in negative direction
            # If centre is at negative side (near -0.5), push in positive direction
            if centres[min_i][min_k] > 0:
                grad[min_i][min_k] -= 1.0
            else:
                grad[min_i][min_k] += 1.0
        
        # Add small random noise to escape local minima
        noise = np.random.normal(0, 0.001, size=centres.shape)
        grad += noise
        
        # Update with momentum
        momentum = beta * momentum + (1 - beta) * grad
        centres = centres + lr * momentum
        
        # Project back into the cube with margin
        centres = np.clip(centres, -0.5, 0.5)
        
        # Check new radius
        new_r = _compute_radius_full(centres)
        if new_r > best_radius:
            best_radius = new_r
            best_centres = centres.copy()
        
        # Decay learning rate
        lr *= lr_decay
        if lr < 1e-6:
            break
    
    return best_centres, best_radius

def _optimize_landscape(centres, max_iters=500, deadline=None):
    """Simple landscape optimization with random perturbations."""
    centres = centres.copy().astype(np.float64)
    best_centres = centres.copy()
    best_radius = _compute_radius_full(centres)
    
    step_size = 0.01
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random perturbation
        perturbed = centres + np.random.normal(0, step_size, size=centres.shape)
        perturbed = np.clip(perturbed, -0.5, 0.5)
        
        new_r = _compute_radius_full(perturbed)
        if new_r > best_radius:
            best_radius = new_r
            best_centres = perturbed.copy()
            centres = perturbed.copy()
            step_size = max(0.001, step_size * 1.05)
        else:
            step_size = max(0.001, step_size * 0.95)
    
    return best_centres, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Try multiple seeds and optimization methods
    seeds = []
    
    # Cubic grid
    seeds.append(('cubic', _generate_cubic_seed()))
    
    # BCC-inspired
    seeds.append(('bcc', _generate_bcc_seed()))
    
    # Random with structure
    for s in range(5):
        seeds.append((f'rand_{s}', _generate_random_seed(s * 42)))
    
    best_centres = None
    best_radius = -np.inf
    
    for name, seed_centres in seeds:
        if time.time() > deadline - 5.0:
            break
        
        # Try gradient optimization
        opt_centres, opt_radius = _optimize_gradient(seed_centres, max_iters=1000, deadline=deadline - 5.0)
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centres = opt_centres.copy()
        
        # Try landscape optimization from the gradient result
        if time.time() > deadline - 3.0:
            break
        opt_centres2, opt_radius2 = _optimize_landscape(opt_centres, max_iters=200, deadline=deadline - 2.0)
        if opt_radius2 > best_radius:
            best_radius = opt_radius2
            best_centres = opt_centres2.copy()
    
    # Final polish: small local search
    if best_centres is not None and time.time() < deadline - 1.0:
        # Try very small perturbations
        for _ in range(50):
            if time.time() > deadline - 1.0:
                break
            perturbed = best_centres + np.random.normal(0, 0.002, size=best_centres.shape)
            perturbed = np.clip(perturbed, -0.5, 0.5)
            new_r = _compute_radius_full(perturbed)
            if new_r > best_radius:
                best_radius = new_r
                best_centres = perturbed.copy()
    
    if best_centres is None:
        best_centres = _generate_cubic_seed()
    
    # Ensure valid output
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres
# EVOLVE-BLOCK-END
