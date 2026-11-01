import numpy as np
import time
import random

N = 100
DEADLINE = 145.0  # Safety margin

def _score(centers):
    """Compute the radius for a given set of centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    D = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=-1))
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    r = min(min_pair / 2.0, min_wall)
    return r

def _delta_score(centers, i, new_pos, old_min_dist, old_min_pair):
    """
    Efficiently update the minimum distance when only center i changes.
    Returns new_min_dist and new_min_pair.
    """
    # Distance from i to all others
    diffs = centers - new_pos
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    dists[i] = np.inf
    
    # Wall distance for i
    wall_dist_i = np.min(0.5 - np.abs(new_pos))
    
    # We need the global min.
    # The old min_pair might still be valid if it doesn't involve i.
    # If it involves i, it's changed.
    # Also, any new pair involving i could be smaller.
    
    # Candidate 1: The minimum distance among pairs NOT involving i.
    # This is hard to get without recomputing all.
    # However, if we maintain the second minimum distance, we can be smarter.
    # For simplicity and robustness with N=100, let's just recompute the global min
    # but using the precomputed distances for non-i pairs?
    # Actually, O(N) is fast enough. Let's just compute min(dists, wall_dist_i) 
    # BUT we must compare against the min of all OTHER pairs.
    
    # Let's just compute the full score for the candidate?
    # No, that defeats the purpose.
    
    # Let's maintain a KDTree?
    # For N=100, cdist is ~10k ops. Python loop overhead is the issue.
    # Let's try to stick to NumPy vectorization for the full matrix, 
    # but optimize the SA loop to use C-level loops via NumPy as much as possible.
    
    # Alternative: Just use the full score but make the SA step count high.
    # 150s / 100us per step = 1.5M steps.
    # Full score takes ~50us? 
    # Let's benchmark mentally. 100x100 matrix mult is fast.
    
    pass

def _build_bcc_init():
    """Initialize with Body-Centered Cubic lattice points."""
    # BCC unit cell has 2 points: (0,0,0) and (0.5, 0.5, 0.5)
    # We need 100 points.
    # Let's generate a grid of BCC cells.
    # Side length of cube is 1.
    # If we use n cells per side, we have n^3 cells, 2*n^3 points.
    # 2*n^3 >= 100 => n^3 >= 50 => n=4 (2*64=128 points).
    n = 4
    scale = 1.0 / n
    points = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                # Corner point
                points.append([x * scale, y * scale, z * scale])
                # Center point
                points.append([(x + 0.5) * scale, (y + 0.5) * scale, (z + 0.5) * scale])
    
    points = np.array(points)
    # Center the lattice in [-0.5, 0.5]
    # Current range [0, 1]. Shift by -0.5.
    points -= 0.5
    
    # We have 128 points. We need 100.
    # Randomly select 100? Or take the first 100?
    # Taking first 100 might create a bias. 
    # Let's randomize the selection to break symmetry.
    indices = np.random.choice(len(points), N, replace=False)
    return points[indices].copy()

def _build_halton_init():
    """Initialize with Halton sequence."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    points = []
    for i in range(1, N + 1):
        points.append([halton(i, 2), halton(i, 3), halton(i, 5)])
    
    points = np.array(points)
    # Scale to [-0.5, 0.5]
    points = points * 1.0 - 0.5
    return points

def _optimize_sa(centers, time_limit):
    """
    Simulated Annealing optimization.
    """
    centers = centers.copy()
    current_r = _score(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    start_time = time.time()
    
    # Cooling schedule
    T = 0.05
    alpha = 0.9995 # Cooling rate
    min_T = 1e-6
    
    # Move size
    step_size = 0.05
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        # Pick a random sphere
        i = np.random.randint(N)
        
        # Generate a new position
        # Gaussian perturbation
        new_pos = centers[i] + np.random.normal(0, step_size, size=3)
        
        # Keep inside bounds (soft constraint in score, but let's clip to avoid huge penalties)
        new_pos = np.clip(new_pos, -0.4999, 0.4999)
        
        # Calculate new score
        old_pos = centers[i].copy()
        centers[i] = new_pos
        new_r = _score(centers)
        
        # Acceptance criterion
        delta_r = new_r - current_r
        if delta_r > 0 or np.random.random() < np.exp(delta_r / T):
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            centers[i] = old_pos
            
        # Cool down
        T *= alpha
        
    return best_centers

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Multiple starts
    inits = [
        _build_bcc_init(),
        _build_halton_init(),
        np.random.uniform(-0.45, 0.45, size=(N, 3))
    ]
    
    best_overall_r = 0.0
    best_overall_centers = inits[0]
    
    # Allocate time to each start
    # 3 starts, ~40s each
    time_per_start = (DEADLINE - 5.0) / len(inits) # Leave 5s for final check
    
    for init in inits:
        if time.time() > deadline - 5.0:
            break
            
        # Run SA
        optimized = _optimize_sa(init, time_limit=time_per_start)
        r = _score(optimized)
        
        if r > best_overall_r:
            best_overall_r = r
            best_overall_centers = optimized
    
    # Final validation
    if _score(best_overall_centers) > 0:
        return best_overall_centers
    else:
        return _build_bcc_init()
