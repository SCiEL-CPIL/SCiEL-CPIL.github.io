"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_score(centers):
    """
    Compute the maximum radius for a given set of centers.
    Returns -inf if invalid.
    """
    if centers is None or len(centers) != N:
        return -np.inf
    
    # Check for non-finite values
    if not np.all(np.isfinite(centers)):
        return -np.inf
        
    # Check bounds: centers must be in [-0.5, 0.5]
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return -np.inf
        
    # Compute min distance to boundary
    # distance from center c to boundary of [-0.5, 0.5] is min(0.5 - c, c + 0.5)
    # Actually, distance to nearest face is min(0.5 - c_k) for k in 0,1,2?
    # No, distance to face x=0.5 is 0.5-c_x. Distance to face x=-0.5 is c_x+0.5.
    # So min dist to boundary for sphere i is min(0.5 - c_i, c_i + 0.5) for each dim?
    # Wait, the sphere must be inside. So c_i + r <= 0.5 and c_i - r >= -0.5.
    # r <= 0.5 - c_i and r <= c_i + 0.5.
    # So max r for sphere i is min(0.5 - c_i, c_i + 0.5).
    
    dists_to_boundary = np.minimum(0.5 - centers, centers + 0.5)
    min_boundary_dist = np.min(dists_to_boundary)
    
    if min_boundary_dist < 0:
        return -np.inf
        
    # Compute min pairwise distance / 2
    # Use scipy spatial for efficiency if available, else manual
    try:
        from scipy.spatial.distance import pdist
        dists = pdist(centers)
        if len(dists) == 0:
            min_pair_dist = np.inf
        else:
            min_pair_dist = np.min(dists)
    except ImportError:
        # Fallback for small N or no scipy (N=100 is small enough for O(N^2) in loop? 
        # 100*100/2 = 5000 ops, very fast)
        min_pair_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_pair_dist:
                    min_pair_dist = d
                    
    min_pair_radius = min_pair_dist / 2.0
    r = min(min_boundary_dist, min_pair_radius)
    
    return r

def _random_init():
    """Generate random initial centers inside the cube."""
    # Random points in [-0.5, 0.5]
    centers = np.random.uniform(-0.5, 0.5, (N, 3))
    return centers

def _push_apart(centers, iterations=100, step_size=0.05):
    """
    Simple iterative push-apart to resolve overlaps.
    This is a basic form of relaxation.
    """
    centers = centers.copy()
    for _ in range(iterations):
        moved = False
        for i in range(N):
            for j in range(i+1, N):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                # If they are very close, push apart
                # We don't know the optimal radius yet, so we push if dist < some threshold?
                # Better: just push if dist is small.
                if dist < 0.1: # Arbitrary threshold
                    if dist == 0:
                        d = np.random.normal(0, 0.01, 3)
                        dist = np.linalg.norm(d)
                    # Push each away by half the overlap? 
                    # Let's just push them apart by a small amount along d
                    # We want to increase dist.
                    # Move i by +d/dist * eps, j by -d/dist * eps
                    move = (d / dist) * (0.01 * step_size)
                    centers[i] += move
                    centers[j] -= move
                    # Clamp to box
                    centers[i] = np.clip(centers[i], -0.5, 0.5)
                    centers[j] = np.clip(centers[j], -0.5, 0.5)
                    moved = True
        if not moved:
            break
    return centers

def _simulated_annealing(centers, time_limit=50.0):
    """
    Simulated Annealing to maximize the minimum radius.
    Objective: maximize r = min(min_pair_dist/2, min_boundary_dist)
    Energy: E = -r (we want to minimize E)
    
    Moves:
    1. Move a random sphere by a small random vector.
    2. Swap two spheres? (Not really useful for indistinguishable spheres, but maybe for symmetry breaking? No, positions matter).
    3. "Jiggle" the closest pair apart.
    """
    start_time = time.time()
    current_centers = centers.copy()
    current_score = _compute_score(current_centers)
    
    best_centers = current_centers.copy()
    best_score = current_score
    
    T = 0.05 # Initial temperature
    T_min = 0.001
    cooling = 0.995
    
    # Precompute indices for efficiency
    indices = np.arange(N)
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Choose a move
        # 80%: Move a single random sphere
        # 20%: "Fix" the worst pair (closest pair)
        
        new_centers = current_centers.copy()
        
        if random.random() < 0.8:
            i = random.randint(0, N-1)
            # Step size proportional to T
            step = np.random.normal(0, T * 0.5, 3)
            new_centers[i] += step
            new_centers[i] = np.clip(new_centers[i], -0.5, 0.5)
        else:
            # Find closest pair
            # This is expensive O(N^2) every step. 
            # Optimization: only do this if we haven't done it recently? 
            # For N=100, O(N^2) is 10k ops. 10k * 1000 steps = 10M ops. Fast enough in Python? 
            # Maybe 1s for 1000 steps. 100s limit -> 100k steps. 100k * 10k = 10^9 ops. Too slow in pure Python loop.
            # Let's stick to single particle moves for SA, or use vectorized distance calc.
            
            # Vectorized closest pair search
            # Compute all pairwise distances
            # This creates a 100x100 matrix.
            # dists = np.linalg.norm(new_centers[:, np.newaxis, :] - new_centers[np.newaxis, :, :], axis=2)
            # This is memory intensive (100*100*8 bytes = 80KB, fine).
            # But doing this every step is heavy.
            
            # Alternative: Just do single particle moves but with larger steps sometimes.
            # Or, identify the "worst" sphere (the one with smallest local clearance) and move it.
            
            # Let's compute local clearance for all spheres efficiently?
            # Still O(N^2).
            
            # Let's just use single particle moves for now, but vary the step size more.
            # Or, pick 2 random spheres and swap? No.
            
            # Let's try: Pick the sphere with the smallest distance to its nearest neighbor.
            # We can approximate this by just picking a random sphere.
            # To make the "20%" case useful, let's pick a random pair (i, j) and push them apart if they are close.
            i, j = random.sample(range(N), 2)
            d = new_centers[i] - new_centers[j]
            dist = np.linalg.norm(d)
            if dist < 0.2:
                if dist > 1e-9:
                    # Push apart
                    push_dir = d / dist
                    # How much to push?
                    # Try to increase dist by a small amount
                    delta = T * 0.2
                    new_centers[i] += push_dir * delta
                    new_centers[j] -= push_dir * delta
                    new_centers[i] = np.clip(new_centers[i], -0.5, 0.5)
                    new_centers[j] = np.clip(new_centers[j], -0.5, 0.5)
                else:
                    # Coincident, randomize
                    new_centers[i] += np.random.normal(0, T*0.1, 3)
                    new_centers[j] -= np.random.normal(0, T*0.1, 3)
                    new_centers[i] = np.clip(new_centers[i], -0.5, 0.5)
                    new_centers[j] = np.clip(new_centers[j], -0.5, 0.5)

        new_score = _compute_score(new_centers)
        
        # Acceptance criterion
        if new_score > current_score:
            current_centers = new_centers
            current_score = new_score
            if current_score > best_score:
                best_centers = current_centers.copy()
                best_score = current_score
        else:
            # Metropolis criterion
            # Energy difference dE = new_score - current_score (negative)
            # Accept with probability exp(dE / T)
            # Note: Our score is radius (positive). We want to maximize.
            # So "energy" is -score.
            # dE = -new_score - (-current_score) = current_score - new_score
            # If new_score < current_score, dE > 0.
            # Prob = exp(-dE / T) = exp((new_score - current_score) / T)
            
            delta_score = new_score - current_score
            # If delta_score is negative, exp(negative/T) is < 1
            if random.random() < np.exp(delta_score / T):
                current_centers = new_centers
                current_score = new_score
                
        T *= cooling
        
    return best_centers, best_score

def _polish_best_pair(centers, iterations=2000, time_limit=30.0):
    """
    High precision polish: Iteratively find the closest pair and push them apart.
    Also check boundary constraints.
    """
    start_time = time.time()
    centers = centers.copy()
    best_score = _compute_score(centers)
    best_centers = centers.copy()
    
    for _ in range(iterations):
        if time.time() - start_time > time_limit:
            break
            
        # Find closest pair
        # Vectorized
        # dists matrix
        # To save memory/time, we can just loop, but vectorized is faster for N=100.
        # Let's use scipy if available
        try:
            from scipy.spatial.distance import pdist, squareform
            dists = pdist(centers)
            # We need indices of min
            # pdist doesn't return indices directly.
            # Alternative: use cdist
            from scipy.spatial.distance import cdist
            dist_mat = cdist(centers, centers)
            np.fill_diagonal(dist_mat, np.inf)
            min_idx = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
            i, j = min_idx
            min_dist = dist_mat[i, j]
        except ImportError:
            min_dist = np.inf
            i, j = 0, 1
            for a in range(N):
                for b in range(a+1, N):
                    d = np.linalg.norm(centers[a] - centers[b])
                    if d < min_dist:
                        min_dist = d
                        i, j = a, b
                        
        # Check boundary
        dists_to_boundary = np.minimum(0.5 - centers, centers + 0.5)
        min_b_idx = np.argmin(dists_to_boundary)
        min_b_dist = dists_to_boundary[min_b_idx]
        
        current_r = min(min_dist / 2.0, min_b_dist)
        
        # If pair distance is limiting
        if min_dist / 2.0 < min_b_dist:
            # Push i and j apart
            d = centers[i] - centers[j]
            if np.linalg.norm(d) < 1e-9:
                d = np.random.normal(0, 0.01, 3)
            d_norm = np.linalg.norm(d)
            dir_vec = d / d_norm
            
            # Try a small push
            # We want to increase min_dist.
            # Push amount
            push = 0.001
            new_centers = centers.copy()
            new_centers[i] += dir_vec * push
            new_centers[j] -= dir_vec * push
            new_centers[i] = np.clip(new_centers[i], -0.5, 0.5)
            new_centers[j] = np.clip(new_centers[j], -0.5, 0.5)
            
            new_score = _compute_score(new_centers)
            if new_score > current_r:
                centers = new_centers
                if new_score > best_score:
                    best_score = new_score
                    best_centers = centers.copy()
            else:
                # If push didn't help, maybe move other spheres?
                # Or try smaller push?
                # For now, just break or try a different move.
                # Let's try moving a random sphere slightly to break symmetry
                k = random.randint(0, N-1)
                if k not in (i, j):
                    new_centers = centers.copy()
                    new_centers[k] += np.random.normal(0, 0.005, 3)
                    new_centers[k] = np.clip(new_centers[k], -0.5, 0.5)
                    new_score = _compute_score(new_centers)
                    if new_score > current_r:
                        centers = new_centers
                        if new_score > best_score:
                            best_score = new_score
                            best_centers = centers.copy()
        else:
            # Boundary is limiting
            # Push the sphere closest to boundary away from boundary
            k = min_b_idx
            # Determine which boundary
            # If center is close to 0.5, push towards -0.5
            # If center is close to -0.5, push towards 0.5
            # Actually, just push towards center of cube (0,0,0)
            dir_vec = -centers[k]
            d_norm = np.linalg.norm(dir_vec)
            if d_norm > 1e-9:
                dir_vec /= d_norm
            else:
                dir_vec = np.random.normal(0, 1, 3)
                dir_vec /= np.linalg.norm(dir_vec)
                
            push = 0.001
            new_centers = centers.copy()
            new_centers[k] += dir_vec * push
            new_centers[k] = np.clip(new_centers[k], -0.5, 0.5)
            
            new_score = _compute_score(new_centers)
            if new_score > current_r:
                centers = new_centers
                if new_score > best_score:
                    best_score = new_score
                    best_centers = centers.copy()
            else:
                # Try moving another sphere
                m = random.randint(0, N-1)
                if m != k:
                    new_centers = centers.copy()
                    new_centers[m] += np.random.normal(0, 0.005, 3)
                    new_centers[m] = np.clip(new_centers[m], -0.5, 0.5)
                    new_score = _compute_score(new_centers)
                    if new_score > current_r:
                        centers = new_centers
                        if new_score > best_score:
                            best_score = new_score
                            best_centers = centers.copy()

    return best_centers, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    best_overall = None
    best_score_overall = -np.inf
    
    # Multiple restarts
    n_restarts = 5
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        # 1. Random Init
        centers = _random_init()
        
        # 2. Push Apart (Relaxation)
        centers = _push_apart(centers, iterations=50, step_size=0.1)
        
        # 3. Simulated Annealing
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        if remaining < 5:
            break
        sa_time = min(remaining * 0.7, 40.0)
        centers, score = _simulated_annealing(centers, time_limit=sa_time)
        
        # 4. Polish
        remaining = deadline - time.time()
        if remaining < 5:
            break
        polish_time = min(remaining * 0.8, 30.0)
        centers, score = _polish_best_pair(centers, iterations=1000, time_limit=polish_time)
        
        if score > best_score_overall:
            best_score_overall = score
            best_overall = centers.copy()
            
    if best_overall is None:
        # Fallback to grid if something went wrong
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
        
    return best_overall
# EVOLVE-BLOCK-END
