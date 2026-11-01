"""Optimized circle packing: simulated annealing + local refinement."""
import numpy as np
import time

N = 50


def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    # Distance to walls
    wall_dists = np.min(np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # Use cdist for efficiency
    from scipy.spatial.distance import cdist
    D = cdist(centers, centers)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    
    return min(min_wall, min_pair)


def _soft_radius(centers, beta=50.0):
    """Soft surrogate for the radius using log-sum-exp style soft-min."""
    # Wall constraints
    wx = 0.5 - np.abs(centers[:, 0])
    wy = 0.5 - np.abs(centers[:, 1])
    wall_min = np.minimum(wx, wy)
    
    # Soft min over walls: -logsumexp(-beta * wall_min) / beta
    # But we want a smooth approximation of min
    # softmin(x) = -(1/beta) * log(sum(exp(-beta * x)))
    # For large beta, this approaches min(x)
    
    # Pairwise distances / 2
    from scipy.spatial.distance import cdist
    D = cdist(centers, centers)
    np.fill_diagonal(D, 0.0)
    D_half = D / 2.0
    
    # Flatten pairwise (i<j)
    iu = np.triu_indices(N, k=1)
    pairs = D_half[iu]
    
    # Soft min over pairs
    # softmin(pairs) = -(1/beta)*log(sum(exp(-beta*pairs)))
    # Use stable logsumexp
    def softmin(x, beta):
        m = np.min(x)
        return m - np.log(np.sum(np.exp(-beta * (x - m)))) / beta
    
    soft_wall = softmin(wall_min, beta)
    soft_pair = softmin(pairs, beta)
    
    # Soft min of the two
    return softmin(np.array([soft_wall, soft_pair]), beta)


def _init_hexagonal():
    """Initialize with a hexagonal packing pattern, clipped to square."""
    # Start with a grid and perturb
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random perturbation
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.01, size=centers.shape)
    
    # Clip to valid range
    centers[:, 0] = np.clip(centers[:, 0], -0.48, 0.48)
    centers[:, 1] = np.clip(centers[:, 1], -0.48, 0.48)
    
    return centers


def _local_refinement(centers, max_iter=2000, tol=1e-12):
    """Local refinement using gradient ascent on the radius."""
    from scipy.spatial.distance import cdist
    
    best_r = _compute_radius(centers)
    centers = centers.copy()
    
    for it in range(max_iter):
        # Compute gradient of radius w.r.t. each center
        # The radius is min of all pairwise distances/2 and wall distances
        # We do a simple subgradient ascent
        
        D = cdist(centers, centers)
        np.fill_diagonal(D, np.inf)
        
        # Find closest pair
        flat_idx = np.argmin(D)
        i, j = np.unravel_index(flat_idx, (N, N))
        
        # Wall distances
        wx = 0.5 - np.abs(centers[:, 0])
        wy = 0.5 - np.abs(centers[:, 1])
        wall_min_idx = np.argmin(np.minimum(wx, wy))
        
        # Determine which constraint is active
        min_pair_dist = D[i, j] / 2.0
        min_wall_dist = np.min(np.minimum(wx, wy))
        
        grad = np.zeros_like(centers)
        
        if min_pair_dist < min_wall_dist:
            # Pair constraint active
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-15:
                unit = diff / dist
                grad[i] += unit
                grad[j] -= unit
        else:
            # Wall constraint active
            idx = wall_min_idx
            if wx[idx] <= wy[idx]:
                # x-wall
                if centers[idx, 0] > 0:
                    grad[idx, 0] -= 1.0
                else:
                    grad[idx, 0] += 1.0
            else:
                # y-wall
                if centers[idx, 1] > 0:
                    grad[idx, 1] -= 1.0
                else:
                    grad[idx, 1] += 1.0
        
        # Also add contributions from all near-active constraints for smoother ascent
        # Add small repulsion from all pairs
        for a in range(N):
            for b in range(a+1, N):
                diff = centers[a] - centers[b]
                dist = np.linalg.norm(diff)
                if dist < 0.15:  # Only nearby pairs
                    if dist > 1e-15:
                        weight = (0.15 - dist) / 0.15
                        unit = diff / dist
                        grad[a] += weight * unit * 0.1
                        grad[b] -= weight * unit * 0.1
        
        # Wall repulsion for near-boundary points
        for a in range(N):
            for axis in range(2):
                val = np.abs(centers[a, axis])
                if val > 0.4:
                    wall_dist = 0.5 - val
                    if wall_dist < 0.05:
                        weight = (0.05 - wall_dist) / 0.05
                        if centers[a, axis] > 0:
                            grad[a, axis] -= weight * 0.1
                        else:
                            grad[a, axis] += weight * 0.1
        
        # Update
        step = 0.005
        new_centers = centers + step * grad
        
        # Clip to valid range
        new_centers[:, 0] = np.clip(new_centers[:, 0], -0.499, 0.499)
        new_centers[:, 1] = np.clip(new_centers[:, 1], -0.499, 0.499)
        
        new_r = _compute_radius(new_centers)
        
        if new_r > best_r:
            centers = new_centers
            best_r = new_r
            if best_r - _compute_radius(centers) < tol:
                break
        else:
            # Try smaller step
            step *= 0.5
            if step < 1e-8:
                break
    
    return centers, best_r


def construct():
    start_time = time.time()
    deadline = 125.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(12345)
    
    # Initialize
    centers = _init_hexagonal()
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    # Simulated annealing
    T_start = 0.05
    T_end = 0.0001
    n_iterations = 50000
    cooling_rate = (T_end / T_start) ** (1.0 / n_iterations)
    
    T = T_start
    
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        
        # Propose a move
        idx = rng.integers(0, N)
        move_size = T * 2.0
        delta = rng.normal(0, move_size, size=2)
        
        new_centers = centers.copy()
        new_centers[idx] += delta
        new_centers[idx, 0] = np.clip(new_centers[idx, 0], -0.499, 0.499)
        new_centers[idx, 1] = np.clip(new_centers[idx, 1], -0.499, 0.499)
        
        new_r = _compute_radius(new_centers)
        
        # Accept if better or with probability exp((new_r - old_r)/T)
        if new_r >= best_r:
            centers = new_centers
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
        else:
            delta_r = new_r - _compute_radius(centers)
            if delta_r > 0:
                centers = new_centers
            else:
                prob = np.exp(delta_r / T)
                if rng.random() < prob:
                    centers = new_centers
        
        T *= cooling_rate
        
        # Periodic local refinement
        if it % 5000 == 0 and it > 0:
            if time.time() > deadline - 10:
                break
            refined, refined_r = _local_refinement(centers, max_iter=500)
            if refined_r > best_r:
                best_r = refined_r
                best_centers = refined.copy()
                centers = refined
    
    # Final local refinement with more iterations
    if time.time() < deadline - 5:
        refined, refined_r = _local_refinement(best_centers, max_iter=5000)
        if refined_r > best_r:
            best_r = refined_r
            best_centers = refined
    
    return best_centers
