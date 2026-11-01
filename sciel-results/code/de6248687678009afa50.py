import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _softmin_radius_and_grad(centers, beta):
    """
    Compute a smooth approximation of the minimum radius using soft-min via log-sum-exp.
    
    The radius r is the minimum of:
    - d_ij / 2 for all pairs i<j
    - 0.5 - max(|c_ix|, |c_iy|, |c_iz|) for all i,k
    
    We use a soft-min: -log(exp(-x_1/beta) + ... + exp(-x_m/beta)) / beta
    as beta -> 0, this approaches the true minimum.
    
    Returns: (r_soft, gradient wrt centers)
    """
    c = centers.reshape(N, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1e-15)  # avoid zero
    dist = np.sqrt(dist2)
    
    # Get upper triangle (i < j) to avoid double counting
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu] / 2.0  # r from pair distance
    
    # Wall distances: 0.5 - max(|cx|, |cy|, |cz|)
    abs_c = np.abs(c)
    max_coord = np.max(abs_c, axis=1)
    wall_dists = 0.5 - max_coord
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft-min via log-sum-exp: r_soft = -log(sum(exp(-d/beta))) / beta
    # For numerical stability, shift by min
    min_d = np.min(all_dists)
    shifted = -all_dists / beta
    shifted -= np.max(shifted)  # shift for stability
    log_sum = np.log(np.sum(np.exp(shifted)))
    r_soft = min_d - beta * log_sum
    
    # Gradient of r_soft wrt centers
    # d(r_soft)/d(centers) = -sum_i (exp(-d_i/beta) / sum_j exp(-d_j/beta)) * d(d_i)/d(centers)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    
    # Gradient of pair distances wrt centers
    # d(d_ij/2)/d(c_i) = (c_i - c_j) / (2 * d_ij)
    # d(d_ij/2)/d(c_j) = (c_j - c_i) / (2 * d_ij)
    
    grad = np.zeros_like(c)
    
    # Pair contributions
    for idx in range(len(pair_dists)):
        i, j = iu[0][idx], iu[1][idx]
        d_ij = dist[i, j]
        w = weights[idx] / beta
        # dr/d(c_i) = w * (c_i - c_j) / (2 * d_ij)
        # dr/d(c_j) = w * (c_j - c_i) / (2 * d_ij)
        if d_ij > 1e-10:
            factor = w / (2.0 * d_ij)
            grad[i] += factor * (c[i] - c[j])
            grad[j] += factor * (c[j] - c[i])
    
    # Wall contributions
    wall_start = len(pair_dists)
    for idx in range(len(wall_dists)):
        i = idx
        w = weights[wall_start + idx] / beta
        # d(0.5 - max(|c_ik|))/d(c_ik) = -sign(c_ik) if this is the max component
        # For soft version, we need to handle this more carefully
        # Actually, the wall distance is 0.5 - max(|cx|, |cy|, |cz|)
        # The gradient of max(|c_ik|) is sign(c_ik) for the argmax component
        # But for a smooth version, we'd need soft-max. Let's use a simple approach:
        # Approximate gradient as -sign(c_ik) * w for each component where it's close to max
        for k in range(3):
            if abs_c[i, k] > 0.01:  # only for non-zero components
                # Weight by how dominant this component is
                dominance = abs_c[i, k] / (np.sum(abs_c[i]) + 1e-10)
                grad[i, k] -= w * dominance * np.sign(c[i, k])
    
    return r_soft, grad

def _anneal_objective(centers_flat, beta):
    """Objective function for annealing: maximize soft-min radius."""
    centers = centers_flat.reshape(N, 3)
    r_soft, _ = _softmin_radius_and_grad(centers, beta)
    return -r_soft  # minimize negative radius

def _simulate_anneal(centers, beta_start, beta_end, n_steps, timeout):
    """
    Simulated annealing on the soft-min radius.
    Returns best centers found.
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_r = -np.inf
    current_r = -np.inf
    
    for step in range(n_steps):
        if time.time() - start_time > timeout:
            break
        
        # Linear annealing schedule
        t = step / max(1, n_steps - 1)
        beta = beta_start * (beta_end / beta_start) ** t
        
        # Try to improve using L-BFGS-B
        result = minimize(
            _anneal_objective,
            centers.flatten(),
            args=(beta,),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (3 * N),
            options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        new_centers = result.x.reshape(N, 3)
        new_r, _ = _softmin_radius_and_grad(new_centers, beta)
        
        if new_r > current_r or (step < 10):
            current_r = new_r
            centers = new_centers
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers.copy()
        else:
            # Accept with probability exp((new_r - current_r) / temperature)
            # Temperature related to beta
            temp = beta * 0.1
            if temp > 1e-10:
                prob = np.exp((new_r - current_r) / temp)
                if np.random.random() < prob:
                    current_r = new_r
                    centers = new_centers
                    if new_r > best_r:
                        best_r = new_r
                        best_centers = new_centers.copy()
    
    return best_centers

def _compute_radius(centers):
    """Compute the exact minimum radius for a configuration."""
    c = centers.reshape(N, 3)
    
    # Check bounds
    abs_c = np.abs(c)
    if np.any(abs_c > 0.5 + 1e-10):
        return 0.0
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1e-15)
    dist = np.sqrt(dist2)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu] / 2.0
    
    # Wall distances
    max_coord = np.max(abs_c, axis=1)
    wall_dists = 0.5 - max_coord
    
    all_dists = np.concatenate([pair_dists, wall_dists])
    return np.min(all_dists)

def _polish(centers, max_iter=200):
    """High-precision polish using Nelder-Mead on the exact radius."""
    def objective(centers_flat):
        return -_compute_radius(centers_flat.reshape(N, 3))
    
    result = minimize(
        objective,
        centers.flatten(),
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-12}
    )
    
    return result.x.reshape(N, 3)

def _lloyd_step(centers, n_iter=10):
    """
    Lloyd's algorithm step: move each center to the centroid of its Voronoi cell.
    This helps spread out the points.
    """
    c = centers.copy()
    
    for _ in range(n_iter):
        new_c = c.copy()
        
        for i in range(N):
            # Find the region where point i is the closest
            # For simplicity, use a local neighborhood approach
            # Compute distances to all other points
            diffs = c[i] - c
            dists = np.sqrt(np.sum(diffs * diffs, axis=1))
            
            # Find points within some radius (local Voronoi approximation)
            # Use a fixed radius based on average spacing
            avg_spacing = 0.15
            mask = dists < avg_spacing
            mask[i] = False
            
            if np.sum(mask) > 0:
                # Move away from neighbors
                neighbor_diffs = c[i] - c[mask]
                neighbor_dists = dists[mask]
                
                # Weighted repulsion
                forces = np.zeros(3)
                for j in range(len(neighbor_diffs)):
                    d = neighbor_dists[j]
                    if d > 1e-10:
                        force = neighbor_diffs[j] / (d * d)
                        forces += force
                
                new_c[i] = c[i] + 0.01 * forces
        
        # Project back to cube
        new_c = np.clip(new_c, -0.5 + 1e-6, 0.5 - 1e-6)
        c = new_c
    
    return c

def _local_search(centers, n_restarts=3, timeout=30):
    """Local search with multiple restarts."""
    start_time = time.time()
    best_centers = centers.copy()
    best_r = _compute_radius(best_centers)
    
    for restart in range(n_restarts):
        if time.time() - start_time > timeout:
            break
        
        # Jitter the centers
        jittered = centers + np.random.normal(0, 0.01, size=centers.shape)
        jittered = np.clip(jittered, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Polish
        polished = _polish(jittered, max_iter=100)
        r = _compute_radius(polished)
        
        if r > best_r:
            best_r = r
            best_centers = polished
    
    return best_centers

def _initial_fcc_like():
    """
    Initialize with an FCC-like lattice pattern, then adjust for 50 spheres.
    FCC has 4 atoms per unit cell. For 50 spheres, use about 12.5 cells.
    """
    # Create a 4x4x4 grid (64 points), then remove 14 to get 50
    # Or use a 3x4x5 grid = 60, remove 10
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Add FCC offset points
    fcc_offsets = [
        (0.5/m, 0.5/m, 0.5/m),
        (0.5/m, 0.5/m, -0.5/m),
        (0.5/m, -0.5/m, 0.5/m),
        (-0.5/m, 0.5/m, 0.5/m),
    ]
    
    all_points = [grid]
    for offset in fcc_offsets:
        fcc_grid = grid + np.array(offset)
        # Keep only points inside the cube
        fcc_grid = fcc_grid[np.all(np.abs(fcc_grid) < 0.5 - 1e-6, axis=1)]
        all_points.append(fcc_grid)
    
    combined = np.vstack(all_points)
    
    # Randomly select 50 points, preferring well-distributed ones
    if len(combined) >= N:
        # Use a simple greedy selection: pick points that maximize minimum distance
        selected = [combined[0]]
        while len(selected) < N and len(combined) > len(selected):
            candidates = combined[len(selected):]
            if len(candidates) == 0:
                break
            
            # Compute min distance from each candidate to selected points
            min_dists = []
            for cand in candidates:
                dists = np.sqrt(np.sum((cand - np.array(selected))**2, axis=1))
                min_dists.append(np.min(dists) if len(dists) > 0 else 0)
            
            # Pick the candidate with maximum min distance
            max_idx = np.argmax(min_dists)
            selected.append(candidates[max_idx])
        
        return np.array(selected[:N])
    else:
        # Fallback to simple grid
        m2 = int(np.ceil(N ** (1.0/3.0)))
        ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
        grid2 = np.array([(x, y, z) for z in ticks2 for y in ticks2 for x in ticks2])
        return grid2[:N]

def construct():
    start_time = time.time()
    timeout = 115  # Safety margin before 120s limit
    
    # Initialize
    centers = _initial_fcc_like()
    
    # Phase 1: Annealing with soft-min
    beta_start = 0.1
    beta_end = 0.001
    n_anneal_steps = 30
    anneal_timeout = 60
    
    centers = _simulate_anneal(centers, beta_start, beta_end, n_anneal_steps, anneal_timeout)
    
    # Phase 2: Lloyd's algorithm to improve distribution
    if time.time() - start_time < timeout - 20:
        centers = _lloyd_step(centers, n_iter=5)
    
    # Phase 3: Local search with restarts
    local_timeout = 20
    if time.time() - start_time < timeout - local_timeout:
        centers = _local_search(centers, n_restarts=2, timeout=local_timeout)
    
    # Phase 4: Final high-precision polish
    if time.time() - start_time < timeout - 5:
        centers = _polish(centers, max_iter=300)
    
    # Ensure centers are strictly inside the cube
    centers = np.clip(centers, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return centers
# EVOLVE-BLOCK-END
