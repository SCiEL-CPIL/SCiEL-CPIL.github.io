import numpy as np
import time
from scipy.spatial.distance import pdist

N = 50

def _compute_score(centers):
    """Compute the maximum radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    if N < 2:
        min_dist = np.inf
    else:
        dists = pdist(centers, metric='euclidean')
        min_dist = np.min(dists)
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_dist / 2.0, min_wall)

def _hilbert_curve_3d(n_points, seed=42):
    """Generate 3D points using a Hilbert-like space-filling curve mapping.
    We map 1D indices to 3D coordinates using a bit-interleaving approach
    that approximates a Hilbert curve for better space-filling properties."""
    rng = np.random.default_rng(seed)
    
    # Use a simple approach: map 1D index to 3D via base-4 digit decomposition
    # This gives a reasonable space-filling pattern
    points = np.zeros((n_points, 3))
    
    # For a better space-filling, use a 3D Z-order (Morton) curve with randomization
    # Map indices to Morton codes
    for i in range(n_points):
        # Convert i to 3D coordinates using bit interleaving
        # Use a 6-bit resolution per axis (64^3 = 262144 possible positions)
        bits = i
        x = 0
        y = 0
        z = 0
        for bit in range(12):  # 12 bits = 4 bits per axis
            b = (bits >> bit) & 1
            if bit % 3 == 0:
                x |= (b << (bit // 3))
            elif bit % 3 == 1:
                y |= (b << (bit // 3))
            else:
                z |= (b << (bit // 3))
        # Normalize to [0, 1)
        points[i, 0] = x / 4096.0
        points[i, 1] = y / 4096.0
        points[i, 2] = z / 4096.0
    
    # Shuffle to break any remaining correlation
    perm = rng.permutation(n_points)
    points = points[perm]
    
    # Map to [-0.5, 0.5] with small margin
    margin = 0.02
    points = points * (1.0 - 2 * margin) + (-0.5 + margin)
    
    return points

def _optimize_sa(centers, time_budget, target_r=0.135):
    """Simulated annealing optimization."""
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_score = _compute_score(current)
    
    # Simulated annealing parameters
    T = 0.005
    T_min = 1e-10
    cooling = 0.99995
    step_size = 0.008
    
    rng = np.random.default_rng(123)
    
    # Track the minimum pair distance to adaptively adjust target
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
        
        # Propose a move
        idx = rng.integers(N)
        delta = rng.standard_normal(3) * step_size
        new_centers = current.copy()
        new_centers[idx] += delta
        
        # Keep within bounds with small margin
        new_centers = np.clip(new_centers, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Compute energy difference
        # Energy = sum of (max(0, 2*target_r - d_ij))^2 for pairs + wall violations
        # Use a faster approximation: only check moved point's interactions
        
        # Distance from moved point to all others
        diffs = new_centers[idx] - current
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        dists[idx] = np.inf  # Ignore self
        
        # Overlap penalty for pairs involving idx
        overlaps = np.maximum(0, 2 * target_r - dists)
        pair_energy_new = np.sum(overlaps**2)
        
        # Wall violation for idx
        wall_viol = np.maximum(0, np.abs(new_centers[idx]) - (0.5 - target_r))
        wall_energy_new = np.sum(wall_viol**2)
        
        new_energy = pair_energy_new + 10.0 * wall_energy_new
        
        # Old energy for idx (approximate)
        diffs_old = current[idx] - current
        dists_old = np.sqrt(np.sum(diffs_old**2, axis=1))
        dists_old[idx] = np.inf
        overlaps_old = np.maximum(0, 2 * target_r - dists_old)
        pair_energy_old = np.sum(overlaps_old**2)
        wall_viol_old = np.maximum(0, np.abs(current[idx]) - (0.5 - target_r))
        wall_energy_old = np.sum(wall_viol_old**2)
        old_energy = pair_energy_old + 10.0 * wall_energy_old
        
        delta_energy = new_energy - old_energy
        
        # Accept if better or with probability exp(-deltaE/T)
        if delta_energy < 0 or rng.random() < np.exp(-delta_energy / max(T, 1e-15)):
            current = new_centers
            # Periodically check full score
            if rng.random() < 0.01:
                current_score = _compute_score(current)
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()
        
        T *= cooling
        # Gradually reduce step size
        step_size *= 0.99999
        
        # Adaptively update target_r to current best + small margin
        if best_score > 0.13:
            target_r = min(0.14, best_score * 1.05)
    
    # Final score check
    final_score = _compute_score(current)
    if final_score > best_score:
        best_score = final_score
        best = current.copy()
    
    return best, best_score

def _optimize_gradient(centers, time_budget):
    """Gradient-based optimization using smooth repulsive potential."""
    start_time = time.time()
    pts = centers.copy()
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    lr = 0.005
    momentum = 0.9
    velocity = np.zeros_like(pts)
    
    for step in range(100000):
        if time.time() - start_time > time_budget:
            break
        
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1e10)
        
        # Smooth repulsive force: F_ij = diff_ij / (dist_sq_ij + eps)
        # This is the gradient of -sum 1/dist_sq (negative gradient = descent)
        # Actually, to maximize min distance, minimize sum 1/dist^2
        # d(1/d^2)/d(c_i) = -2 * (c_i - c_j) / d^4 = -2 * diff / dist_sq^2
        # So gradient of energy = -2 * sum_j diff_ij / dist_sq_ij^2
        # Force (negative gradient) = 2 * sum_j diff_ij / dist_sq_ij^2
        
        factors = 1.0 / (dist_sq**2)
        np.fill_diagonal(factors, 0.0)
        
        pair_forces = np.sum(factors[:, :, np.newaxis] * diff, axis=1)
        
        # Wall forces: potential U = sum 1/(0.5 - |x|)^2
        # Force = -dU/dx = -2 * sign(x) / (0.5 - |x|)^3
        wall_dist = 0.5 - np.abs(pts)
        wall_dist_safe = np.maximum(wall_dist, 1e-4)
        signs = np.sign(pts)
        wall_forces = -2.0 * signs / (wall_dist_safe**3)
        
        # Scale wall forces to be comparable
        # Pair forces scale as ~1/d^4, wall forces as ~1/w^3
        # At d~0.13, pair ~ 1/0.0003 ~ 3000; at w~0.13, wall ~ 1/0.002 ~ 500
        # Scale wall forces up a bit
        total_forces = pair_forces + 0.5 * wall_forces
        
        # Normalize to prevent explosion
        norms = np.linalg.norm(total_forces, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = total_forces / norms
        
        # Momentum update
        velocity = momentum * velocity + (1 - momentum) * normalized
        pts = pts + lr * velocity
        
        # Clamp
        pts = np.clip(pts, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Check score periodically
        if step % 50 == 0:
            current_score = _compute_score(pts)
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
            else:
                # Decay learning rate if no improvement
                lr *= 0.999
        
        # Early stop if learning rate too small
        if lr < 1e-8:
            break
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    best_radius = 0.0
    best_centers = None
    
    # Multiple initialization strategies
    initializations = []
    
    # 1. Hilbert-curve based initialization
    initializations.append(_hilbert_curve_3d(N, seed=42))
    
    # 2. Random initialization
    rng = np.random.default_rng(42)
    initializations.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
    
    # 3. Grid-based (4x4x4, take 50)
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    indices = rng.choice(len(grid), size=N, replace=False)
    initializations.append(grid[indices])
    
    # 4. Another Hilbert with different seed
    initializations.append(_hilbert_curve_3d(N, seed=7))
    
    # Phase 1: Simulated annealing on each initialization
    for i, init in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        
        remaining = deadline - 30 - time.time()
        if remaining < 5:
            break
        
        centers, radius = _optimize_sa(init, time_budget=min(remaining, 30), target_r=0.135)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers
    
    # Phase 2: Gradient-based polish on the best result
    if time.time() < deadline - 10 and best_centers is not None:
        remaining = deadline - 10 - time.time()
        if remaining > 5:
            centers, radius = _optimize_gradient(best_centers, time_budget=min(remaining, 40))
            if radius > best_radius:
                best_radius = radius
                best_centers = centers
    
    # Phase 3: Final fine-tuning with very small SA steps
    if time.time() < deadline - 5 and best_centers is not None:
        remaining = deadline - 5 - time.time()
        if remaining > 3:
            centers, radius = _optimize_sa(best_centers, time_budget=min(remaining, 15), target_r=max(0.136, best_radius))
            if radius > best_radius:
                best_radius = radius
                best_centers = centers
    
    if best_centers is None:
        # Fallback
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
