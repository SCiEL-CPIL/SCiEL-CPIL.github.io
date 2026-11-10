"""Improved packing search: force-directed Lennard-Jones with simulated annealing."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(c):
    """Compute the maximum common radius for centers c (N,3)."""
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _compute_forces(c, sigma=0.22, epsilon=1.0, wall_stiffness=50.0):
    """
    Compute Lennard-Jones forces between spheres and wall forces.
    Returns: forces (N,3)
    """
    c = np.asarray(c)
    n = c.shape[0]
    forces = np.zeros_like(c)
    
    # Pairwise Lennard-Jones forces
    # F_ij = 24*epsilon * (2*(sigma/d)^12 - (sigma/d)^6) * (sigma^6/d^5) * (c_i - c_j)/d
    # Simplified: F_ij = 24*epsilon * (2*sigma^12/d^13 - sigma^6/d^7) * (c_i - c_j)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]  # (n,n,3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n,n)
    np.fill_diagonal(dist_sq, np.inf)
    
    dist = np.sqrt(dist_sq + 1e-30)
    
    # LJ potential: U = 4*epsilon*((sigma/r)^12 - (sigma/r)^6)
    # Force magnitude: F = -dU/dr = 24*epsilon*(2*sigma^12/r^13 - sigma^6/r^7)
    # Force direction: (c_i - c_j)/r
    
    sigma6 = sigma ** 6
    sigma12 = sigma ** 12
    
    # Only compute for pairs within cutoff (4*sigma)
    cutoff_sq = (4.0 * sigma) ** 2
    mask = dist_sq < cutoff_sq
    
    inv_dist = np.zeros_like(dist)
    inv_dist[mask] = 1.0 / dist[mask]
    
    inv_dist2 = inv_dist ** 2
    inv_dist5 = inv_dist2 * inv_dist2 * inv_dist
    inv_dist11 = inv_dist5 * inv_dist5 * inv_dist
    
    # Force coefficient: 24*epsilon*(2*sigma^12/r^13 - sigma^6/r^7)
    # = 24*epsilon*(2*sigma^12/r^13 - sigma^6/r^7)
    # = 24*epsilon*inv_dist5*(2*sigma^12*inv_dist2 - sigma^6)
    coeff = 24.0 * epsilon * inv_dist5 * (2.0 * sigma12 * inv_dist2 - sigma6)
    
    # Zero out diagonal and non-cutoff pairs
    coeff[~mask] = 0.0
    np.fill_diagonal(coeff, 0.0)
    
    # Force on i from j: coeff[i,j] * (c_i - c_j)
    # Total force on i: sum_j coeff[i,j] * (c_i - c_j)
    # = c_i * sum_j coeff[i,j] - sum_j coeff[i,j] * c_j
    
    coeff_sum = np.sum(coeff, axis=1)  # (n,)
    coeff_c = coeff[:, :, np.newaxis] * c[np.newaxis, :, :]  # (n,n,3)
    coeff_c_sum = np.sum(coeff_c, axis=1)  # (n,3)
    
    forces += c * coeff_sum[:, np.newaxis] - coeff_c_sum
    
    # Wall forces: push spheres back into cube
    # For each coordinate, if c > 0.5 - margin, push left; if c < -0.5 + margin, push right
    margin = 0.05
    for k in range(3):
        # Right wall
        excess = np.maximum(0.0, c[:, k] - (0.5 - margin))
        forces[:, k] -= wall_stiffness * excess
        # Left wall
        excess = np.maximum(0.0, (-0.5 + margin) - c[:, k])
        forces[:, k] += wall_stiffness * excess
    
    return forces


def _simulated_annealing(c0, n_steps=5000, T_init=0.01, T_final=1e-6, 
                         sigma=0.22, epsilon=1.0, wall_stiffness=50.0,
                         step_size=0.005, deadline=None, start_time=None):
    """
    Simulated annealing with Lennard-Jones forces.
    Returns: optimized centers, best radius
    """
    if start_time is None:
        start_time = time.time()
    
    c = c0.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Precompute sigma powers
    sigma6 = sigma ** 6
    sigma12 = sigma ** 12
    
    for step in range(n_steps):
        # Check time
        if deadline is not None and time.time() > deadline:
            break
        
        # Temperature schedule (exponential cooling)
        alpha = step / max(1, n_steps - 1)
        T = T_init * (T_final / T_init) ** alpha
        
        # Compute forces
        forces = _compute_forces(c, sigma=sigma, epsilon=epsilon, wall_stiffness=wall_stiffness)
        
        # Update positions: c += step_size * forces + noise
        noise = np.random.normal(0, T * 0.1, size=c.shape)
        c += step_size * forces + noise
        
        # Keep in cube (hard constraint)
        c = np.clip(c, -0.4999, 0.4999)
        
        # Check if improved
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    return best_c, best_r


def _polish(c, n_iter=2000, step_size=0.001, deadline=None, start_time=None):
    """
    Final polishing: gradient descent on negative radius with small steps.
    Uses subgradient of the min constraint.
    """
    if start_time is None:
        start_time = time.time()
    
    c = c0 = c.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    tri_idx = np.triu_indices(N, k=1)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2) + 1e-30)
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(c)
        min_wall = np.min(wall_dists)
        
        min_pair = np.min(dists)
        r = min(min_pair / 2.0, min_wall)
        
        if r > best_r:
            best_r = r
            best_c = c.copy()
        
        # Find which constraint is active
        pair_dists = dists[tri_idx] / 2.0
        wall_vals = wall_dists.flatten()
        all_vals = np.concatenate([pair_dists, wall_vals])
        
        # Active constraint (minimum)
        min_idx = np.argmin(all_vals)
        
        # Gradient of -r
        grad = np.zeros_like(c)
        
        if min_idx < len(pair_dists):
            # Pair constraint active
            i, j = tri_idx[0][min_idx], tri_idx[1][min_idx]
            d = c[i] - c[j]
            dist = np.linalg.norm(d) + 1e-30
            unit = d / dist
            # To increase distance, move i in direction of unit, j in opposite
            grad[i] -= unit  # moving i away from j increases distance
            grad[j] += unit
        else:
            # Wall constraint active
            wall_idx = min_idx - len(pair_dists)
            i = wall_idx // 3
            k = wall_idx % 3
            # c[i,k] is too close to wall
            # If c[i,k] > 0, it's near +0.5 wall, move left (negative)
            # If c[i,k] < 0, it's near -0.5 wall, move right (positive)
            if c[i, k] >= 0:
                grad[i, k] = -1.0
            else:
                grad[i, k] = 1.0
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad /= grad_norm
        
        # Update
        c += step_size * grad
        c = np.clip(c, -0.4999, 0.4999)
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_c = None
    best_r = -np.inf
    
    np.random.seed(42)
    
    # Phase 1: Simulated annealing from multiple starts
    n_sa_restarts = 8
    
    for restart in range(n_sa_restarts):
        if time.time() > deadline - 20:
            break
        
        # Generate initial configuration
        if restart == 0:
            # 5x5x5 grid (first 100 points)
            m = 5
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            c0 = grid[:N].copy()
        elif restart == 1:
            # Different grid offset
            m = 5
            ticks = (np.arange(m) + 0.2) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            c0 = grid[:N].copy()
        elif restart == 2:
            # FCC-like structure
            # Simple cubic with face diagonals
            base = np.array([(x, y, z) for z in [0, 0.5] for y in [0, 0.5] for x in [0, 0.5]])
            base = base / 2.0 - 0.5
            c0 = base + np.random.normal(0, 0.01, size=(8, 3))
            # Repeat to get 100
            c0 = np.tile(c0, (13, 1))[:N]
            c0 += np.random.normal(0, 0.02, size=(N, 3))
        else:
            # Random with clustering
            n_clusters = 5
            cluster_centers = np.random.uniform(-0.35, 0.35, size=(n_clusters, 3))
            c0 = np.zeros((N, 3))
            for i in range(N):
                cl = i % n_clusters
                c0[i] = cluster_centers[cl] + np.random.normal(0, 0.08, size=3)
            c0 = np.clip(c0, -0.45, 0.45)
        
        # Run simulated annealing
        c_opt, r_opt = _simulated_annealing(
            c0, n_steps=3000, T_init=0.005, T_final=1e-6,
            sigma=0.22, epsilon=1.0, wall_stiffness=30.0,
            step_size=0.003, deadline=deadline, start_time=start_time
        )
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
        
        # Polish the result
        if time.time() < deadline - 15:
            c_polished, r_polished = _polish(
                c_opt, n_iter=1500, step_size=0.0005,
                deadline=deadline, start_time=start_time
            )
            if r_polished > best_r:
                best_r = r_polished
                best_c = c_polished.copy()
    
    # Phase 2: Fine-tune with very small steps
    if best_c is not None and time.time() < deadline - 5:
        c_final, r_final = _polish(
            best_c, n_iter=3000, step_size=0.0001,
            deadline=deadline, start_time=start_time
        )
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
    
    # Fallback
    if best_c is None:
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    # Ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c

# EVOLVE-BLOCK-END
