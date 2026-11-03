# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _hex_init(n, seed=0):
    """Generate n points from a hexagonal lattice inside the unit disk."""
    rng = np.random.RandomState(seed)
    best_pts = None
    best_r = -1.0
    
    # Try a few different spacings to ensure we get at least n points
    for attempt in range(50):
        spacing = 0.05 + rng.random() * 0.02
        k = int(2.0 / spacing) + 5
        xs, ys = np.meshgrid(
            np.arange(-k, k + 1) * spacing,
            np.arange(-k, k + 1) * spacing * np.sqrt(3.0) / 2.0
        )
        for j in range(2 * k + 1):
            if j % 2 == 1:
                xs[j, :] += spacing / 2.0
        pts = np.column_stack([xs.ravel(), ys.ravel()])
        r = np.hypot(pts[:, 0], pts[:, 1])
        mask = r <= 1.0 - spacing / 2.0 - 0.001
        pts = pts[mask]
        
        if len(pts) >= n:
            # Select n points closest to center to maximize potential r
            idx = np.argsort(r[mask])[:n]
            cand_pts = pts[idx].copy()
            
            # Calculate current radius for this candidate
            diff = cand_pts[:, np.newaxis, :] - cand_pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            boundary_dists = 1.0 - np.hypot(cand_pts[:, 0], cand_pts[:, 1])
            min_boundary = np.min(boundary_dists)
            r_val = min(min_pair_dist / 2.0, min_boundary)
            
            if r_val > best_r:
                best_r = r_val
                best_pts = cand_pts
    
    if best_pts is not None:
        return best_pts
    
    # Fallback: random points
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = rng.uniform(0, 0.85, n)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _energy(pts, n, r_target=None):
    """
    Calculate the energy of the configuration.
    We use a repulsive force model.
    E = sum_{i<j} 1/d_ij^2 + lambda * sum_i 1/(1-|c_i|)^2
    This encourages large distances and keeping points inside.
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1e10) # Avoid division by zero
    
    # Repulsive term: sum 1/d^2
    # To avoid numerical issues with very small distances, we clip
    dists_sq_clipped = np.maximum(dists_sq, 1e-6)
    repulsive = np.sum(1.0 / dists_sq_clipped)
    
    # Boundary term: push points away from boundary
    radii = np.hypot(pts[:, 0], pts[:, 1])
    boundary_dist = 1.0 - radii
    # If a point is outside, penalty is huge
    boundary_dist_clipped = np.maximum(boundary_dist, 1e-6)
    boundary_penalty = np.sum(1.0 / boundary_dist_clipped ** 2)
    
    # Weight for boundary term
    lambda_b = 10.0
    
    return repulsive + lambda_b * boundary_penalty


def _simulated_annealing(pts, n, time_limit=60.0):
    """
    Simulated annealing to minimize the energy.
    """
    start_time = time.time()
    pts = pts.copy().astype(np.float64)
    current_energy = _energy(pts, n)
    best_pts = pts.copy()
    best_energy = current_energy
    
    T = 0.01
    T_min = 1e-5
    cooling_rate = 0.999
    
    # Estimate number of steps we can afford
    # Each step involves O(N^2) distance calculation
    # N=300, N^2=90000. 
    # 120s limit. Let's say we can do ~10000-20000 steps safely.
    # We will loop until time limit or T < T_min.
    
    rng = np.random.RandomState(42)
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Randomly select a point
        i = rng.randint(n)
        
        # Generate a random perturbation
        # Scale perturbation by temperature
        step_size = 0.01 * T / 0.01 # Normalize so initial step is 0.01
        # Actually, let's use a fixed step size that decays or is scaled by T
        # A common approach: delta = N(0, sigma^2) where sigma decreases
        sigma = 0.005
        delta = rng.normal(0, sigma, 2)
        
        new_pt = pts[i].copy()
        new_pt += delta
        
        # Check if new point is valid (inside unit circle)
        r_new = np.hypot(new_pt[0], new_pt[1])
        if r_new > 0.999:
            # Reject immediately if it goes outside
            T *= cooling_rate
            continue
            
        old_pt = pts[i].copy()
        pts[i] = new_pt
        
        new_energy = _energy(pts, n)
        
        # Acceptance criterion
        if new_energy < current_energy:
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        else:
            # Metropolis criterion
            delta_e = new_energy - current_energy
            if delta_e < 0:
                accept = True
            else:
                accept_prob = math.exp(-delta_e / T)
                accept = rng.random() < accept_prob
            
            if accept:
                current_energy = new_energy
            else:
                pts[i] = old_pt
        
        T *= cooling_rate
        
    return best_pts


def construct():
    n = N
    deadline = time.time() + 140  # Leave 10s margin
    
    # Get initial hexagonal packing
    x0 = _hex_init(n, seed=42)
    best_pts = x0.copy()
    
    # Calculate initial radius
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    boundary_dists = 1.0 - np.hypot(best_pts[:, 0], best_pts[:, 1])
    min_boundary = np.min(boundary_dists)
    best_r = min(min_pair_dist / 2.0, min_boundary)
    
    # Run Simulated Annealing
    # Use most of the time budget
    time_for_sa = 100.0
    if time.time() + time_for_sa > deadline:
        time_for_sa = max(1.0, deadline - time.time() - 10.0)
        
    optimized = _simulated_annealing(x0, n, time_limit=time_for_sa)
    
    # Calculate radius for optimized
    diff = optimized[:, np.newaxis, :] - optimized[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    boundary_dists = 1.0 - np.hypot(optimized[:, 0], optimized[:, 1])
    min_boundary = np.min(boundary_dists)
    r_val = min(min_pair_dist / 2.0, min_boundary)
    
    if r_val > best_r:
        best_r = r_val
        best_pts = optimized.copy()
        
    return best_pts
# EVOLVE-BLOCK-END
