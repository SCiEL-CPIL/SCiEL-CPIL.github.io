"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    t_start = time.time()
    deadline = 140.0  # Safety margin
    
    # Define ring structure: [count, initial_radius, initial_angle_offset]
    # 1 + 6 + 12 + 18 + 13 = 50
    ring_counts = [1, 6, 12, 18, 13]
    
    def get_points_from_rings(params):
        """
        params: [R1, R2, R3, R4, phi1, phi2, phi3, phi4]
        Returns array of shape (50, 2)
        """
        points = np.zeros((50, 2))
        idx = 0
        # Ring 0: Center
        points[0] = [0.0, 0.0]
        idx = 1
        
        for k in range(1, 5):
            count = ring_counts[k]
            R = params[k-1] # R1 is params[0]
            phi = params[4 + k - 1] # phi1 is params[4]
            angles = phi + 2 * np.pi * np.arange(count) / count
            x = R * np.cos(angles)
            y = R * np.sin(angles)
            points[idx:idx+count, 0] = x
            points[idx:idx+count, 1] = y
            idx += count
        return points

    def compute_score(points):
        """
        Computes the score: min(min_pair_dist/2, min_boundary_dist)
        """
        if not np.all(np.isfinite(points)):
            return 0.0
        # Boundary check
        radii = np.hypot(points[:, 0], points[:, 1])
        if np.any(radii > 1.0):
            return 0.0
        min_boundary = np.min(1.0 - radii)
        
        # Pairwise distances
        # Use broadcasting for N=50, 50x50 matrix is small
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, k=1)
        min_pair = np.min(dists[iu]) / 2.0
        
        return min(min_pair, min_boundary)

    def objective_ring(params):
        """
        Objective for ring parameters. Returns negative soft-min score.
        """
        points = get_points_from_rings(params)
        # Quick validity check
        radii = np.hypot(points[:, 0], points[:, 1])
        if np.any(radii > 1.0):
            return 10.0 # Large penalty
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, k=1)
        min_pair_dists = dists[iu]
        boundary_dists = 1.0 - radii
        
        all_dists = np.concatenate([min_pair_dists, boundary_dists])
        
        # Soft min
        beta = 50.0
        scaled_dists = all_dists
        # We want to maximize min(scaled_dists)
        # soft_min = -1/beta * log(sum(exp(-beta * x)))
        # Objective to minimize = -soft_min
        
        # Numerical stability
        max_val = np.max(-beta * scaled_dists)
        if not np.isfinite(max_val):
            return 10.0
        log_sum_exp = max_val + np.log(np.sum(np.exp(-beta * scaled_dists - max_val)))
        soft_min = -log_sum_exp / beta
        
        return -soft_min

    # Initial guess for ring parameters
    # Radii roughly increasing, angles staggered
    x0 = np.array([0.2, 0.4, 0.6, 0.8,  # R1..R4
                   0.0, np.pi/6, 0.0, np.pi/26]) # phi1..phi4
    
    # Optimize ring parameters
    res_ring = minimize(
        objective_ring,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    if time.time() > deadline:
        # Fallback to initial hexagonal or just return the ring result if valid
        points_ring = get_points_from_rings(res_ring.x)
        return points_ring if compute_score(points_ring) > 0 else np.zeros((N,2))

    points_ring = get_points_from_rings(res_ring.x)
    
    # Refine with full coordinate optimization
    # Start from the ring-optimized points
    x_full = points_ring.flatten()
    
    def objective_full(params_flat):
        points = params_flat.reshape(N, 2)
        radii = np.hypot(points[:, 0], points[:, 1])
        if np.any(radii > 1.0):
            # Penalize going outside
            penalty = np.sum(np.maximum(0, radii - 1.0))
            return 10.0 + penalty * 100.0
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, k=1)
        min_pair_dists = dists[iu]
        boundary_dists = 1.0 - radii
        
        all_dists = np.concatenate([min_pair_dists, boundary_dists])
        
        beta = 50.0
        scaled_dists = all_dists
        max_val = np.max(-beta * scaled_dists)
        if not np.isfinite(max_val):
            return 10.0
        log_sum_exp = max_val + np.log(np.sum(np.exp(-beta * scaled_dists - max_val)))
        soft_min = -log_sum_exp / beta
        
        return -soft_min

    res_full = minimize(
        objective_full,
        x_full,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    points_full = res_full.x.reshape(N, 2)
    
    # Check time
    if time.time() > deadline:
        # Return the better of ring or full
        score_ring = compute_score(points_ring)
        score_full = compute_score(points_full)
        if score_full > score_ring:
            return points_full
        else:
            return points_ring

    # Final validation and selection
    score_ring = compute_score(points_ring)
    score_full = compute_score(points_full)
    
    if score_full > score_ring:
        return points_full
    else:
        return points_ring

# EVOLVE-BLOCK-END
