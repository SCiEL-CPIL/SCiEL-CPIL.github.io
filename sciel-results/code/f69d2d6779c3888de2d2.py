"""
Packing 300 equal circles in a unit circle.
Two-stage approach:
  1. Hertzian dynamics (overlap^4 repulsion) to find a good global structure.
  2. Soft-min / log-sum-exp polish to maximize the minimum distance.
"""

import numpy as np
import time
from scipy.spatial.distance import cdist

N = 300

# EVOLVE-BLOCK-START

def _hex_lattice_init(n_points):
    """Generate points on a hexagonal lattice, keep those inside unit disk,
    select n_points closest to origin (or random subset if too many)."""
    # Estimate spacing for ~n_points in unit disk
    # Area of unit disk = pi, area per hex cell ~ spacing^2 * sqrt(3)/2
    # n_points * spacing^2 * sqrt(3)/2 ~ pi
    spacing = np.sqrt(2.0 * np.pi / (n_points * np.sqrt(3.0) / 2.0))
    # Add some margin
    spacing *= 0.95
    
    max_r = 1.0 + spacing
    nx = int(2.0 * max_r / spacing) + 2
    ny = int(2.0 * max_r / (spacing * np.sqrt(3.0) / 2.0)) + 2
    
    points = []
    for i in range(-nx, nx + 1):
        for j in range(-ny, ny + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0 - spacing * 0.5:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) >= n_points:
        # Sort by distance from origin, take closest
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists, kind='stable')[:n_points]
        return points[idx]
    else:
        # Need to add more points - this shouldn't happen with good spacing
        # Fall back to random
        angles = np.random.uniform(0, 2 * np.pi, n_points)
        radii = np.sqrt(np.random.uniform(0, 1, n_points))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return pts


def _hertzian_dynamics(points, n_iter=5000, dt=0.001, cooldown=0.999):
    """
    Hertzian dynamics: force proportional to overlap^4.
    This provides smooth, physically-motivated repulsion.
    """
    n = len(points)
    points = points.copy()
    
    # Initial spacing estimate
    min_dist = np.inf
    for i in range(n):
        d = np.hypot(points[i, 0] - points[:, 0], points[i, 1] - points[:, 1])
        d[i] = np.inf
        min_dist = min(min_dist, np.min(d))
    
    # Target radius ~ min_dist / 2
    target_r = min_dist / 2.0
    
    for it in range(n_iter):
        # Compute pairwise distances
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        
        # Overlap: max(0, 2*r - dist)
        overlap = np.maximum(0.0, 2.0 * target_r - dists)
        
        # Hertzian force: proportional to overlap^4
        force_mag = overlap ** 4
        # Normalize by distance to get direction
        with np.errstate(divide='ignore', invalid='ignore'):
            force_dir = np.where(dists > 1e-12, 1.0 / dists, 0.0)
        
        # Force on each point = sum of forces from all others
        forces = np.zeros_like(points)
        for i in range(n):
            for j in range(n):
                if i != j:
                    f = force_mag[i, j] * force_dir[i, j]
                    forces[i, 0] += f * (points[i, 0] - points[j, 0])
                    forces[i, 1] += f * (points[i, 1] - points[j, 1])
        
        # Containment force: push back if outside
        radii = np.hypot(points[:, 0], points[:, 1])
        for i in range(n):
            if radii[i] > 1.0 - target_r:
                # Push inward
                norm = radii[i] if radii[i] > 1e-12 else 1.0
                forces[i, 0] -= (points[i, 0] / norm) * (radii[i] - (1.0 - target_r)) ** 2
                forces[i, 1] -= (points[i, 1] / norm) * (radii[i] - (1.0 - target_r)) ** 2
        
        # Update positions
        points += dt * forces
        
        # Cool down
        dt *= cooldown
        
        # Check if converged
        max_force = np.max(np.hypot(forces[:, 0], forces[:, 1]))
        if max_force < 1e-10:
            break
    
    return points


def _soft_min_polish(points, n_iter=2000, lr=0.01, beta=50.0):
    """
    Soft-min polish using log-sum-exp for smooth minimum.
    Maximizes the minimum of (pairwise distances / 2, containment margins).
    """
    n = len(points)
    points = points.copy()
    
    for it in range(n_iter):
        # Compute pairwise distances
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        
        # Pairwise half-distances
        half_dists = dists / 2.0
        
        # Containment margins
        radii = np.hypot(points[:, 0], points[:, 1])
        margins = 1.0 - radii
        
        # Combine all constraints
        # We want to maximize min(all constraints)
        # Use soft-min: -log-sum-exp(-beta * values) / beta
        
        all_vals = np.concatenate([half_dists[half_dists < np.inf].flatten(), margins])
        
        # Soft min
        beta_vals = beta * all_vals
        max_beta = np.max(beta_vals)
        log_sum_exp = max_beta + np.log(np.sum(np.exp(beta_vals - max_beta)))
        soft_min_val = log_sum_exp / beta
        
        # Gradient of soft min w.r.t. points
        # d(soft_min)/d(x_i) = -sum_j exp(-beta * v_j) / sum_k exp(-beta * v_k) * dv_j/d(x_i)
        # But we want to MAXIMIZE soft_min, so we add gradient
        
        # Actually, let's compute gradient numerically or analytically
        # For efficiency, let's use a simpler approach: 
        # For each constraint, compute weight = exp(-beta * (max_val - val)) / sum(exp(-beta * (max_val - val_k)))
        
        max_val = np.max(all_vals)
        weights = np.exp(-beta * (max_val - all_vals))
        weights /= np.sum(weights)
        
        # Now compute gradient
        grad = np.zeros_like(points)
        
        # Pairwise distances contribute
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                w = weights[idx]
                d = dists[i, j]
                if d > 1e-12:
                    # d(half_dist)/d(x_i) = (x_i - x_j) / (2 * d)
                    grad[i, 0] += w * (points[i, 0] - points[j, 0]) / (2.0 * d)
                    grad[i, 1] += w * (points[i, 1] - points[j, 1]) / (2.0 * d)
                    grad[j, 0] += w * (points[j, 0] - points[i, 0]) / (2.0 * d)
                    grad[j, 1] += w * (points[j, 1] - points[i, 1]) / (2.0 * d)
                idx += 1
        
        # Containment margins contribute
        for i in range(n):
            w = weights[idx]
            r = radii[i]
            if r > 1e-12:
                # d(margin)/d(x_i) = -x_i / r
                grad[i, 0] += w * (-points[i, 0] / r)
                grad[i, 1] += w * (-points[i, 1] / r)
            idx += 1
        
        # Update: ascend gradient
        points += lr * grad
        
        # Project back inside container
        radii = np.hypot(points[:, 0], points[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            points[mask, 0] *= (1.0 / radii[mask])
            points[mask, 1] *= (1.0 / radii[mask])
        
        # Decay learning rate
        lr *= 0.999
    
    return points


def _compute_score(points):
    """Compute the packing radius."""
    n = len(points)
    if n < 2:
        return 0.0
    
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    radii = np.hypot(points[:, 0], points[:, 1])
    min_margin = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_margin)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Stage 1: Initialize with hex lattice
    points = _hex_lattice_init(N)
    
    # Stage 2: Hertzian dynamics
    elapsed = time.time() - start_time
    if elapsed < deadline - 30.0:
        points = _hertzian_dynamics(points, n_iter=3000, dt=0.002, cooldown=0.9995)
    
    # Stage 3: Soft-min polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        points = _soft_min_polish(points, n_iter=1000, lr=0.005, beta=30.0)
    
    # Final check and clamp
    radii = np.hypot(points[:, 0], points[:, 1])
    mask = radii > 1.0
    if np.any(mask):
        points[mask, 0] *= (0.999 / radii[mask])
        points[mask, 1] *= (0.999 / radii[mask])
    
    return points

# EVOLVE-BLOCK-END
