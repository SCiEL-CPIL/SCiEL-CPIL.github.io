"""Improved packing of 50 equal spheres in a unit cube using bisection + projected gradient descent."""
import numpy as np
import time
import math
from itertools import combinations

N = 50
TARGET_RADIUS = 0.1359545126417

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    # Minimum pairwise distance / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_pair = np.min(dists[iu]) / 2.0
    
    # Minimum distance to walls
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)

def _project_to_feasible(centers, r, max_iters=50):
    """Project centers to feasible set for radius r.
    Feasible: |c_i - c_j| >= 2r for all i<j, and |c_ik| <= 0.5 - r for all i,k."""
    centers = centers.copy()
    bounds = 0.5 - r
    
    for _ in range(max_iters):
        # Project to boundary constraints
        centers = np.clip(centers, -bounds, bounds)
        
        # Project to pairwise distance constraints
        converged = True
        for i in range(N):
            for j in range(i+1, N):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                min_dist = 2.0 * r
                if dist < min_dist - 1e-12:
                    converged = False
                    if dist < 1e-12:
                        # Points coincident, push apart randomly
                        direction = np.random.randn(3)
                        direction /= np.linalg.norm(direction)
                        centers[i] += direction * min_dist / 2.0
                        centers[j] -= direction * min_dist / 2.0
                    else:
                        # Push apart along the line connecting them
                        correction = (min_dist - dist) / 2.0
                        unit_d = d / dist
                        centers[i] += correction * unit_d
                        centers[j] -= correction * unit_d
                        # Re-clip to bounds
                        centers[i] = np.clip(centers[i], -bounds, bounds)
                        centers[j] = np.clip(centers[j], -bounds, bounds)
        
        if converged:
            break
    
    return centers

def _feasible_radius(centers, r):
    """Check if centers are feasible for radius r."""
    bounds = 0.5 - r
    if np.any(np.abs(centers) > bounds + 1e-10):
        return False
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 2.0 * r - 1e-10:
        return False
    
    return True

def _solve_for_radius(r, initial_centers=None, max_iter=2000, lr=0.001, seed=42):
    """Try to find a feasible packing for radius r using projected gradient descent.
    Returns (success, centers)."""
    rng = np.random.RandomState(seed)
    bounds = 0.5 - r
    
    if initial_centers is not None:
        centers = _project_to_feasible(initial_centers.copy(), r)
    else:
        # Initialize with a grid-like structure
        m = int(np.ceil(N ** (1.0/3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        centers = grid[:N].copy()
        centers = _project_to_feasible(centers, r)
    
    # Check if already feasible
    if _feasible_radius(centers, r):
        return True, centers
    
    # Gradient descent: minimize energy = sum of violations
    for it in range(max_iter):
        # Compute gradient of energy
        grad = np.zeros_like(centers)
        energy = 0.0
        
        # Pairwise violations
        for i in range(N):
            for j in range(i+1, N):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                min_dist = 2.0 * r
                if dist < min_dist:
                    if dist < 1e-12:
                        direction = rng.randn(3)
                        direction /= np.linalg.norm(direction)
                        grad[i] += direction
                        grad[j] -= direction
                        energy += (min_dist - 0.0) ** 2
                    else:
                        unit_d = d / dist
                        violation = min_dist - dist
                        grad[i] += unit_d * violation
                        grad[j] -= unit_d * violation
                        energy += violation ** 2
        
        # Boundary violations
        for i in range(N):
            for k in range(3):
                if centers[i, k] > bounds:
                    grad[i, k] -= (centers[i, k] - bounds)
                    energy += (centers[i, k] - bounds) ** 2
                elif centers[i, k] < -bounds:
                    grad[i, k] -= (centers[i, k] + bounds)
                    energy += (centers[i, k] + bounds) ** 2
        
        # Update
        centers -= lr * grad
        centers = np.clip(centers, -bounds, bounds)
        
        # Periodically check feasibility
        if it % 100 == 0 and _feasible_radius(centers, r):
            return True, centers
    
    # Final check
    if _feasible_radius(centers, r):
        return True, centers
    
    return False, centers

def _bisection_search(initial_centers=None, r_low=0.1, r_high=0.14, tol=1e-6, max_iter=30, seed=42):
    """Bisection search for the maximum feasible radius."""
    best_r = r_low
    best_centers = None
    
    if initial_centers is not None:
        # Check if initial centers are feasible for r_high
        if _feasible_radius(initial_centers, r_high):
            return r_high, initial_centers
    
    for it in range(max_iter):
        r_mid = (r_low + r_high) / 2.0
        
        success, centers = _solve_for_radius(r_mid, initial_centers, max_iter=1500, lr=0.0005, seed=seed+it)
        
        if success:
            best_r = r_mid
            best_centers = centers
            r_low = r_mid
        else:
            r_high = r_mid
        
        if r_high - r_low < tol:
            break
    
    return best_r, best_centers

def _generate_initial_guess(seed=42):
    """Generate a good initial guess using a structured grid with perturbations."""
    rng = np.random.RandomState(seed)
    
    # Use a body-centered cubic-like structure
    # Start with a 4x4x4 grid (64 points), take 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Select 50 points, preferring corners and edges
    # Add small random perturbations
    centers = grid[:N].copy()
    centers += rng.uniform(-0.01, 0.01, size=(N, 3))
    centers = np.clip(centers, -0.45, 0.45)
    
    return centers

def _polish(centers, r, max_iter=5000, lr=0.0001, seed=42):
    """Polish the packing to maximize the minimum distance."""
    rng = np.random.RandomState(seed)
    best = centers.copy()
    best_score = _compute_radius(best)
    current = centers.copy()
    current_score = best_score
    
    for it in range(max_iter):
        # Adaptive learning rate
        lr_current = lr * (1.0 + 0.5 * np.sin(it / 100.0))
        
        # Compute gradient of the soft-min objective
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        
        # Soft minimum
        beta = 50.0
        log_sum_exp = np.log(np.sum(np.exp(-beta * pair_dists)))
        soft_min = -np.log(np.sum(np.exp(-beta * pair_dists))) / beta
        
        # Gradient of soft-min with respect to centers
        weights = np.exp(-beta * pair_dists)
        weights /= np.sum(weights)
        
        grad = np.zeros_like(current)
        for idx, (i, j) in enumerate(zip(*iu)):
            d = current[i] - current[j]
            dist = pair_dists[idx]
            if dist > 1e-12:
                unit_d = d / dist
                grad[i] -= weights[idx] * unit_d
                grad[j] += weights[idx] * unit_d
        
        # Boundary gradient
        bounds = 0.5
        for i in range(N):
            for k in range(3):
                if current[i, k] > bounds - 0.05:
                    grad[i, k] -= 1.0
                elif current[i, k] < -bounds + 0.05:
                    grad[i, k] += 1.0
        
        # Update
        current -= lr_current * grad
        current = np.clip(current, -0.5, 0.5)
        
        new_score = _compute_radius(current)
        if new_score > best_score:
            best = current.copy()
            best_score = new_score
            current_score = new_score
        else:
            # Revert if it got worse
            current = best.copy()
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # Generate initial guess
    initial = _generate_initial_guess(seed=42)
    
    # Try bisection search
    best_r, best_centers = _bisection_search(initial, r_low=0.12, r_high=0.138, tol=1e-5, max_iter=25, seed=42)
    
    # Polish the result
    if time.time() < deadline - 10:
        polished, polished_score = _polish(best_centers, best_r, max_iter=3000, lr=0.00005, seed=42)
        if polished_score > best_r:
            best_centers = polished
            best_r = polished_score
    
    # Try a few more restarts if time permits
    for seed in [123, 456, 789]:
        if time.time() > deadline - 15:
            break
        
        initial2 = _generate_initial_guess(seed=seed)
        r2, centers2 = _bisection_search(initial2, r_low=0.12, r_high=0.138, tol=1e-5, max_iter=20, seed=seed)
        
        if r2 > best_r:
            best_r = r2
            best_centers = centers2
        
        if time.time() < deadline - 5:
            polished2, polished2_score = _polish(best_centers, best_r, max_iter=2000, lr=0.00005, seed=seed)
            if polished2_score > best_r:
                best_centers = polished2
                best_r = polished2_score
    
    # Final validation
    if not np.all(np.isfinite(best_centers)):
        # Fallback to a simple grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()
    
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

# EVOLVE-BLOCK-END
