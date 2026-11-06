"""
Physics-Based Repulsion Packing Strategy for 50 circles in a unit square.

Strategy:
1. Initialize circles in a random or grid configuration.
2. Use a physics-based relaxation method (Molecular Dynamics) where circles 
   repel each other strongly when close and are pushed inward by the boundaries.
   This allows the system to find a dense, non-lattice packing naturally.
3. Once the system stabilizes (energy minimized), perform a rigorous local search 
   on individual coordinates to maximize the minimum radius (min-pair distance 
   and min-boundary distance).
4. This approach avoids the local minima traps of structured lattices by allowing 
   the geometry to deform freely under repulsive forces.
"""

import time
import numpy as np
import random

N = 50
DEADLINE_FRACTION = 0.90

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    
    if min_boundary <= 0:
        return 0.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def _fast_radius(centers):
    """Fast radius computation without validity checks."""
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_boundary, min_pair)

def _physics_relaxation(centers, n_steps=1000, dt=0.01, repulsion_strength=1.0, boundary_strength=1.0):
    """
    Simulate repulsion forces to pack circles.
    Force on i: sum_{j != i} (1/d_ij^2) * (c_i - c_j) + boundary_forces
    """
    centers = centers.copy()
    # Soft boundary: if |c| > 0.5, push back. We want them INSIDE, so if they touch boundary, 
    # we actually want to push them IN if they are too close? 
    # No, standard packing: circles want to be far from each other. 
    # To fit in square, we need to balance repulsion with a "confinement" force that keeps them in.
    # Actually, simpler: just repel each other. If they fly apart, we rescale?
    # Better: Use a potential that penalizes overlap (d < 2r) and also penalizes going out of bounds.
    # But we don't know r. 
    # Alternative: Minimize the "stress" or just maximize min distance.
    # Let's use a gradient ascent on the min-distance objective? No, non-differentiable.
    # Let's use a force that pushes circles apart if they are closer than a target distance, 
    # and pushes them in if they are outside.
    
    # Since we don't know the target radius, we can use a heuristic:
    # Assume a target radius slightly larger than current estimate to force expansion?
    # Or just use pure repulsion and then normalize the bounding box to fit the square.
    
    # Strategy: 
    # 1. Repel all pairs with force ~ 1/d^2.
    # 2. Apply a weak centering force to prevent drift.
    # 3. After simulation, rescale the entire configuration to fit inside [-0.5, 0.5]^2.
    # 4. This rescaling preserves the relative packing structure (ratios of distances).
    # 5. The "quality" of the packing is determined by the ratio of min_pair_dist to max_extent.
    
    for step in range(n_steps):
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        # O(N^2) is fine for N=50
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                d2 = np.dot(diff, diff)
                if d2 < 1e-10:
                    # Random jitter
                    diff = np.random.normal(0, 0.01, 2)
                    d2 = np.dot(diff, diff)
                
                # Force magnitude ~ 1/d^2, direction diff
                # Soften to avoid singularity
                f_mag = repulsion_strength / (d2 * np.sqrt(d2) + 1e-6)
                f_vec = diff * f_mag
                forces[i] += f_vec
                forces[j] -= f_vec
        
        # Boundary handling: 
        # We don't strictly enforce boundary in physics step if we plan to rescale.
        # But if they fly to infinity, rescaling makes them tiny. 
        # So we need to keep them bounded. 
        # Let's add a restoring force to keep them near the center, but weak.
        forces -= centers * 0.01
        
        # Update positions
        centers += forces * dt
        
        # Optional: Damping
        # Not using velocity, just position update with small dt acts as implicit damping if dt is small enough
        
    # Rescale to fit in square
    # Find max extent
    min_x, min_y = np.min(centers, axis=0)
    max_x, max_y = np.max(centers, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    scale = 1.0 / max(width, height)
    
    # Center the cluster
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    centers = (centers - np.array([center_x, center_y])) * scale
    
    return centers

def _local_search(centers, n_iterations=5000, deadline=None):
    """
    Local search on individual coordinates.
    Tries to move one circle at a time to increase the minimum radius.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    rng = np.random.RandomState(42)
    
    current_step = 0.02
    no_improve_count = 0
    
    for i in range(n_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        idx = rng.randint(N)
        orig = best_centers[idx].copy()
        
        improved = False
        # Try several random perturbations
        for attempt in range(5):
            s = current_step * (0.5 ** attempt)
            perturbation = rng.normal(0, s, size=2)
            candidate = orig + perturbation
            
            # Keep inside square
            np.clip(candidate, -0.5 + 1e-8, 0.5 - 1e-8, out=candidate)
            
            best_centers[idx] = candidate
            new_radius = _fast_radius(best_centers)
            
            if new_radius > best_radius:
                best_radius = new_radius
                improved = True
                break
            else:
                best_centers[idx] = orig
        
        if improved:
            no_improve_count = 0
            # Increase step slightly if improving
            current_step = min(0.05, current_step * 1.05)
        else:
            no_improve_count += 1
            if no_improve_count > 100:
                current_step = max(0.001, current_step * 0.8)
                no_improve_count = 0
                
    return best_centers, best_radius

def construct():
    start_time = time.time()
    total_time_budget = 150.0
    deadline = start_time + total_time_budget * DEADLINE_FRACTION
    
    best_centers = None
    best_radius = 0.0
    
    # Multiple restarts with different initial configurations
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
            
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
        
        # Initialize
        if restart == 0:
            # Grid init
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            centers = grid[:N].astype(float)
            # Add small noise to break symmetry
            centers += np.random.normal(0, 0.01, size=centers.shape)
        else:
            # Random init within square
            centers = np.random.uniform(-0.4, 0.4, size=(N, 2))
        
        # Physics Relaxation
        # Allocate time: 50% of remaining for physics, 50% for local search
        phys_time_budget = remaining * 0.4
        phys_deadline = time.time() + phys_time_budget
        
        # Estimate steps based on time
        # Each step is O(N^2) = 2500 ops. Very fast.
        # Let's run for a fixed number of steps or until time up.
        n_steps = 2000
        step_deadline = time.time() + phys_time_budget
        
        # Simple loop with time check
        for step in range(n_steps):
            if time.time() > step_deadline:
                break
            # Inline physics step for speed
            forces = np.zeros_like(centers)
            for i in range(N):
                for j in range(i + 1, N):
                    diff = centers[i] - centers[j]
                    d2 = np.dot(diff, diff)
                    if d2 < 1e-10:
                        diff = np.random.normal(0, 0.01, 2)
                        d2 = np.dot(diff, diff)
                    f_mag = 1.0 / (d2 * np.sqrt(d2) + 1e-6)
                    f_vec = diff * f_mag
                    forces[i] += f_vec
                    forces[j] -= f_vec
            forces -= centers * 0.01
            centers += forces * 0.005
            
        # Rescale
        min_x, min_y = np.min(centers, axis=0)
        max_x, max_y = np.max(centers, axis=0)
        width = max_x - min_x
        height = max_y - min_y
        scale = 1.0 / max(width, height)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        centers = (centers - np.array([center_x, center_y])) * scale
        
        # Local Search
        ls_deadline = deadline - 5.0 # Leave 5s safety
        n_iters = 2000
        centers, r = _local_search(centers, n_iterations=n_iters, deadline=ls_deadline)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
    # Final safety check
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        m = 8
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N].astype(float)
        
    final_radius = _compute_radius(best_centers)
    if final_radius <= 0:
        m = 8
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N].astype(float)

    return best_centers

# EVOLVE-BLOCK-END
