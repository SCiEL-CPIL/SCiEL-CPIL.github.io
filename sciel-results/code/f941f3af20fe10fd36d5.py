"""
Strategy: Physics-Based Molecular Dynamics (MD) Relaxation
The current approach uses L-BFGS on a non-smooth objective (min distance). 
This new approach treats the circles as soft repulsive spheres in a box.
We minimize a smooth energy function:
E = sum_{i<j} k * (max(0, r_i + r_j - d_ij))^2 + sum_i k_b * (max(0, r_i - d_i_boundary))^2
By using a soft penalty, the landscape is differentiable, and the "pressure" 
from repulsion drives the system towards a dense packing. We use a simple 
velocity-Verlet integrator with damping (Langevin dynamics) to let the 
system relax. This is robust against local minima because the forces 
propagate globally through the chain of interactions.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

def _radius_score(centers):
    """Compute the maximum common radius for given centers in [-0.5, 0.5]^2."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
        return 0.0
    
    centers = np.clip(centers, -0.5, 0.5)
    
    # Boundary distance
    dist_to_boundary = np.min(np.stack([
        centers[:, 0] + 0.5,
        0.5 - centers[:, 0],
        centers[:, 1] + 0.5,
        0.5 - centers[:, 1]
    ]), axis=0)
    min_boundary = np.min(dist_to_boundary)
    
    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2
    
    return min(min_boundary, min_pairwise)

def _md_relax(centers, time_budget=10.0):
    """
    Perform Molecular Dynamics relaxation to find a dense packing.
    Returns the relaxed centers.
    """
    start_time = time.time()
    centers = centers.copy()
    
    # Initial radius estimate for scaling
    r0 = _radius_score(centers)
    if r0 <= 0:
        r0 = 0.05
    
    # Simulation parameters
    # We want to maximize the minimum distance. 
    # We use a repulsive potential that is active when distance < 2*r_target.
    # We iteratively increase r_target.
    
    current_r = r0 * 0.8
    dt = 0.001
    damping = 0.9
    k_rep = 100.0  # Repulsion strength
    k_bound = 100.0 # Boundary strength
    
    velocities = np.zeros_like(centers)
    
    # Annealing schedule: slowly increase target radius
    # We simulate for a fixed number of steps or time budget
    max_steps = 5000
    steps_per_check = 100
    
    for step in range(max_steps):
        if time.time() - start_time > time_budget:
            break
            
        # Gradually increase the target radius to push circles apart
        # This is like increasing temperature/pressure
        if step % steps_per_check == 0:
            # Increase target radius slightly
            current_r *= 1.001
            # Check if we are violating constraints heavily, if so, back off
            # (Simple heuristic: if energy is too high, reduce r)
            
        # Calculate forces
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        # We want to enforce d_ij >= 2 * current_r
        # Potential U = 0.5 * k * max(0, 2*r - d)^2
        # Force F = -dU/dr = k * max(0, 2*r - d) * (r_vec / d)
        
        # Efficiently compute distances
        # For N=50, O(N^2) is fine (2500 pairs)
        for i in range(N):
            for j in range(i + 1, N):
                r_vec = centers[i] - centers[j]
                d = np.linalg.norm(r_vec)
                
                # Boundary check is handled separately, but we also 
                # need to ensure they don't overlap.
                min_dist = 2.0 * current_r
                
                if d < min_dist and d > 1e-10:
                    # Overlap penalty
                    overlap = min_dist - d
                    # Force magnitude
                    f_mag = k_rep * overlap
                    # Force direction (repulsive)
                    f_vec = (r_vec / d) * f_mag
                    forces[i] += f_vec
                    forces[j] -= f_vec
                elif d <= 1e-10:
                    # Coincident points, random kick
                    kick = np.random.normal(0, 0.01, size=2)
                    forces[i] += kick
                    forces[j] -= kick

        # Boundary forces
        # Keep centers within [-0.5, 0.5]
        # Soft wall: if center is within 'current_r' of boundary, push back
        # Actually, the constraint is center must be at least 'r' away from boundary?
        # No, the circle of radius r must be inside. So center must be in [-0.5+r, 0.5-r].
        # So if center < -0.5 + current_r, push right.
        
        min_coord = -0.5 + current_r
        max_coord = 0.5 - current_r
        
        for i in range(N):
            # X-axis
            if centers[i, 0] < min_coord:
                forces[i, 0] += k_bound * (min_coord - centers[i, 0])
            elif centers[i, 0] > max_coord:
                forces[i, 0] -= k_bound * (centers[i, 0] - max_coord)
            
            # Y-axis
            if centers[i, 1] < min_coord:
                forces[i, 1] += k_bound * (min_coord - centers[i, 1])
            elif centers[i, 1] > max_coord:
                forces[i, 1] -= k_bound * (centers[i, 1] - max_coord)

        # Update velocities and positions (Velocity Verlet / Langevin)
        velocities = velocities * damping + forces * dt
        centers = centers + velocities * dt
        
        # Hard clip to prevent numerical explosion (shouldn't happen with soft walls, but safe)
        centers = np.clip(centers, -0.5, 0.5)
        
        # Optional: If we reach a valid state with higher radius, we can stop increasing r
        # But for now, we just let it run.
        
    return centers

def _polish(centers, time_budget=5.0):
    """
    Final local optimization using L-BFGS to fine-tune the MD result.
    """
    start_time = time.time()
    
    def _objective(centers):
        r = _radius_score(centers)
        return -r

    try:
        result = minimize(
            _objective,
            centers,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * N * 2,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.success or result.fun < -_radius_score(centers):
             # Check if it actually improved
             r_new = _radius_score(result.x)
             r_old = _radius_score(centers)
             if r_new > r_old:
                 return result.x
    except Exception:
        pass
        
    return centers

def _construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_radius = -np.inf
    
    # 1. Generate diverse initial configurations
    initial_configs = []
    
    # Grid
    m, n = 7, 8
    points = []
    for i in range(n):
        for j in range(m):
            if len(points) >= N:
                break
            x = (j + 0.5) / m - 0.5
            y = (i + 0.5) / n - 0.5
            points.append([x, y])
    initial_configs.append(np.array(points[:N]))
    
    # Hex-like
    points = []
    row_height = 0.09
    col_width = 0.1
    offset = col_width / 2
    for row in range(8):
        y = -0.5 + 0.05 + row * row_height
        for col in range(7):
            if len(points) >= N:
                break
            if row % 2 == 0:
                x = -0.5 + 0.05 + col * col_width
            else:
                x = -0.5 + 0.05 + offset + col * col_width
            points.append([x, y])
    initial_configs.append(np.array(points[:N]))
    
    # Random
    for _ in range(5):
        initial_configs.append(rng.uniform(-0.4, 0.4, size=(N, 2)))
        
    # 2. Relax each configuration using MD
    for i, init in enumerate(initial_configs):
        if time.time() > deadline:
            break
            
        # Allocate time for MD
        md_time = 5.0
        if i == 0:
            md_time = 8.0 # Give more time to the first one
        
        relaxed = _md_relax(init, time_budget=md_time)
        
        # Polish
        polished = _polish(relaxed, time_budget=2.0)
        
        r = _radius_score(polished)
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()
            
        # Check time
        if time.time() > deadline - 10.0:
            break

    # 3. If time remains, do a global search/polish on the best found
    if best_centers is not None and time.time() < deadline - 5.0:
        # Try a few perturbations of the best solution
        for _ in range(3):
            if time.time() > deadline - 2.0:
                break
            perturbation = rng.normal(0, 0.01, size=(N, 2))
            perturbed = best_centers + perturbation
            perturbed = np.clip(perturbed, -0.49, 0.49)
            
            # Quick MD relaxation
            relaxed = _md_relax(perturbed, time_budget=2.0)
            polished = _polish(relaxed, time_budget=1.0)
            
            r = _radius_score(polished)
            if r > best_radius:
                best_radius = r
                best_centers = polished.copy()

    # Fallback
    if best_centers is None:
        m, n = 7, 8
        points = []
        for i in range(n):
            for j in range(m):
                if len(points) >= N:
                    break
                x = (j + 0.5) / m - 0.5
                y = (i + 0.5) / n - 0.5
                points.append([x, y])
        best_centers = np.array(points[:N])
    else:
        best_centers = np.clip(best_centers, -0.5, 0.5)
        
    return best_centers

def construct():
    return _construct()
