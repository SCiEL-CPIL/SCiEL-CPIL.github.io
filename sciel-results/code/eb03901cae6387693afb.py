"""
Optimization of 100 equal spheres in a unit cube.
Uses a physics-based Langevin dynamics approach with iterative radius increase.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    # Use cdist for efficiency
    from scipy.spatial.distance import cdist
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _energy_and_grad(x, R):
    """
    Computes the penalty energy and gradient for target radius R.
    Energy = sum of squared violations of pair distances and wall distances.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    # Compute all pairwise differences
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    v_ij = np.maximum(0, 2.0 * R - d_ij)
    
    if np.any(v_ij > 0):
        energy += np.sum(v_ij**2)
        
        # Gradient accumulation
        # d/dx_i ( (2R - d_ij)^2 ) = 2(2R - d_ij) * (-1/d_ij) * (x_i - x_j)
        # Only for violating pairs
        valid = v_ij > 0
        if np.any(valid):
            i_v = i_idx[valid]
            j_v = j_idx[valid]
            v_v = v_ij[valid]
            d_v = d_ij[valid]
            
            # Factor: 2 * v / d
            factors = 2.0 * v_v / d_v
            vecs = centers[i_v] - centers[j_v]
            forces = factors[:, np.newaxis] * vecs
            
            np.add.at(grad, i_v, forces)
            np.add.at(grad, j_v, -forces)

    # 2. Wall terms
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    v_wall = np.maximum(0, R - wall_dists)
    
    if np.any(v_wall > 0):
        energy += np.sum(v_wall**2)
        
        valid_wall = v_wall > 0
        if np.any(valid_wall):
            i_v = np.where(valid_wall)[0]
            v_v = v_wall[i_v]
            
            # Gradient of wall distance: -sign(x_k) for the max coordinate k
            # We need to identify the max coordinate for each violating sphere
            for i in i_v:
                max_coord = np.argmax(abs_centers[i])
                sign = np.sign(centers[i, max_coord])
                # d/dx_i ( (R - w_i)^2 ) = 2(R - w_i) * (-d w_i / d x_i)
                # d w_i / d x_i = -sign(x_k)
                # So grad = 2 v * (- (-sign)) = 2 v * sign
                # Wait, w_i = 0.5 - |x_k|. d w_i / d x_k = -sign(x_k).
                # Energy term: (R - w_i)^2.
                # dE/dx_k = 2(R - w_i) * (-1) * d w_i / d x_k = -2(R - w_i) * (-sign) = 2(R - w_i) sign.
                # So we add 2 * v * sign to grad[i, max_coord].
                grad[i, max_coord] += 2.0 * v_v[np.where(i_v == i)[0][0]] * sign
            
    return energy, grad.flatten()

def _langevin_step(centers, velocities, R, dt, temp, damping):
    """
    Perform one Langevin dynamics step.
    """
    centers = centers.copy()
    velocities = velocities.copy()
    
    # Compute forces (negative gradient of energy)
    _, grad_flat = _energy_and_grad(centers.flatten(), R)
    forces = -grad_flat.reshape(-1, 3)
    
    # Update velocities: v_new = (v + F*dt) * (1 - damping*dt) + noise
    velocities = (velocities + forces * dt) * (1.0 - damping * dt)
    noise = np.random.normal(0, np.sqrt(2 * temp * dt), size=velocities.shape)
    velocities += noise
    
    # Update positions
    centers = centers + velocities * dt
    
    # Clip to cube
    centers = np.clip(centers, -0.499999, 0.499999)
    
    return centers, velocities

def _optimize_packing(centers, R, time_budget):
    """
    Optimize centers for a fixed radius R using Langevin dynamics.
    """
    start_time = time.time()
    current_centers = centers.copy()
    velocities = np.zeros_like(current_centers)
    
    # Tuned for speed: larger dt, fewer steps
    dt = 0.05
    temp = 0.05
    damping = 0.5
    max_steps = 200
    
    # Simulate
    for step in range(max_steps):
        if time.time() - start_time > time_budget:
            break
            
        current_centers, velocities = _langevin_step(current_centers, velocities, R, dt, temp, damping)
        
        # Cool down temperature over time
        progress = step / max_steps
        temp = 0.05 * (1.0 - progress) + 0.001
        # Increase damping as we cool to settle
        damping = 0.5 + 0.5 * progress
    
    # Final gradient descent polish
    def obj(x):
        e, _ = _energy_and_grad(x, R)
        return e
        
    def jac(x):
        _, g = _energy_and_grad(x, R)
        return g
        
    try:
        res = minimize(obj, current_centers.flatten(), jac=jac, method='L-BFGS-B', 
                       options={'maxiter': 100, 'disp': False})
        current_centers = res.x.reshape(-1, 3)
        current_centers = np.clip(current_centers, -0.499999, 0.499999)
    except Exception:
        pass
        
    return current_centers

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Initial guess: Grid
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    grid = np.array([[x, y, z] for z in zs for y in ys for x in xs])
    centers = grid
    
    # Start with a small radius and increase
    R_curr = 0.05
    R_max_possible = 0.12 # Rough upper bound
    R_step = 0.005
    
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    while time.time() < deadline - 10.0:
        # Check if current packing is valid for R_curr
        # We optimize for R_curr
        time_left = deadline - time.time()
        if time_left < 5.0:
            break
            
        # Allocate time for this R step
        # Keep it small to allow many radius increases
        step_time = min(3.0, time_left - 5.0)
        
        optimized_centers = _optimize_packing(centers, R_curr, step_time)
        
        # Check validity
        r_actual = _compute_radius(optimized_centers)
        
        if r_actual >= R_curr * 0.99: # Allow small tolerance
            # Valid, accept and increase R
            centers = optimized_centers
            best_centers = centers.copy()
            best_radius = r_actual
            R_curr += R_step
        else:
            # Invalid, try to recover or stop?
            # If we can't satisfy R_curr, we might be stuck.
            # Try a smaller step?
            R_step *= 0.8
            R_curr = max(0.05, R_curr - R_step * 2)
            if R_step < 0.0001:
                break

    # Final polish on best_centers
    if time.time() < deadline - 5.0:
        # Try to polish with the best known radius
        final_R = best_radius
        # Slightly reduce R to ensure validity during optimization, then check
        polish_R = final_R * 0.99
        time_left = deadline - time.time() - 2.0
        if time_left > 1.0:
            polished = _optimize_packing(best_centers, polish_R, time_left)
            r_polished = _compute_radius(polished)
            if r_polished > best_radius:
                best_centers = polished
                best_radius = r_polished

    return best_centers
