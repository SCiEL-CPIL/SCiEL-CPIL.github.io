"""Simulated annealing packing of 50 equal circles in a unit circle."""
import numpy as np
from scipy.optimize import minimize
import random

N = 50

def compute_radius(pos):
    """Compute the actual radius of the packing."""
    pos = pos.reshape(-1, 2)
    dists = np.sqrt(np.sum(pos**2, axis=1))
    if np.any(dists > 1.0):
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    min_pair_dist = np.min(dist_mat)
    min_boundary_dist = np.min(1.0 - dists)
    return min(min_pair_dist / 2.0, min_boundary_dist)

def optimize_scale(pos):
    """Find optimal uniform scale to maximize radius."""
    pos = pos.reshape(-1, 2)
    dists = np.sqrt(np.sum(pos**2, axis=1))
    max_dist = np.max(dists)
    if max_dist < 1e-10:
        return pos, compute_radius(pos)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    min_pair_dist = np.min(dist_mat)
    r_pair = min_pair_dist / 2.0
    
    def radius_at_scale(s):
        if s * max_dist > 1.0:
            return 0.0
        r_p = s * r_pair
        r_b = 1.0 - s * max_dist
        return min(r_p, r_b)
    
    s_opt = 1.0 / (r_pair + max_dist)
    s_opt = np.clip(s_opt, 0.5, 1.5)
    
    r_final = radius_at_scale(s_opt)
    r_orig = radius_at_scale(1.0)
    
    if r_final > r_orig:
        return pos * s_opt, r_final
    else:
        return pos, r_orig

def simulated_annealing(pos_initial, max_iter=80000, seed=42):
    """Run simulated annealing from given initial position."""
    np.random.seed(seed)
    pos = pos_initial.copy()
    
    def energy(pos_flat):
        pos = pos_flat.reshape(-1, 2)
        dists_to_center = np.sqrt(np.sum(pos**2, axis=1))
        # Penalize being outside or too close to boundary
        boundary_penalty = np.sum(np.maximum(0, dists_to_center - 0.995)**2 * 1000)
        boundary_term = np.sum(1.0 / (1.0 - np.clip(dists_to_center, 0, 0.995))**2)
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_mat, 1.0)
        safe_dist_mat = np.clip(dist_mat, 0.001, 2.0)
        pair_term = np.sum(1.0 / safe_dist_mat**2)
        
        return boundary_penalty + boundary_term + pair_term

    T = 50.0
    T_min = 1e-6
    cooling = 0.9995
    step_size = 0.15
    
    best_pos = pos.copy()
    best_energy = energy(pos.reshape(-1))
    current_energy = best_energy
    
    for it in range(max_iter):
        new_pos = pos.copy()
        idx = np.random.randint(0, N)
        new_pos[idx] += np.random.normal(0, step_size, 2)
        
        new_energy = energy(new_pos.reshape(-1))
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
            pos = new_pos
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
            
        if it % 10000 == 0 and it > 0:
            step_size *= 0.85

    pos = best_pos
    
    # Local refinement
    def objective(pos_flat):
        return energy(pos_flat)
    
    try:
        result = minimize(objective, pos.reshape(-1), method='L-BFGS-B', 
                          options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
        if result.fun < best_energy:
            pos = result.x.reshape(-1, 2)
    except Exception:
        pass
    
    return pos

def construct():
    np.random.seed(42)
    random.seed(42)
    
    best_pos = None
    best_r = 0.0
    
    # Multiple restarts with different initial configurations
    for restart in range(5):
        # Different initial seeds for diversity
        seed = 42 + restart * 1000
        
        # Generate initial hexagonal lattice with some randomness
        points = []
        r_hex = 0.13 + 0.01 * (restart % 3)
        sqrt3 = np.sqrt(3.0)
        
        for i in range(-10, 11):
            for j in range(-10, 11):
                x = i * r_hex + (j % 2) * r_hex / 2.0
                y = j * r_hex * sqrt3 / 2.0
                # Add small random perturbation
                x += np.random.uniform(-0.02, 0.02)
                y += np.random.uniform(-0.02, 0.02)
                if x*x + y*y <= 0.95:
                    points.append([x, y])
        
        points = np.array(points)
        
        if len(points) > N:
            dists = np.sqrt(np.sum(points**2, axis=1))
            idx = np.argsort(dists)[:N]
            points = points[idx]
        elif len(points) < N:
            while len(points) < N:
                angle = np.random.uniform(0, 2*np.pi)
                r = np.random.uniform(0, 0.8)
                points = np.vstack([points, [[r*np.cos(angle), r*np.sin(angle)]]])
        
        pos_initial = points.copy()
        
        # Run simulated annealing
        pos = simulated_annealing(pos_initial, max_iter=60000, seed=seed)
        
        # Optimize scale
        pos, r = optimize_scale(pos)
        
        # Final validation
        dists = np.sqrt(np.sum(pos**2, axis=1))
        if np.any(dists > 1.0):
            mask = dists > 1.0
            pos[mask] = pos[mask] / dists[mask, np.newaxis] * 0.9999
            r = compute_radius(pos)
        
        if r > best_r:
            best_r = r
            best_pos = pos.copy()
        
        # Early exit if we're close to known best
        if best_r > 0.1255:
            break
    
    if best_pos is None:
        # Fallback: return hexagonal lattice
        points = []
        r_hex = 0.14
        sqrt3 = np.sqrt(3.0)
        for i in range(-10, 11):
            for j in range(-10, 11):
                x = i * r_hex + (j % 2) * r_hex / 2.0
                y = j * r_hex * sqrt3 / 2.0
                if x*x + y*y <= 0.95:
                    points.append([x, y])
        points = np.array(points)
        if len(points) > N:
            dists = np.sqrt(np.sum(points**2, axis=1))
            idx = np.argsort(dists)[:N]
            points = points[idx]
        best_pos = points
    
    return best_pos
