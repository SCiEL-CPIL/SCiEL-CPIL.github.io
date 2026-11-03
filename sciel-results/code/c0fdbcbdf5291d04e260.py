"""Simulated annealing packing of 50 equal circles in a unit circle."""
import numpy as np
from scipy.optimize import minimize
import random

N = 50

def compute_radius(pos):
    """Compute the actual radius of the packing."""
    pos = np.asarray(pos).reshape(-1, 2)
    dists = np.sqrt(np.sum(pos**2, axis=1))
    if np.any(dists > 1.0 + 1e-12):
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    min_pair_dist = np.min(dist_mat)
    min_boundary_dist = np.min(1.0 - dists)
    return min(min_pair_dist / 2.0, min_boundary_dist)

def optimize_scale(pos):
    """Find optimal uniform scale to maximize radius."""
    pos = np.asarray(pos).reshape(-1, 2)
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
    
    # The optimal scale s satisfies s * r_pair = 1 - s * max_dist => s = 1 / (r_pair + max_dist)
    s_opt = 1.0 / (r_pair + max_dist)
    s_opt = np.clip(s_opt, 0.5, 1.5)
    
    r_final = radius_at_scale(s_opt)
    r_orig = radius_at_scale(1.0)
    
    if r_final > r_orig:
        return pos * s_opt, r_final
    else:
        return pos, r_orig

def expand_radius(pos, max_iter=200, verbose=False):
    """
    Iteratively increase the target radius and solve for positions that satisfy it.
    This is more robust than pure SA for finding high-quality packings.
    """
    pos = np.asarray(pos).reshape(-1, 2).copy()
    current_r = compute_radius(pos)
    if current_r == 0.0:
        return pos, 0.0
    
    # Start with a small increment
    delta_r = 0.001
    min_delta_r = 1e-5
    
    for i in range(max_iter):
        target_r = current_r + delta_r
        
        def objective(x):
            p = x.reshape(-1, 2)
            # Boundary penalty
            dists = np.sqrt(np.sum(p**2, axis=1))
            b_viol = np.maximum(0, dists - (1.0 - target_r))
            b_pen = np.sum(b_viol**2)
            
            # Pairwise penalty
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist_mat = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist_mat, np.inf)
            p_viol = np.maximum(0, 2.0 * target_r - dist_mat)
            p_pen = np.sum(p_viol**2)
            
            return b_pen + p_pen
        
        # Use L-BFGS-B to minimize overlaps for the new target radius
        try:
            res = minimize(objective, pos.reshape(-1), method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            new_pos = res.x.reshape(-1, 2)
            new_r = compute_radius(new_pos)
            
            if new_r >= target_r - 1e-6:
                # Success, we found a valid packing for the larger radius
                pos = new_pos
                current_r = new_r
                # Keep delta_r or increase slightly if we had easy success
                if i % 10 == 0:
                    delta_r *= 1.1
            else:
                # Failed to reach target_r, reduce delta_r
                delta_r *= 0.5
                if delta_r < min_delta_r:
                    break
        except Exception:
            delta_r *= 0.5
            if delta_r < min_delta_r:
                break
                
    return pos, current_r

def construct():
    np.random.seed(42)
    random.seed(42)
    
    best_pos = None
    best_r = 0.0
    
    # Multiple restarts with different initial configurations
    for restart in range(10):
        seed = 42 + restart * 137
        
        # Generate initial hexagonal lattice with varying parameters
        points = []
        
        # Vary the density and distortion
        base_spacing = 0.25 - restart * 0.002
        if base_spacing < 0.20:
            base_spacing = 0.20
            
        sqrt3 = np.sqrt(3.0)
        
        # Generate a larger grid and filter
        for i in range(-15, 16):
            for j in range(-15, 16):
                x = i * base_spacing + (j % 2) * base_spacing / 2.0
                y = j * base_spacing * sqrt3 / 2.0
                # Add random perturbation
                x += np.random.uniform(-0.05, 0.05)
                y += np.random.uniform(-0.05, 0.05)
                if x*x + y*y <= 0.98:
                    points.append([x, y])
        
        points = np.array(points)
        
        if len(points) == 0:
            continue
            
        if len(points) > N:
            # Select N points closest to center or random subset?
            # Random subset might be better to avoid bias, but center bias helps stability.
            # Let's take the first N after shuffling? Or closest to center?
            # Closest to center ensures they are inside.
            dists = np.sqrt(np.sum(points**2, axis=1))
            idx = np.argsort(dists)[:N]
            points = points[idx]
        elif len(points) < N:
            while len(points) < N:
                angle = np.random.uniform(0, 2*np.pi)
                r = np.random.uniform(0, 0.8)
                points = np.vstack([points, [[r*np.cos(angle), r*np.sin(angle)]]])
        
        pos_initial = points.copy()
        
        # 1. Initial quick relaxation using expand_radius with small steps
        pos, r = expand_radius(pos_initial, max_iter=50)
        
        # 2. Scale optimization
        pos, r = optimize_scale(pos)
        
        # 3. More aggressive expansion
        pos, r = expand_radius(pos, max_iter=100)
        
        # 4. Final scale optimization
        pos, r = optimize_scale(pos)
        
        # 5. One last fine-tuning pass
        pos, r = expand_radius(pos, max_iter=50)
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
            
        if best_r > 0.125825:
            break
        
    if best_pos is None:
        # Fallback
        points = []
        r_hex = 0.13
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
        best_r = compute_radius(best_pos)

    return best_pos
