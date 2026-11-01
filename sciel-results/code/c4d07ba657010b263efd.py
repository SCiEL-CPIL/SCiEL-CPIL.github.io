"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import numpy as np
import time
import math
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _fibonacci_sphere(n):
    """Generate points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_energy(x):
    """
    Compute logarithmic energy efficiently.
    E = sum_{i<j} log(1/||x_i - x_j||)
    """
    # x is (N, 3)
    # diff is (N, N, 3) where diff[i,j] = x_i - x_j
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    # dist is (N, N)
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero on diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Energy
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    return energy

def _compute_grad(x):
    """
    Compute gradient of logarithmic energy.
    grad_i = - sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    
    grad = -np.sum(weighted_diff, axis=1)
    
    return grad

def _rotate_points(x, i, j, angle, axis):
    """
    Rotate points i and j around the axis defined by their difference vector.
    This keeps both points on the sphere and changes their relative configuration.
    """
    x_new = x.copy()
    v = x[i] - x[j]
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-12:
        return x_new
    
    axis = v / norm_v
    
    # Rodrigues' rotation formula
    c = np.cos(angle)
    s = np.sin(angle)
    k = axis
    v1 = x[i]
    v2 = x[j]
    
    # Rotate v1
    v1_rot = v1 * c + np.cross(k, v1) * s + k * np.dot(k, v1) * (1 - c)
    # Rotate v2
    v2_rot = v2 * c + np.cross(k, v2) * s + k * np.dot(k, v2) * (1 - c)
    
    # Normalize to ensure they stay on the sphere (numerical safety)
    v1_rot /= np.linalg.norm(v1_rot)
    v2_rot /= np.linalg.norm(v2_rot)
    
    x_new[i] = v1_rot
    x_new[j] = v2_rot
    
    return x_new

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Initial configuration: Fibonacci sphere
    x = _fibonacci_sphere(N)
    best_x = x.copy()
    best_energy = _compute_energy(x)
    
    # Simulated Annealing parameters
    T_start = 5.0
    T_end = 0.01
    # We will run for a fixed number of iterations or until time runs out.
    # Given N=150, computing energy is O(N^2) ~ 22500 ops, very fast.
    # We can afford many iterations.
    
    rng = np.random.default_rng(42)
    
    # Pre-compute mask for energy calculation to speed things up
    mask = ~np.eye(N, dtype=bool)
    
    current_energy = best_energy
    
    # Determine number of iterations we can afford.
    # Let's just run until deadline.
    
    while time.time() < deadline:
        # Update temperature
        # Linear cooling or exponential? Exponential is standard.
        # We don't know total iterations, so let's use a simple decay based on time?
        # Or just a large number of iterations with exponential decay.
        # Let's estimate max iterations. 120s / (cost per iter).
        # Cost per iter: energy calc (~22k ops) + rotation (cheap).
        # Let's say 100,000 iterations is safe?
        # Actually, let's just use a counter and a fixed T schedule for that counter.
        
        # Since we don't know exact iterations, let's use a time-based temperature decay?
        # No, SA needs a defined schedule. Let's assume we can do ~50k iterations.
        # 50k * 1ms = 50s. We have 115s. So 100k iterations is likely safe.
        
        # Let's define a max_iter for the T schedule
        max_iter = 100000
        # If we exit early, T will be higher than T_end, which is fine (less cooling).
        
        # For this specific step, we need T.
        # We need to track iteration count.
        # Let's restructure to track iter.
        
        pass # Placeholder, logic below

    # Re-implement loop with proper tracking
    x = _fibonacci_sphere(N)
    best_x = x.copy()
    best_energy = _compute_energy(x)
    current_energy = best_energy
    
    max_iter = 100000
    alpha = (T_end / T_start) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        T = T_start * (alpha ** it)
        
        # Pick two random distinct points
        i, j = rng.choice(N, 2, replace=False)
        
        # Random rotation angle
        # Small angles for fine tuning, larger for exploration?
        # Standard SA uses fixed perturbation size.
        # Let's use a random angle in [-pi, pi] but scaled?
        # Actually, rotating by any angle is valid.
        # Let's pick angle uniformly in [-pi, pi].
        angle = rng.uniform(-math.pi, math.pi)
        
        x_new = _rotate_points(x, i, j, angle)
        
        # Calculate new energy
        # Optimization: Only recalculate affected terms?
        # E = sum_{k<l} log(1/d_kl)
        # Changing i and j affects terms (i,j), (i,k) for k!=j, (j,k) for k!=i.
        # Terms (k,l) for k,l != i,j are unchanged.
        # This is a huge speedup!
        
        # Let's implement incremental energy update.
        
        # Old energy components involving i or j:
        # E_old_partial = sum_{k!=i} log(1/d_ik) + sum_{k!=j} log(1/d_jk) + log(1/d_ij)
        # Note: sum_{k!=i} includes j. sum_{k!=j} includes i.
        # So d_ij is counted twice in the sum over k!=i and k!=j?
        # Let's be precise.
        # E = 0.5 * sum_{a!=b} log(1/d_ab)
        # Terms involving i: 0.5 * sum_{b!=i} log(1/d_ib)
        # Terms involving j: 0.5 * sum_{a!=j} log(1/d_aj)
        # Intersection: term (i,j) and (j,i).
        # In the full sum sum_{a!=b}, (i,j) and (j,i) are distinct.
        # So E = 0.5 * ( ... + log(1/d_ij) + log(1/d_ji) + sum_{k!=i,j} log(1/d_ik) + log(1/d_jk) + ... )
        # Since d_ij = d_ji, this is log(1/d_ij) + sum_{k!=i,j} (log(1/d_ik) + log(1/d_jk)).
        
        # Let's calculate the change in energy.
        # d_ij_old, d_ik_old, d_jk_old
        # d_ij_new, d_ik_new, d_jk_new
        
        # Extract old distances
        d_ij_old = np.linalg.norm(x[i] - x[j])
        
        # Distances from i to all others
        diff_i_old = x[i] - x
        d_i_old = np.linalg.norm(diff_i_old, axis=1)
        
        # Distances from j to all others
        diff_j_old = x[j] - x
        d_j_old = np.linalg.norm(diff_j_old, axis=1)
        
        # New distances
        d_ij_new = np.linalg.norm(x_new[i] - x_new[j])
        
        diff_i_new = x_new[i] - x_new
        d_i_new = np.linalg.norm(diff_i_new, axis=1)
        
        diff_j_new = x_new[j] - x_new
        d_j_new = np.linalg.norm(diff_j_new, axis=1)
        
        # Calculate partial energies
        # We need to be careful with the 0.5 factor and double counting.
        # E = sum_{i<j} log(1/d_ij)
        
        # Let's just compute the sum of log(1/d) for pairs involving i or j.
        # Pairs involving i: (i, k) for k != i
        # Pairs involving j: (j, k) for k != j
        # Pair (i, j) is in both sets.
        
        # Sum_i = sum_{k != i} log(1/d_ik)
        # Sum_j = sum_{k != j} log(1/d_jk)
        # Sum_ij = log(1/d_ij)
        
        # Total contribution of i and j to E:
        # E_ij = Sum_i + Sum_j - Sum_ij  (because (i,j) is counted in Sum_i and Sum_j, but should be counted once in E? No.)
        # E = sum_{a<b} ...
        # (i,j) is one pair.
        # (i,k) is one pair.
        # (j,k) is one pair.
        
        # Let's define:
        # E_old_part = sum_{k != i, k != j} log(1/d_ik) + sum_{k != i, k != j} log(1/d_jk) + log(1/d_ij)
        
        # Using vectors:
        # d_i_old has d_ii=0. Mask it.
        # d_j_old has d_jj=0. Mask it.
        
        mask_i = np.ones(N, dtype=bool)
        mask_i[i] = False
        mask_i[j] = False # Exclude j from "others" for i, we handle (i,j) separately
        
        mask_j = np.ones(N, dtype=bool)
        mask_j[i] = False
        mask_j[j] = False # Exclude i from "others" for j
        
        # Sum over k != i, j
        log_terms_i_old = np.sum(-np.log(d_i_old[mask_i]))
        log_terms_j_old = np.sum(-np.log(d_j_old[mask_j]))
        log_term_ij_old = -np.log(d_ij_old)
        
        E_old_part = log_terms_i_old + log_terms_j_old + log_term_ij_old
        
        log_terms_i_new = np.sum(-np.log(d_i_new[mask_i]))
        log_terms_j_new = np.sum(-np.log(d_j_new[mask_j]))
        log_term_ij_new = -np.log(d_ij_new)
        
        E_new_part = log_terms_i_new + log_terms_j_new + log_term_ij_new
        
        delta_E = E_new_part - E_old_part
        
        # Acceptance probability
        if delta_E < 0:
            accept = True
        else:
            # Avoid log(0) if T is very small
            if T < 1e-10:
                accept = False
            else:
                accept = rng.random() < math.exp(-delta_E / T)
        
        if accept:
            x = x_new
            current_energy += delta_E
            
            # Check for new best
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()
    
    # Final normalization to be safe
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    
    return best_x

# EVOLVE-BLOCK-END
