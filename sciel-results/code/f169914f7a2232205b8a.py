import numpy as np
import time
import random
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def compute_radius(pos):
    """
    Computes the largest common radius r for the given centers.
    r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    """
    # Pairwise distances
    # pos shape (N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # Walls are at +/- 0.5
    # dist to wall = min(0.5 - x, 0.5 + x) for each dimension
    # Actually, min over dimensions of (0.5 - |x|)
    abs_pos = np.abs(pos)
    dists_to_walls = 0.5 - abs_pos
    min_wall_dist = np.min(dists_to_walls)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def compute_energy_smooth(pos):
    """
    Smooth energy function for optimization.
    E = sum_{i<j} 1/d_ij^2 + sum_i sum_a [ 1/(0.5-x_ia)^2 + 1/(0.5+x_ia)^2 ]
    """
    # Pairwise
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    # Avoid division by zero if any dist is 0 (should not happen if initialized well)
    dists_sq = np.maximum(dists_sq, 1e-9)
    pair_energy = np.sum(1.0 / dists_sq)
    
    # Wall terms
    # x in [-0.5, 0.5]
    # Terms: 1/(0.5-x)^2 + 1/(0.5+x)^2
    # Note: if x is close to 0.5, 0.5-x is small.
    # We assume pos is inside the cube.
    wall_energy = 0.0
    for a in range(3):
        x = pos[:, a]
        term1 = 1.0 / np.maximum((0.5 - x) ** 2, 1e-9)
        term2 = 1.0 / np.maximum((0.5 + x) ** 2, 1e-9)
        wall_energy += np.sum(term1) + np.sum(term2)
        
    return pair_energy + wall_energy

def compute_energy_grad(pos):
    """
    Returns energy and gradient for the smooth energy function.
    """
    N = len(pos)
    grad = np.zeros_like(pos)
    energy = 0.0
    
    # Pairwise part
    # E_pair = sum_{i<j} 1/d_ij^2
    # d_ij^2 = sum_k (x_ik - x_jk)^2
    # d/dx_ik (1/d_ij^2) = -1/d_ij^4 * 2(x_ik - x_jk)
    
    # Vectorized calculation for energy
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists_sq_safe = np.maximum(dists_sq, 1e-9)
    inv_dists_sq = 1.0 / dists_sq_safe
    energy += np.sum(inv_dists_sq)
    
    # Gradient for pairwise part
    # grad_i = sum_{j != i} -2 * diff_ij / dists_sq_ij^2
    # diff_ij = pos_i - pos_j
    # term = -2 * (pos_i - pos_j) / dists_sq_ij^2
    # We can compute this using broadcasting
    # weights shape (N, N), weights[i,j] = -2 / dists_sq_ij^2
    weights = -2.0 * (inv_dists_sq ** 2)
    np.fill_diagonal(weights, 0.0)
    
    # grad_i = sum_j weights[i,j] * (pos_i - pos_j)
    # sum_j weights[i,j] * pos_i = pos_i * sum_j weights[i,j]
    # sum_j weights[i,j] * (-pos_j) = - sum_j (weights[i,j] * pos_j)
    
    sum_weights = np.sum(weights, axis=1) # shape (N,)
    weighted_pos_sum = np.dot(weights, pos) # shape (N, 3)
    
    grad_pair = pos * sum_weights[:, np.newaxis] - weighted_pos_sum
    grad += grad_pair
    
    # Wall part
    # E_wall = sum_i sum_a [ 1/(0.5-x_ia)^2 + 1/(0.5+x_ia)^2 ]
    # d/dx 1/(0.5-x)^2 = -1 * (0.5-x)^-3 * (-1) = 1/(0.5-x)^3
    # d/dx 1/(0.5+x)^2 = -1 * (0.5+x)^-3 * (1) = -1/(0.5+x)^3
    
    for a in range(3):
        x = pos[:, a]
        term1 = 1.0 / np.maximum((0.5 - x) ** 3, 1e-9)
        term2 = -1.0 / np.maximum((0.5 + x) ** 3, 1e-9)
        grad[:, a] += term1 + term2
        energy += np.sum(1.0 / np.maximum((0.5 - x) ** 2, 1e-9)) + np.sum(1.0 / np.maximum((0.5 + x) ** 2, 1e-9))
        
    return energy, grad

def sa_optimize(pos, T0, alpha, steps, rng):
    """
    Simulated Annealing optimization.
    """
    current_pos = pos.copy()
    current_energy = compute_energy_smooth(current_pos)
    T = T0
    
    for _ in range(steps):
        # Pick a random particle
        i = rng.randint(0, N - 1)
        
        # Propose move
        # Move size proportional to T
        step_size = 0.05 * T
        delta = rng.normal(0, step_size, size=3)
        
        new_pos_i = current_pos[i] + delta
        # Keep within bounds loosely during SA, polish will fix
        # But avoid huge jumps outside
        new_pos_i = np.clip(new_pos_i, -0.6, 0.6)
        
        # Calculate delta energy efficiently
        # Only recalculate terms involving i
        old_pos_i = current_pos[i]
        
        # Old pairwise energy involving i
        dists_old_sq = np.sum((current_pos - old_pos_i) ** 2, axis=1)
        dists_old_sq[i] = np.inf
        dists_old_sq = np.maximum(dists_old_sq, 1e-9)
        pair_old = np.sum(1.0 / dists_old_sq)
        
        # New pairwise energy involving i
        current_pos[i] = new_pos_i
        dists_new_sq = np.sum((current_pos - new_pos_i) ** 2, axis=1)
        dists_new_sq[i] = np.inf
        dists_new_sq = np.maximum(dists_new_sq, 1e-9)
        pair_new = np.sum(1.0 / dists_new_sq)
        
        # Wall energy for i
        def wall_energy_single(p):
            e = 0.0
            for a in range(3):
                x = p[a]
                e += 1.0 / max((0.5 - x) ** 2, 1e-9)
                e += 1.0 / max((0.5 + x) ** 2, 1e-9)
            return e
            
        wall_old = wall_energy_single(old_pos_i)
        wall_new = wall_energy_single(new_pos_i)
        
        delta_e = (pair_new - pair_old) + (wall_new - wall_old)
        
        # Acceptance
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            current_energy += delta_e
            # current_pos[i] is already new_pos_i
        else:
            current_pos[i] = old_pos_i
            
        # Cool
        T *= alpha
        
    return current_pos

def polish(pos):
    """
    Local optimization using L-BFGS-B.
    """
    # Flatten position
    x0 = pos.flatten()
    
    def objective(x):
        p = x.reshape(N, 3)
        e, _ = compute_energy_grad(p)
        return e
        
    def gradient(x):
        p = x.reshape(N, 3)
        _, g = compute_energy_grad(p)
        return g.flatten()
        
    # Bounds: keep inside cube with small margin to avoid singularities
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    try:
        res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
        return res.x.reshape(N, 3)
    except Exception:
        return pos

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_pos = None
    best_r = -1.0
    rng = np.random.default_rng(42)
    
    # Initial simple grid to have a valid fallback
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best_pos = grid[:N].copy()
    best_r = compute_radius(best_pos)
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Check if we have time for a full restart (SA + Polish)
        # Estimate: SA 5000 steps is fast, Polish might take 1-5s.
        # If less than 10s left, break.
        if time.time() > deadline - 10.0:
            break
            
        # 1. Initialize
        # Random positions in a smaller cube to avoid immediate wall issues
        pos = rng.uniform(-0.4, 0.4, size=(N, 3))
        
        # 2. Simulated Annealing
        # Parameters tuned for N=100
        T0 = 0.5
        alpha = 0.995
        steps = 5000
        
        pos = sa_optimize(pos, T0, alpha, steps, rng)
        
        # 3. Polish
        pos_polished = polish(pos)
        
        # 4. Evaluate
        r = compute_radius(pos_polished)
        
        if r > best_r:
            best_r = r
            best_pos = pos_polished.copy()
            
        # Optional: Small SA step after polish to escape local minima?
        # Given time constraints, we'll just do restarts.
        
    return best_pos
# EVOLVE-BLOCK-END
