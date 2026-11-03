"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(centres):
    """Compute the packing radius for a set of centres."""
    if centres.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Distance to walls
    wall_dists = 0.5 - np.abs(centres)
    min_wall = np.min(wall_dists)
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_pair / 2.0, min_wall)

def _energy_and_grad(centres, t):
    """
    Energy function: penalize overlaps and wall violations.
    We want to maximize the minimum radius, which is equivalent to
    minimizing a barrier function that pushes apart close pairs and
    pulls centers away from walls.
    
    Using a soft-min / log-sum-exp style approach:
    E = -log(sum(exp(-beta * d_ij))) - log(sum(exp(-beta * w_i)))
    where d_ij are pairwise distances and w_i are wall distances.
    But this is expensive. Instead, use a simpler penalty:
    E = sum(max(0, 2r - d_ij)^2) + sum(max(0, r - w_i)^2)
    with r being a target radius.
    
    Actually, let's use a different approach: minimize the sum of
    inverse distances (with a floor) to encourage spreading.
    """
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1e10)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centres)
    
    # Energy: sum of 1/d for pairs (encourages large min distance)
    # and sum of 1/w for walls (encourages staying away from walls)
    # Use a softened version to avoid singularities
    eps = 1e-6
    pair_energy = np.sum(1.0 / (dists + eps))
    wall_energy = np.sum(1.0 / (np.abs(wall_dists) + eps))
    
    energy = pair_energy + wall_energy
    
    # Gradient
    # d/dx_i [1/|x_i - x_j|] = -(x_i - x_j) / |x_i - x_j|^3
    grad = np.zeros_like(centres)
    for i in range(N):
        for j in range(i+1, N):
            d = dists[i, j]
            diff_ij = centres[i] - centres[j]
            factor = -1.0 / (d**3 + eps**3)
            grad[i] += factor * diff_ij
            grad[j] -= factor * diff_ij
    
    # Wall gradient: d/dx_i [1/|0.5 - x_ik|]
    for i in range(N):
        for k in range(3):
            w = wall_dists[i, k]
            if w > 0:
                # 1/w, derivative w.r.t. x_ik is -1/w^2 * sign
                # Actually wall_dists[i,k] = 0.5 - x_ik if x_ik >= 0, or 0.5 + x_ik if x_ik < 0
                # Let's just use the absolute value derivative
                grad[i, k] -= np.sign(centres[i, k]) / (w**2 + eps**2)
            else:
                # If already outside, push back
                grad[i, k] += np.sign(centres[i, k]) * 100.0
    
    return energy, grad

def _energy_and_grad_fast(centres, t):
    """Vectorized energy and gradient computation."""
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1e10)
    
    wall_dists = 0.5 - np.abs(centres)
    
    eps = 1e-8
    pair_energy = np.sum(1.0 / (dists + eps))
    wall_energy = np.sum(1.0 / (np.abs(wall_dists) + eps))
    
    energy = pair_energy + wall_energy
    
    # Gradient for pairs
    # factor[i,j] = -1 / d_ij^3
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = -1.0 / (dists**3 + eps)
    np.fill_diagonal(factor, 0)
    
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    # Wall gradient
    signs = np.sign(centres)
    with np.errstate(divide='ignore', invalid='ignore'):
        wall_grad = -signs / (wall_dists**2 + eps)
    # Where wall_dists <= 0, add a strong push
    wall_grad[wall_dists <= 0] = signs[wall_dists <= 0] * 100.0
    grad += wall_grad
    
    return energy, grad

def _sgld_step(centres, t, lr, noise_scale):
    """One step of Stochastic Gradient Langevin Dynamics."""
    energy, grad = _energy_and_grad_fast(centres, t)
    noise = noise_scale * np.random.randn_like(centres)
    new_centres = centres - lr * grad + noise
    # Project back into cube
    new_centres = np.clip(new_centres, -0.5 + 1e-10, 0.5 - 1e-10)
    return new_centres, energy

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    # Initial guess: cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centres = grid[:N].copy()
    
    best_score = _score(centres)
    best_centres = centres.copy()
    
    # Multi-start with SGLD
    np.random.seed(42)
    
    for restart in range(10):
        if time.time() > deadline - 10.0:
            break
        
        if restart == 0:
            current = centres.copy()
        else:
            # Random perturbation of best
            current = best_centres + 0.05 * np.random.randn(N, 3)
            current = np.clip(current, -0.5 + 1e-10, 0.5 - 1e-10)
        
        # SGLD phase
        lr = 0.001
        noise_scale = 0.01
        for step in range(5000):
            if time.time() > deadline - 5.0:
                break
            
            current, energy = _sgld_step(current, step, lr, noise_scale)
            
            # Anneal
            lr = max(0.0001, 0.001 * (1.0 - step / 5000.0))
            noise_scale = max(0.001, 0.01 * (1.0 - step / 5000.0))
        
        # Gradient descent polish
        for polish in range(2000):
            if time.time() > deadline - 3.0:
                break
            
            energy, grad = _energy_and_grad_fast(current, 0)
            # Adaptive learning rate
            lr = 0.0005 / (1.0 + polish / 100.0)
            current = current - lr * grad
            current = np.clip(current, -0.5 + 1e-10, 0.5 - 1e-10)
        
        # Evaluate
        score = _score(current)
        if score > best_score:
            best_score = score
            best_centres = current.copy()
        
        # Final polish with L-BFGS-B if time permits
        if time.time() < deadline - 20.0:
            def objective(x):
                c = x.reshape(N, 3)
                # Use negative of a proxy for the score
                diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff * diff, axis=2))
                np.fill_diagonal(dists, 1e10)
                wall_dists = 0.5 - np.abs(c)
                # Minimize max violation: use sum of squares of min distances
                min_pair = np.min(dists)
                min_wall = np.min(wall_dists)
                r = min(min_pair / 2.0, min_wall)
                return -r
            
            x0 = best_centres.flatten()
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                              options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < -best_score and time.time() < deadline - 5.0:
                    candidate = res.x.reshape(N, 3)
                    candidate = np.clip(candidate, -0.5 + 1e-10, 0.5 - 1e-10)
                    cand_score = _score(candidate)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_centres = candidate
            except Exception:
                pass
    
    return best_centres
# EVOLVE-BLOCK-END
