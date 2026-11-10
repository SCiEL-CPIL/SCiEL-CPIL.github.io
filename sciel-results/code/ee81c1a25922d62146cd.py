import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _energy_and_grad(pos):
    """
    Calculate repulsion energy and gradient.
    pos: (N, 3) array
    """
    N = pos.shape[0]
    energy = 0.0
    grad = np.zeros_like(pos)
    
    # Pairwise repulsion
    # d_ij = pos[i] - pos[j]
    # E_pair = sum 1/d^2
    # grad_i = -2/d^3 * d_ij
    
    # Vectorized calculation for speed
    # diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :] # (N, N, 3)
    # dist2 = np.sum(diff**2, axis=2) # (N, N)
    # np.fill_diagonal(dist2, 1.0) # Avoid div by zero
    # inv_dist2 = 1.0 / dist2
    # np.fill_diagonal(inv_dist2, 0.0)
    # energy = np.sum(inv_dist2)
    # 
    # grad contribution from pair (i,j) to pos[i] is -2 * inv_dist2[i,j] * diff[i,j]
    # grad contribution to pos[j] is +2 * inv_dist2[i,j] * diff[i,j]
    # Total grad[i] = sum_j -2 * inv_dist2[i,j] * diff[i,j]
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    # Ensure no division by zero
    dist2[dist2 < 1e-12] = 1e-12
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    energy = np.sum(inv_dist2)
    
    # grad = -2 * sum_j (inv_dist2[i,j] * diff[i,j])
    # inv_dist2 is (N,N), diff is (N,N,3)
    # Multiply inv_dist2[:, :, np.newaxis] with diff
    weighted_diff = inv_dist2[:, :, np.newaxis] * diff
    grad = -2.0 * np.sum(weighted_diff, axis=1)
    
    # Wall repulsion
    # w_ik = 0.5 - |pos[i,k]|
    # E_wall = sum 1/w^2
    # grad_ik = 2 * sign(pos[i,k]) / w^3
    
    abs_pos = np.abs(pos)
    w = 0.5 - abs_pos
    # Prevent zero
    w[w < 1e-12] = 1e-12
    inv_w2 = 1.0 / (w**2)
    inv_w3 = 1.0 / (w**3)
    
    energy += np.sum(inv_w2)
    
    # grad_wall[i,k] = 2 * sign(pos[i,k]) * inv_w3[i,k]
    signs = np.sign(pos)
    grad += 2.0 * signs * inv_w3
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_radius = 0.0
    best_pos = None
    
    # Initial best: simple grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_pos = grid[:N].copy()
    
    # Evaluate initial
    # Calculate radius for a given position
    def calc_radius(pos):
        # min distance between centers / 2
        d = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_dist = np.min(d)
        
        # min distance to wall
        min_wall = np.min(0.5 - np.abs(pos))
        
        return min(min_dist / 2.0, min_wall)
        
    r_init = calc_radius(init_pos)
    if r_init > best_radius:
        best_radius = r_init
        best_pos = init_pos.copy()
        
    # Optimization loop
    # We use L-BFGS-B to minimize energy
    # Since the problem is symmetric, we might get stuck in local minima.
    # We do multiple restarts.
    
    for restart in range(20): # Max 20 restarts
        if time.time() > deadline:
            break
            
        # Generate initial guess
        # Jittered grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        pos0 = grid[:N].copy()
        
        # Add random jitter
        jitter_scale = 0.05
        pos0 += np.random.uniform(-jitter_scale, jitter_scale, pos0.shape)
        
        # Clip to be strictly inside to avoid initial singularity if jitter pushes out
        # Actually, wall repulsion handles it, but better to start valid
        pos0 = np.clip(pos0, -0.49, 0.49)
        
        # Run optimization
        try:
            res = minimize(_energy_and_grad, pos0, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20})
            
            if res.success or res.status == 0:
                pos_opt = res.x
                r_opt = calc_radius(pos_opt)
                
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_pos = pos_opt.copy()
        except Exception:
            pass
            
    return best_pos
