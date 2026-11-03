import numpy as np
import time
from scipy.optimize import minimize

N = 200

def compute_radius(centers):
    # centers shape (N, 2)
    # check bounds
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers[:, 0] < -0.5) or np.any(centers[:, 0] > 0.5) or np.any(centers[:, 1] < -0.5) or np.any(centers[:, 1] > 0.5):
        return 0.0
    # distances
    dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # wall distances
    wall_dists = np.min(np.stack([
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ]), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_dist / 2.0, min_wall_dist)

def energy_and_grad(flat):
    x = flat.reshape(N, 2)
    # ensure inside bounds for stability, though optimizer handles bounds
    # Energy: sum 1/d^2
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=-1)
    np.fill_diagonal(d2, 1.0) # avoid div by 0
    # Avoid extremely small d2 causing overflow/instability
    d2 = np.maximum(d2, 1e-6)
    
    E_pair = np.sum(1.0 / d2)
    
    # Wall energy
    dx = 0.5 - np.abs(x[:, 0])
    dy = 0.5 - np.abs(x[:, 1])
    dw2 = np.minimum(dx**2, dy**2)
    dw2 = np.maximum(dw2, 1e-6)
    E_wall = np.sum(1.0 / dw2)
    
    E = E_pair + E_wall
    
    # Gradient
    # dE/dx_i for pair term: sum_j d(1/d_ij^2)/dx_i = sum_j -2/d_ij^4 * (x_i - x_j)
    inv_d4 = 1.0 / (d2**2)
    np.fill_diagonal(inv_d4, 0.0)
    grad_pair = -2.0 * np.sum(diff * inv_d4[:, :, np.newaxis], axis=1)
    
    # Wall gradient
    # d(1/dw^2)/dx = -2/dw^4 * dw * d(dw)/dx
    # dw = min(dx, dy). If dx < dy, dw=dx, d(dw)/dx = sign(x). Else sign(y) term.
    # This is tricky with min. Let's just use numerical gradient or simplify wall penalty.
    # Actually, L-BFGS-B can handle bounds, so we might not need wall energy if we just bound the box.
    # But bounding doesn't prevent circles from touching the wall too closely if they are pushed by others.
    # We need the wall repulsion to push them away if they are too close.
    
    # Simpler: Use only pair repulsion and rely on bounds? No, they will pile up on walls.
    # Let's use a smooth wall potential: $E_{wall} = \sum (\frac{1}{dx^2} + \frac{1}{dy^2})$? 
    # That would be $1/(0.5-x)^2 + 1/(0.5+x)^2$ etc.
    
    # Let's stick to the min distance logic but make it differentiable.
    # Or just use Nelder-Mead which doesn't need gradients and handles non-smooth mins better.
    # Given N=200, Nelder-Mead might be slow but 150s is plenty for a few runs.
    # Let's try Nelder-Mead with a simple objective.

def objective(flat):
    x = flat.reshape(N, 2)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=-1)
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-10)
    E_pair = np.sum(1.0 / d2)
    
    dx = 0.5 - np.abs(x[:, 0])
    dy = 0.5 - np.abs(x[:, 1])
    # Smooth approximation of min(dx, dy) or just sum of inverse squares of distances to each wall
    # Distance to left: 0.5 + x, right: 0.5 - x
    # Using sum of 1/d^2 for each wall ensures they stay away from all walls.
    d_left = 0.5 + x[:, 0]
    d_right = 0.5 - x[:, 0]
    d_down = 0.5 + x[:, 1]
    d_up = 0.5 - x[:, 1]
    
    # Clip to avoid inf
    d_left = np.maximum(d_left, 1e-6)
    d_right = np.maximum(d_right, 1e-6)
    d_down = np.maximum(d_down, 1e-6)
    d_up = np.maximum(d_up, 1e-6)
    
    E_wall = np.sum(1.0/d_left**2 + 1.0/d_right**2 + 1.0/d_down**2 + 1.0/d_up**2)
    
    return E_pair + E_wall

def construct():
    start_time = time.time()
    best_r = 0
    best_x = None
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    base_grid = grid[:N]
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    for i in range(20): # 20 restarts
        if time.time() - start_time > 120: break
        
        # Jitter
        np.random.seed(i)
        jitter = np.random.uniform(-0.01, 0.01, size=(N, 2))
        x0 = base_grid + jitter
        x0 = np.clip(x0, -0.49, 0.49)
        
        res = minimize(objective, x0.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        x_opt = res.x.reshape(N, 2)
        r = compute_radius(x_opt)
        
        if r > best_r:
            best_r = r
            best_x = x_opt
            
    return best_x
