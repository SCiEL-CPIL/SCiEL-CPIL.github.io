import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_radius(centers):
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    x = centers[:, 0]
    y = centers[:, 1]
    
    dx = 0.5 - np.abs(x)
    dy = 0.5 - np.abs(y)
    wall_dists = np.minimum(dx, dy)
    
    if np.any(wall_dists < 0):
        return 0.0
        
    min_wall = np.min(wall_dists)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)

def energy_and_grad(centers):
    centers = np.asarray(centers)
    x = centers[:, 0]
    y = centers[:, 1]
    
    E = 0.0
    grad = np.zeros_like(centers)
    
    # Wall term
    dx = 0.5 - np.abs(x)
    dy = 0.5 - np.abs(y)
    w = np.minimum(dx, dy)
    w = np.maximum(w, 1e-9)
    
    E_wall = np.sum(1.0 / (w**2))
    E += E_wall
    
    # Gradient of wall term
    # dw/dx = -sign(x) if dx < dy else 0
    # dw/dy = -sign(y) if dy < dx else 0
    
    mask_x = dx < dy
    mask_y = dy < dx
    # If equal, assign to x arbitrarily
    
    dw_dx = np.zeros(N)
    dw_dy = np.zeros(N)
    
    dw_dx[mask_x] = -np.sign(x[mask_x])
    dw_dy[mask_y] = -np.sign(y[mask_y])
    
    # d(1/w^2)/dx = -2/w^3 * dw/dx
    grad[:, 0] += -2.0 / (w**3) * dw_dx
    grad[:, 1] += -2.0 / (w**3) * dw_dy
    
    # Pairwise term
    # diff: (N, N, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, 1e9)
    dists_sq = np.maximum(dists_sq, 1e-9)
    
    # E_pair = sum_{i<j} 1/dists_sq[i,j]
    # We can compute the full matrix and take upper triangle
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    E_pair = 0.5 * np.sum(inv_d2) # 0.5 because of double counting
    E += E_pair
    
    # Gradient
    # d(1/d^2)/dx_k = -2 * sum_{j != k} (x_k - x_j) / d^4
    # Let's compute the sum for each k
    
    # diff_x: (N, N) matrix of x_i - x_j
    diff_x = x[:, np.newaxis] - x[np.newaxis, :]
    diff_y = y[:, np.newaxis] - y[np.newaxis, :]
    
    # d4 = dists_sq^2
    d4 = dists_sq**2
    d4 = np.maximum(d4, 1e-9)
    
    # Sum over j for each i
    # grad_x_i = -2 * sum_j (x_i - x_j) / d4[i,j]
    # Note: diagonal is 0 because diff is 0
    
    sum_x = np.sum(diff_x / d4, axis=1)
    sum_y = np.sum(diff_y / d4, axis=1)
    
    grad[:, 0] += -2.0 * sum_x
    grad[:, 1] += -2.0 * sum_y
    
    return E, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_radius = 0.0
    best_centers = None
    
    inits = []
    
    # 1. Grid
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid_50 = np.vstack([grid, [[0.0, 0.0]]])
    inits.append(grid_50)
    
    # 2. Random
    np.random.seed(42)
    for _ in range(50):
        pts = np.random.uniform(-0.4, 0.4, (N, 2))
        inits.append(pts)

    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                lambda x: energy_and_grad(x)[0], 
                init, 
                jac=lambda x: energy_and_grad(x)[1],
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14},
                bounds=[(-0.5, 0.5) for _ in range(2*N)]
            )
            
            if res.fun < 1e6:
                current_radius = compute_radius(res.x)
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = res.x.copy()
        except Exception:
            pass

    if best_centers is None:
        m = 7
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = np.vstack([grid, [[0.0, 0.0]]])

    return best_centers
