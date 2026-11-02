"""Optimized packing of 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """Multi-start local optimization with diverse initializations."""
    
    def objective(x):
        x = x.reshape(N, 2)
        x = np.clip(x, -0.5, 0.5)
        diff = x[:, None, :] - x[None, :, :]
        d = np.sqrt((diff**2).sum(-1))
        np.fill_diagonal(d, np.inf)
        mp = d.min()
        b = np.minimum(np.minimum(0.5 - x[:,0], x[:,0]+0.5),
                       np.minimum(0.5 - x[:,1], x[:,1]+0.5))
        return -min(mp, b.min())
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    best_x = None
    best_val = -np.inf
    
    # Multiple starts: grid, hexagonal, random
    rng = np.random.RandomState(42)
    
    starts = []
    
    # Grid start
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    starts.append(grid[:N].copy())
    
    # Hexagonal start
    s = 0.072
    sqrt3 = np.sqrt(3)
    hex_pts = []
    for i in range(20):
        y = -0.5 + i * s * sqrt3 / 2
        if y > 0.5:
            break
        x_off = (i % 2) * s / 2
        for j in range(20):
            x = -0.5 + j * s + x_off
            if x > 0.5:
                break
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    if len(hex_pts) >= N:
        d = np.sum(hex_pts**2, axis=1)
        starts.append(hex_pts[np.argsort(d)[:N]])
    
    # Random starts
    for _ in range(5):
        starts.append(rng.uniform(-0.45, 0.45, (N, 2)))
    
    for x0 in starts:
        x0 = np.clip(x0, -0.499, 0.499)
        res = minimize(objective, x0.flatten(), method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10})
        val = -res.fun
        if val > best_val:
            best_val = val
            best_x = res.x.reshape(N, 2)
    
    return np.clip(best_x, -0.4999, 0.4999)
# EVOLVE-BLOCK-END
