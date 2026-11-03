"""Circle packing in a unit square: maximize common radius for 50 equal circles."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # Safety margin before 120s kill

def _score(pos):
    """Compute the maximum valid radius for given positions."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(pos)):
        return -1.0
    # Wall constraint
    wall = 0.5 - np.abs(pos).max(axis=1)
    r_wall = wall.min()
    if r_wall < 0:
        return -1.0
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    r_pair = (dist.min() / 2.0)
    return min(r_wall, r_pair)

def _relax(c, R, iters, eta0, eta1, t0=None, deadline=None):
    """
    Relaxation step: push overlapping circles apart and keep them inside the square.
    c: (N, 2) array of centers
    R: target radius
    iters: number of relaxation iterations
    eta0, eta1: learning rate start and end
    """
    c = c.copy()
    n = len(c)
    for it in range(iters):
        if t0 is not None and deadline is not None:
            if time.time() - t0 > deadline - 3:
                break
        t = it / max(iters, 1)
        eta = eta0 + (eta1 - eta0) * t
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2 * R - d, 0.0) / d
        np.fill_diagonal(mag, 0.0)
        f = (mag[:, :, np.newaxis] * diff).sum(1)
        
        # Wall forces
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
        
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    return c

def _grow(start, R0, R1, step, iters, eta0, eta1, t0, deadline):
    """
    Grow the radius from R0 to R1 in steps, relaxing at each step.
    Returns (best_centers, best_radius).
    """
    c = start.copy()
    best = (c.copy(), _score(c))
    R = R0
    while R <= R1 and (time.time() - t0) < deadline - 5:
        c = _relax(c, R, iters, eta0, eta1, t0, deadline)
        r = _score(c)
        if r > best[1]:
            best = (c.copy(), r)
        R += step
        if (time.time() - t0) > deadline - 5:
            break
    return best

def _init_grid_7x7():
    """7x7 grid (49 circles) + 1 at center."""
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    return np.vstack([g, extra])

def _init_jitter8(rng):
    """8x8 grid, pick 50, add jitter."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-0.03, 0.03, c.shape)
    return c

def _init_rand(rng):
    """Random uniform positions."""
    return rng.uniform(-0.45, 0.45, (N, 2))

def _init_hex():
    """Hexagonal lattice starting position."""
    positions = []
    spacing = 0.14
    row_spacing = spacing * np.sqrt(3) / 2
    for row in range(8):
        for col in range(7):
            x = (col - 3) * spacing + (0.5 * spacing if row % 2 else 0)
            y = (row - 3.5) * row_spacing
            positions.append([x, y])
    positions = np.array(positions[:N])
    max_abs = np.abs(positions).max()
    if max_abs > 0.45:
        positions = positions * (0.45 / max_abs)
    return positions

def _init_grid_8x8():
    """8x8 grid, pick 50."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    return g[:N].copy()

def _langevin_refine(c, r, t0, deadline, steps=5000, T0=0.001, T1=0.0001, lr=0.002):
    """
    Langevin dynamics refinement: add stochastic kicks to escape local minima.
    """
    c = c.copy()
    best_c = c.copy()
    best_r = r
    rng = np.random.default_rng(999)
    
    for step in range(steps):
        if time.time() - t0 > deadline - 5:
            break
        t = step / steps
        T = T0 + (T1 - T0) * t
        lr_t = lr * (1.0 - 0.5 * t)
        
        # Compute forces (same as relaxation)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2 * r - d, 0.0) / d
        np.fill_diagonal(mag, 0.0)
        f = (mag[:, :, np.newaxis] * diff).sum(1)
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - r), 0.0) + np.maximum((-0.5 + r) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - r), 0.0) + np.maximum((-0.5 + r) - c[:, 1], 0.0)
        
        # Langevin update
        noise = T * rng.standard_normal(c.shape)
        c += lr_t * f + noise
        c = np.clip(c, -0.5, 0.5)
        
        r_new = _score(c)
        if r_new > best_r:
            best_r = r_new
            best_c = c.copy()
    
    return best_c, best_r

def _soft_min_radius(pos, beta=1000.0):
    """
    Smooth surrogate for the radius: soft-min of wall distances and half pairwise distances.
    Returns the value (negative for minimization).
    """
    pos = np.asarray(pos, dtype=np.float64)
    n = pos.shape[0]
    
    # Wall distances
    wall = 0.5 - np.abs(pos)  # shape (n, 2)
    wall_vals = wall.flatten()  # 2n values
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist2 = (diff ** 2).sum(axis=-1)
    np.fill_diagonal(dist2, np.inf)
    dist = np.sqrt(dist2)
    np.fill_diagonal(dist, np.inf)
    
    pair_idx = np.triu_indices(n, k=1)
    pair_vals = dist[pair_idx] / 2.0  # N(N-1)/2 values
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    
    # Soft-min: - (1/beta) * log(sum(exp(-beta * vals)))
    # To avoid overflow/underflow, use log-sum-exp trick
    # min(vals) approx = m - (1/beta) * log(sum(exp(-beta * (vals - m))))
    m = all_vals.min()
    # Clip to prevent numerical issues in exp
    exp_args = -beta * (all_vals - m)
    exp_args = np.clip(exp_args, -700, 700)
    exp_vals = np.exp(exp_args)
    soft_min = m - (1.0 / beta) * np.log(exp_vals.sum())
    
    return -soft_min  # Return negative for minimization

def _soft_min_grad(pos, beta=1000.0):
    """
    Gradient of the soft-min radius surrogate.
    """
    pos = np.asarray(pos, dtype=np.float64)
    n = pos.shape[0]
    grad = np.zeros_like(pos)
    
    # Wall distances
    wall = 0.5 - np.abs(pos)
    wall_vals = wall.flatten()
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist2 = (diff ** 2).sum(axis=-1)
    np.fill_diagonal(dist2, np.inf)
    dist = np.sqrt(dist2)
    np.fill_diagonal(dist, np.inf)
    
    pair_idx = np.triu_indices(n, k=1)
    pair_vals = dist[pair_idx] / 2.0
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    m = all_vals.min()
    exp_args = -beta * (all_vals - m)
    exp_args = np.clip(exp_args, -700, 700)
    exp_vals = np.exp(exp_args)
    weights = exp_vals / exp_vals.sum()
    
    # Wall gradients
    # d(0.5 - |x|)/dx = -sign(x)
    for i in range(n):
        w0 = weights[2*i]
        w1 = weights[2*i+1]
        # d(-soft_min)/dx = - d(soft_min)/dx
        # d(soft_min)/dx = sum(w_k * d(val_k)/dx)
        # We are minimizing -soft_min, so we want gradient of -soft_min
        # Let's just compute gradient of soft_min and negate it later if needed, 
        # but scipy minimize expects gradient of the function we return.
        # We return -soft_min. So grad = - grad(soft_min).
        
        # d(wall_val_0)/dx_i = -sign(x_i)
        # d(wall_val_1)/dy_i = -sign(y_i)
        gx0 = -np.sign(pos[i, 0]) if pos[i, 0] != 0 else 0.0
        gy1 = -np.sign(pos[i, 1]) if pos[i, 1] != 0 else 0.0
        
        # Contribution to d(soft_min)/dx_i
        grad[i, 0] += w0 * gx0
        grad[i, 1] += w1 * gy1

    # Pair gradients
    # f = dist_ij / 2
    # df/dx_i = (x_i - x_j) / (2 * dist_ij)
    for k, (i, j) in enumerate(zip(pair_idx[0], pair_idx[1])):
        w = weights[2*n + k]
        d = diff[i, j]
        dist_ij = dist[i, j]
        if dist_ij > 1e-15:
            factor = 1.0 / (2.0 * dist_ij)
            # d(dist_ij/2)/dx_i = d / (2 * dist_ij)
            grad[i] += w * factor * d
            grad[j] -= w * factor * d  # d(dist_ij/2)/dx_j = -d / (2 * dist_ij)
            
    # Since we return -soft_min, the gradient is -grad(soft_min)
    return -grad

def _l_bfgs_polish(c, t0, deadline, beta=1000.0, maxiter=500):
    """
    Use L-BFGS-B to polish the configuration using the soft-min objective.
    """
    x0 = c.flatten()
    
    def func(x):
        pos = x.reshape(N, 2)
        return _soft_min_radius(pos, beta)
        
    def grad(x):
        pos = x.reshape(N, 2)
        return _soft_min_grad(pos, beta).flatten()
        
    # Bounds: keep circles inside the square
    bounds = [(-0.5, 0.5)] * (N * 2)
    
    try:
        res = minimize(func, x0, jac=grad, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
        c_new = res.x.reshape(N, 2)
        return c_new
    except Exception:
        return c

def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    
    # Multiple diverse initializations
    inits = [
        _init_grid_7x7(),
        _init_jitter8(rng),
        _init_hex(),
        _init_grid_8x8(),
        _init_rand(np.random.default_rng(42)),
        _init_rand(np.random.default_rng(77)),
    ]
    
    best = None
    best_r = -1.0
    
    # Phase 1: Grow radius from each starting configuration
    n_inits = len(inits)
    # Reserve 30s for refinement phases
    time_per_init = (DEADLINE - (time.time() - t0) - 30) / n_inits
    
    for i, start in enumerate(inits):
        remaining = DEADLINE - (time.time() - t0)
        if remaining < 15:
            break
        
        budget = min(time_per_init, remaining * 0.85 / (n_inits - i))
        budget = max(budget, 3.0)
        
        # Grow from 0.05 to 0.0715
        c, r = _grow(start, 0.05, 0.0715, 0.0004, 1500, 0.02, 0.002, t0, DEADLINE)
        
        if r > best_r:
            best_r = r
            best = c.copy()
        
        if time.time() - t0 > DEADLINE - 35:
            break
    
    if best is None:
        best = _init_grid_7x7()
        best_r = _score(best)
    
    # Phase 2: Langevin refinement on the best configuration
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 10:
        c, r = _langevin_refine(best, best_r, t0, DEADLINE, steps=8000, T0=0.0015, T1=0.00005, lr=0.003)
        if r > best_r:
            best_r = r
            best = c
    
    # Phase 3: L-BFGS Polish
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 5:
        # Use a high beta for high precision
        c_polished = _l_bfgs_polish(best, t0, DEADLINE, beta=1000.0, maxiter=200)
        r_polished = _score(c_polished)
        if r_polished > best_r:
            best_r = r_polished
            best = c_polished
            
        # If we have more time, try a second polish with even higher beta
        remaining = DEADLINE - (time.time() - t0)
        if remaining > 5 and best_r > 0.0713:
            c_polished2 = _l_bfgs_polish(best, t0, DEADLINE, beta=5000.0, maxiter=100)
            r_polished2 = _score(c_polished2)
            if r_polished2 > best_r:
                best_r = r_polished2
                best = c_polished2

    # Phase 4: One more Langevin pass if time allows (to escape L-BFGS local min)
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 8:
        c, r = _langevin_refine(best, best_r, t0, DEADLINE, steps=5000, T0=0.0008, T1=0.00002, lr=0.002)
        if r > best_r:
            best_r = r
            best = c
            
        # Final quick polish
        remaining = DEADLINE - (time.time() - t0)
        if remaining > 3:
            c_polished = _l_bfgs_polish(best, t0, DEADLINE, beta=2000.0, maxiter=50)
            r_polished = _score(c_polished)
            if r_polished > best_r:
                best_r = r_polished
                best = c_polished

    # Final check
    if not np.all(np.isfinite(best)) or best.shape != (N, 2):
        best = _init_grid_7x7()
    
    return best
