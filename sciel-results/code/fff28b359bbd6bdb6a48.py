import numpy as np
from scipy.optimize import minimize
import time

N = 200

def _compute_radius(centres):
    """Compute the maximum common radius for the given centres."""
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    wall_dists = np.minimum(np.minimum(0.5 - centres[:, 0], 0.5 + centres[:, 0]),
                            np.minimum(0.5 - centres[:, 1], 0.5 + centres[:, 1]))
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _soft_min(values, beta=50.0):
    """Smooth approximation of min using log-sum-exp."""
    m = np.min(values)
    return m - np.log(np.sum(np.exp(beta * (values - m)))) / beta

def _energy_objective(centres, beta=50.0, gamma=2.0):
    """
    Smooth energy: maximize soft-min of (pair distances / 2, wall distances).
    Lower energy = better packing.
    """
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Pair distances / 2 (these are the "radii" from pair constraints)
    pair_r = dist / 2.0
    
    # Wall distances
    wall_dists = np.minimum(np.minimum(0.5 - centres[:, 0], 0.5 + centres[:, 0]),
                            np.minimum(0.5 - centres[:, 1], 0.5 + centres[:, 1]))
    
    # Combine all candidate radii
    all_r = np.concatenate([pair_r.ravel(), wall_dists.ravel()])
    all_r = all_r[~np.isinf(all_r)]
    
    # Soft-min gives a smooth surrogate for min
    return -_soft_min(all_r, beta)

def _initial_hexagonal_layout(n):
    """Create an initial hexagonal-like layout for n circles."""
    best_centres = None
    best_radius = 0
    
    for rows in range(int(np.sqrt(n)) - 2, int(np.sqrt(n)) + 6):
        if rows <= 0:
            continue
        cols_per_row = int(np.ceil(n / rows))
        
        centres = []
        for i in range(rows):
            for j in range(cols_per_row):
                if len(centres) >= n:
                    break
                x = (j + 0.5) / cols_per_row - 0.5
                y = (i + 0.5) / rows - 0.5
                if i % 2 == 1:
                    x += 0.5 / cols_per_row
                centres.append([x, y])
        
        if len(centres) == n:
            centres = np.array(centres)
            r = _compute_radius(centres)
            if r > best_radius:
                best_radius = r
                best_centres = centres
    
    if best_centres is None:
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centres = grid[:n]
    
    return best_centres

def _polish_centres(centres, max_iter=50, ftol=1e-14):
    """High-precision polish using smooth energy minimization."""
    x0 = centres.flatten()
    bounds = [(-0.499, 0.499)] * (2 * N)
    
    def obj(x):
        c = x.reshape(-1, 2)
        return _energy_objective(c, beta=80.0)
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-10})
        if res.success:
            new_centres = res.x.reshape(-1, 2)
            new_centres = np.clip(new_centres, -0.5, 0.5)
            if _compute_radius(new_centres) >= _compute_radius(centres) - 1e-12:
                return new_centres
    except Exception:
        pass
    
    return centres

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    # Initial best solution
    best_centres = _initial_hexagonal_layout(N)
    best_radius = _compute_radius(best_centres)
    
    np.random.seed(42)
    
    # Phase 1: Smooth energy minimization with multiple restarts
    # Use soft-min surrogate to get differentiable objective
    for i in range(15):
        if time.time() > deadline - 30:
            break
        
        if i == 0:
            x0 = best_centres.flatten()
        elif i == 1:
            # Perturb best
            perturbation = np.random.randn(2 * N) * 0.02
            x0 = best_centres.flatten() + perturbation
            x0 = np.clip(x0, -0.5, 0.5)
        else:
            # Larger perturbations for diversity
            scale = 0.03 + 0.01 * i
            perturbation = np.random.randn(2 * N) * scale
            x0 = best_centres.flatten() + perturbation
            x0 = np.clip(x0, -0.5, 0.5)
        
        # Use smooth energy objective
        bounds = [(-0.499, 0.499)] * (2 * N)
        
        def obj_energy(x):
            c = x.reshape(-1, 2)
            return _energy_objective(c, beta=60.0)
        
        try:
            res = minimize(obj_energy, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-8})
            if res.success:
                new_centres = res.x.reshape(-1, 2)
                new_centres = np.clip(new_centres, -0.5, 0.5)
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
        except Exception:
            pass
    
    # Phase 2: Additional spread-based optimization for better global structure
    for i in range(5):
        if time.time() > deadline - 20:
            break
        
        if i == 0:
            x0 = best_centres.flatten()
        else:
            perturbation = np.random.randn(2 * N) * 0.015
            x0 = best_centres.flatten() + perturbation
            x0 = np.clip(x0, -0.5, 0.5)
        
        def obj_spread(x):
            c = x.reshape(-1, 2)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, np.inf)
            min_dists = np.min(dist, axis=1)
            return -np.sum(min_dists**2)
        
        bounds = [(-0.499, 0.499)] * (2 * N)
        
        try:
            res = minimize(obj_spread, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-12})
            if res.success:
                x0 = res.x
        except Exception:
            pass
        
        # Refine with energy
        if time.time() > deadline - 15:
            continue
        
        def obj_energy2(x):
            c = x.reshape(-1, 2)
            return _energy_objective(c, beta=70.0)
        
        try:
            res = minimize(obj_energy2, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-13})
            if res.success:
                new_centres = res.x.reshape(-1, 2)
                new_centres = np.clip(new_centres, -0.5, 0.5)
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
        except Exception:
            pass
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 5:
        best_centres = _polish_centres(best_centres, max_iter=80, ftol=1e-15)
        best_radius = _compute_radius(best_centres)
    
    # Final validation and clipping
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres
