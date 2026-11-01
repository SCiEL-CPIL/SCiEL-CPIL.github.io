"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 140  # safety margin before 150s limit


def _min_dist_and_wall(c):
    """Compute the minimum distance between any pair of centres and to walls.
    Returns (min_pair_dist, min_wall_dist)."""
    n = len(c)
    # Pairwise distances
    d = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Wall distances: for each coordinate, distance to nearest wall
    # Cube is [-0.5, 0.5]^3, so wall distance for coordinate k is 0.5 - |c_ik|
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min_pair, min_wall


def _radius(c):
    """Compute the common radius for a given set of centres."""
    min_pair, min_wall = _min_dist_and_wall(c)
    return min(min_pair / 2.0, min_wall)


def _pair_dist_matrix(c):
    """(N,N) pairwise distance matrix, diagonal = inf."""
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(d, np.inf)
    return d


def _softmin_radius(c, t):
    """Smooth approximation of the true radius via log-sum-exp over all
    constraints (pair/2 and wall)."""
    d = _pair_dist_matrix(c)
    pair = d / 2.0
    wall = 0.5 - np.abs(c)
    vals = np.concatenate([pair.ravel(), wall.ravel()])
    vals = vals[np.isfinite(vals)]
    m = np.min(vals)
    # Avoid overflow/underflow in exp
    s = np.sum(np.exp(-(vals - m) / t))
    return m - t * np.log(s)


def _smooth_obj(x, t):
    """Objective to MINIMIZE: negative smooth radius plus boundary penalty."""
    c = x.reshape(N, 3)
    r = _softmin_radius(c, t)
    pen = np.sum(np.maximum(0.0, np.abs(c) - 0.5 + 1e-6) ** 2)
    return -r + 1e3 * pen


def _generate_initial_grid():
    """Generate a 5x5x2 grid of 50 points inside the unit cube."""
    # 5 points in x, 5 in y, 2 in z
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.array([-0.25, 0.25])
    
    points = []
    for zi in z:
        for yi in y:
            for xi in x:
                points.append([xi, yi, zi])
    return np.array(points)


def _random_initial():
    """Generate random initial positions inside the cube."""
    return np.random.uniform(-0.45, 0.45, size=(N, 3))


def _repulsion_anneal(c, deadline, steps=3000):
    """Vectorized repulsion + wall-attraction gradient descent."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    
    for it in range(steps):
        if time.time() > deadline:
            break
        d = _pair_dist_matrix(c)
        inv = 1.0 / np.maximum(d, 1e-6) ** 2
        diff = c[:, None, :] - c[None, :, :]
        F = np.sum(diff * inv[:, :, None], axis=1)
        
        wall_force = np.zeros_like(c)
        for k in range(3):
            sgn = np.sign(c[:, k])
            sgn[sgn == 0] = 1.0
            wall_force[:, k] = sgn * (0.5 - np.abs(c[:, k]))
        
        frac = it / max(steps - 1, 1)
        rep_scale = 1e-3 * (1.0 - 0.7 * frac)
        wall_scale = 1e-3 * (0.3 + 1.0 * frac)
        lr = 1e-2 * (1.0 - 0.5 * frac)
        c = c + lr * (rep_scale * F + wall_scale * wall_force)
        c = np.clip(c, -0.49, 0.49)
    
    return c


def _smooth_polish(c, deadline):
    """Polish with smooth softmin max-min via L-BFGS-B, annealing temperature."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _radius(c)
    x0 = c.flatten()
    bounds = [(-0.49, 0.49)] * (N * 3)
    
    for t in (0.05, 0.02, 0.008, 0.003, 0.001, 0.0005, 0.0001):
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _smooth_obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                args=(t,),
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-11,
                         'maxls': 50},
            )
            c = np.clip(res.x.reshape(N, 3), -0.49, 0.49)
            r = _radius(c)
            if r > best_r:
                best_r = r
                best = c.copy()
            x0 = c.flatten()
        except Exception:
            pass
    return best, best_r


def _active_set_polish(c, deadline, iters=500):
    """Iteratively identify the most violated constraint (closest pair or wall)
    and perform a targeted local search to increase the minimum clearance."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _radius(c)
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        # Find the most violated constraint
        d = _pair_dist_matrix(c)
        min_pair_idx = np.unravel_index(np.argmin(d), d.shape)
        min_pair_dist = d[min_pair_idx]
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        min_wall_dist = wall_dists[min_wall_idx]
        
        if min_pair_dist / 2.0 < min_wall_dist:
            # Pair constraint is limiting
            i, j = min_pair_idx
            if i == j:
                continue
            # Push i and j apart along their connecting vector
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-12:
                continue
            vec /= norm
            # Move each by half the needed amount to increase distance
            move = 0.001 * (1.0 - it / iters)
            c[i] += move * vec
            c[j] -= move * vec
        else:
            # Wall constraint is limiting
            i, k = min_wall_idx
            # Push away from the wall
            sign = np.sign(c[i, k])
            if sign == 0:
                sign = 1.0
            move = 0.001 * (1.0 - it / iters)
            c[i, k] += sign * move
        
        c = np.clip(c, -0.49, 0.49)
        r = _radius(c)
        if r > best_r:
            best_r = r
            best = c.copy()
    
    return best, best_r


def _metropolis_mc(c, deadline, iters=5000):
    """Metropolis Monte Carlo: random displacements with acceptance based on radius.
    Allows occasional worsening moves to escape local minima."""
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _radius(c)
    best_c = c.copy()
    rng = np.random.default_rng(123)
    
    # Adaptive temperature that cools over time
    T0 = 0.002
    T_min = 0.0001
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        frac = it / max(iters - 1, 1)
        T = T0 * (T_min / T0) ** frac
        
        # Random displacement
        trial = c.copy()
        idx = rng.integers(0, N)
        disp = rng.normal(0, 0.01, size=3)
        trial[idx] += disp
        trial = np.clip(trial, -0.499, 0.499)
        
        r_trial = _radius(trial)
        r_current = _radius(c)
        
        if r_trial >= r_current:
            c = trial
            if r_trial > best_r:
                best_r = r_trial
                best_c = c.copy()
        else:
            # Accept with probability exp(-(r_current - r_trial) / T)
            prob = np.exp(-(r_current - r_trial) / T)
            if rng.random() < prob:
                c = trial
    
    return best_c, best_r


def construct():
    start_time = time.time()
    
    # Start with a good initial configuration
    best_c = _generate_initial_grid()
    best_r = _radius(best_c)
    
    # Try a few different initializations
    inits = [
        _generate_initial_grid(),
    ]
    # Add a couple random starts
    for _ in range(2):
        inits.append(_random_initial())
    
    for c0 in inits:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        r0 = _radius(c0)
        if r0 > best_r:
            best_c = c0.copy()
            best_r = r0
        
        # Phase 1: Repulsion annealing to spread points
        if time.time() - start_time < DEADLINE - 10:
            c_rep = _repulsion_anneal(c0, deadline=start_time + DEADLINE - 10)
            r_rep = _radius(c_rep)
            if r_rep > best_r:
                best_c = c_rep.copy()
                best_r = r_rep
        
        # Phase 2: Smooth polish with L-BFGS-B
        if time.time() - start_time < DEADLINE - 10:
            c_smooth, r_smooth = _smooth_polish(best_c, deadline=start_time + DEADLINE - 10)
            if r_smooth > best_r:
                best_c = c_smooth.copy()
                best_r = r_smooth
        
        # Phase 3: Active set polish
        if time.time() - start_time < DEADLINE - 5:
            c_active, r_active = _active_set_polish(best_c, deadline=start_time + DEADLINE - 5, iters=200)
            if r_active > best_r:
                best_c = c_active.copy()
                best_r = r_active
    
    # Final Metropolis MC to escape local minima
    if time.time() - start_time < DEADLINE - 5:
        try:
            c_mc, r_mc = _metropolis_mc(best_c, deadline=start_time + DEADLINE - 5, iters=3000)
            if r_mc > best_r:
                best_c = c_mc.copy()
                best_r = r_mc
        except Exception:
            pass
    
    # Final polish
    if time.time() - start_time < DEADLINE - 2:
        try:
            c_final, r_final = _smooth_polish(best_c, deadline=start_time + DEADLINE - 2)
            if r_final > best_r:
                best_c = c_final.copy()
                best_r = r_final
        except Exception:
            pass
    
    # Ensure the result is valid
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
