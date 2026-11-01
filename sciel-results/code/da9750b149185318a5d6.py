"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 115.0  # seconds; leave margin before 120s limit


def _score(c):
    """Compute the radius for a configuration c of shape (N,3)."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    # Check inside cube [-0.5, 0.5]^3
    if np.any(c < -0.5) or np.any(c > 0.5):
        return -1.0
    # Pairwise distances
    # Use vectorized computation
    # For N=50, this is 1225 pairs, manageable
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    # Distance to walls
    wall_dist = np.min(0.5 - np.abs(c), axis=1)
    min_wall = np.min(wall_dist)
    r = 0.5 * min(min_pair, min_wall)
    return r


def _initial_random(seed):
    """Generate a random initial configuration."""
    rng = np.random.default_rng(seed)
    # Place spheres in a grid-like pattern with some randomness
    # Start with a simple packing: divide cube into layers
    # For 50 spheres, try a 5x5x2 grid with jitter
    m = 5
    k = 2
    positions = []
    for z in range(k):
        for y in range(m):
            for x in range(m):
                if len(positions) >= N:
                    break
                # Center of each cell
                cx = (x + 0.5) / m - 0.5
                cy = (y + 0.5) / m - 0.5
                cz = (z + 0.5) / k - 0.5
                # Add small random jitter
                jitter = 0.05
                cx += rng.uniform(-jitter, jitter)
                cy += rng.uniform(-jitter, jitter)
                cz += rng.uniform(-jitter, jitter)
                # Clamp to cube
                cx = np.clip(cx, -0.45, 0.45)
                cy = np.clip(cy, -0.45, 0.45)
                cz = np.clip(cz, -0.45, 0.45)
                positions.append([cx, cy, cz])
    pos = np.array(positions[:N])
    return pos


def _local_refine(c, seed, max_iter=2000):
    """Refine a configuration using gradient-free optimization."""
    rng = np.random.default_rng(seed)
    
    def neg_score(x):
        c = x.reshape(N, 3)
        # Penalize going outside the cube
        if np.any(c < -0.5 + 1e-9) or np.any(c > 0.5 - 1e-9):
            return 1.0
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)
        wall_dist = np.min(0.5 - np.abs(c), axis=1)
        min_wall = np.min(wall_dist)
        r = 0.5 * min(min_pair, min_wall)
        return -r
    
    best_c = c.copy()
    best_score = _score(best_c)
    
    # Try multiple local optimization runs
    for trial in range(3):
        x0 = c.flatten().copy()
        # Add small perturbation for diversity
        if trial > 0:
            x0 += rng.normal(0, 0.01, size=x0.shape)
            x0 = np.clip(x0.reshape(N, 3), -0.49, 0.49).flatten()
        
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
            new_c = res.x.reshape(N, 3)
            new_score = _score(new_c)
            if new_score > best_score:
                best_c = new_c
                best_score = new_score
        except Exception:
            pass
    
    return best_c, best_score


def _hill_climbing(c, seed, max_steps=500):
    """Simple hill climbing with random perturbations."""
    rng = np.random.default_rng(seed)
    best_c = c.copy()
    best_score = _score(best_c)
    
    step_size = 0.02
    for step in range(max_steps):
        # Random perturbation
        new_c = best_c + rng.normal(0, step_size, size=(N, 3))
        new_c = np.clip(new_c, -0.49, 0.49)
        new_score = _score(new_c)
        
        if new_score > best_score:
            best_c = new_c
            best_score = new_score
            step_size = max(0.001, step_size * 0.95)
        else:
            step_size = max(0.001, step_size * 0.9)
    
    return best_c, best_score


def construct():
    start_time = time.time()
    best_c = None
    best_score = -1.0
    
    # Try multiple random initializations
    for seed in range(20):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:
            break
        
        c0 = _initial_random(seed)
        score0 = _score(c0)
        
        # Local refinement
        c_refined, score_refined = _local_refine(c0, seed, max_iter=1000)
        
        if score_refined > best_score:
            best_score = score_refined
            best_c = c_refined.copy()
        
        # Hill climbing
        c_hc, score_hc = _hill_climbing(c_refined, seed + 1000, max_steps=300)
        
        if score_hc > best_score:
            best_score = score_hc
            best_c = c_hc.copy()
    
    # Final check
    if best_c is None or not np.all(np.isfinite(best_c)):
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]
    
    return best_c
# EVOLVE-BLOCK-END
