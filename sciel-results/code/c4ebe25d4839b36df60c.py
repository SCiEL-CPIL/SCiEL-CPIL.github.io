"""Optimized sphere packing: 50 equal spheres in unit cube."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 140.0  # Safety margin before 150s kill

def _score(c):
    """Compute the radius score for a configuration."""
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Minimum pairwise distance
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Distance to walls
    wall_dists = np.min(0.5 - np.abs(c), axis=1)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)

def _objective(c, n=N):
    """Negative of the radius score (to be minimized)."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall_dists = np.min(0.5 - np.abs(c), axis=1)
    min_wall = np.min(wall_dists)
    r = min(min_pair / 2.0, min_wall)
    return -r

def _gradient(c, n=N):
    """Gradient of the objective (negative radius)."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair = np.min(dists)
    wall_dists = np.min(0.5 - np.abs(c), axis=1)
    min_wall = np.min(wall_dists)
    
    r = min(min_pair / 2.0, min_wall)
    
    grad = np.zeros_like(c)
    
    # If the binding constraint is pairwise distance
    if min_pair / 2.0 <= min_wall:
        # Find the pair(s) achieving min_pair
        mask = np.isclose(dists, min_pair, atol=1e-12)
        np.fill_diagonal(mask, False)
        for i in range(n):
            for j in range(i+1, n):
                if mask[i, j]:
                    d = c[i] - c[j]
                    norm = np.linalg.norm(d)
                    if norm > 1e-15:
                        grad[i] += d / norm
                        grad[j] -= d / norm
        grad *= -0.5  # derivative of -min_pair/2
    else:
        # Binding constraint is wall distance
        min_idx = np.argmin(wall_dists)
        c_min = c[min_idx]
        for k in range(3):
            if c_min[k] >= 0:
                grad[min_idx, k] = 1.0  # derivative of -(0.5 - c) = 1
            else:
                grad[min_idx, k] = -1.0  # derivative of -(0.5 + c) = -1
        grad *= -1  # We're minimizing -r, so gradient of -r
    
    return grad

def _fcc_init(n=N):
    """Generate FCC lattice points and select n closest to center."""
    # FCC lattice: points at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) + integer shifts
    # Scale to fit in [-0.5, 0.5]^3
    # Use a scaled FCC lattice
    s = 0.25  # scale factor
    points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                # FCC basis vectors
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i, j, k]) * s + np.array([dx, dy, dz]) * s
                    if -0.45 <= p[0] <= 0.45 and -0.45 <= p[1] <= 0.45 and -0.45 <= p[2] <= 0.45:
                        points.append(p)
    
    points = np.array(points)
    # Remove duplicates
    unique_points = []
    for p in points:
        if not any(np.allclose(p, q, atol=1e-6) for q in unique_points):
            unique_points.append(p)
    points = np.array(unique_points)
    
    # Select n points closest to center
    if len(points) >= n:
        center_dists = np.sum(points**2, axis=1)
        idx = np.argsort(center_dists)[:n]
        return points[idx]
    else:
        # Pad with random points
        extra = np.random.uniform(-0.4, 0.4, size=(n - len(points), 3))
        return np.vstack([points, extra])

def _random_init(n=N):
    """Random initialization within the cube."""
    return np.random.uniform(-0.4, 0.4, size=(n, 3))

def _refine(c, time_budget):
    """Refine a configuration using L-BFGS-B with time budget."""
    start_time = time.time()
    best_c = c.copy()
    best_score = _score(c)
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.49, 0.49)] * (n * 3)
    
    while time.time() - start_time < time_budget:
        # Reshape for scipy
        c_flat = c.flatten()
        result = minimize(
            _objective, c_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        new_c = result.x.reshape(n, 3)
        new_score = _score(new_c)
        
        if new_score > best_score:
            best_c = new_c
            best_score = new_score
        
        # Check if we've made progress
        if not result.success:
            break
        
        # If improvement is tiny, try a perturbation to escape local minimum
        if _score(c) >= best_score - 1e-10:
            # Perturb slightly
            c = best_c + np.random.normal(0, 0.01, size=(n, 3))
            c = np.clip(c, -0.49, 0.49)
        else:
            c = new_c
        
        if time.time() - start_time > time_budget * 0.8:
            break
    
    return best_c, best_score

def construct():
    """Main entry point: find best packing of 50 spheres in unit cube."""
    start_time = time.time()
    
    best_c = None
    best_score = -1.0
    
    # Try multiple initializations
    inits = []
    
    # FCC initialization
    inits.append(_fcc_init())
    
    # Random initializations
    for _ in range(3):
        inits.append(_random_init())
    
    # Also try a simple grid for baseline
    m = int(np.ceil(N ** (1.0/3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    if len(grid) >= N:
        inits.append(grid[:N])
    
    for init_c in inits:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Allocate remaining time for this init
        remaining = DEADLINE - 10 - (time.time() - start_time)
        time_budget = max(5.0, remaining / 2.0)
        
        refined_c, score = _refine(init_c.copy(), time_budget)
        
        if score > best_score:
            best_score = score
            best_c = refined_c
    
    # Final polish: try a few more refinement steps on the best solution
    if best_c is not None and time.time() - start_time < DEADLINE - 5:
        final_time_budget = DEADLINE - 5 - (time.time() - start_time)
        if final_time_budget > 2:
            polished_c, polished_score = _refine(best_c.copy(), final_time_budget)
            if polished_score > best_score:
                best_c = polished_c
                best_score = polished_score
    
    # Ensure valid output
    if best_c is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0/3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_c = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    # Clip to ensure inside cube
    best_c = np.clip(best_c, -0.499999, 0.499999)
    
    return best_c
