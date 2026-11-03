"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def _pairwise_min_dist(centers):
    """Compute the minimum pairwise distance among centers."""
    n = len(centers)
    min_d = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_d:
                min_d = d
    return min_d


def _radius(centers):
    """Compute the maximum common radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    # Compute min pairwise distance
    min_d = _pairwise_min_dist(centers)
    # Compute min distance to boundary
    min_boundary = np.inf
    for c in centers:
        min_boundary = min(min_boundary, 0.5 - abs(c[0]), 0.5 - abs(c[1]))
    return min(min_d / 2.0, min_boundary)


def _soft_min_radius(centers, beta=100.0):
    """
    Smooth surrogate for the radius objective using log-sum-exp.
    We want to maximize r = min(min_pair_dist/2, min_boundary_dist).
    Equivalently, minimize -r.
    
    For smooth optimization, we can use a soft-min approach.
    """
    n = len(centers)
    
    # Pairwise distances / 2
    pair_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j]) / 2.0
            pair_vals.append(d)
    pair_vals = np.array(pair_vals)
    
    # Boundary distances
    bound_vals = []
    for c in centers:
        bound_vals.append(0.5 - abs(c[0]))
        bound_vals.append(0.5 - abs(c[1]))
    bound_vals = np.array(bound_vals)
    
    # All values
    all_vals = np.concatenate([pair_vals, bound_vals])
    
    if len(all_vals) == 0:
        return 0.0
    
    # Soft min using log-sum-exp: -log-sum-exp(-beta * x) / beta
    # This approximates min(x) as beta -> inf
    max_val = np.max(all_vals)
    if not np.isfinite(max_val):
        return 0.0
    
    # Use stable log-sum-exp
    shifted = -beta * (all_vals - max_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_val - log_sum / beta
    
    return soft_min


def _negative_soft_min_radius(centers, beta=100.0):
    """Negative of soft min radius for minimization."""
    return -_soft_min_radius(centers, beta)


def _gradient_soft_min(centers, beta=100.0):
    """
    Compute gradient of the soft min radius objective.
    This is complex; we'll use numerical gradient or a simplified approach.
    """
    # For simplicity, use finite differences in the optimizer
    return None


def _initial_grid():
    """Create an initial grid configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with grid
    best_centers = _initial_grid()
    best_radius = _radius(best_centers)
    
    # Try differential evolution for global search
    try:
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        def de_objective(flat):
            centers = flat.reshape(N, 2)
            # Penalize if out of bounds
            if np.any(np.abs(centers) > 0.5):
                return -100.0
            return -_radius(centers)
        
        result = differential_evolution(
            de_objective,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=True
        )
        
        if result.success or result.fun < de_objective(best_centers.flatten()):
            new_centers = result.x.reshape(N, 2)
            if np.all(np.isfinite(new_centers)) and np.all(np.abs(new_centers) <= 0.5):
                r = _radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = new_centers
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 20.0:
            return best_centers
        
    except Exception:
        pass
    
    # Local refinement with L-BFGS-B using smooth surrogate
    try:
        # Use a moderate beta for smoothness
        beta = 50.0
        
        def objective(flat):
            centers = flat.reshape(N, 2)
            # Add penalty for going out of bounds
            penalty = 0.0
            for c in centers:
                if c[0] < -0.5:
                    penalty += 100 * (-0.5 - c[0]) ** 2
                elif c[0] > 0.5:
                    penalty += 100 * (c[0] - 0.5) ** 2
                if c[1] < -0.5:
                    penalty += 100 * (-0.5 - c[1]) ** 2
                elif c[1] > 0.5:
                    penalty += 100 * (c[1] - 0.5) ** 2
            return -_soft_min_radius(centers, beta) + penalty
        
        result = minimize(
            objective,
            best_centers.flatten(),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.fun < -_soft_min_radius(best_centers, beta):
            new_centers = result.x.reshape(N, 2)
            if np.all(np.isfinite(new_centers)) and np.all(np.abs(new_centers) <= 0.5 + 1e-10):
                # Clip to bounds
                new_centers = np.clip(new_centers, -0.5, 0.5)
                r = _radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = new_centers
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 15.0:
            return best_centers
        
    except Exception:
        pass
    
    # Final high-precision polish pass
    try:
        # Increase beta for more accurate approximation
        beta = 200.0
        
        def objective_polish(flat):
            centers = flat.reshape(N, 2)
            penalty = 0.0
            for c in centers:
                if c[0] < -0.5:
                    penalty += 1000 * (-0.5 - c[0]) ** 2
                elif c[0] > 0.5:
                    penalty += 1000 * (c[0] - 0.5) ** 2
                if c[1] < -0.5:
                    penalty += 1000 * (-0.5 - c[1]) ** 2
                elif c[1] > 0.5:
                    penalty += 1000 * (c[1] - 0.5) ** 2
            return -_soft_min_radius(centers, beta) + penalty
        
        result = minimize(
            objective_polish,
            best_centers.flatten(),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        
        new_centers = result.x.reshape(N, 2)
        if np.all(np.isfinite(new_centers)) and np.all(np.abs(new_centers) <= 0.5 + 1e-10):
            new_centers = np.clip(new_centers, -0.5, 0.5)
            r = _radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        
    except Exception:
        pass
    
    # Final validation and return
    if not np.all(np.isfinite(best_centers)) or np.any(np.abs(best_centers) > 0.5 + 1e-12):
        # Fallback to grid
        return _initial_grid()
    
    return np.clip(best_centers, -0.5, 0.5)
# EVOLVE-BLOCK-END
