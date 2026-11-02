"""Optimized placement of 11 points in a unit-side equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

# Precompute all 165 triplets of indices
_indices = list(itertools.combinations(range(N), 3))
_idx_arr = np.array(_indices)

def _bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) array of shape (N, 2) to Cartesian points.
    u corresponds to weight of V2 (top), v corresponds to weight of V1 (right).
    V0 is (0,0).
    x = v * 1.0 + (1 - u - v) * 0.5
    y = (1 - u - v) * H
    """
    u = np.clip(bary[:, 0], 0.0, None)
    v = np.clip(bary[:, 1], 0.0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u = np.where(mask, u / s, u)
        v = np.where(mask, v / s, v)
        
    pts = np.empty((N, 2))
    pts[:, 0] = v * 1.0 + (1.0 - u - v) * 0.5
    pts[:, 1] = (1.0 - u - v) * H
    return pts

def _compute_areas(points):
    """
    Compute all 165 triangle areas from 11 points using vectorized operations.
    """
    i = _idx_arr[:, 0]
    j = _idx_arr[:, 1]
    k = _idx_arr[:, 2]
    
    xi, yi = points[i, 0], points[i, 1]
    xj, yj = points[j, 0], points[j, 1]
    xk, yk = points[k, 0], points[k, 1]
    
    # Area = 0.5 * |xi(yj-yk) + xj(yk-yi) + xk(yi-yj)|
    det = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
    areas = 0.5 * np.abs(det)
    return areas

def _min_area(points):
    """Return the minimum triangle area."""
    return float(np.min(_compute_areas(points)))

def _soft_min_area(points, temperature=0.001):
    """
    Smooth approximation of min area using log-sum-exp.
    soft_min(a) = -T * log(sum(exp(-a_i / T)))
    """
    areas = _compute_areas(points)
    areas = np.maximum(areas, 1e-15)
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -temperature * log_sum

def _objective(params, temperature=0.001):
    """
    Objective to minimize: negative soft-min area.
    params: 22 values representing barycentric coordinates (u, v) for each point.
    """
    uv = params.reshape(N, 2)
    pts = _bary_to_cart(uv)
    soft_min = _soft_min_area(pts, temperature)
    return -soft_min

def _initial_guess_grid():
    """Generate a structured initial guess based on a grid pattern."""
    bary = np.empty((N, 2))
    # Approximate hexagonal packing or grid
    # (u, v) where u is "up" weight, v is "right" weight
    # Center is (1/3, 1/3)
    positions = [
        (1/3, 1/3),
        (0.1, 0.1), (0.4, 0.1), (0.7, 0.1),
        (0.2, 0.3), (0.5, 0.3), (0.8, 0.3),
        (0.15, 0.5), (0.45, 0.5), (0.75, 0.5),
        (0.3, 0.7), (0.6, 0.7),
    ]
    for m, (u, v) in enumerate(positions):
        if u + v > 1.0:
            s = 1.0 / (u + v)
            u *= s
            v *= s
        bary[m, 0] = u
        bary[m, 1] = v
    return bary

def _random_valid_barycentric(rng):
    """Generate random valid barycentric coordinates."""
    bary = np.empty((N, 2))
    for m in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                bary[m, 0] = u
                bary[m, 1] = v
                break
    return bary

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    # Bounds for barycentric coordinates: each in [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_points = None
    best_score = -np.inf
    
    def _try_optimize(initial_params, temperature, maxiter, method='L-BFGS-B'):
        """Try to optimize from initial params, return (points, score) or (None, -inf)."""
        try:
            result = minimize(
                _objective,
                initial_params,
                args=(temperature,),
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < -np.inf + 1: # Check if result is finite
                pts = _bary_to_cart(result.x.reshape(N, 2))
                # Verify validity
                if np.all(pts[:, 0] >= -1e-8) and np.all(pts[:, 0] <= 1.0 + 1e-8) and \
                   np.all(pts[:, 1] >= -1e-8) and np.all(pts[:, 1] <= H + 1e-8):
                    # Check triangle containment more carefully
                    x = pts[:, 0]
                    y = pts[:, 1]
                    if np.all(y >= -1e-8) and np.all(y <= SQRT3 * x + 1e-8) and \
                       np.all(y <= -SQRT3 * x + SQRT3 + 1e-8):
                        score = _min_area(pts)
                        return pts, score
        except Exception:
            pass
        return None, -np.inf

    # Initial guesses
    initial_guesses = []
    
    # 1. Grid
    initial_guesses.append(_initial_guess_grid())
    
    # 2. Random
    for _ in range(10):
        initial_guesses.append(_random_valid_barycentric(rng))
        
    # 3. Symmetric patterns (e.g. vertices scaled, centroid, etc.)
    # Just add a few more random ones with different seeds if time permits
    
    # Optimization Loop
    # We will iterate through guesses, refining the temperature and using previous best as new start
    # This mimics Simulated Annealing / Multi-start
    
    # Start with high temperature for exploration, lower for exploitation
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    # We will run a sequence of optimizations.
    # Each step takes the best point found so far, or a random start, and optimizes.
    
    # To manage time, we will allocate budget per optimization call.
    # L-BFGS-B is fast. Let's try to do as many as possible.
    
    # Strategy:
    # 1. Run a few random starts with high T to find good basins.
    # 2. Refine the best ones with lower T.
    
    # Phase 1: Exploration
    for i in range(min(15, len(initial_guesses))):
        if time.time() > deadline - 10.0: # Leave 10s for final refinement
            break
            
        start_params = initial_guesses[i].flatten()
        # Use a moderately high temperature
        temp = 0.05
        # Limit iterations for exploration
        max_iter = 200
        
        pts, score = _try_optimize(start_params, temp, max_iter)
        
        if pts is not None and score > best_score:
            best_score = score
            best_points = pts
            
    # Phase 2: Refinement
    # Use the best points found so far as starting points for lower temperature runs
    # Also try random perturbations of the best points
    
    refinement_starts = []
    if best_points is not None:
        # Convert best points back to barycentric for restarting
        # x = v + 0.5(1-u-v) => v - u = 2x - 1
        # y = H(1-u-v) => u+v = 1 - y/H
        # v = (2x - 1 + 1 - y/H)/2 = x - y/(2H)
        # u = 1 - x - y/(2H)
        bary_best = np.empty((N, 2))
        for m in range(N):
            x = best_points[m, 0]
            y = best_points[m, 1]
            bary_best[m, 1] = x - y / (2.0 * H)
            bary_best[m, 0] = 1.0 - x - y / (2.0 * H)
        refinement_starts.append(bary_best)
        
        # Add perturbed versions
        for _ in range(5):
            perturbed = bary_best + rng.normal(0, 0.05, (N, 2))
            refinement_starts.append(perturbed)
            
    # Continue with more random starts if time permits, but prioritize refinement
    for temp in temperatures[1:]: # Skip 0.05, start from 0.02
        if time.time() > deadline - 5.0:
            break
            
        for start_bary in refinement_starts:
            if time.time() > deadline - 5.0:
                break
                
            start_params = start_bary.flatten()
            max_iter = 500
            pts, score = _try_optimize(start_params, temp, max_iter)
            
            if pts is not None and score > best_score:
                best_score = score
                best_points = pts
                # Update refinement starts with the new best
                bary_new = np.empty((N, 2))
                for m in range(N):
                    x = best_points[m, 0]
                    y = best_points[m, 1]
                    bary_new[m, 1] = x - y / (2.0 * H)
                    bary_new[m, 0] = 1.0 - x - y / (2.0 * H)
                refinement_starts = [bary_new] + [bary_new + rng.normal(0, 0.02, (N, 2)) for _ in range(3)]
                
    # Final Polish: Very low temperature, many iterations on the best point
    if best_points is not None and time.time() < deadline - 2.0:
        bary_best = np.empty((N, 2))
        for m in range(N):
            x = best_points[m, 0]
            y = best_points[m, 1]
            bary_best[m, 1] = x - y / (2.0 * H)
            bary_best[m, 0] = 1.0 - x - y / (2.0 * H)
            
        # Try a few very low temperatures
        for temp in [0.0001, 0.00001]:
            if time.time() > deadline - 1.0:
                break
            pts, score = _try_optimize(bary_best.flatten(), temp, 1000)
            if pts is not None and score > best_score:
                best_score = score
                best_points = pts

    # Fallback if no valid points found (should not happen)
    if best_points is None:
        best_points = _bary_to_cart(_initial_guess_grid())
        
    return best_points

