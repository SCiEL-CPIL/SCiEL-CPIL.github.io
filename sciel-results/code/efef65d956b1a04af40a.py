"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import time

N = 14

# EVOLVE-BLOCK-START

def _compute_min_triangle_area_vectorized(angles):
    """
    Compute the minimum triangle area for points on a unit circle defined by angles.
    Uses vectorization for speed.
    """
    # Sort angles to ensure convex ordering (though for area, order doesn't matter, 
    # but it helps with potential hull area calculation if needed, but here hull area is constant 
    # for points on a circle if we assume they are in convex position, which they are 
    # unless some are inside the hull of others. For points on a circle, ALL are on the hull 
    # unless they are not distinct or collinear. We assume distinct.)
    angles = np.sort(angles)
    
    # Compute coordinates
    x = np.cos(angles)
    y = np.sin(angles)
    
    # We need to check all triples (i, j, k) with i < j < k
    # To do this efficiently, we can use broadcasting or pre-compute differences.
    # N=14, so N^3 is ~2744 operations, which is small, but let's be efficient.
    
    # Area of triangle (i,j,k) = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
    
    # Create index arrays
    idx = np.arange(N)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    
    # Filter i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    # Calculate areas
    x_i, y_i = x[i_flat], y[i_flat]
    x_j, y_j = x[j_flat], y[j_flat]
    x_k, y_k = x[k_flat], y[k_flat]
    
    # Vectorized area calculation
    areas = 0.5 * np.abs((x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i))
    
    return np.min(areas)

def _objective(angles):
    """
    Objective function for minimization (Differential Evolution minimizes).
    We want to maximize min_triangle_area, so we return -min_triangle_area.
    """
    min_area = _compute_min_triangle_area_vectorized(angles)
    return -min_area

def construct():
    """
    Place 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    
    Approach: Represent points as inscribed in a unit circle, parameterized
    by central angles. Use Differential Evolution to optimize these angles.
    """
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer for safety
    
    # Bounds for each angle: [0, 2*pi]
    bounds = [(0, 2 * np.pi) for _ in range(N)]
    
    # Use Differential Evolution
    # strategy='best1bin' is standard
    # maxiter and popsize control the effort
    # We run in a loop, increasing effort if time permits, or just setting a high maxiter 
    # and letting it stop. However, DE in scipy doesn't have a built-in time limit.
    # We can estimate the number of iterations we can afford.
    # A rough estimate: 1 evaluation takes ~0.1ms? Let's assume 1ms to be safe.
    # 110s / 1ms = 110,000 evaluations.
    # With popsize=15 (default is 15*N? No, default popsize is 15), 
    # iterations = evals / popsize.
    # Let's set popsize=20.
    # We can run multiple DE instances or just one long one.
    # One long one is usually better for convergence.
    
    # To be safe and robust, we'll run DE with a high maxiter and check time in a callback?
    # Scipy's differential_evolution doesn't support callbacks for early stopping easily in older versions.
    # Instead, we can run it with a conservative maxiter and if time is left, run it again with the best result as start?
    # Or just set maxiter high and hope it finishes. 
    # Actually, for N=14, 14 dimensions, DE is quite fast.
    # Let's try maxiter=1000, popsize=20. 
    # 1000 * 20 = 20,000 evaluations.
    # If each eval is 0.5ms, that's 10 seconds.
    # We can afford more.
    
    # Let's use a time-aware loop:
    # 1. Run DE with a moderate maxiter.
    # 2. If time remains, run DE again starting from the best result found? 
    #    DE doesn't take a single start point, it uses a population.
    #    We can use the best result to seed a new population? 
    #    Or just run another independent DE?
    
    # Better strategy: Run DE with a large maxiter and rely on the vectorized speed.
    # Let's benchmark mentally: 2744 triangle areas per eval. NumPy handles this in microseconds.
    # Overhead of Python loop in DE is the bottleneck.
    
    # Let's try:
    # popsize = 15 (default)
    # maxiter = 500
    # This should take a few seconds.
    
    # We will run DE until time is up or maxiter is reached.
    # Since we can't easily stop DE mid-run, we will run it with a maxiter that we *estimate* fits in the time.
    # Let's be conservative: 200 iterations.
    # If time is left, we run it again with a different seed or strategy?
    
    # Actually, we can just run one instance with a high maxiter and see if it finishes.
    # If it's too slow, the evaluator will kill it. 
    # To be safe, let's use a loop of smaller DE runs.
    
    best_result = None
    best_min_area = -np.inf
    
    # Run several independent DE optimizations with different seeds
    # This increases the chance of finding the global optimum.
    
    seed = 42
    for run_idx in range(5):
        if time.time() - start_time > time_limit - 10:
            break
            
        try:
            # maxiter=100, popsize=15 -> 1500 evals.
            # Should be fast.
            result = differential_evolution(
                _objective, 
                bounds, 
                seed=seed + run_idx, 
                maxiter=100, 
                popsize=15,
                tol=1e-8,
                polish=True  # Polish the result using L-BFGS-B
            )
            
            if result.success:
                min_area = -result.fun
                if min_area > best_min_area:
                    best_min_area = min_area
                    best_result = result.x
            else:
                # Even if not "success" (convergence), we might have a good result
                min_area = -result.fun
                if min_area > best_min_area:
                    best_min_area = min_area
                    best_result = result.x
                    
        except Exception:
            pass

    if best_result is None:
        # Fallback to evenly spaced if optimization fails
        best_result = np.linspace(0, 2 * np.pi, N, endpoint=False)
        
    # Convert to points
    points = np.column_stack([np.cos(best_result), np.sin(best_result)])
    
    return points

# EVOLVE-BLOCK-END
