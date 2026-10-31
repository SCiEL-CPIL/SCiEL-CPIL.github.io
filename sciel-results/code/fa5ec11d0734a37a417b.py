"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist


def _score(points):
    """Compute (max_dist / min_dist)^2. Returns inf if invalid."""
    if points.shape != (N, D):
        return np.inf
    # Check distinctness
    if np.any(np.diff(np.sort(points[:, 0])) < 1e-12) and np.any(np.diff(np.sort(points[:, 1])) < 1e-12):
        # More robust check: all pairwise distances > 0
        pass
    dists = pdist(points, metric='euclidean')
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d < 1e-12:
        return np.inf
    return (max_d / min_d) ** 2


def _grid_with_deformation(params):
    """
    Generate 16 points on a 4x4 grid with sinusoidal deformation.
    params: [amp_x, freq_x, amp_y, freq_y, phase_x, phase_y, scale]
    """
    side = 4
    base_x, base_y = np.meshgrid(np.arange(side), np.arange(side), indexing='ij')
    # Flatten to 16 points
    bx = base_x.ravel()
    by = base_y.ravel()
    
    amp_x, freq_x, amp_y, freq_y, phase_x, phase_y, scale = params
    
    # Deform x coordinate based on y position
    x = bx + amp_x * np.sin(freq_x * by + phase_x)
    # Deform y coordinate based on x position
    y = by + amp_y * np.sin(freq_y * bx + phase_y)
    
    points = np.column_stack([x, y]) * scale
    return points


def _objective(params):
    """Objective function for differential evolution."""
    points = _grid_with_deformation(params)
    return _score(points)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Initial grid
    side = 4
    grid = np.array(list(np.ndindex(side, side)), dtype=float)
    best_points = grid.copy()
    best_score = _score(grid)
    
    # Define bounds for deformation parameters
    # amp in [-1, 1], freq in [0.1, 3.0], phase in [0, 2*pi], scale in [0.5, 2.0]
    bounds = [
        (-1.0, 1.0),   # amp_x
        (0.1, 3.0),    # freq_x
        (-1.0, 1.0),   # amp_y
        (0.1, 3.0),    # freq_y
        (0.0, 2 * np.pi),  # phase_x
        (0.0, 2 * np.pi),  # phase_y
        (0.5, 2.0),    # scale
    ]
    
    # Run differential evolution
    try:
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=True,
            callback=None,
            init='sobol',
            polish_kwargs={'maxiter': 1000}
        )
        
        if result.fun < best_score:
            best_points = _grid_with_deformation(result.x)
            best_score = result.fun
            
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 5:
        return best_points
    
    # If time remains, try a second round with different initialization
    remaining = deadline - time.time()
    if remaining > 10:
        try:
            # Try with different seed and bounds
            bounds2 = [
                (-0.5, 0.5),
                (0.2, 2.0),
                (-0.5, 0.5),
                (0.2, 2.0),
                (0.0, 2 * np.pi),
                (0.0, 2 * np.pi),
                (0.7, 1.5),
            ]
            result2 = differential_evolution(
                _objective,
                bounds2,
                maxiter=50,
                popsize=10,
                tol=1e-5,
                seed=123,
                polish=True,
                init='latinhypercube',
                polish_kwargs={'maxiter': 500}
            )
            
            if result2.fun < best_score:
                best_points = _grid_with_deformation(result2.x)
                best_score = result2.fun
                
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
