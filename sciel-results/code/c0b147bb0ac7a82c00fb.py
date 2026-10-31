"""Improved program: sheared sinusoidal grid + differential evolution."""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 16
D = 2

# Deadline guard: kill after 150s wall-clock, safety margin 5s
_DEADLINE = 150.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _score(points):
    """Compute (max pairwise dist / min pairwise dist)^2."""
    n = len(points)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d < 1e-12:
        return 1e10
    return (max_d / min_d) ** 2


def _params_to_points(params):
    """
    Convert parameter vector to 16 points in 2D.
    
    Parameters (12 total):
      - base_spacing (1): overall scale
      - row_offsets (4): vertical offset per row
      - col_mod_amp (1): amplitude of sinusoidal horizontal modulation
      - col_mod_freq (1): frequency of sinusoidal horizontal modulation
      - shear (1): overall shear factor
      - phase (1): phase of sinusoidal modulation
      - vertical_scale (1): vertical scaling
      - extra_x (1): extra horizontal shift
    
    Layout: 4 rows x 4 cols
    """
    base_spacing = params[0]
    row_offsets = params[1:5]
    col_mod_amp = params[5]
    col_mod_freq = params[6]
    shear = params[7]
    phase = params[8]
    vertical_scale = params[9]
    extra_x = params[10]
    # params[11] reserved
    
    points = np.zeros((16, 2))
    for i in range(4):  # row
        for j in range(4):  # col
            x = j * base_spacing + col_mod_amp * np.sin(col_mod_freq * i + phase) * j + shear * i * base_spacing + extra_x
            y = i * base_spacing * vertical_scale + row_offsets[i]
            points[4 * i + j] = [x, y]
    
    return points


def _objective(params):
    """Objective function for differential evolution."""
    if _time_left() < 5.0:
        return 1e10
    points = _params_to_points(params)
    # Check distinctness
    if len(np.unique(points, axis=0)) < N:
        return 1e10
    return _score(points)


def _grid_baseline():
    """Return a simple 4x4 grid as fallback."""
    pts = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    return pts


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_points = _grid_baseline()
    best_score = _score(best_points)
    
    # Define bounds for parameters
    bounds = [
        (0.5, 3.0),    # base_spacing
        (-1.0, 1.0),   # row_offset_0
        (-1.0, 1.0),   # row_offset_1
        (-1.0, 1.0),   # row_offset_2
        (-1.0, 1.0),   # row_offset_3
        (-1.0, 1.0),   # col_mod_amp
        (0.5, 3.0),    # col_mod_freq
        (-1.0, 1.0),   # shear
        (0.0, 2 * np.pi),  # phase
        (0.5, 2.0),    # vertical_scale
        (-1.0, 1.0),   # extra_x
        (0.0, 1.0),    # reserved
    ]
    
    # Try differential evolution with limited iterations
    try:
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            seed=42,
            polish=False,
            init='sobol',
        )
        
        if result.success or result.fun < best_score:
            candidate_points = _params_to_points(result.x)
            candidate_score = _score(candidate_points)
            if candidate_score < best_score:
                best_points = candidate_points
                best_score = candidate_score
    except Exception:
        pass
    
    # If we still have time, try a second run with different seed
    if _time_left() > 30.0:
        try:
            result2 = differential_evolution(
                _objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-4,
                seed=123,
                polish=False,
                init='latinhypercube',
            )
            
            if result2.success or result2.fun < best_score:
                candidate_points = _params_to_points(result2.x)
                candidate_score = _score(candidate_points)
                if candidate_score < best_score:
                    best_points = candidate_points
                    best_score = candidate_score
        except Exception:
            pass
    
    # Final polish with local optimization if time permits
    if _time_left() > 15.0:
        try:
            from scipy.optimize import minimize
            # Use the best params from last DE result
            last_params = result2.x if 'result2' in dir() else result.x
            res = minimize(
                _objective,
                last_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6},
            )
            if res.fun < best_score:
                candidate_points = _params_to_points(res.x)
                candidate_score = _score(candidate_points)
                if candidate_score < best_score:
                    best_points = candidate_points
                    best_score = candidate_score
        except Exception:
            pass
    
    # Ensure points are distinct
    if len(np.unique(best_points, axis=0)) < N:
        best_points = _grid_baseline()
    
    return best_points


# Pre-compute for faster import
if __name__ == "__main__":
    pts = construct()
    print(_score(pts))
