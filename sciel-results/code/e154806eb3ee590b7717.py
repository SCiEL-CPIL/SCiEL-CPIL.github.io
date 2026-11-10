# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

def _compute_score(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points using vectorized operations."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Vectorized triangle area computation
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n), indexing='ij'
    )
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    x1 = points[i_idx, 0]
    y1 = points[i_idx, 1]
    x2 = points[j_idx, 0]
    y2 = points[j_idx, 1]
    x3 = points[k_idx, 0]
    y3 = points[k_idx, 1]
    
    areas = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
    A_min = np.min(areas)
    
    return A_min / A_hull

def _fourier_points(params: np.ndarray) -> np.ndarray:
    """
    Generate 13 points from Fourier mode parameters.
    params: array of shape (n_modes, 3) where each row is [amplitude, phase, frequency]
    Points are generated as:
      theta_k = 2*pi*k/13
      r_k = 1.0 + sum_m params[m,0] * cos(params[m,1] + params[m,2]*theta_k)
      x_k = r_k * cos(theta_k), y_k = r_k * sin(theta_k)
    """
    n = 13
    theta = 2 * np.pi * np.arange(n) / n
    radii = np.ones(n)
    for m in range(params.shape[0]):
        amp, phase, freq = params[m]
        radii += amp * np.cos(phase + freq * theta)
    
    # Ensure convexity: radii must be positive and not too varied
    if np.any(radii <= 0.3):
        radii = np.clip(radii, 0.3, 2.0)
    
    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    return np.column_stack([x, y])

def _fourier_local_search(params: np.ndarray, deadline: float, max_iters: int = 50000) -> np.ndarray:
    """
    Local search in Fourier parameter space.
    """
    n_modes = params.shape[0]
    params = params.copy()
    current_score = _compute_score(_fourier_points(params))
    
    # Step sizes for each parameter
    step_sizes = np.ones((n_modes, 3)) * 0.01
    # Amplitude steps smaller, phase steps medium, frequency steps larger
    step_sizes[:, 0] = 0.005
    step_sizes[:, 1] = 0.05
    step_sizes[:, 2] = 0.1
    
    iter_count = 0
    stuck_count = 0
    
    while iter_count < max_iters:
        if time.time() > deadline:
            break
        
        improved = False
        for m in range(n_modes):
            for p in range(3):
                for direction in [-1, 1]:
                    for step_mult in [1.0, 0.5, 0.1, 2.0]:
                        new_params = params.copy()
                        new_params[m, p] += direction * step_sizes[m, p] * step_mult
                        
                        # Constraints
                        new_params[m, 0] = np.clip(new_params[m, 0], -0.3, 0.3)
                        new_params[m, 2] = np.clip(new_params[m, 2], 1.0, 6.0)
                        
                        new_points = _fourier_points(new_params)
                        if not np.all(np.isfinite(new_points)):
                            continue
                        
                        new_score = _compute_score(new_points)
                        
                        if new_score > current_score:
                            params = new_params
                            current_score = new_score
                            improved = True
                            step_sizes[m, p] *= 0.9
                            break
                    else:
                        step_sizes[m, p] *= 0.85
                        if step_sizes[m, p] < 1e-8:
                            step_sizes[m, p] = 0.01
        
        if not improved:
            stuck_count += 1
            if stuck_count > 50:
                # Perturb randomly
                idx = np.random.randint(n_modes)
                params[idx] += np.random.normal(0, 0.01, 3)
                params[idx, 0] = np.clip(params[idx, 0], -0.3, 0.3)
                params[idx, 2] = np.clip(params[idx, 2], 1.0, 6.0)
                current_score = _compute_score(_fourier_points(params))
                stuck_count = 0
        else:
            stuck_count = 0
        
        iter_count += 1
    
    return params

def _refine_with_direct_coords(points: np.ndarray, deadline: float, max_iters: int = 20000) -> np.ndarray:
    """
    Fine-tune points directly in 2D coordinate space.
    """
    n = points.shape[0]
    points = points.copy()
    current_score = _compute_score(points)
    
    step_sizes = np.ones((n, 2)) * 0.005
    
    directions = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]])
    
    iter_count = 0
    stuck_count = 0
    
    while iter_count < max_iters:
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            for d in directions:
                for step_mult in [1.0, 0.5, 0.1, 2.0]:
                    new_points = points.copy()
                    new_points[i] = points[i] + step_sizes[i] * step_mult * d
                    
                    if not np.all(np.isfinite(new_points)):
                        continue
                    
                    new_score = _compute_score(new_points)
                    
                    if new_score > current_score:
                        points = new_points
                        current_score = new_score
                        improved = True
                        step_sizes[i] *= 0.9
                        break
                else:
                    step_sizes[i] *= 0.85
                    if step_sizes[i, 0] < 1e-9:
                        step_sizes[i] = 0.005
        
        if not improved:
            stuck_count += 1
            if stuck_count > 100:
                idx = np.random.randint(n)
                points[idx] += np.random.normal(0, 0.005, 2)
                current_score = _compute_score(points)
                stuck_count = 0
        else:
            stuck_count = 0
        
        iter_count += 1
    
    return points

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area.
    Uses Fourier parametrization for structural symmetry, then refines in direct coordinates.
    """
    start_time = time.time()
    deadline = start_time + 590  # 590 second budget, leave 40s safety margin
    
    n = 13
    best_points = None
    best_score = -1.0
    
    # Start with regular 13-gon as baseline
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    regular_poly = np.column_stack([np.cos(angles), np.sin(angles)])
    score_regular = _compute_score(regular_poly)
    best_score = score_regular
    best_points = regular_poly.copy()
    
    # Try different numbers of Fourier modes
    n_modes_list = [1, 2, 3, 4, 5]
    rng = np.random.default_rng(seed=12345)
    
    for n_modes in n_modes_list:
        if time.time() > deadline - 60:
            break
        
        # Try multiple random initializations for each mode count
        for trial in range(3):
            if time.time() > deadline - 30:
                break
            
            # Initialize parameters
            params = np.zeros((n_modes, 3))
            for m in range(n_modes):
                params[m, 0] = rng.uniform(-0.1, 0.1)  # amplitude
                params[m, 1] = rng.uniform(0, 2 * np.pi)  # phase
                params[m, 2] = rng.choice([1, 2, 3, 4, 5, 6])  # frequency
            
            # Evaluate initial
            points = _fourier_points(params)
            score = _compute_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
            
            # Fourier local search
            if time.time() > deadline - 20:
                break
            params_opt = _fourier_local_search(params, deadline - 10, max_iters=20000)
            points_opt = _fourier_points(params_opt)
            score_opt = _compute_score(points_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_points = points_opt.copy()
            
            # Direct coordinate refinement
            if time.time() > deadline - 10:
                break
            points_refined = _refine_with_direct_coords(points_opt, deadline - 5, max_iters=10000)
            score_refined = _compute_score(points_refined)
            if score_refined > best_score:
                best_score = score_refined
                best_points = points_refined.copy()
    
    # Final safety: ensure valid output
    if best_points is None:
        return regular_poly
    
    if not np.all(np.isfinite(best_points)):
        return regular_poly
    
    try:
        hull = ConvexHull(best_points)
    except Exception:
        return regular_poly
    
    return best_points

# EVOLVE-BLOCK-END
