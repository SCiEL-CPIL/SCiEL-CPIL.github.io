# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def _soft_min_max_ratio(points: np.ndarray, alpha: float = 50.0) -> float:
    """
    Compute a smooth surrogate for -log(d_max/d_min) = log(d_min) - log(d_max).
    We minimize this to maximize d_min/d_max.
    Uses log-sum-exp for soft max and soft min.
    """
    d = pdist(points)
    if np.any(d <= 0):
        return 1e10  # penalize coincident points heavily
    
    # soft_max = (1/alpha) * log(sum(exp(alpha * d)))
    # soft_min = -(1/alpha) * log(sum(exp(-alpha * d)))
    # We want to minimize log(soft_max) - log(soft_min)
    
    # For numerical stability, use the log-sum-exp trick
    max_d = np.max(d)
    min_d = np.min(d)
    
    # soft_max approximation
    # log(sum(exp(alpha*d))) = alpha*max_d + log(sum(exp(alpha*(d - max_d))))
    shifted_max = alpha * (d - max_d)
    log_sum_exp_max = alpha * max_d + np.log(np.sum(np.exp(shifted_max)))
    soft_max = np.exp(log_sum_exp_max) / alpha
    
    # soft_min approximation  
    # log(sum(exp(-alpha*d))) = -alpha*min_d + log(sum(exp(-alpha*(d - min_d))))
    shifted_min = -alpha * (d - min_d)
    log_sum_exp_min = -alpha * min_d + np.log(np.sum(np.exp(shifted_min)))
    soft_min = np.exp(log_sum_exp_min) / alpha
    
    # We want to minimize log(soft_max) - log(soft_min) = log(soft_max/soft_min)
    if soft_min <= 0:
        return 1e10
    return np.log(soft_max) - np.log(soft_min)


def _compute_ratio(points: np.ndarray) -> float:
    """Compute R = (d_min / d_max)^2 for the given points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2


def _make_initial_configurations(n: int = 16) -> list:
    """Generate several good initial configurations."""
    configs = []
    
    # Config 1: Regular 16-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points1 = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(points1)
    
    # Config 2: 1 center + 6 on hexagon + 9 on outer circle
    center = np.array([[0.0, 0.0]])
    hex_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    hex_radius = 0.5
    hex_points = hex_radius * np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    outer_angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    outer_radius = 1.0
    outer_points = outer_radius * np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    points2 = np.vstack([center, hex_points, outer_points])
    configs.append(points2)
    
    # Config 3: 1 center + 6 on hexagon + 9 on slightly different outer
    outer_angles3 = np.linspace(np.pi/9, 2 * np.pi + np.pi/9, 9, endpoint=False)
    outer_points3 = 1.0 * np.column_stack([np.cos(outer_angles3), np.sin(outer_angles3)])
    points3 = np.vstack([center, hex_points, outer_points3])
    configs.append(points3)
    
    # Config 4: Grid-like arrangement
    # 4x4 grid
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid = np.array([[x, y] for x in xs for y in ys])
    configs.append(grid)
    
    # Config 5: 1 center + 7 on inner ring + 8 on outer ring
    inner_angles5 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    inner_points5 = 0.55 * np.column_stack([np.cos(inner_angles5), np.sin(inner_angles5)])
    outer_angles5 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_points5 = 1.0 * np.column_stack([np.cos(outer_angles5), np.sin(outer_angles5)])
    points5 = np.vstack([center, inner_points5, outer_points5])
    configs.append(points5)
    
    # Config 6: Random with seed
    np.random.seed(42)
    points6 = np.random.randn(n, 2)
    # Normalize
    points6 = points6 / np.max(np.abs(points6))
    configs.append(points6)
    
    # Config 7: 2 centers + 14 on outer
    center7a = np.array([[-0.3, 0.0]])
    center7b = np.array([[0.3, 0.0]])
    outer_angles7 = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    outer_points7 = 1.0 * np.column_stack([np.cos(outer_angles7), np.sin(outer_angles7)])
    points7 = np.vstack([center7a, center7b, outer_points7])
    configs.append(points7)
    
    return configs


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the points.
    """
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin out of 630s
    
    n = 16
    best_score = -1
    best_points = None
    
    initial_configs = _make_initial_configurations(n)
    
    # Try multiple alphas for softness
    alphas = [20.0, 50.0, 100.0, 200.0]
    
    for config in initial_configs:
        if time.time() > deadline:
            break
        
        for alpha in alphas:
            if time.time() > deadline:
                break
            
            def objective(x):
                pts = x.reshape(n, 2)
                return _soft_min_max_ratio(pts, alpha=alpha)
            
            x0 = config.flatten()
            
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.success or result.fun < 1e8:
                    pts = result.x.reshape(n, 2)
                    score = _compute_ratio(pts)
                    
                    if score > best_score:
                        best_score = score
                        best_points = pts.copy()
                        
            except Exception:
                continue
        
        # Also try a few random perturbations of this config
        for _ in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(np.random.randint(0, 1000000))
            perturbed = config + 0.1 * np.random.randn(n, 2)
            
            for alpha in [50.0, 100.0]:
                if time.time() > deadline:
                    break
                
                def objective(x):
                    pts = x.reshape(n, 2)
                    return _soft_min_max_ratio(pts, alpha=alpha)
                
                x0 = perturbed.flatten()
                
                try:
                    result = minimize(
                        objective,
                        x0,
                        method='L-BFGS-B',
                        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    
                    if result.success or result.fun < 1e8:
                        pts = result.x.reshape(n, 2)
                        score = _compute_ratio(pts)
                        
                        if score > best_score:
                            best_score = score
                            best_points = pts.copy()
                            
                except Exception:
                    continue
    
    # Final polish: use high-alpha soft objective on best solution
    if best_points is not None and time.time() < deadline - 30:
        for alpha in [200.0, 500.0, 1000.0]:
            if time.time() > deadline - 10:
                break
            
            def objective(x):
                pts = x.reshape(n, 2)
                return _soft_min_max_ratio(pts, alpha=alpha)
            
            x0 = best_points.flatten()
            
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                if result.success or result.fun < 1e8:
                    pts = result.x.reshape(n, 2)
                    score = _compute_ratio(pts)
                    
                    if score > best_score:
                        best_score = score
                        best_points = pts.copy()
                        
            except Exception:
                continue
    
    # Ensure valid output
    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points


# EVOLVE-BLOCK-END
