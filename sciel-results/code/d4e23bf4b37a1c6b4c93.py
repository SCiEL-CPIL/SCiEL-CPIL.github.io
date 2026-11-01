import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 14
D = 3
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def get_pairwise_sq_dists(points):
    """Returns flat array of squared pairwise distances for N points."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Compute the actual score: (max_dist / min_dist)^2 = max_sq / min_sq."""
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-15:
        return np.inf
    return np.max(sq_dists) / np.min(sq_dists)

def fibonacci_sphere(n_points):
    """Generate n_points on a unit sphere using Fibonacci lattice."""
    if n_points == 0:
        return np.empty((0, 3))
    points = np.zeros((n_points, 3))
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    for i in range(n_points):
        y = 1 - 2 * i / (n_points - 1) if n_points > 1 else 0
        radius = np.sqrt(1 - y * y)
        theta = 2 * np.pi * i / phi
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        points[i] = [x, y, z]
    return points

def objective_epigraph(x, tau=50.0):
    """
    Epigraph formulation: minimize t such that all pairwise sq dists are in [1/t, t].
    x = [points (N*D), t]
    We use a smooth penalty for constraint violations.
    """
    points = x[:N*D].reshape(N, D)
    t = x[N*D]
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Constraints: 1/t <= sq_dist <= t
    # Violations: max(0, 1/t - sq_dist) + max(0, sq_dist - t)
    # Smooth approximation using softplus-like: log(1+exp(k*violation))/k
    k = 50.0
    
    lower_viol = np.maximum(0, 1.0 / max(t, 1e-10) - sq_dists)
    upper_viol = np.maximum(0, sq_dists - t)
    
    # Smooth penalty
    penalty_lower = np.sum(np.log1p(np.exp(k * lower_viol))) / k
    penalty_upper = np.sum(np.log1p(np.exp(k * upper_viol))) / k
    
    # Also add a small term to push t down
    return t + 100.0 * (penalty_lower + penalty_upper)

def objective_log_ratio_smooth(x, tau=50.0):
    """Smooth log ratio objective."""
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    exp_max_terms = np.exp(tau * (sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    exp_min_terms = np.exp(-tau * (sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    return np.log(soft_max) - np.log(soft_min)

def optimize_from_start(init_points, method='L-BFGS-B', max_iter=500):
    """Optimize from a given starting point using epigraph formulation."""
    points = init_points.copy()
    # Initial t estimate
    sq_dists = get_pairwise_sq_dists(points)
    t_init = np.sqrt(np.max(sq_dists) / np.min(sq_dists))
    
    x0 = np.concatenate([points.flatten(), [t_init]])
    
    try:
        res = minimize(
            objective_epigraph,
            x0,
            method=method,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-14}
        )
        if res.fun < np.inf:
            pts = res.x[:N*D].reshape(N, D)
            # Center the points
            pts = pts - np.mean(pts, axis=0)
            return pts, raw_score(pts)
    except Exception:
        pass
    
    # Fallback: try log ratio
    try:
        res = minimize(
            objective_log_ratio_smooth,
            init_points.flatten(),
            method=method,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-14}
        )
        pts = res.x.reshape(N, D)
        pts = pts - np.mean(pts, axis=0)
        return pts, raw_score(pts)
    except Exception:
        pass
    
    return init_points, raw_score(init_points)

def local_refinement(points, n_iter=200):
    """Additional local refinement using coordinate descent style updates."""
    best_pts = points.copy()
    best_score = raw_score(points)
    
    for _ in range(n_iter):
        improved = False
        for i in range(N):
            # Try perturbing point i
            for axis in range(D):
                for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                    new_pts = best_pts.copy()
                    new_pts[i, axis] += delta
                    s = raw_score(new_pts)
                    if s < best_score * 0.999:
                        best_pts = new_pts
                        best_score = s
                        improved = True
        if not improved:
            break
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    
    best_points = None
    best_score = np.inf
    
    def time_remaining():
        return TIME_LIMIT - (time.time() - start_time)
    
    # Generate diverse initial points
    candidates = []
    
    # 1. Fibonacci sphere
    fib_pts = fibonacci_sphere(N)
    candidates.append(fib_pts)
    
    # 2. Random starts
    rng = np.random.default_rng(42)
    for _ in range(15):
        pts = rng.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * rng.uniform(1.5, 2.5)
        candidates.append(pts)
    
    # 3. Grid-based
    grid_pts = []
    for x in range(3):
        for y in range(3):
            for z in range(2):
                grid_pts.append([x, y, z])
    grid_pts = np.array(grid_pts[:N], dtype=float)
    grid_pts = grid_pts - np.mean(grid_pts, axis=0)
    candidates.append(grid_pts)
    
    # 4. Spherical shell with random angles
    for _ in range(5):
        pts = rng.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * 2.0
        candidates.append(pts)
    
    # Optimize each candidate
    for i, init_pts in enumerate(candidates):
        if time_remaining() < 10:
            break
        
        pts, score = optimize_from_start(init_pts, max_iter=300)
        if score < best_score:
            best_score = score
            best_points = pts.copy()
        
        # Local refinement if promising
        if score < best_score * 1.05 and time_remaining() > 15:
            pts_ref, score_ref = local_refinement(pts, n_iter=100)
            if score_ref < best_score:
                best_score = score_ref
                best_points = pts_ref.copy()
    
    # Final refinement on best point
    if best_points is not None and time_remaining() > 20:
        pts_final, score_final = local_refinement(best_points, n_iter=200)
        if score_final < best_score:
            best_score = score_final
            best_points = pts_final
    
    if best_points is None:
        # Fallback
        best_points = fibonacci_sphere(N)
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
