"""Improved packing for 200 equal circles in a unit square."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 200
DEADLINE_MARGIN = 5.0  # seconds to spare before the 150s kill
START_TIME = time.time()


def _remaining_time():
    return 150.0 - (time.time() - START_TIME) - DEADLINE_MARGIN


def _hexagonal_init(n):
    """Generate a jittered hexagonal lattice initialization inside [-0.5, 0.5]^2."""
    # Hexagonal packing: rows offset by half spacing
    # Approximate number of points per row
    # For n=200, try rows of ~14-15 points
    # Hex spacing: dx = s, dy = s * sqrt(3)/2
    # We want to fit in [0.5, 0.5] square (side 1)
    # Let's use a grid approach with hexagonal offset
    
    # Try different configurations
    best_init = None
    best_score = 0
    
    rng = np.random.RandomState(42)
    
    # Start with a square grid and add hexagonal perturbation
    m = int(np.ceil(np.sqrt(n)))
    
    # Try a few different m values
    for m_try in [13, 14, 15, 16]:
        # Create a grid of m_try x m_try
        xs = np.linspace(-0.5 + 0.5/m_try, 0.5 - 0.5/m_try, m_try)
        ys = np.linspace(-0.5 + 0.5/m_try, 0.5 - 0.5/m_try, m_try)
        
        # Generate hexagonal pattern
        points = []
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                # Offset every other row by half the x-spacing
                offset = (0.5 / m_try) * (i % 2)
                px = x + offset
                # Wrap around if needed
                if px > 0.5 - 0.5/m_try:
                    px -= 1.0 / m_try
                points.append([px, y])
        
        points = np.array(points[:n])
        
        # Add small random jitter
        jitter = 0.001 * rng.randn(n, 2)
        points = points + jitter
        
        # Clip to stay in bounds
        points[:, 0] = np.clip(points[:, 0], -0.49, 0.49)
        points[:, 1] = np.clip(points[:, 1], -0.49, 0.49)
        
        # Score this initialization
        if len(points) == n:
            d = pdist(points)
            min_d = d.min() if len(d) > 0 else 0
            boundary = min(0.5 - np.abs(points[:, 0]).min(), 0.5 - np.abs(points[:, 1]).min())
            score = min(min_d / 2, boundary)
            if score > best_score:
                best_score = score
                best_init = points.copy()
    
    if best_init is None:
        # Fallback to simple grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_init = grid[:n]
    
    return best_init


def _compute_radius(points):
    """Compute the maximum radius for given center positions."""
    if len(points) < 2:
        return 0.5 - np.max(np.abs(points))
    
    # Pairwise distances
    d = pdist(points)
    min_pair_dist = d.min() if len(d) > 0 else float('inf')
    
    # Boundary distances
    boundary = min(0.5 - np.min(points[:, 0]), 
                   0.5 - np.min(points[:, 1]),
                   np.max(points[:, 0]) - (-0.5),
                   np.max(points[:, 1]) - (-0.5))
    
    # Actually, we need min distance to each boundary for each point
    # Distance to left: point_x + 0.5
    # Distance to right: 0.5 - point_x
    # Distance to bottom: point_y + 0.5
    # Distance to top: 0.5 - point_y
    left = points[:, 0] + 0.5
    right = 0.5 - points[:, 0]
    bottom = points[:, 1] + 0.5
    top = 0.5 - points[:, 1]
    
    boundary_min = min(left.min(), right.min(), bottom.min(), top.min())
    
    return min(min_pair_dist / 2, boundary_min)


def _soft_min_radius(points, beta=100):
    """
    Smooth surrogate for the minimum radius using log-sum-exp style soft-min.
    We maximize min(min_pair_dist/2, boundary_min).
    Use a penalty-based approach: maximize a smooth approximation.
    """
    n = len(points)
    if n < 2:
        return 0.5 - np.max(np.abs(points))
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Soft-min of pairwise distances / 2
    # Use log-sum-exp trick for soft-min: -log(sum(exp(-x/beta))) * beta
    # For numerical stability
    min_pair = dist.min()
    exp_term = np.exp(-(dist - min_pair) / beta)
    exp_term[~np.isfinite(exp_term)] = 0
    soft_min_pair = min_pair - beta * np.log(exp_term.sum())
    
    # Boundary soft-min
    left = points[:, 0] + 0.5
    right = 0.5 - points[:, 0]
    bottom = points[:, 1] + 0.5
    top = 0.5 - points[:, 1]
    
    boundary_vals = np.concatenate([left, right, bottom, top])
    min_b = boundary_vals.min()
    exp_b = np.exp(-(boundary_vals - min_b) / beta)
    soft_min_b = min_b - beta * np.log(exp_b.sum())
    
    # Combine: soft-min of the two
    vals = np.array([soft_min_pair / 2, soft_min_b])
    min_v = vals.min()
    exp_v = np.exp(-(vals - min_v) / beta)
    result = min_v - beta * np.log(exp_v.sum())
    
    return result


def _soft_min_radius_grad(points, beta=100):
    """Gradient of the soft-min radius approximation."""
    n = len(points)
    if n < 2:
        return np.zeros_like(points)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-30)
    np.fill_diagonal(dist, np.inf)
    
    # Soft-min of pairwise distances
    min_pair = dist.min()
    exp_term = np.exp(-(dist - min_pair) / beta)
    exp_term[~np.isfinite(exp_term)] = 0
    sum_exp = exp_term.sum()
    soft_min_pair = min_pair - beta * np.log(sum_exp)
    
    # Gradient of soft_min_pair w.r.t. points
    # d(soft_min_pair)/d(points[i]) = -beta * d(log(sum_exp))/d(points[i])
    # d(log(sum_exp))/d(points[i]) = (1/sum_exp) * sum_j d(exp_term[i,j])/d(points[i])
    # d(exp_term[i,j])/d(points[i]) = exp_term[i,j] * (-1/beta) * d(dist[i,j])/d(points[i])
    # d(dist[i,j])/d(points[i]) = (points[i] - points[j]) / dist[i,j]
    
    grad_pair = np.zeros_like(points)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if not np.isfinite(dist[i, j]):
                continue
            d_dist = (points[i] - points[j]) / dist[i, j]
            grad_pair[i] += exp_term[i, j] * (-1 / beta) * d_dist / sum_exp
    grad_pair *= -beta
    
    # Boundary soft-min
    left = points[:, 0] + 0.5
    right = 0.5 - points[:, 0]
    bottom = points[:, 1] + 0.5
    top = 0.5 - points[:, 1]
    
    boundary_vals = np.concatenate([left, right, bottom, top])
    min_b = boundary_vals.min()
    exp_b = np.exp(-(boundary_vals - min_b) / beta)
    sum_exp_b = exp_b.sum()
    soft_min_b = min_b - beta * np.log(sum_exp_b)
    
    # Gradient of soft_min_b
    grad_b = np.zeros_like(points)
    # left: d(left_i)/d(points[i,0]) = 1
    # right: d(right_i)/d(points[i,0]) = -1
    # bottom: d(bottom_i)/d(points[i,1]) = 1
    # top: d(top_i)/d(points[i,1]) = -1
    
    for i in range(n):
        # left contribution
        idx = i
        grad_b[i, 0] += exp_b[idx] * (-1 / beta) * 1 / sum_exp_b
        # right contribution
        idx = n + i
        grad_b[i, 0] += exp_b[idx] * (-1 / beta) * (-1) / sum_exp_b
        # bottom contribution
        idx = 2 * n + i
        grad_b[i, 1] += exp_b[idx] * (-1 / beta) * 1 / sum_exp_b
        # top contribution
        idx = 3 * n + i
        grad_b[i, 1] += exp_b[idx] * (-1 / beta) * (-1) / sum_exp_b
    grad_b *= -beta
    
    # Combine gradients
    vals = np.array([soft_min_pair / 2, soft_min_b])
    min_v = vals.min()
    exp_v = np.exp(-(vals - min_v) / beta)
    sum_v = exp_v.sum()
    
    # Gradient of combined soft-min
    # d/dx min_v - beta*log(sum_v) where vals depend on x
    # = d(min_v)/dx - beta * d(log(sum_v))/dx
    # But min_v is piecewise constant, so gradient comes from the log term
    # d(log(sum_v))/dx = (1/sum_v) * sum_k exp_v[k] * d(vals[k])/dx
    
    grad_combined = np.zeros_like(points)
    # vals[0] = soft_min_pair / 2, d(vals[0])/dx = grad_pair / 2
    grad_combined += exp_v[0] * (grad_pair / 2) / sum_v
    # vals[1] = soft_min_b, d(vals[1])/dx = grad_b
    grad_combined += exp_v[1] * grad_b / sum_v
    grad_combined *= -beta
    
    return grad_combined


def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Initial guess
    points = _hexagonal_init(N)
    current_radius = _compute_radius(points)
    best_points = points.copy()
    best_radius = current_radius
    
    # Optimization using L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Flatten points for scipy
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 2)
        # We want to maximize the radius, so minimize negative radius
        r = _soft_min_radius(pts, beta=50)
        return -r
    
    def gradient(x):
        pts = x.reshape(-1, 2)
        g = _soft_min_radius_grad(pts, beta=50)
        return g.flatten()
    
    # Run optimization with time check
    result = None
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            jac=gradient,
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
        )
    except Exception:
        pass
    
    if result is not None and result.success:
        new_points = result.x.reshape(-1, 2)
        new_radius = _compute_radius(new_points)
        if new_radius > best_radius:
            best_points = new_points
            best_radius = new_radius
    
    # Check time and do a second round with tighter beta if time permits
    if _remaining_time() > 20:
        # Second round with smaller beta for more accurate result
        x0 = best_points.flatten()
        
        def objective2(x):
            pts = x.reshape(-1, 2)
            r = _soft_min_radius(pts, beta=20)
            return -r
        
        def gradient2(x):
            pts = x.reshape(-1, 2)
            g = _soft_min_radius_grad(pts, beta=20)
            return g.flatten()
        
        try:
            result2 = minimize(
                objective2,
                x0,
                method='L-BFGS-B',
                jac=gradient2,
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 1000}
            )
        except Exception:
            result2 = None
        
        if result2 is not None and result2.success:
            new_points = result2.x.reshape(-1, 2)
            new_radius = _compute_radius(new_points)
            if new_radius > best_radius:
                best_points = new_points
                best_radius = new_radius
    
    # Final polish: local search with small perturbations if time permits
    if _remaining_time() > 10:
        # Try a few random perturbations and keep the best
        rng = np.random.RandomState(123)
        for _ in range(50):
            if _remaining_time() < 5:
                break
            perturbed = best_points + 0.001 * rng.randn(N, 2)
            perturbed[:, 0] = np.clip(perturbed[:, 0], -0.49, 0.49)
            perturbed[:, 1] = np.clip(perturbed[:, 1], -0.49, 0.49)
            r = _compute_radius(perturbed)
            if r > best_radius:
                best_points = perturbed
                best_radius = r
    
    return best_points
