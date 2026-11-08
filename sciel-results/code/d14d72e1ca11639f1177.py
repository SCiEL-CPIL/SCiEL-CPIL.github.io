"""Rewritten program for Heilbronn convex regions n=14."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _hull_area(p):
    """Return area of the convex hull of point set p."""
    try:
        hull = ConvexHull(p)
        return hull.volume
    except Exception:
        return 1e-10

def _min_triangle_area(p):
    """Return minimum triangle area (2*area) over all triples."""
    n = len(p)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = abs(np.cross(p[j] - p[i], p[k] - p[i]))
                if a < min_a:
                    min_a = a
    return min_a

def _all_triangle_areas(p):
    """Return array of 2*areas for all triples."""
    n = len(p)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = abs(np.cross(p[j] - p[i], p[k] - p[i]))
                areas.append(a)
    return np.array(areas)

def _score(p):
    """Return score: min_triangle_area / hull_area."""
    ha = _hull_area(p)
    if ha < 1e-10:
        return 0.0
    return _min_triangle_area(p) / ha

def _soft_min(areas, T):
    """Smooth approximation of min(areas) using log-sum-exp."""
    areas_clipped = np.maximum(areas, 1e-12)
    return -T * np.log(np.sum(np.exp(-areas_clipped / T)))

def _objective_and_grad(x, T=0.05):
    """
    Objective: minimize -soft_min(areas) / hull_area approximation.
    Since hull area is not smooth, we use a simpler approach:
    Minimize -soft_min(areas) with points constrained to unit circle.
    The score is scale-invariant, so we can fix the hull area by
    scaling at the end. But for optimization, we work with points
    in a bounded region and then normalize.
    
    Actually, let's use a different approach: maximize soft_min(areas)
    subject to hull_area = 1. Since hull area is hard to handle,
    let's instead use the fact that score is scale-invariant.
    We can optimize the shape and then scale to unit hull area.
    
    For the objective, let's minimize -soft_min(areas) with points
    in [-1, 1]^2. Then at the end, we scale to unit hull area.
    """
    p = x.reshape(-1, 2)
    areas = _all_triangle_areas(p)
    sm = _soft_min(areas, T)
    
    # Gradient of -soft_min with respect to points
    # soft_min = -T * log(sum(exp(-a_i/T)))
    # d(soft_min)/d(x) = -T * (1/sum(exp(-a_i/T))) * sum(exp(-a_i/T) * (-1/T) * da_i/dx)
    #                   = (sum(exp(-a_i/T) * da_i/dx)) / sum(exp(-a_i/T))
    # So d(-soft_min)/d(x) = -(sum(exp(-a_i/T) * da_i/dx)) / sum(exp(-a_i/T))
    
    weights = np.exp(-areas / T)
    weights /= np.sum(weights)
    
    # Gradient of each area with respect to points
    # area(i,j,k) = |cross(p_j - p_i, p_k - p_i)|
    # For simplicity, we'll use numerical gradient or approximate
    
    # Let's use numerical gradient for the full objective
    # This is expensive but more robust
    eps = 1e-7
    grad = np.zeros_like(x)
    for idx in range(len(x)):
        x_plus = x.copy()
        x_plus[idx] += eps
        p_plus = x_plus.reshape(-1, 2)
        areas_plus = _all_triangle_areas(p_plus)
        sm_plus = _soft_min(areas_plus, T)
        
        x_minus = x.copy()
        x_minus[idx] -= eps
        p_minus = x_minus.reshape(-1, 2)
        areas_minus = _all_triangle_areas(p_minus)
        sm_minus = _soft_min(areas_minus, T)
        
        grad[idx] = -(sm_plus - sm_minus) / (2 * eps)
    
    return -sm, grad

def _regular_polygon(n, r=1.0):
    """Generate a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])

def _two_ring(n, r1, r2, offset=0.0):
    """Generate points on two concentric rings."""
    n1 = n // 2
    n2 = n - n1
    angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False) + offset
    angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + offset + np.pi / n2
    p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    p2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    return np.vstack([p1, p2])

def _optimize_from_initial(p0, T=0.05, maxiter=500):
    """Optimize from initial points using L-BFGS-B."""
    x0 = p0.flatten()
    
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    try:
        result = minimize(
            _objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        p_opt = result.x.reshape(-1, 2)
        return p_opt
    except Exception:
        return p0

def _normalize_to_unit_hull(p):
    """Scale points so that hull area is 1."""
    ha = _hull_area(p)
    if ha < 1e-10:
        return p
    scale = 1.0 / np.sqrt(ha)
    return p * scale

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin for 150s limit
    
    best_score = 0.0
    best_points = None
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Regular polygon
    initial_configs.append(_regular_polygon(N, r=1.0))
    
    # Two-ring configurations with various parameters
    for r1 in [0.3, 0.5, 0.7]:
        for r2 in [0.8, 1.0, 1.2]:
            for offset in [0.0, np.pi / 14]:
                initial_configs.append(_two_ring(N, r1, r2, offset))
    
    # Random configurations
    rng = np.random.default_rng(42)
    for _ in range(5):
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        radii = rng.uniform(0.5, 1.0, N)
        p = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initial_configs.append(p)
    
    T = 0.05
    
    for i, p0 in enumerate(initial_configs):
        if time.time() > deadline - 10.0:
            break
        
        # Optimize
        p_opt = _optimize_from_initial(p0, T=T, maxiter=300)
        
        # Normalize to unit hull area
        p_norm = _normalize_to_unit_hull(p_opt)
        
        # Compute score
        score = _score(p_norm)
        
        if score > best_score:
            best_score = score
            best_points = p_norm.copy()
        
        # Also try with smaller T for more precise optimization
        if i % 3 == 0 and time.time() < deadline - 20.0:
            p_opt2 = _optimize_from_initial(p_opt, T=0.01, maxiter=200)
            p_norm2 = _normalize_to_unit_hull(p_opt2)
            score2 = _score(p_norm2)
            if score2 > best_score:
                best_score = score2
                best_points = p_norm2.copy()
    
    # If no valid points found, return regular polygon
    if best_points is None:
        best_points = _normalize_to_unit_hull(_regular_polygon(N, r=1.0))
    
    return best_points
# EVOLVE-BLOCK-END
