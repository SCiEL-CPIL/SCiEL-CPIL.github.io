"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _triangle_areas_fast(points):
    """Compute areas of all triangles formed by triples of points using vectorized operations."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                areas.append(0.5 * abs(cross))
    return np.array(areas)


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    hull = ConvexHull(points)
    return hull.volume


def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    min_tri = np.min(_triangle_areas_fast(points))
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_tri / hull_area


def _generate_points(params):
    """
    Generate 13 points from angular perturbation parameters.
    params: [a1, b1, a2, b2] where:
      angle_k = k*2*pi/13 + a1*cos(k*2*pi/13) + b1*sin(k*2*pi/13) + a2*cos(2*k*2*pi/13) + b2*sin(2*k*2*pi/13)
    Points lie on the unit circle at these perturbed angles, ensuring smooth convexity.
    """
    n = 13
    a1, b1, a2, b2 = params
    
    base_angles = np.arange(n) * 2 * np.pi / n
    
    # Angular perturbation with 2 harmonics
    angle_perturbation = (a1 * np.cos(base_angles) + b1 * np.sin(base_angles) +
                         a2 * np.cos(2 * base_angles) + b2 * np.sin(2 * base_angles))
    
    angles = base_angles + angle_perturbation
    
    # Points on unit circle at perturbed angles
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points


def _is_convex(points):
    """Check if all 13 points are on the convex hull."""
    try:
        hull = ConvexHull(points)
        return len(hull.vertices) == 13
    except Exception:
        return False


def _objective(params):
    """Negative score for minimization with convexity penalty."""
    points = _generate_points(params)
    if not _is_convex(points):
        return -0.005  # Penalty for non-convex configurations
    return -_score(points)


def _cma_es(objective, x0, sigma0, bounds, max_iter=500, pop_size=15, seed=42):
    """
    Simple CMA-ES implementation for low-dimensional optimization.
    """
    rng = np.random.RandomState(seed)
    n = len(x0)
    
    # Strategy parameters
    lambda_ = pop_size
    mu = lambda_ // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = w / np.sum(w)
    mu_eff = 1.0 / np.sum(w**2)
    
    # Learning rates
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c_1 = 2 / ((n + 1.3)**2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2)**2 + mu_eff))
    
    # Initial values
    x = np.array(x0, dtype=float)
    sigma = sigma0
    B = np.eye(n)
    D = np.ones(n)
    p_sigma = np.zeros(n)
    p_c = np.zeros(n)
    hsig = 0.0
    
    # Population history
    best_x = x.copy()
    best_f = objective(x)
    
    for it in range(max_iter):
        # Generate population
        pops = np.zeros((lambda_, n))
        for i in range(lambda_):
            z = rng.standard_normal(n)
            pops[i] = x + sigma * (B @ (D * z))
        
        # Evaluate
        fs = np.array([objective(pops[i]) for i in range(lambda_)])
        
        # Sort by fitness
        order = np.argsort(fs)
        ys = pops[order]
        fs = fs[order]
        
        # Update best
        if fs[0] < best_f:
            best_f = fs[0]
            best_x = ys[0].copy()
        
        # Mean update
        y_w = np.zeros(n)
        for i in range(mu):
            y_w += w[i] * (ys[i] - x) / sigma
        x_new = x + sigma * y_w
        
        # Sigma update
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D @ (y_w / sigma)))
        
        chi_n = np.sqrt(n) * (1 - 1 / (2 * n * n) + 1 / (12 * n**3))
        if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (it + 1))) < (1.4 + 2 / (n + 1)) * chi_n:
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
        
        # Covariance matrix update
        p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
        hsig = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**(it + 1)) < (1.4 + 2 / (n + 1)) * chi_n else 0
        rank_1 = (1 + c_1 * c_c * hsig - c_1) * np.outer(p_c, p_c)
        rank_mu = np.zeros((n, n))
        for i in range(mu):
            zi = (ys[i] - x) / sigma
            rank_mu += w[i] * np.outer(zi, zi)
        
        B = B @ (np.eye(n) + 0.5 * c_1 * (rank_1 + rank_mu - np.eye(n)))
        # Reorthogonalize and compute D
        from scipy.linalg import eigh
        H = B @ B.T
        eigvals, eigvecs = eigh(H)
        eigvals = np.maximum(eigvals, 1e-10)
        D = np.sqrt(eigvals)
        B = eigvecs
        
        x = x_new
        sigma = max(sigma, 1e-10)
    
    return best_x, best_f


def construct():
    """
    Construct 13 points to maximize the minimum triangle area over convex hull area.
    Uses CMA-ES to optimize angular perturbations (2 harmonics) of a regular 13-gon.
    """
    best_score = 0.0
    best_points = None
    
    # Parameter bounds: [a1, b1, a2, b2]
    # Angular perturbations should be small to maintain convexity
    bounds = [
        (-0.3, 0.3),   # a1
        (-0.3, 0.3),   # b1
        (-0.15, 0.15), # a2
        (-0.15, 0.15), # b2
    ]
    
    rng = np.random.RandomState(42)
    
    # Multiple restarts with CMA-ES
    for trial in range(5):
        if trial == 0:
            # Start from regular polygon
            init_params = np.zeros(4)
        else:
            # Random initialization within bounds
            init_params = np.array([
                rng.uniform(bounds[0][0], bounds[0][1]),
                rng.uniform(bounds[1][0], bounds[1][1]),
                rng.uniform(bounds[2][0], bounds[2][1]),
                rng.uniform(bounds[3][0], bounds[3][1]),
            ])
        
        try:
            # Run CMA-ES
            sigma0 = 0.1
            result_params, result_f = _cma_es(
                _objective,
                init_params,
                sigma0,
                bounds,
                max_iter=400,
                pop_size=12,
                seed=42 + trial
            )
            
            # Clip to bounds
            for i in range(len(result_params)):
                result_params[i] = np.clip(result_params[i], bounds[i][0], bounds[i][1])
            
            points = _generate_points(result_params)
            
            # Verify convexity
            if not _is_convex(points):
                continue
            
            score = _score(points)
            
            if score > best_score:
                best_score = score
                best_points = points
        except Exception:
            continue
    
    if best_points is None:
        # Fallback: regular 13-gon
        angles = np.arange(N) * 2 * np.pi / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
