"""Heilbronn convex regions: place 13 points to maximize min triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

N = 13
DEADLINE = 150.0  # seconds
START_TIME = time.time()


def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)


def _min_triangle_area(pts):
    """Compute the minimum area over all triples of points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(pts):
    """Compute the area of the convex hull of pts."""
    # Use the shoelace formula on the convex hull vertices
    # First, find convex hull using a simple method
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use all points (degenerate case)
        return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area


def _soft_min_area(pts, beta):
    """Soft-min surrogate for the minimum triangle area using log-sum-exp."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    areas = np.array(areas)
    # Soft-min: -1/beta * log(sum(exp(-beta * areas)))
    max_a = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a)))) + (-beta * max_a)
    return -log_sum / beta


def _soft_min_area_grad(pts, beta):
    """Gradient of the soft-min area surrogate."""
    n = len(pts)
    areas = []
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
                indices.append((i, j, k))
    areas = np.array(areas)
    
    # Soft-min value
    max_a = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_a))
    sum_exp = np.sum(exp_terms)
    soft_min_val = max_a - np.log(sum_exp) / beta
    
    # Gradient with respect to each point
    grad = np.zeros_like(pts)
    for idx, (i, j, k) in enumerate(indices):
        weight = exp_terms[idx] / sum_exp
        # d(area)/d(pts) for triangle (i,j,k)
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[i]
        # area = 0.5 * |v1 x v2|
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        sign = 1.0 if cross >= 0 else -1.0
        
        # d(area)/d(pts[i]) = -0.5 * sign * (v2, -v1) rotated... 
        # Actually, area = 0.5 * |x_j - x_i, y_j - y_i, x_k - x_i, y_k - y_i| cross
        # Let's compute gradients properly
        # area = 0.5 * | (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i) |
        # d/dx_i = 0.5 * sign * (-(y_k-y_i) + (y_j-y_i)) = 0.5 * sign * (y_j - y_k)
        # d/dy_i = 0.5 * sign * ((x_k-x_i) - (x_j-x_i)) = 0.5 * sign * (x_k - x_j)
        # d/dx_j = 0.5 * sign * (y_k-y_i)
        # d/dy_j = 0.5 * sign * (-(x_k-x_i))
        # d/dx_k = 0.5 * sign * (-(y_j-y_i))
        # d/dy_k = 0.5 * sign * (x_j-x_i)
        
        dy_jk = pts[j][1] - pts[k][1]
        dx_kj = pts[k][0] - pts[j][0]
        dy_ki = pts[k][1] - pts[i][1]
        dx_ki = pts[k][0] - pts[i][0]
        dy_ji = pts[j][1] - pts[i][1]
        dx_ji = pts[j][0] - pts[i][0]
        
        grad[i][0] += weight * 0.5 * sign * dy_jk
        grad[i][1] += weight * 0.5 * sign * dx_kj
        grad[j][0] += weight * 0.5 * sign * dy_ki
        grad[j][1] += weight * 0.5 * sign * (-dx_ki)
        grad[k][0] += weight * 0.5 * sign * (-dy_ji)
        grad[k][1] += weight * 0.5 * sign * dx_ji
    
    # Soft-min gradient: d/dx [-1/beta * log(sum(exp(-beta*areas)))] 
    # = -1/beta * 1/sum(exp) * sum(-beta * exp * d(area)/dx)
    # = sum(exp/sum(exp)) * d(area)/dx  (for the soft-min, the gradient is weighted sum of area gradients)
    # But soft-min = max - log(sum(exp(-beta*(a-max))))/beta
    # d(soft_min)/dx = -1/beta * 1/sum(exp) * sum(-beta * exp * d(a)/dx)
    # = sum(exp/sum(exp)) * d(a)/dx
    # Wait, that's not right. Let me redo.
    # soft_min = max_a - (1/beta)*log(sum(exp(-beta*(a_i - max_a))))
    # d(soft_min)/dx = -(1/beta) * (1/sum) * sum(-beta * exp(-beta*(a_i-max)) * d(a_i)/dx)
    # = (1/sum) * sum(exp(-beta*(a_i-max)) * d(a_i)/dx)
    # = sum(w_i * d(a_i)/dx) where w_i = exp(-beta*(a_i-max))/sum(exp(...))
    
    # So the gradient of soft_min is the weighted sum of individual area gradients
    # But we computed grad as sum(weight * d(area)/dx) already, which is correct for soft-min gradient
    # However, we need to negate because soft_min = -logsumexp(-beta*a)/beta
    # d(soft_min)/dx = -(1/beta) * d(logsumexp(-beta*a))/dx
    # d(logsumexp(-beta*a))/dx = sum(exp(-beta*a_i)/sum(exp(-beta*a))) * (-beta * d(a_i)/dx)
    # So d(soft_min)/dx = -(1/beta) * (-beta) * sum(w_i * d(a_i)/dx) = sum(w_i * d(a_i)/dx)
    
    return grad


def _objective(params, beta):
    """Objective function for optimization: negative soft-min area."""
    pts = params.reshape(-1, 2)
    return -_soft_min_area(pts, beta)


def _gradient(params, beta):
    """Gradient of the objective function."""
    pts = params.reshape(-1, 2)
    return -_soft_min_area_grad(pts, beta)


def _regular_polygon(n):
    """Generate a regular n-gon centered at origin with radius 1."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _radial_jitter(points, rng, sigma=0.1):
    """Add radial jitter to points."""
    jittered = points.copy()
    for i in range(len(points)):
        angle = np.arctan2(points[i][1], points[i][0])
        r = np.sqrt(points[i][0]**2 + points[i][1]**2)
        new_r = r * (1 + rng.gauss(0, sigma))
        jittered[i] = [new_r * np.cos(angle), new_r * np.sin(angle)]
    return jittered


def construct():
    """Construct 13 points maximizing min triangle area / convex hull area."""
    global START_TIME
    START_TIME = time.time()
    
    best_score = 0.0
    best_points = _regular_polygon(N)
    
    # Start with regular polygon
    current_score = _score(best_points)
    if current_score > best_score:
        best_score = current_score
        best_points = best_points.copy()
    
    # Use L-BFGS-B with soft-min surrogate
    rng = np.random.RandomState(42)
    
    # Progressive beta annealing
    betas = [50, 100, 200, 500, 1000]
    
    for beta in betas:
        if _remaining_time() < 20:
            break
        
        # Multiple restarts with different initializations
        for restart in range(3):
            if _remaining_time() < 15:
                break
            
            if restart == 0:
                x0 = best_points.flatten().copy()
            elif restart == 1:
                x0 = _radial_jitter(_regular_polygon(N), rng, sigma=0.05).flatten()
            else:
                x0 = _radial_jitter(_regular_polygon(N), rng, sigma=0.15).flatten()
            
            try:
                result = minimize(
                    _objective,
                    x0,
                    jac=_gradient,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < _objective(x0, beta):
                    new_pts = result.x.reshape(-1, 2)
                    new_score = _score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_pts.copy()
            except Exception:
                pass
        
        # After each beta stage, try a Nelder-Mead polish
        if _remaining_time() < 15:
            break
        
        try:
            def nm_objective(params):
                pts = params.reshape(-1, 2)
                return -_score(pts)
            
            nm_result = minimize(
                nm_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if nm_result.fun < -best_score:
                new_pts = nm_result.x.reshape(-1, 2)
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_pts.copy()
        except Exception:
            pass
    
    # Final polish with Nelder-Mead
    if _remaining_time() > 10:
        try:
            def nm_objective(params):
                pts = params.reshape(-1, 2)
                return -_score(pts)
            
            nm_result = minimize(
                nm_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if nm_result.fun < -best_score:
                new_pts = nm_result.x.reshape(-1, 2)
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_pts.copy()
        except Exception:
            pass
    
    return best_points
