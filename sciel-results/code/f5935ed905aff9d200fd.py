"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

# Precompute all triangle indices
TRI_INDICES = np.array(list(itertools.combinations(range(N), 3)))
NUM_TRIS = len(TRI_INDICES)

def triangle_areas(points):
    """Compute areas of all triangles formed by triples of points."""
    i, j, k = TRI_INDICES.T
    pi = points[i]
    pj = points[j]
    pk = points[k]
    # Area = 0.5 * |cross(pj - pi, pk - pi)|
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except:
        # Fallback: if points are collinear or degenerate
        return 0.0

def score(points):
    """Compute the score: min triangle area / convex hull area."""
    areas = triangle_areas(points)
    min_area = np.min(areas)
    hull_area = convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def logsumexp_neg_min_triangle_area(points, beta=100.0):
    """
    Compute -log(sum(exp(-beta * area))) for all triangles.
    This is a smooth approximation of -min(area).
    Maximizing this is equivalent to maximizing min(area).
    """
    areas = triangle_areas(points)
    # Use log-sum-exp trick for numerical stability
    min_a = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - min_a)))) - beta * min_a
    # We want to maximize min area, so minimize -min area
    # logsumexp approximation of -min(a) is: -(1/beta) * log(sum(exp(-beta*a)))
    # So the objective to minimize is: (1/beta) * log(sum(exp(-beta*a)))
    # But we also need to divide by hull area. Let's handle that separately.
    return (1.0 / beta) * log_sum

def objective_smooth(points_flat, beta=100.0):
    """
    Objective function for optimization.
    We want to maximize min_triangle_area / hull_area.
    Equivalently, minimize -log(min_area / hull_area) = -log(min_area) + log(hull_area).
    
    Using log-sum-exp approximation for -log(min_area):
    -log(min_area) ≈ (1/beta) * log(sum(exp(-beta * area)))
    
    So objective ≈ (1/beta) * log(sum(exp(-beta * area))) + log(hull_area)
    """
    points = points_flat.reshape(N, 2)
    
    # Compute triangle areas
    areas = triangle_areas(points)
    
    # Compute hull area
    hull_area = convex_hull_area(points)
    if hull_area <= 1e-10:
        return 1e10
    
    # Log-sum-exp approximation for -log(min_area)
    min_a = np.min(areas)
    if min_a <= 1e-10:
        return 1e10
    
    log_sum_exp = np.log(np.sum(np.exp(-beta * (areas - min_a)))) - beta * min_a
    neg_log_min_area = (1.0 / beta) * log_sum_exp
    
    # Objective: neg_log_min_area + log(hull_area)
    # We want to minimize this
    obj = neg_log_min_area + np.log(hull_area)
    
    return obj

def gradient_smooth(points_flat, beta=100.0):
    """
    Compute gradient of the smooth objective.
    """
    points = points_flat.reshape(N, 2)
    
    # Compute triangle areas
    areas = triangle_areas(points)
    
    # Compute hull area
    hull_area = convex_hull_area(points)
    if hull_area <= 1e-10:
        return np.zeros_like(points_flat)
    
    # Log-sum-exp approximation
    min_a = np.min(areas)
    if min_a <= 1e-10:
        return np.zeros_like(points_flat)
    
    # weights = exp(-beta * (areas - min_a)) / sum(exp(-beta * (areas - min_a)))
    exp_terms = np.exp(-beta * (areas - min_a))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    # Gradient of (1/beta) * log(sum(exp(-beta * area)))
    # d/dx [(1/beta) * log(sum(exp(-beta * area)))] = (1/beta) * sum(weights * (-beta) * d(area)/dx)
    # = -sum(weights * d(area)/dx)
    
    # Compute gradient of each triangle area with respect to points
    # Area of triangle (i,j,k) = 0.5 * |cross(pj - pi, pk - pi)|
    # For the absolute value, we need to handle the sign.
    # Let's compute the signed area and use the sign for the gradient.
    
    i, j, k = TRI_INDICES.T
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    signed_areas = 0.5 * cross
    signs = np.sign(signed_areas)
    signs[signs == 0] = 1.0
    
    # Gradient of signed area w.r.t. points
    # d/dpi (0.5 * cross) = -0.5 * (v2 rotated 90 deg) ... let me think more carefully.
    # cross = (pj_x - pi_x)(pk_y - pi_y) - (pj_y - pi_y)(pk_x - pi_x)
    # d/dpi_x: -(pk_y - pi_y) - (-(pj_y - pi_y)) = -pk_y + pi_y + pj_y - pi_y = pj_y - pk_y
    # d/dpi_y: -(pj_x - pi_x) - (-(pk_x - pi_x)) = -pj_x + pi_x + pk_x - pi_x = pk_x - pj_x
    # d/dpj_x: (pk_y - pi_y)
    # d/dpj_y: -(pk_x - pi_x)
    # d/dpk_x: -(pj_y - pi_y)
    # d/dpk_y: (pj_x - pi_x)
    
    grad = np.zeros((N, 2))
    
    # Contribution from each triangle
    # For triangle t with vertices i, j, k:
    # d(area_t)/dpi = 0.5 * sign * (pj_y - pk_y, pk_x - pj_x)
    # d(area_t)/dpj = 0.5 * sign * (pk_y - pi_y, -(pk_x - pi_x))
    # d(area_t)/dpk = 0.5 * sign * (-(pj_y - pi_y), (pj_x - pi_x))
    
    for t in range(NUM_TRIS):
        w = weights[t]
        if w < 1e-10:
            continue
        ii, jj, kk = i[t], j[t], k[t]
        s = 0.5 * w * signs[t]
        
        grad[ii, 0] += s * (points[jj, 1] - points[kk, 1])
        grad[ii, 1] += s * (points[kk, 0] - points[jj, 0])
        
        grad[jj, 0] += s * (points[kk, 1] - points[ii, 1])
        grad[jj, 1] += s * (-(points[kk, 0] - points[ii, 0]))
        
        grad[kk, 0] += s * (-(points[jj, 1] - points[ii, 1]))
        grad[kk, 1] += s * (points[jj, 0] - points[ii, 0])
    
    # Gradient of (1/beta) * log(sum(exp(-beta * area))) is -sum(weights * grad_area)
    grad_obj = -grad
    
    # Add gradient of log(hull_area)
    # This is tricky. The hull area is a piecewise linear function of the points.
    # For a convex polygon, the area is 0.5 * sum of cross products of consecutive vertices.
    # Let's compute the hull and its gradient.
    
    try:
        hull = ConvexHull(points)
        hull_indices = hull.vertices
        n_hull = len(hull_indices)
        
        if n_hull >= 3:
            # Area = 0.5 * sum_{i} (x_i * y_{i+1} - x_{i+1} * y_i)
            # d/dx_i: 0.5 * (y_{i-1} - y_{i+1})
            # d/dy_i: 0.5 * (x_{i+1} - x_{i-1})
            
            hull_grad = np.zeros((N, 2))
            for idx in range(n_hull):
                curr = hull_indices[idx]
                prev = hull_indices[(idx - 1) % n_hull]
                next_pt = hull_indices[(idx + 1) % n_hull]
                
                hull_grad[curr, 0] += 0.5 * (points[prev, 1] - points[next_pt, 1])
                hull_grad[curr, 1] += 0.5 * (points[next_pt, 0] - points[prev, 0])
            
            grad_obj += hull_grad / hull_area
    except:
        pass
    
    return grad_obj.flatten()

def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Try multiple random initializations
    num_restarts = 20
    beta = 100.0
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        np.random.seed(N * 1000 + restart)
        points = np.random.rand(N, 2)
        
        # Normalize to unit area
        hull_area = convex_hull_area(points)
        if hull_area > 1e-10:
            points = points / np.sqrt(hull_area)
        
        current_score = score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
        
        # Optimize
        points_flat = points.flatten()
        
        result = minimize(
            objective_smooth,
            points_flat,
            args=(beta,),
            method='L-BFGS-B',
            jac=gradient_smooth,
            options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < 1e10:
            opt_points = result.x.reshape(N, 2)
            
            # Normalize to unit area
            hull_area = convex_hull_area(opt_points)
            if hull_area > 1e-10:
                opt_points = opt_points / np.sqrt(hull_area)
            
            opt_score = score(opt_points)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # If we didn't find any valid points, return random
    if best_points is None:
        np.random.seed(N)
        best_points = np.random.rand(N, 2)
        hull_area = convex_hull_area(best_points)
        if hull_area > 1e-10:
            best_points = best_points / np.sqrt(hull_area)
    
    return best_points
