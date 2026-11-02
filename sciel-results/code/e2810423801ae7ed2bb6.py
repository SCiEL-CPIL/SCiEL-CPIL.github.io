import time
import numpy as np
from scipy.spatial import ConvexHull

N = 13
DEADLINE = 145.0  # Safety margin before 150s kill

# Pre-compute all triples for fast vectorized scoring
# There are C(13, 3) = 286 triples
_triple_i = []
_triple_j = []
_triple_k = []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _triple_i.append(_i)
            _triple_j.append(_j)
            _triple_k.append(_k)

_triple_i = np.array(_triple_i, dtype=np.int32)
_triple_j = np.array(_triple_j, dtype=np.int32)
_triple_k = np.array(_triple_k, dtype=np.int32)

def _compute_areas(pts):
    """Compute all triangle areas for a point set."""
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-12

def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    
    areas = _compute_areas(pts)
    min_area = np.min(areas)
    
    if min_area < 1e-15:
        return 0.0
    
    return min_area / hull_area

def _soft_min_score(pts, beta=100.0):
    """
    Compute a smooth approximation of the min triangle area / hull area.
    """
    pts = np.asarray(pts, dtype=np.float64)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return -np.inf
        
    areas = _compute_areas(pts)
    
    vals = -beta * areas
    v_max = np.max(vals)
    if v_max == -np.inf:
        return -np.inf
        
    log_sum = v_max + np.log(np.sum(np.exp(vals - v_max)))
    soft_min_val = -log_sum / beta
    
    return soft_min_val / hull_area

def _neg_soft_score(params, beta):
    """Negative soft score for minimization."""
    pts = params.reshape(N, 2)
    return -_soft_min_score(pts, beta)

def _force_balance_gradient(pts, beta=100.0):
    """
    Compute the gradient of the soft-minimum area with respect to point positions.
    This is the negative of the repulsive force field.
    
    The soft-minimum is: S = -1/beta * log(sum(exp(-beta * A_i))) / H
    
    We want to maximize S, so we move in the direction of +grad(S).
    The force on each point is the negative gradient: F = -grad(S)
    
    For each triangle (i,j,k) with area A_t:
    dA_t/dp_i = 0.5 * (cross product contribution)
    
    Using the signed area formula: A = 0.5 * |cross(pj-pi, pk-pi)|
    
    The gradient of the absolute value requires care. We use the soft-min surrogate
    which is differentiable.
    
    For the soft-min: S_soft = -1/beta * log(sum_t exp(-beta * A_t))
    
    dS_soft/dp_m = -1/beta * (1/sum_t exp(-beta*A_t)) * sum_t (-beta * exp(-beta*A_t) * dA_t/dp_m)
                 = (sum_t exp(-beta*A_t) * dA_t/dp_m) / (beta * sum_t exp(-beta*A_t))
                 = (1/beta) * (sum_t w_t * dA_t/dp_m)
    
    where w_t = exp(-beta*A_t) / sum_t exp(-beta*A_t)
    
    Then dS/dp_m = dS_soft/dp_m / H - S_soft * dH/dp_m
    """
    pts = np.asarray(pts, dtype=np.float64)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return np.zeros_like(pts)
        
    areas = _compute_areas(pts)
    
    # Compute weights for soft-min
    vals = -beta * areas
    v_max = np.max(vals)
    exp_vals = np.exp(vals - v_max)
    weights = exp_vals / np.sum(exp_vals)
    
    # Compute gradients of each triangle area with respect to each point
    # For triangle (i, j, k):
    # A = 0.5 * |(pj - pi) x (pk - pi)|
    # 
    # dA/dpi = 0.5 * (perpendicular to (pj-pi) scaled by pk-pi component)
    # More precisely, using the cross product:
    # cross = (pj_x - pi_x)*(pk_y - pi_y) - (pj_y - pi_y)*(pk_x - pi_x)
    # A = 0.5 * |cross|
    #
    # dA/dpi_x = 0.5 * sign(cross) * (-(pk_y - pi_y) + (pj_y - pi_y))
    #           = 0.5 * sign(cross) * (pj_y - pk_y)
    # dA/dpi_y = 0.5 * sign(cross) * ((pk_x - pi_x) - (pj_x - pi_x))
    #           = 0.5 * sign(cross) * (pk_x - pj_x)
    #
    # dA/dpj_x = 0.5 * sign(cross) * (pk_y - pi_y)
    # dA/dpj_y = 0.5 * sign(cross) * (-(pk_x - pi_x))
    #           = 0.5 * sign(cross) * (pi_x - pk_x)
    #
    # dA/dpk_x = 0.5 * sign(cross) * (-(pj_y - pi_y))
    #           = 0.5 * sign(cross) * (pi_y - pj_y)
    # dA/dpk_y = 0.5 * sign(cross) * (pj_x - pi_x)
    
    grad = np.zeros_like(pts)
    
    for idx in range(len(_triple_i)):
        i, j, k = _triple_i[idx], _triple_j[idx], _triple_k[idx]
        w = weights[idx]
        
        if w < 1e-15:
            continue
            
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        
        dx1 = pj[0] - pi[0]
        dy1 = pj[1] - pi[1]
        dx2 = pk[0] - pi[0]
        dy2 = pk[1] - pi[1]
        
        cross = dx1 * dy2 - dy1 * dx2
        
        # Use the sign of the cross product to handle the absolute value
        # For the soft-min, we can use the signed area gradient
        # Since we're using abs, we need to be careful. 
        # Actually, for the soft-min of absolute areas, we can use:
        # d|A|/dp = sign(A) * dA/dp
        
        sign_cross = 1.0 if cross >= 0 else -1.0
        
        # Gradient contributions
        # dA/dpi
        grad[i, 0] += (w / beta) * 0.5 * sign_cross * (pj[1] - pk[1])
        grad[i, 1] += (w / beta) * 0.5 * sign_cross * (pk[0] - pj[0])
        
        # dA/dpj
        grad[j, 0] += (w / beta) * 0.5 * sign_cross * (pk[1] - pi[1])
        grad[j, 1] += (w / beta) * 0.5 * sign_cross * (pi[0] - pk[0])
        
        # dA/dpk
        grad[k, 0] += (w / beta) * 0.5 * sign_cross * (pi[1] - pj[1])
        grad[k, 1] += (w / beta) * 0.5 * sign_cross * (pj[0] - pi[0])
    
    # Normalize by hull area (approximate, ignoring dH/dp for simplicity in first pass)
    grad /= hull_area
    
    return grad

def _damped_newton_equilibrium(pts_init, beta=100.0, max_iter=50, damping=0.1, t_start=None):
    """
    Solve the force-balance equations using damped Newton-Raphson iteration.
    We find the equilibrium where the net force on each point is zero.
    
    The force on point m is: F_m = -grad(S)_m
    
    At equilibrium: F_m = 0 for all m, which means grad(S)_m = 0
    
    We use a damped approach: p_new = p + damping * step
    where step is computed to reduce the gradient magnitude.
    """
    if t_start is None:
        t_start = time.time()
        
    pts = np.array(pts_init, dtype=np.float64)
    
    for it in range(max_iter):
        if (time.time() - t_start) > (DEADLINE - 10.0):
            break
            
        # Compute gradient
        grad = _force_balance_gradient(pts, beta)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-8:
            break
            
        # Damped update: move in the direction of the gradient (to maximize S)
        # Use adaptive step size based on gradient magnitude
        step_size = damping / (1.0 + grad_norm)
        pts = pts + step_size * grad
        
        # Project back to maintain reasonable scale
        # Normalize to unit area if needed
        hull_area = _hull_area(pts)
        if hull_area > 1e-12:
            scale = np.sqrt(1.0 / hull_area)
            pts = pts * scale
            
    return pts

def _local_search_worst_triangle(pts, t_start=None, max_iter=100):
    """
    Local search that iteratively identifies the worst (smallest area) triangle
    and moves its vertices to increase the minimum area.
    """
    if t_start is None:
        t_start = time.time()
        
    pts = np.array(pts, dtype=np.float64)
    
    for it in range(max_iter):
        if (time.time() - t_start) > (DEADLINE - 10.0):
            break
            
        areas = _compute_areas(pts)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        i, j, k = _triple_i[min_idx], _triple_j[min_idx], _triple_k[min_idx]
        
        # Try to increase this triangle's area by moving one of its vertices
        # Use a simple perturbation approach
        best_score = _score_points(pts)
        best_pts = pts.copy()
        
        # Try moving each vertex slightly outward from the centroid
        centroid = (pts[i] + pts[j] + pts[k]) / 3.0
        
        for vertex in [i, j, k]:
            # Direction from centroid to vertex
            direction = pts[vertex] - centroid
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-10:
                continue
            direction /= dir_norm
            
            # Try several step sizes
            for step in [0.01, 0.005, 0.001]:
                test_pts = pts.copy()
                test_pts[vertex] = pts[vertex] + step * direction
                
                # Check if this improves the minimum area
                test_areas = _compute_areas(test_pts)
                test_min = np.min(test_areas)
                
                if test_min > min_area * 1.01:
                    test_score = _score_points(test_pts)
                    if test_score > best_score:
                        best_score = test_score
                        best_pts = test_pts
        
        if np.array_equal(best_pts, pts):
            break
            
        pts = best_pts
        
    return pts

def _generate_initial_guesses(rng):
    """Generate diverse initial guesses for the equilibrium solver."""
    guesses = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    guesses.append(p_reg)
    
    # 2. Ring structures
    ring_structures = [
        [1, 12],
        [1, 6, 6],
        [1, 4, 8],
        [1, 5, 7],
        [1, 3, 9],
        [2, 11],
        [3, 10],
        [4, 9],
        [5, 8],
        [6, 7],
    ]
    
    for sizes in ring_structures:
        pts = []
        for ring_idx, size in enumerate(sizes):
            if size == 1:
                # Center point
                pts.append([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)])
            else:
                radius = (ring_idx + 1) / len(sizes)
                theta = np.linspace(0, 2 * np.pi, size, endpoint=False) + rng.uniform(0, 2 * np.pi)
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                pts.append(np.column_stack([x, y]))
        
        if len(np.vstack(pts)) == N:
            guesses.append(np.vstack(pts))
    
    # 3. Random points in unit circle
    for _ in range(5):
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = rng.uniform(0.3, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        guesses.append(pts)
    
    return guesses

def construct():
    t_start = time.time()
    
    best_pts = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initial guesses
    guesses = _generate_initial_guesses(rng)
    
    # For each guess, run the equilibrium solver and local search
    for guess_idx, pts_init in enumerate(guesses):
        if (time.time() - t_start) > (DEADLINE - 15.0):
            break
            
        # Run damped Newton equilibrium
        try:
            pts_eq = _damped_newton_equilibrium(pts_init, beta=100.0, max_iter=30, damping=0.1, t_start=t_start)
            
            # Run local search on the result
            pts_ls = _local_search_worst_triangle(pts_eq, t_start=t_start, max_iter=50)
            
            score = _score_points(pts_ls)
            
            if score > best_score:
                best_score = score
                best_pts = pts_ls
                
        except Exception as e:
            continue
            
    # If no good result, fallback to regular polygon
    if best_pts is None or best_score < 1e-10:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score_points(best_pts)
        
    return best_pts
