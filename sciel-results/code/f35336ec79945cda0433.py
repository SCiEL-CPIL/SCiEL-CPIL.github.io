import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import itertools

N = 14
START_TIME = time.time()
DEADLINE = START_TIME + 145.0

def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

TRIPLE_INDICES = _get_triple_indices(N)
I_IDX = TRIPLE_INDICES[:, 0]
J_IDX = TRIPLE_INDICES[:, 1]
K_IDX = TRIPLE_INDICES[:, 2]

def _score(points):
    """Compute min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    p_i = points[I_IDX]
    p_j = points[J_IDX]
    p_k = points[K_IDX]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_score(points, sharpness=20.0):
    """
    Compute soft-min of triangle areas / convex hull area.
    Soft-min provides smooth gradient for optimization.
    """
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    p_i = points[I_IDX]
    p_j = points[J_IDX]
    p_k = points[K_IDX]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft-min: -log(sum(exp(-areas/sharpness))) * sharpness
    # This is smooth and approximates min as sharpness -> inf
    min_log = np.min(areas)  # For numerical stability
    log_sum = min_log + np.log(np.sum(np.exp(-(areas - min_log) / sharpness)))
    soft_min = -sharpness * log_sum
    
    return soft_min / hull_area

def _neg_score(flat_points):
    points = flat_points.reshape((N, 2))
    return -_score(points)

def _neg_soft_score(flat_points, sharpness=20.0):
    points = flat_points.reshape((N, 2))
    return -_soft_min_score(points, sharpness)

def _make_regular_polygon(n, radius=1.0, offset=0.0):
    angles = np.linspace(offset, 2*np.pi + offset, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_ring_points(n_out, n_in, r, theta):
    """
    Generate points for a two-ring configuration.
    Outer ring: n_out points on unit circle.
    Inner ring: n_in points on circle of radius r, rotated by theta.
    """
    if n_out + n_in != N:
        return None
        
    pts_out = _make_regular_polygon(n_out, 1.0, 0.0)
    pts_in = _make_regular_polygon(n_in, r, theta)
    
    return np.vstack([pts_out, pts_in])

def _neg_ring_score(params, n_out, n_in, use_soft=True):
    """
    Objective function for differential evolution.
    params: [r, theta]
    """
    r, theta = params
    if r < 0.01 or r > 0.99:
        return 0.0
        
    points = _generate_ring_points(n_out, n_in, r, theta)
    if points is None:
        return 0.0
        
    if use_soft:
        s = _soft_min_score(points, sharpness=15.0)
    else:
        s = _score(points)
    return -s

def _fourier_points(coeffs, n_pts=N):
    """
    Generate points from Fourier coefficients.
    coeffs: array of shape (2*K,) where K is number of harmonics.
    Point i at angle theta_i:
      x_i = 1 + sum_{k=1}^{K-1} (a_k cos(k*theta_i) + b_k sin(k*theta_i))
      y_i = sum_{k=1}^{K-1} (c_k cos(k*theta_i) + d_k sin(k*theta_i))
    
    This parameterization ensures points are near a unit circle base,
    allowing flexible deformation while maintaining approximate convexity.
    """
    K = len(coeffs) // 4  # 4 coefficients per harmonic (a,b,c,d)
    theta = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    
    x = np.ones(n_pts)  # Base radius
    y = np.zeros(n_pts)
    
    for k in range(1, K + 1):
        a = coeffs[4*(k-1)]
        b = coeffs[4*(k-1) + 1]
        c = coeffs[4*(k-1) + 2]
        d = coeffs[4*(k-1) + 3]
        
        kt = k * theta
        x += a * np.cos(kt) + b * np.sin(kt)
        y += c * np.cos(kt) + d * np.sin(kt)
    
    return np.column_stack([x, y])

def _neg_fourier_score(params, K=3, use_soft=True):
    """
    Objective for Fourier parameters.
    params: array of length 4*K
    """
    coeffs = params
    pts = _fourier_points(coeffs)
    
    # Check if points form a valid convex shape
    if np.any(np.abs(pts) > 3.0):  # Points too far from origin
        return 0.0
    
    if use_soft:
        s = _soft_min_score(pts, sharpness=15.0)
    else:
        s = _score(pts)
    return -s

def _polish_points(points, time_budget=10.0):
    """
    Multi-stage coordinate polishing.
    Stage 1: Nelder-Mead with soft-min
    Stage 2: Coordinate descent with hard score
    """
    start_time = time.time()
    deadline = start_time + time_budget
    best_score = _score(points)
    best_points = points.copy()
    
    # Stage 1: Nelder-Mead with soft-min objective
    if time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: _neg_soft_score(x, sharpness=10.0),
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'maxfev': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12
                }
            )
            opt_pts = res.x.reshape((N, 2))
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts
        except Exception:
            pass
    
    # Stage 2: Coordinate descent with hard score
    if time.time() < deadline - 1:
        try:
            pts = best_points.copy()
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                for i in range(N):
                    for coord in range(2):
                        # Try small perturbations
                        for delta in [0.001, -0.001, 0.01, -0.01, 0.05, -0.05]:
                            new_pts = pts.copy()
                            new_pts[i, coord] += delta
                            new_score = _score(new_pts)
                            if new_score > best_score + 1e-10:
                                best_score = new_score
                                best_points = new_pts.copy()
                                pts = new_pts.copy()
                                improved = True
                                break
        except Exception:
            pass
    
    return best_score, best_points

def _optimize_ring_split(n_out, n_in, time_budget):
    """
    Optimize a specific split (n_out, n_in) using DE + Polish.
    Returns (score, points)
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g
    
    bounds = [(0.05, 0.95), (0.0, theta_max)]
    
    best_score = -np.inf
    best_points = None
    
    popsize = 15
    maxiter = 30
    
    try:
        res = differential_evolution(
            lambda p: _neg_ring_score(p, n_out, n_in, use_soft=True),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-6,
            polish=False,
        )
        
        if res.fun < 0:
            best_score = -res.fun
            best_params = res.x
            best_points = _generate_ring_points(n_out, n_in, best_params[0], best_params[1])
        else:
            best_params = None
            
    except Exception:
        best_params = None
        best_points = None

    if best_points is None or best_score < 0.5:
        heuristic_starts = []
        for r in [0.3, 0.5, 0.7, 0.9]:
            for theta in [0, theta_max/2, theta_max/3]:
                heuristic_starts.append([r, theta])
        
        for start in heuristic_starts:
            if time.time() > deadline - 10:
                break
            s = -_neg_ring_score(start, n_out, n_in, use_soft=False)
            if s > best_score:
                best_score = s
                best_points = _generate_ring_points(n_out, n_in, start[0], start[1])
                best_params = start

    if best_points is not None and time.time() < deadline - 5:
        best_score, best_points = _polish_points(best_points, time_budget=min(15.0, deadline - time.time() - 2))
        
    return best_score, best_points

def _optimize_fourier(time_budget, K=3):
    """
    Optimize Fourier parameters using DE + Polish.
    Returns (score, points)
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    n_params = 4 * K
    bounds = [(-0.3, 0.3)] * n_params
    
    best_score = -np.inf
    best_points = None
    
    popsize = 15
    maxiter = 20
    
    try:
        res = differential_evolution(
            lambda p: _neg_fourier_score(p, K=K, use_soft=True),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-6,
            polish=False,
        )
        
        if res.fun < 0:
            best_score = -res.fun
            best_params = res.x
            best_points = _fourier_points(best_params)
        else:
            best_params = None
            
    except Exception:
        best_params = None
        best_points = None
    
    # Heuristic starts if DE failed or score low
    if best_points is None or best_score < 0.5:
        # Start from regular polygon (all Fourier coeffs = 0)
        heuristic_starts = [
            np.zeros(n_params),  # Regular polygon
        ]
        
        # Add some perturbations
        for _ in range(5):
            pert = np.random.uniform(-0.1, 0.1, n_params)
            heuristic_starts.append(pert)
        
        for start in heuristic_starts:
            if time.time() > deadline - 10:
                break
            s = -_neg_fourier_score(start, K=K, use_soft=False)
            if s > best_score:
                best_score = s
                best_points = _fourier_points(start)
    
    # Coordinate Polish
    if best_points is not None and time.time() < deadline - 5:
        best_score, best_points = _polish_points(best_points, time_budget=min(15.0, deadline - time.time() - 2))
        
    return best_score, best_points

def construct():
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = START_TIME + 145.0
    
    np.random.seed(42)
    
    best_global_score = -np.inf
    best_global_points = None
    
    # Baseline: Regular polygon
    reg_poly = _make_regular_polygon(N)
    s_reg = _score(reg_poly)
    best_global_score = s_reg
    best_global_points = reg_poly.copy()
    
    # 1. Try Fourier Optimization (most flexible)
    if time.time() < DEADLINE - 50:
        s_fourier, pts_fourier = _optimize_fourier(45.0, K=3)
        if pts_fourier is not None and s_fourier > best_global_score:
            best_global_score = s_fourier
            best_global_points = pts_fourier
            
    # 2. Try Ring Optimization
    if time.time() < DEADLINE - 30:
        s_ring, pts_ring = _optimize_ring_split(7, 7, 25.0)
        if pts_ring is not None and s_ring > best_global_score:
            best_global_score = s_ring
            best_global_points = pts_ring
    
    # 3. Try different ring splits
    if time.time() < DEADLINE - 15:
        for n_out in [6, 8]:
            n_in = N - n_out
            if time.time() < DEADLINE - 10:
                s_ring2, pts_ring2 = _optimize_ring_split(n_out, n_in, 10.0)
                if pts_ring2 is not None and s_ring2 > best_global_score:
                    best_global_score = s_ring2
                    best_global_points = pts_ring2
    
    # 4. Final Polish on the absolute best
    if best_global_points is not None and time.time() < DEADLINE - 5:
        best_global_score, best_global_points = _polish_points(
            best_global_points, 
            time_budget=min(10.0, DEADLINE - time.time() - 2)
        )

    return best_global_points
