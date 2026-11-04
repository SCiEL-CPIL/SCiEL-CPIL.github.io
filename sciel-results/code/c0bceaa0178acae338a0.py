import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

N = 14
START_TIME = time.time()
DEADLINE = START_TIME + 115.0

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
NUM_TRIPLES = len(TRIPLE_INDICES)

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

def _soft_min_score(points, beta=50.0):
    """Soft minimum of triangle areas for smoother optimization."""
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
    
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    log_sum = np.log(np.sum(np.exp(-beta * areas)))
    soft_min = -log_sum / beta
    
    return soft_min / hull_area

def _jittered_polygon_points(fourier_coeffs, rotation=0.0, scale=1.0):
    """
    Generate jittered regular 14-gon points.
    
    fourier_coeffs: array of shape (2*K,) where K is number of Fourier modes.
    The radial displacement is: r(theta) = 1 + sum_k (a_k * cos(k*theta) + b_k * sin(k*theta))
    
    rotation: rotation angle of the entire configuration
    scale: global scale factor
    """
    base_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # Compute radial displacements
    n_coeffs = len(fourier_coeffs)
    K = n_coeffs // 2
    
    radial = np.ones(N)
    for k in range(1, K + 1):
        a_k = fourier_coeffs[2*(k-1)]
        b_k = fourier_coeffs[2*(k-1) + 1]
        radial += a_k * np.cos(k * base_angles) + b_k * np.sin(k * base_angles)
    
    # Ensure positive radii
    radial = np.maximum(radial, 0.1)
    
    # Apply rotation
    angles = base_angles + rotation
    
    x = scale * radial * np.cos(angles)
    y = scale * radial * np.sin(angles)
    
    return np.column_stack([x, y])

def _neg_score_jittered(params):
    """Negative soft-min score for jittered polygon optimization."""
    # params: [fourier_coeffs..., rotation, scale]
    n_fourier = 12  # 6 modes * 2
    fourier_coeffs = params[:n_fourier]
    rotation = params[n_fourier]
    scale = params[n_fourier + 1]
    
    # Clamp scale to reasonable range
    scale = np.clip(scale, 0.5, 2.0)
    
    try:
        pts = _jittered_polygon_points(fourier_coeffs, rotation, scale)
        return -_soft_min_score(pts, beta=50.0)
    except Exception:
        return 0.0

def _neg_score_jittered_hard(params):
    """Negative hard min score for jittered polygon optimization."""
    n_fourier = 12
    fourier_coeffs = params[:n_fourier]
    rotation = params[n_fourier]
    scale = params[n_fourier + 1]
    scale = np.clip(scale, 0.5, 2.0)
    
    try:
        pts = _jittered_polygon_points(fourier_coeffs, rotation, scale)
        return -_score(pts)
    except Exception:
        return 0.0

def _neg_score(points_flat):
    points = points_flat.reshape((N, 2))
    return -_score(points)

def _neg_soft_score(points_flat):
    points = points_flat.reshape((N, 2))
    return -_soft_min_score(points, beta=50.0)

def construct():
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0
    
    np.random.seed(42)
    
    best_score = -1
    best_points = None
    
    # === Phase 1: Jittered Regular Polygon Optimization ===
    # Use 6 Fourier modes (12 coefficients) + rotation + scale = 14 parameters
    n_fourier = 12
    
    # Multiple starting points with different random Fourier coefficients
    start_configs = []
    
    # Zero jitter (regular polygon)
    start_configs.append(np.zeros(n_fourier + 2))
    start_configs[-1][-2] = 0.0  # rotation
    start_configs[-1][-1] = 1.0  # scale
    
    # Small random jitters
    for _ in range(8):
        coeffs = np.random.randn(n_fourier) * 0.05
        rot = np.random.uniform(0, 2*np.pi)
        scale = np.random.uniform(0.9, 1.1)
        start_configs.append(np.concatenate([coeffs, [rot, scale]]))
    
    # Medium jitters
    for _ in range(4):
        coeffs = np.random.randn(n_fourier) * 0.1
        rot = np.random.uniform(0, 2*np.pi)
        scale = np.random.uniform(0.8, 1.2)
        start_configs.append(np.concatenate([coeffs, [rot, scale]]))
    
    for sp in start_configs:
        if time.time() > DEADLINE - 25:
            break
        
        try:
            # First optimize with soft-min for smoother landscape
            res = minimize(
                _neg_score_jittered,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 500, 'maxfev': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            opt_params = res.x
            opt_pts = _jittered_polygon_points(opt_params[:n_fourier], opt_params[n_fourier], opt_params[n_fourier+1])
            opt_score = _score(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
                
        except Exception:
            pass
        
        # Polish with hard min
        if best_points is not None and time.time() < DEADLINE - 15:
            try:
                # Convert to full point representation for direct optimization
                res2 = minimize(
                    _neg_score,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'maxfev': 500, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                final_pts = res2.x.reshape((N, 2))
                final_score = _score(final_pts)
                
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_pts.copy()
                    
            except Exception:
                pass
    
    # === Phase 2: Direct Point Optimization from Best ===
    if best_points is not None and time.time() < DEADLINE - 10:
        try:
            # Use soft-min for initial polish
            res = minimize(
                _neg_soft_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'maxfev': 400, 'xatol': 1e-9, 'fatol': 1e-11}
            )
            soft_pts = res.x.reshape((N, 2))
            soft_score = _score(soft_pts)
            
            if soft_score > best_score:
                best_score = soft_score
                best_points = soft_pts.copy()
                
        except Exception:
            pass
    
    # Final hard-min polish
    if best_points is not None and time.time() < DEADLINE - 5:
        try:
            res = minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'maxfev': 400, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            final_pts = res.x.reshape((N, 2))
            final_score = _score(final_pts)
            
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts.copy()
                
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
