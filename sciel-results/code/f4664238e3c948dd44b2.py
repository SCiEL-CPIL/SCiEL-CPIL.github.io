import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
from scipy.spatial import ConvexHull

N = 14

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    return min_area / hull_area

def _generate_fourier_points(params):
    """
    Generate 14 points on a distorted circle.
    params: [a2, a3, a4, p2, p3, p4]
    Radii: R_k = 1 + a2*cos(2*theta_k + p2) + a3*cos(3*theta_k + p3) + a4*cos(4*theta_k + p4)
    """
    a2, a3, a4, p2, p3, p4 = params
    
    # Regular 14-gon angles
    angles = np.arange(N) * 2 * np.pi / N
    
    # Calculate radii
    radii = 1.0
    radii += a2 * np.cos(2 * angles + p2)
    radii += a3 * np.cos(3 * angles + p3)
    radii += a4 * np.cos(4 * angles + p4)
    
    # Ensure convexity/safety: if any radius is too small, clamp or penalize
    # For optimization, we can just let it be, but if it becomes non-convex or self-intersecting,
    # the hull area calculation handles the "hull" part, but the "points in convex region"
    # constraint implies the points themselves should form a convex set or be inside one.
    # The score normalizes by the hull area of the points. If points are non-convex,
    # the hull area is larger than the area "occupied" by the shape, but the definition
    # says "normalized by the point set convex hull area".
    # So non-convex sets are allowed, but likely suboptimal.
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    return np.column_stack((x, y))

def _neg_score_fourier(params):
    pts = _generate_fourier_points(params)
    score = _calculate_score(pts)
    return -score

def _optimize_fourier():
    """
    Optimize the Fourier coefficients.
    """
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    # Initial guess: regular polygon (all 0)
    # Try small perturbations
    best_score = 0.0
    best_params = np.zeros(6)
    
    # Random restarts in the 6D space
    # Amplitudes should be small to maintain convexity, e.g., [-0.2, 0.2]
    # Phases in [0, 2*pi]
    
    n_restarts = 10
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        if i == 0:
            params0 = np.zeros(6)
        else:
            # Random initialization
            a2 = np.random.uniform(-0.1, 0.1)
            a3 = np.random.uniform(-0.1, 0.1)
            a4 = np.random.uniform(-0.1, 0.1)
            p2 = np.random.uniform(0, 2*np.pi)
            p3 = np.random.uniform(0, 2*np.pi)
            p4 = np.random.uniform(0, 2*np.pi)
            params0 = np.array([a2, a3, a4, p2, p3, p4])
            
        try:
            res = minimize(
                _neg_score_fourier,
                params0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res.fun < 0:
                s = -res.fun
                if s > best_score:
                    best_score = s
                    best_params = res.x
        except Exception:
            pass
            
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # 1. Try Fourier Distortion
    try:
        params, score = _optimize_fourier()
        if score > best_score:
            best_score = score
            best_points = _generate_fourier_points(params)
    except Exception:
        pass
    
    # 2. Fallback / Alternative: Two Ring (7,7)
    # This was the previous method. We can include a quick check if Fourier failed or is poor.
    # But Fourier includes the regular 14-gon as a special case (all params 0).
    # The regular 14-gon score is a baseline.
    # Let's ensure we have at least the regular 14-gon.
    angles = np.arange(N) * 2 * np.pi / N
    x = np.cos(angles)
    y = np.sin(angles)
    regular_pts = np.column_stack((x, y))
    regular_score = _calculate_score(regular_pts)
    
    if regular_score > best_score:
        best_score = regular_score
        best_points = regular_pts
        
    # 3. If we have time, try the old 7-ring method as well?
    # The directive asks for a structural change. The Fourier method is the change.
    # We should stick to it to avoid confusion, unless it's clearly worse.
    # Given the stagnation, the Fourier method is the primary candidate.
    
    if best_points is None:
        best_points = regular_pts
        
    return best_points
