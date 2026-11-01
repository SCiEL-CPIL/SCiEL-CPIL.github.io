import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# Pre-compute triangle indices
def _get_triangle_indices(n):
    i_idx, j_idx, k_idx = np.indices((n, n, n)).reshape(3, -1)
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    return i_idx[mask], j_idx[mask], k_idx[mask]

TRI_I, TRI_J, TRI_K = _get_triangle_indices(N)
NUM_TRI = len(TRI_I)

def compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    Vectorized for speed.
    """
    if len(points) < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
        
    p_i = points[TRI_I]
    p_j = points[TRI_J]
    p_k = points[TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
        
    return min_area / hull_area

def _smooth_min_area_and_grad(points, beta):
    """
    Compute smooth approximation of min triangle area and its gradient.
    Uses log-sum-exp: -log(sum(exp(-beta * areas))) / beta
    """
    p_i = points[TRI_I]
    p_j = points[TRI_J]
    p_k = points[TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Smooth minimum using log-sum-exp
    # min(a) ≈ -1/beta * log(sum(exp(-beta * a)))
    max_area = np.max(areas)
    shifted = -beta * (areas - max_area)
    log_sum_exp = max_area - (1.0 / beta) * np.log(np.sum(np.exp(shifted)))
    smooth_min = log_sum_exp
    
    # Gradient: d/dx smooth_min = -sum(exp(-beta*areas) * d(areas)/dx) / sum(exp(-beta*areas))
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    
    # Gradient of each area w.r.t. points
    # area = 0.5 * |cross|, cross = v1_x * v2_y - v1_y * v2_x
    # sign = sign(cross)
    signs = np.sign(cross)
    
    grad = np.zeros_like(points)
    # d(cross)/d(p_i) = -d(v1)/d(p_i) * v2_y + d(v2)/d(p_i) * v1_y ... 
    # Actually: cross = (p_j - p_i)_x * (p_k - p_i)_y - (p_j - p_i)_y * (p_k - p_i)_x
    # d(cross)/d(p_i_x) = -(p_k - p_i)_y + (p_j - p_i)_y = p_j_y - p_k_y
    # d(cross)/d(p_i_y) = (p_k - p_i)_x - (p_j - p_i)_x = p_k_x - p_j_x
    # d(cross)/d(p_j_x) = (p_k - p_i)_y
    # d(cross)/d(p_j_y) = -(p_k - p_i)_x
    # d(cross)/d(p_k_x) = -(p_j - p_i)_y
    # d(cross)/d(p_k_y) = (p_j - p_i)_x
    
    # For area = 0.5 * sign(cross) * cross
    # d(area)/d(p_i_x) = 0.5 * sign * (p_j_y - p_k_y)
    # etc.
    
    # Use vectorized approach: accumulate gradients
    # d(area_t)/d(points[i, 0]) for triangle t where i is TRI_I[t]
    # d(area_t)/d(points[j, 0]) for triangle t where j is TRI_J[t]
    # d(area_t)/d(points[k, 0]) for triangle t where k is TRI_K[t]
    
    w = weights * signs * 0.5
    
    # For point i in each triangle
    di_x = w * (points[TRI_J, 1] - points[TRI_K, 1])
    di_y = w * (points[TRI_K, 0] - points[TRI_J, 0])
    
    dj_x = w * (points[TRI_K, 1] - points[TRI_I, 1])
    dj_y = w * (-(points[TRI_K, 0] - points[TRI_I, 0]))
    
    dk_x = w * (-(points[TRI_J, 1] - points[TRI_I, 1]))
    dk_y = w * (points[TRI_J, 0] - points[TRI_I, 0])
    
    # Accumulate
    np.add.at(grad[:, 0], TRI_I, di_x)
    np.add.at(grad[:, 1], TRI_I, di_y)
    np.add.at(grad[:, 0], TRI_J, dj_x)
    np.add.at(grad[:, 1], TRI_J, dj_y)
    np.add.at(grad[:, 0], TRI_K, dk_x)
    np.add.at(grad[:, 1], TRI_K, dk_y)
    
    return smooth_min, grad

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _neg_smooth_score_and_grad(x, beta):
    """
    Negative smooth score with gradient.
    score = smooth_min_area / hull_area
    We maximize score, so minimize -score.
    """
    points = x.reshape((N, 2))
    smooth_min, grad_min = _smooth_min_area_and_grad(points, beta)
    ha = _hull_area(points)
    
    if ha < 1e-12:
        return 1e6, np.zeros_like(x)
    
    score = smooth_min / ha
    # Gradient of -score = -grad(smooth_min)/ha
    grad_score = grad_min / ha
    
    return -score, -grad_score.flatten()

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -float('inf')
    
    rng = np.random.default_rng(42)
    
    inits = []
    
    # 1. Regular N-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(p_reg)
    
    # 2. Two concentric rings (7 points each)
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = angles_outer + np.pi/7
    p_outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
    p_inner = np.column_stack((0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)))
    p_two_rings = np.vstack((p_outer, p_inner))
    inits.append(p_two_rings)
    
    # 3. Three rings: 5 outer, 5 middle, 4 inner
    n1, n2, n3 = 5, 5, 4
    r1, r2, r3 = 1.0, 0.6, 0.3
    a1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
    a2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + np.pi/n2
    a3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + np.pi/n3
    p1 = np.column_stack((r1*np.cos(a1), r1*np.sin(a1)))
    p2 = np.column_stack((r2*np.cos(a2), r2*np.sin(a2)))
    p3 = np.column_stack((r3*np.cos(a3), r3*np.sin(a3)))
    p_three_rings = np.vstack((p1, p2, p3))
    inits.append(p_three_rings)
    
    # 4. Random points in unit circle
    for _ in range(5):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        p_rand = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        inits.append(p_rand)
        
    # 5. Jittered regular polygon
    for _ in range(5):
        p_jitter = p_reg + rng.normal(0, 0.1, (N, 2))
        inits.append(p_jitter)

    # Evaluate initial scores
    for p in inits:
        s = compute_score(p)
        if s > best_score:
            best_score = s
            best_points = p.copy()
            
    # Sort inits by score
    scored_inits = [(compute_score(p), p) for p in inits]
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    # Use L-BFGS-B with smooth objective from top initial points
    # Use multiple beta values for annealing
    betas = [50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for init_score, init_points in scored_inits[:5]:  # Top 5 inits
        if time.time() - start_time > deadline:
            break
            
        x0 = init_points.flatten()
        current_x = x0.copy()
        
        for beta in betas:
            if time.time() - start_time > deadline:
                break
                
            try:
                res = minimize(
                    lambda x: _neg_smooth_score_and_grad(x, beta)[0],
                    current_x,
                    jac=lambda x: _neg_smooth_score_and_grad(x, beta)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                current_x = res.x
                new_points = current_x.reshape((N, 2))
                new_score = compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                    
            except Exception:
                pass
    
    # Final polish with high beta
    if best_points is not None and time.time() - start_time < deadline - 10:
        for beta in [1000.0, 2000.0, 5000.0]:
            if time.time() - start_time > deadline - 5:
                break
            try:
                x_best = best_points.flatten()
                res = minimize(
                    lambda x: _neg_smooth_score_and_grad(x, beta)[0],
                    x_best,
                    jac=lambda x: _neg_smooth_score_and_grad(x, beta)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                new_points = res.x.reshape((N, 2))
                new_score = compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
            except Exception:
                pass
    
    return best_points
