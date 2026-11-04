"""Heilbronn convex regions n=14 - optimized two-ring with soft-min L-BFGS-B and coordinate polish."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, minimize_scalar

N = 14
DEADLINE = 120.0
SAFETY = 10.0  # seconds before deadline to stop

def _time_remaining():
    return DEADLINE - SAFETY - time.time()

def _calculate_score(pts):
    """Compute min triangle area / convex hull area."""
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area < 1e-12:
        return 0.0
    
    n = len(pts)
    min_area = float('inf')
    
    # Vectorized computation of all triangle areas
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-10:
                    return 0.0
                    
    return min_area / hull_area

def _soft_min_score(pts, beta=50.0):
    """
    Smooth surrogate for the minimum triangle area using log-sum-exp.
    Returns -soft_min / hull_area (to be minimized).
    """
    n = len(pts)
    if n < 3:
        return 1e9
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 1e9
    if hull_area < 1e-12:
        return 1e9
    
    # Collect all triangle areas
    areas = []
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas.extend((0.5 * crosses).tolist())
    
    areas = np.array(areas)
    if len(areas) == 0:
        return 1e9
    
    # Soft min via log-sum-exp: soft_min = -log(sum(exp(-a_i / beta))) * beta
    # For numerical stability, subtract max
    a_min = np.min(areas)
    # Use a slightly different formulation for stability
    # soft_min ≈ a_min - beta * log(sum(exp(-(a_i - a_min)/beta)))
    # But we want to maximize min, so minimize -min.
    # Let's use: f = -soft_min / hull_area
    # soft_min = min_i a_i (approximately)
    # To make it smooth: g = -beta * log(sum_i exp(-a_i / beta))
    # As beta -> 0, g -> -min(a_i)
    # As beta -> inf, g -> -mean(a_i)
    
    # Numerically stable version:
    # g = -beta * (log_sum_exp(-a_i / beta))
    # log_sum_exp(x) = max(x) + log(sum(exp(x - max(x))))
    x = -areas / beta
    x_max = np.max(x)
    lse = x_max + np.log(np.sum(np.exp(x - x_max)))
    soft_min_val = -beta * lse
    
    return -soft_min_val / hull_area

def _get_points_2p(r, theta):
    """Generate 14 points: 7 outer (unit circle), 7 inner (radius r, rotated by theta)."""
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_inner = angles_outer + theta
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = r * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    return np.vstack([outer_pts, inner_pts])

def _score_ring_params(params, beta=50.0):
    """Score function for ring parameters (r, theta)."""
    r, theta = params
    if r < 0.05 or r > 1.0:
        return 1e9
    pts = _get_points_2p(r, theta)
    return _soft_min_score(pts, beta)

def _optimize_ring(beta=50.0):
    """Optimize ring parameters using L-BFGS-B."""
    def neg_score(params):
        return _score_ring_params(params, beta)
    
    # Multi-start
    best_x = None
    best_f = float('inf')
    
    starts = [
        [0.6, np.pi/7],
        [0.7, np.pi/14],
        [0.5, 0.0],
        [0.8, np.pi/7],
        [0.65, np.pi/10],
        [0.55, np.pi/9],
    ]
    
    for x0 in starts:
        if _time_remaining() < 5.0:
            break
        try:
            res = minimize(neg_score, x0, method='L-BFGS-B', bounds=[(0.05, 1.0), (0, np.pi/7)],
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            pass
    
    if best_x is None:
        best_x = np.array([0.6, np.pi/7])
        best_f = _score_ring_params(best_x, beta)
    
    return best_x, -best_f

def _local_update_polish(pts, max_iters=50, step=0.01):
    """
    Coordinate-level polishing: move one point at a time to improve the minimum triangle area.
    Uses fast local-update scoring.
    """
    pts = pts.copy()
    n = len(pts)
    
    for it in range(max_iters):
        if _time_remaining() < 3.0:
            break
            
        improved = False
        for i in range(n):
            best_move = None
            best_score = _calculate_score(pts)
            
            # Try small moves in x and y directions
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = pts.copy()
                    new_pts[i][0] += dx
                    new_pts[i][1] += dy
                    
                    # Quick check: ensure no points are too close
                    new_score = _calculate_score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_move = (dx, dy)
            
            if best_move is not None:
                pts[i][0] += best_move[0]
                pts[i][1] += best_move[1]
                improved = True
    
    if not improved:
        break
    
    return pts

def _coordinate_polish(pts, max_iters=100, step=0.005):
    """
    More aggressive coordinate polish with adaptive step size.
    """
    pts = pts.copy()
    n = len(pts)
    current_step = step
    
    for it in range(max_iters):
        if _time_remaining() < 3.0:
            break
        
        improved = False
        for i in range(n):
            best_move = None
            best_score = _calculate_score(pts)
            
            # Adaptive step: try larger and smaller steps
            for mult in [1.0, 0.5, 2.0]:
                s = current_step * mult
                for dx in [-s, 0, s]:
                    for dy in [-s, 0, s]:
                        if dx == 0 and dy == 0:
                            continue
                        new_pts = pts.copy()
                        new_pts[i][0] += dx
                        new_pts[i][1] += dy
                        new_score = _calculate_score(new_pts)
                        if new_score > best_score:
                            best_score = new_score
                            best_move = (dx, dy, mult)
            
            if best_move is not None:
                pts[i][0] += best_move[0]
                pts[i][1] += best_move[1]
                improved = True
                if best_move[2] > 1.0:
                    current_step = min(current_step * 1.5, 0.05)
                else:
                    current_step = max(current_step * 0.8, 0.001)
        
        if not improved:
            current_step *= 0.5
            if current_step < 0.0005:
                break
    
    return pts

def _optimize_2p_with_beta_annealing():
    """
    Optimize 2-parameter ring model with beta annealing.
    Start with low beta (smooth) and increase to high beta (sharper min).
    """
    if _time_remaining() < 10.0:
        # Just do a quick optimization
        x, score = _optimize_ring(beta=50.0)
        return x, score
    
    # Beta annealing schedule
    betas = [10.0, 20.0, 50.0, 100.0, 200.0]
    
    current_x = np.array([0.6, np.pi/7])
    current_score = -_score_ring_params(current_x, betas[0])
    
    for beta in betas:
        if _time_remaining() < 5.0:
            break
        
        # Optimize with current beta, starting from previous solution
        def neg_score(params):
            return _score_ring_params(params, beta)
        
        try:
            res = minimize(neg_score, current_x, method='L-BFGS-B', 
                          bounds=[(0.05, 1.0), (0, np.pi/7)],
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < -current_score:
                current_x = res.x
                current_score = -res.fun
        except Exception:
            pass
    
    return current_x, current_score

def _generate_jittered_points(seed=42):
    """Generate jittered regular polygon points."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2*np.pi, 14, endpoint=False)
    # Add small jitter to angles and radii
    angles_jitter = angles + rng.uniform(-0.05, 0.05, 14)
    radii = np.ones(14) + rng.uniform(-0.05, 0.05, 14)
    pts = np.column_stack([radii * np.cos(angles_jitter), radii * np.sin(angles_jitter)])
    return pts

def _soft_min_jittered_score(pts, beta=50.0):
    """Soft min score for jittered points."""
    return _soft_min_score(pts, beta)

def _optimize_jittered(beta=50.0):
    """Optimize jittered points using L-BFGS-B on all 28 coordinates."""
    if _time_remaining() < 15.0:
        return None, None
    
    def neg_score(flat):
        pts = flat.reshape(14, 2)
        return _soft_min_score(pts, beta)
    
    # Multi-start with different jitter seeds
    best_x = None
    best_f = float('inf')
    
    for seed in [42, 123, 456, 789]:
        if _time_remaining() < 5.0:
            break
        pts0 = _generate_jittered_points(seed)
        x0 = pts0.flatten()
        
        try:
            res = minimize(neg_score, x0, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            pass
    
    if best_x is None:
        return None, None
    
    return best_x.reshape(14, 2), -best_f

def construct():
    start_time = time.time()
    best_pts = None
    best_score = -float('inf')
    
    # Phase 1: Optimize 2-parameter ring model with beta annealing
    if _time_remaining() > 20.0:
        ring_params, ring_score = _optimize_2p_with_beta_annealing()
        if ring_params is not None:
            pts_ring = _get_points_2p(ring_params[0], ring_params[1])
            score_ring = _calculate_score(pts_ring)
            if score_ring > best_score:
                best_score = score_ring
                best_pts = pts_ring
    
    # Phase 2: Try jittered polygon optimization
    if _time_remaining() > 30.0:
        jittered_pts, jittered_score = _optimize_jittered(beta=50.0)
        if jittered_pts is not None:
            score_j = _calculate_score(jittered_pts)
            if score_j > best_score:
                best_score = score_j
                best_pts = jittered_pts
    
    # Phase 3: Coordinate polish on the best points found
    if best_pts is not None and _time_remaining() > 10.0:
        polished_pts = _coordinate_polish(best_pts, max_iters=150, step=0.005)
        score_polished = _calculate_score(polished_pts)
        if score_polished > best_score:
            best_score = score_polished
            best_pts = polished_pts
    
    # Phase 4: Final local update polish
    if best_pts is not None and _time_remaining() > 5.0:
        final_pts = _local_update_polish(best_pts, max_iters=30, step=0.002)
        score_final = _calculate_score(final_pts)
        if score_final > best_score:
            best_score = score_final
            best_pts = final_pts
    
    # Fallback: if no valid points found, use a simple regular polygon
    if best_pts is None:
        angles = np.linspace(0, 2*np.pi, 14, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts
