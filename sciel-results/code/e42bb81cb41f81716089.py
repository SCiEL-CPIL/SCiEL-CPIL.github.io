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

def _calculate_score(points):
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
    """Soft minimum of triangle areas divided by hull area, for smoother optimization."""
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
    
    # Soft min approximation
    log_sum = np.log(np.sum(np.exp(-beta * areas)))
    soft_min = -log_sum / beta
    
    return soft_min / hull_area

def _get_points_2p(params):
    """
    Generate 14 points from a two-ring parameterization.
    params: [r1, r2, phase1, phase2, jitter_angles...] 
    Ring 1: 7 points at radius r1, angles phase1 + 2*pi*i/7
    Ring 2: 7 points at radius r2, angles phase2 + 2*pi*i/7
    Optional: small angular jitter for each point
    """
    r1 = params[0]
    r2 = params[1]
    phase1 = params[2]
    phase2 = params[3]
    
    n1 = 7
    n2 = 7
    
    angles1 = phase1 + 2 * np.pi * np.arange(n1) / n1
    angles2 = phase2 + 2 * np.pi * np.arange(n2) / n2
    
    x1 = r1 * np.cos(angles1)
    y1 = r1 * np.sin(angles1)
    x2 = r2 * np.cos(angles2)
    y2 = r2 * np.sin(angles2)
    
    points = np.column_stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])])
    return points

def _score_ring_params(params):
    """Negative score for ring parameter optimization."""
    try:
        pts = _get_points_2p(params)
        return -_calculate_score(pts)
    except Exception:
        return 1e6

def _soft_min_ring_score(params, beta=50.0):
    """Negative soft-min score for ring parameter optimization."""
    try:
        pts = _get_points_2p(params)
        return -_soft_min_score(pts, beta)
    except Exception:
        return 1e6

def _local_update_polish(points, max_iter=50, tol=1e-10):
    """Per-point coordinate descent polish."""
    best_score = _calculate_score(points)
    pts = points.copy()
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            # Try small perturbations in x and y
            best_local = best_score
            best_x = pts[i, 0]
            best_y = pts[i, 1]
            
            for dx in [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05]:
                for dy in [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05]:
                    trial = pts.copy()
                    trial[i, 0] += dx
                    trial[i, 1] += dy
                    s = _calculate_score(trial)
                    if s > best_local + tol:
                        best_local = s
                        best_x = trial[i, 0]
                        best_y = trial[i, 1]
            
            if abs(best_x - pts[i, 0]) > 1e-8 or abs(best_y - pts[i, 1]) > 1e-8:
                pts[i, 0] = best_x
                pts[i, 1] = best_y
                improved = True
        
        if not improved:
            break
        
        best_score = _calculate_score(pts)
        
        if time.time() > DEADLINE - 10:
            break
    
    return pts, best_score

def _coordinate_polish(points, max_iters=100):
    """Full coordinate descent with adaptive step sizes."""
    pts = points.copy()
    best_score = _calculate_score(pts)
    step = 0.1
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 10:
            break
            
        improved = False
        for i in range(N):
            best_local = best_score
            best_x = pts[i, 0]
            best_y = pts[i, 1]
            
            # Multi-scale step search
            for s in [step, step*0.5, step*0.25, step*2]:
                for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                    dx = s * np.cos(angle)
                    dy = s * np.sin(angle)
                    trial = pts.copy()
                    trial[i, 0] += dx
                    trial[i, 1] += dy
                    sc = _calculate_score(trial)
                    if sc > best_local + 1e-12:
                        best_local = sc
                        best_x = trial[i, 0]
                        best_y = trial[i, 1]
            
            if abs(best_x - pts[i, 0]) > 1e-10 or abs(best_y - pts[i, 1]) > 1e-10:
                pts[i, 0] = best_x
                pts[i, 1] = best_y
                improved = True
        
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
        
        best_score = _calculate_score(pts)
    
    return pts, best_score

def _optimize_2p(initial_params, max_iter=500):
    """Optimize two-ring parameters using Nelder-Mead."""
    try:
        res = minimize(
            _score_ring_params,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'maxfev': max_iter*2, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        opt_params = res.x
        opt_pts = _get_points_2p(opt_params)
        opt_score = _calculate_score(opt_pts)
        return opt_pts, opt_score
    except Exception:
        return None, 0.0

def _generate_jittered_points(base_points, jitter_magnitude=0.05, seed=42):
    """Generate jittered versions of base points."""
    rng = np.random.RandomState(seed)
    jittered = base_points + rng.randn(N, 2) * jitter_magnitude
    return jittered

def _soft_min_jittered_score(flat_params, beta=50.0):
    """
    Score for jittered polygon optimization.
    flat_params: [base_params..., jitter_angles...]
    We use the two-ring base + per-point angular jitter.
    """
    try:
        # First 4 params: r1, r2, phase1, phase2
        r1, r2, phase1, phase2 = flat_params[:4]
        # Next 14 params: angular jitter for each point
        jitters = flat_params[4:18]
        
        n1 = 7
        n2 = 7
        
        angles1 = phase1 + 2 * np.pi * np.arange(n1) / n1 + jitters[:7]
        angles2 = phase2 + 2 * np.pi * np.arange(n2) / n2 + jitters[7:]
        
        x1 = r1 * np.cos(angles1)
        y1 = r1 * np.sin(angles1)
        x2 = r2 * np.cos(angles2)
        y2 = r2 * np.sin(angles2)
        
        points = np.column_stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])])
        
        return -_soft_min_score(points, beta)
    except Exception:
        return 1e6

def _optimize_jittered(initial_params, max_iter=300, beta=50.0):
    """Optimize jittered two-ring parameters."""
    try:
        res = minimize(
            lambda x: _soft_min_jittered_score(x, beta),
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'maxfev': max_iter*2, 'xatol': 1e-7, 'fatol': 1e-9}
        )
        opt_params = res.x
        r1, r2, phase1, phase2 = opt_params[:4]
        jitters = opt_params[4:18]
        
        n1 = 7
        n2 = 7
        
        angles1 = phase1 + 2 * np.pi * np.arange(n1) / n1 + jitters[:7]
        angles2 = phase2 + 2 * np.pi * np.arange(n2) / n2 + jitters[7:]
        
        x1 = r1 * np.cos(angles1)
        y1 = r1 * np.sin(angles1)
        x2 = r2 * np.cos(angles2)
        y2 = r2 * np.sin(angles2)
        
        points = np.column_stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])])
        
        opt_score = _calculate_score(points)
        return points, opt_score
    except Exception:
        return None, 0.0

def construct():
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0
    
    np.random.seed(42)
    
    best_score = 0.0
    best_points = None
    
    # Generate initial candidates
    candidates = []
    
    # Two-ring configurations with various parameters
    r1_values = [0.5, 0.6, 0.7, 0.8]
    r2_values = [0.9, 1.0, 1.1, 1.2]
    phase_offsets = [0.0, np.pi/7, np.pi/14, 2*np.pi/7]
    
    for r1 in r1_values:
        for r2 in r2_values:
            for po in phase_offsets:
                params = [r1, r2, 0.0, po]
                pts = _get_points_2p(params)
                candidates.append(pts)
    
    # Also try some random two-ring configs
    for _ in range(20):
        r1 = np.random.uniform(0.4, 0.85)
        r2 = np.random.uniform(0.85, 1.3)
        p1 = np.random.uniform(0, 2*np.pi)
        p2 = np.random.uniform(0, 2*np.pi)
        params = [r1, r2, p1, p2]
        pts = _get_points_2p(params)
        candidates.append(pts)
    
    # Regular 14-gon as fallback
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    candidates.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Evaluate and sort candidates
    scored_candidates = []
    for c in candidates:
        s = _calculate_score(c)
        scored_candidates.append((s, c))
    
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Optimize top candidates
    top_k = min(8, len(scored_candidates))
    for i in range(top_k):
        if time.time() > DEADLINE - 30:
            break
            
        s, c = scored_candidates[i]
        if s > best_score:
            best_score = s
            best_points = c.copy()
        
        # Coordinate polish
        if time.time() < DEADLINE - 25:
            polished, polished_score = _coordinate_polish(c, max_iters=50)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished.copy()
    
    # Two-ring parameter optimization from best candidates
    if time.time() < DEADLINE - 25:
        # Extract ring parameters from top candidates and optimize
        for i in range(min(5, len(scored_candidates))):
            if time.time() > DEADLINE - 20:
                break
            
            s, c = scored_candidates[i]
            # Try to fit a two-ring model to this configuration
            # Use first 7 points as ring 1, last 7 as ring 2
            ring1 = c[:7]
            ring2 = c[7:]
            
            # Estimate radii
            r1_est = np.mean(np.sqrt(ring1[:, 0]**2 + ring1[:, 1]**2))
            r2_est = np.mean(np.sqrt(ring2[:, 0]**2 + ring2[:, 1]**2))
            
            # Estimate phases
            angles1 = np.arctan2(ring1[:, 1], ring1[:, 0])
            angles2 = np.arctan2(ring2[:, 1], ring2[:, 0])
            phase1_est = np.mean(angles1)
            phase2_est = np.mean(angles2)
            
            init_params = [r1_est, r2_est, phase1_est, phase2_est]
            
            opt_pts, opt_score = _optimize_2p(init_params, max_iter=300)
            if opt_pts is not None and opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
                
                # Polish the optimized result
                if time.time() < DEADLINE - 15:
                    polished, polished_score = _coordinate_polish(opt_pts, max_iters=50)
                    if polished_score > best_score:
                        best_score = polished_score
                        best_points = polished.copy()
    
    # Jittered two-ring optimization
    if time.time() < DEADLINE - 15:
        # Start from best two-ring configuration
        if best_points is not None:
            # Estimate initial jittered params
            ring1 = best_points[:7]
            ring2 = best_points[7:]
            r1_est = np.mean(np.sqrt(ring1[:, 0]**2 + ring1[:, 1]**2))
            r2_est = np.mean(np.sqrt(ring2[:, 0]**2 + ring2[:, 1]**2))
            angles1 = np.arctan2(ring1[:, 1], ring1[:, 0])
            angles2 = np.arctan2(ring2[:, 1], ring2[:, 0])
            phase1_est = np.mean(angles1)
            phase2_est = np.mean(angles2)
            
            # Compute initial jitters (deviation from regular spacing)
            expected_angles1 = phase1_est + 2*np.pi*np.arange(7)/7
            expected_angles2 = phase2_est + 2*np.pi*np.arange(7)/7
            jitter1 = angles1 - expected_angles1
            jitter2 = angles2 - expected_angles2
            # Unwrap to avoid jumps
            jitter1 = np.unwrap(jitter1)
            jitter2 = np.unwrap(jitter2)
            
            init_jittered = np.concatenate([[r1_est, r2_est, phase1_est, phase2_est], jitter1, jitter2])
            
            jit_pts, jit_score = _optimize_jittered(init_jittered, max_iter=200, beta=50.0)
            if jit_pts is not None and jit_score > best_score:
                best_score = jit_score
                best_points = jit_pts.copy()
                
                if time.time() < DEADLINE - 10:
                    polished, polished_score = _coordinate_polish(jit_pts, max_iters=50)
                    if polished_score > best_score:
                        best_score = polished_score
                        best_points = polished.copy()
    
    # Final polish
    if best_points is not None and time.time() < DEADLINE - 5:
        polished, polished_score = _coordinate_polish(best_points, max_iters=100)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished.copy()
    
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
