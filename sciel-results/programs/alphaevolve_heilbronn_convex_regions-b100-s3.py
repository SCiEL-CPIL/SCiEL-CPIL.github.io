import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def _all_triangle_areas(points):
    """Vectorized computation of all C(n,3) triangle areas."""
    n = len(points)
    if n < 3:
        return np.array([0.0])
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    return min_area / hull_area

def _soft_min_objective(params, beta=100.0):
    """Negative smooth min area / hull area.
    We minimize - (soft_min(areas) / hull_area).
    """
    points = params.reshape(-1, 2)
    areas = _all_triangle_areas(points)
    
    # Soft min: -logsumexp(-beta * areas) / beta
    # To avoid overflow in exp, we subtract max
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    # If max_val is very large negative, exp will underflow to 0, sum will be 0, log(0) is -inf.
    # But areas are positive, so neg_beta_areas are negative.
    lse = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min = -lse / beta
    
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 1e10
    
    return -soft_min / hull_area

def _two_shell_params_to_points(params):
    """
    Convert two-shell parameters to point coordinates.
    params = [r1, r2, phase1, phase2, n1, n2] where n1+n2=N
    n1 points on circle of radius r1 at angles phase1 + 2*pi*k/n1
    n2 points on circle of radius r2 at angles phase2 + 2*pi*k/n2
    """
    r1 = params[0]
    r2 = params[1]
    phase1 = params[2]
    phase2 = params[3]
    n1 = int(round(params[4]))
    n2 = N - n1
    
    if n1 < 1:
        n1 = 1
        n2 = N - 1
    if n2 < 1:
        n2 = 1
        n1 = N - 1
    if n1 + n2 != N:
        n2 = N - n1
    
    points = []
    for k in range(n1):
        angle = phase1 + 2 * np.pi * k / n1
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    for k in range(n2):
        angle = phase2 + 2 * np.pi * k / n2
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    return np.array(points)

def _two_shell_objective(params):
    """
    Objective for two-shell optimization.
    params = [r1, r2, phase1, phase2, n1_float]
    """
    points = _two_shell_params_to_points(params)
    
    # Normalize: translate to centroid
    points = points - np.mean(points, axis=0)
    
    # Scale: normalize by max distance from origin
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist < 1e-10:
        return 1e10
    points = points / max_dist
    
    score = _score(points)
    return -score

def _three_shell_params_to_points(params):
    """
    Convert three-shell parameters to point coordinates.
    params = [r1, r2, r3, phase1, phase2, phase3, n1, n2, n3] where n1+n2+n3=N
    """
    r1 = params[0]
    r2 = params[1]
    r3 = params[2]
    phase1 = params[3]
    phase2 = params[4]
    phase3 = params[5]
    n1 = int(round(params[6]))
    n2 = int(round(params[7]))
    n3 = N - n1 - n2
    
    # Clamp to valid ranges
    n1 = max(1, min(N-2, n1))
    n2 = max(1, min(N-n1-1, n2))
    n3 = N - n1 - n2
    if n3 < 1:
        n3 = 1
        n2 = N - n1 - n3
    
    points = []
    for k in range(n1):
        angle = phase1 + 2 * np.pi * k / n1
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    for k in range(n2):
        angle = phase2 + 2 * np.pi * k / n2
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    for k in range(n3):
        angle = phase3 + 2 * np.pi * k / n3
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    return np.array(points)

def _three_shell_objective(params):
    """
    Objective for three-shell optimization.
    params = [r1, r2, r3, phase1, phase2, phase3, n1_float, n2_float]
    """
    points = _three_shell_params_to_points(params)
    
    # Normalize: translate to centroid
    points = points - np.mean(points, axis=0)
    
    # Scale: normalize by max distance from origin
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist < 1e-10:
        return 1e10
    points = points / max_dist
    
    score = _score(points)
    return -score

def _generate_two_shell_starts():
    """Generate initial parameter vectors for two-shell optimization."""
    starts = []
    rng = np.random.default_rng(123)
    
    # Try different splits: (1,12), (2,11), (3,10), (4,9), (5,8), (6,7)
    splits = [(1, 12), (2, 11), (3, 10), (4, 9), (5, 8), (6, 7)]
    
    for n1, n2 in splits:
        # Deterministic starts
        for r1 in [0.3, 0.5, 0.7]:
            for r2 in [0.8, 1.0, 1.2]:
                for phase1 in [0, np.pi/n1]:
                    for phase2 in [0, np.pi/n2]:
                        starts.append([r1, r2, phase1, phase2, float(n1)])
        
        # Random starts
        for _ in range(5):
            r1 = rng.uniform(0.2, 0.8)
            r2 = rng.uniform(0.7, 1.3)
            phase1 = rng.uniform(0, 2*np.pi)
            phase2 = rng.uniform(0, 2*np.pi)
            starts.append([r1, r2, phase1, phase2, float(n1)])
    
    return starts

def _generate_three_shell_starts():
    """Generate initial parameter vectors for three-shell optimization."""
    starts = []
    rng = np.random.default_rng(456)
    
    # Try different splits
    splits = [(1, 1, 11), (1, 2, 10), (1, 3, 9), (2, 2, 9), (2, 3, 8), (3, 3, 7), (3, 4, 6), (4, 4, 5)]
    
    for n1, n2, n3 in splits:
        # Deterministic starts
        for r1 in [0.2, 0.4]:
            for r2 in [0.6, 0.8]:
                for r3 in [1.0, 1.2]:
                    for phase1 in [0]:
                        for phase2 in [0, np.pi/n2]:
                            for phase3 in [0, np.pi/n3]:
                                starts.append([r1, r2, r3, phase1, phase2, phase3, float(n1), float(n2)])
        
        # Random starts
        for _ in range(3):
            r1 = rng.uniform(0.1, 0.5)
            r2 = rng.uniform(0.5, 0.9)
            r3 = rng.uniform(0.9, 1.3)
            phase1 = rng.uniform(0, 2*np.pi)
            phase2 = rng.uniform(0, 2*np.pi)
            phase3 = rng.uniform(0, 2*np.pi)
            starts.append([r1, r2, r3, phase1, phase2, phase3, float(n1), float(n2)])
    
    return starts

def _generate_regular_polygon_points():
    """Generate regular N-gon points."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def construct():
    t0 = time.time()
    deadline = 135.0  # Safety margin
    
    best_points = None
    best_score = -1e10
    
    # Start with regular polygon as baseline
    regular = _generate_regular_polygon_points()
    reg_score = _score(regular)
    best_points = regular.copy()
    best_score = reg_score
    
    # Two-shell optimization
    two_shell_starts = _generate_two_shell_starts()
    
    for i, start in enumerate(two_shell_starts):
        if time.time() - t0 > deadline:
            break
        
        try:
            res = minimize(_two_shell_objective, start, method='Nelder-Mead',
                           options={'maxiter': 200, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.x is not None:
                cand = _two_shell_params_to_points(res.x)
                cand_score = _score(cand)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand.copy()
        except Exception:
            pass
    
    # Three-shell optimization if time permits
    if time.time() - t0 < deadline - 30.0:
        three_shell_starts = _generate_three_shell_starts()
        
        for i, start in enumerate(three_shell_starts):
            if time.time() - t0 > deadline:
                break
            
            try:
                res = minimize(_three_shell_objective, start, method='Nelder-Mead',
                               options={'maxiter': 150, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.x is not None:
                    cand = _three_shell_params_to_points(res.x)
                    cand_score = _score(cand)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_points = cand.copy()
            except Exception:
                pass
    
    # Now try the soft-min annealing approach from the best point found so far
    if time.time() - t0 < deadline - 20.0:
        # Optimization 1: L-BFGS-B with numerical gradients using soft-min
        # Start from the best point found so far
        try:
            x0 = best_points.flatten()
            # Use a moderate beta for L-BFGS-B to ensure smoothness
            res1 = minimize(_soft_min_objective, x0, args=(50.0,), method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-9})
            
            if time.time() - t0 > deadline:
                return best_points
                
            candidate = res1.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate.copy()
        except Exception:
            pass
        
        # Optimization 2: Powell method (derivative-free, often good for such problems)
        # Start from the best point found so far
        if time.time() - t0 < deadline:
            try:
                x0 = best_points.flatten()
                res2 = minimize(_soft_min_objective, x0, args=(100.0,), method='Powell',
                                options={'maxiter': 300, 'ftol': 1e-12, 'xtol': 1e-10})
                
                if time.time() - t0 > deadline:
                    return best_points
                    
                candidate = res2.x.reshape(-1, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate.copy()
            except Exception:
                pass
        
        # Final refinement with true min using Nelder-Mead
        if time.time() - t0 < deadline:
            try:
                def _true_min_objective(params):
                    points = params.reshape(-1, 2)
                    score = _score(points)
                    return -score
                
                x0 = best_points.flatten()
                res3 = minimize(_true_min_objective, x0, method='Nelder-Mead',
                                options={'maxiter': 200, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12})
                
                if time.time() - t0 > deadline:
                    return best_points
                    
                candidate = res3.x.reshape(-1, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate.copy()
            except Exception:
                pass
    
    if best_points is None:
        best_points = _generate_regular_polygon_points()
    
    return best_points
