import numpy as np
import time
from scipy.optimize import differential_evolution, minimize, LinearConstraint
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 120.0  # Safety margin before 150s kill

# Precompute triangle indices
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

def compute_score(points):
    """Compute min triangle area / convex hull area."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area

def neg_score(x):
    points = x.reshape((N, 2))
    return -compute_score(points)

def soft_min_score(x, tau=0.05):
    """
    Compute -soft_min(areas) / hull_area.
    soft_min(A) = -tau * log(sum(exp(-A/tau)))
    """
    points = x.reshape((N, 2))
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    areas = np.maximum(areas, 1e-12)
        
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    return -soft_min_val / hull_area

def heptagon_7_7_configuration(params):
    """
    Generate points from two concentric heptagons.
    params: [r1, r2, th1, th2]
    r1: radius of outer heptagon
    r2: radius of inner heptagon
    th1: rotation of outer heptagon
    th2: rotation of inner heptagon
    """
    r1, r2, th1, th2 = params
    
    angles1 = th1 + np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles2 = th2 + np.linspace(0, 2*np.pi, 7, endpoint=False)
    
    points1 = np.column_stack((r1*np.cos(angles1), r1*np.sin(angles1)))
    points2 = np.column_stack((r2*np.cos(angles2), r2*np.sin(angles2)))
    
    return np.vstack([points1, points2])

def neg_heptagon_7_7_score(params):
    try:
        points = heptagon_7_7_configuration(params)
        return -compute_score(points)
    except Exception:
        return 1.0

def ring_configuration(params):
    """
    Generate points from ring parameters.
    params: [r1, r2, r3, r4, theta1, theta2, theta3, theta4, n1, n2, n3, n4]
    """
    r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4 = params
    
    # Round n's to integers, ensuring they sum to 14
    n1_i = int(round(n1))
    n2_i = int(round(n2))
    n3_i = int(round(n3))
    n4_i = int(round(n4))
    
    # Adjust to sum to 14
    total = n1_i + n2_i + n3_i + n4_i
    diff = 14 - total
    
    if diff != 0:
        n4_i += diff
        if n4_i < 1:
            n3_i += (1 - n4_i)
            n4_i = 1
    
    counts = [n1_i, n2_i, n3_i, n4_i]
    for i in range(4):
        if counts[i] < 1:
            counts[i] = 1
    
    while sum(counts) < 14:
        idx = 3
        while idx >= 0 and counts[idx] >= 10:
            idx -= 1
        if idx < 0:
            counts[0] += 1
        else:
            counts[idx] += 1
            
    while sum(counts) > 14:
        idx = 3
        while idx >= 0 and counts[idx] <= 1:
            idx -= 1
        if idx < 0:
            counts[0] -= 1
        else:
            counts[idx] -= 1
            
    n1, n2, n3, n4 = counts

    points = []
    
    if n1 > 0:
        angles = th1 + np.linspace(0, 2*np.pi, n1, endpoint=False)
        points.append(np.column_stack((r1*np.cos(angles), r1*np.sin(angles))))
    
    if n2 > 0:
        angles = th2 + np.linspace(0, 2*np.pi, n2, endpoint=False)
        points.append(np.column_stack((r2*np.cos(angles), r2*np.sin(angles))))
    
    if n3 > 0:
        angles = th3 + np.linspace(0, 2*np.pi, n3, endpoint=False)
        points.append(np.column_stack((r3*np.cos(angles), r3*np.sin(angles))))
    
    if n4 > 0:
        angles = th4 + np.linspace(0, 2*np.pi, n4, endpoint=False)
        points.append(np.column_stack((r4*np.cos(angles), r4*np.sin(angles))))
    
    if not points:
        return np.zeros((N, 2))
    
    result = np.vstack(points)
    
    if len(result) < N:
        extra = np.zeros((N - len(result), 2))
        result = np.vstack([result, extra])
    elif len(result) > N:
        result = result[:N]
    
    return result

def neg_ring_score(params):
    try:
        points = ring_configuration(params)
        return -compute_score(points)
    except Exception:
        return 1.0

def dumbbell_configuration(params):
    """
    Generate points from 7 dumbbell pairs.
    params: [cx1, cy1, cx2, cy2, ..., cx7, cy7, vx, vy]
    First 14 params are pair centers, last 2 are the offset vector (vx, vy).
    Point 2i = center_i + v/2
    Point 2i+1 = center_i - v/2
    """
    centers = params[:14].reshape((7, 2))
    v = params[14:16]
    
    # Create 14 points
    points = np.zeros((14, 2))
    for i in range(7):
        points[2*i] = centers[i] + v / 2
        points[2*i+1] = centers[i] - v / 2
    
    return points

def neg_dumbbell_score(params):
    try:
        points = dumbbell_configuration(params)
        return -compute_score(points)
    except Exception:
        return 1.0

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 10  # Safety margin
    
    best_points = None
    best_score = -float('inf')
    rng = np.random.default_rng(12345)
    
    def try_update(points):
        nonlocal best_points, best_score
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    # --- Phase 1: Heptagon 7+7 Global Search ---
    
    bounds_77 = [
        (0.5, 2.0),  # r1
        (0.0, 1.5),  # r2
        (0, 2*np.pi), # th1
        (0, 2*np.pi), # th2
    ]
    
    try:
        res_de_77 = differential_evolution(
            neg_heptagon_7_7_score,
            bounds_77,
            seed=42,
            maxiter=150,
            tol=1e-8,
            popsize=20,
            mutation=(0.3, 1.2),
            recombination=0.9,
            polish=True,
            init='sobol'
        )
        
        if res_de_77.fun < 0:
            best_params_77 = res_de_77.x
            points_77 = heptagon_7_7_configuration(best_params_77)
            try_update(points_77)
    except Exception:
        pass

    # Local refinement of best 7+7
    if best_points is not None and time.time() < deadline - 20:
        # Nelder-Mead on hard min
        x0 = best_points.flatten()
        try:
            res_nm = minimize(
                neg_score, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            try_update(res_nm.x.reshape((N, 2)))
        except Exception:
            pass
        
        # Powell on hard min
        if time.time() < deadline - 10:
            x0 = best_points.flatten()
            try:
                res_powell = minimize(
                    neg_score, 
                    x0, 
                    method='Powell',
                    options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-12}
                )
                try_update(res_powell.x.reshape((N, 2)))
            except Exception:
                pass

    # --- Phase 2: 4-Ring Global Search ---
    
    bounds_ring = [
        (0.3, 1.5),  # r1
        (0.1, 1.0),  # r2
        (0.0, 0.7),  # r3
        (0.0, 0.4),  # r4
        (0, 2*np.pi), # th1
        (0, 2*np.pi), # th2
        (0, 2*np.pi), # th3
        (0, 2*np.pi), # th4
        (1, 13),  # n1
        (1, 13),  # n2
        (1, 13),  # n3
        (1, 13),  # n4
    ]
    
    if time.time() < deadline - 30:
        try:
            res_de_ring = differential_evolution(
                neg_ring_score,
                bounds_ring,
                seed=43,
                maxiter=80,
                tol=1e-6,
                popsize=15,
                mutation=(0.3, 1.2),
                recombination=0.9,
                polish=True,
                init='sobol'
            )
            
            if res_de_ring.fun < 0:
                best_params_ring = res_de_ring.x
                points_ring = ring_configuration(best_params_ring)
                try_update(points_ring)
        except Exception:
            pass

    # --- Phase 3: Dumbbell 7-pair Global Search ---
    
    # Initialize centers as a heptagon, v as small
    centers_init = heptagon_7_7_configuration([1.0, 0.0, 0, 0])[:7]
    v_init = [0.05, 0.0]
    
    bounds_dumbbell = []
    for i in range(7):
        bounds_dumbbell.append((centers_init[i, 0] - 1.0, centers_init[i, 0] + 1.0))
        bounds_dumbbell.append((centers_init[i, 1] - 1.0, centers_init[i, 1] + 1.0))
    bounds_dumbbell.append((-0.5, 0.5))  # vx
    bounds_dumbbell.append((-0.5, 0.5))  # vy
    
    if time.time() < deadline - 20:
        try:
            res_de_dumbbell = differential_evolution(
                neg_dumbbell_score,
                bounds_dumbbell,
                seed=44,
                maxiter=50,
                tol=1e-6,
                popsize=15,
                mutation=(0.3, 1.2),
                recombination=0.9,
                polish=True,
                init='sobol'
            )
            
            if res_de_dumbbell.fun < 0:
                best_params_dumbbell = res_de_dumbbell.x
                points_dumbbell = dumbbell_configuration(best_params_dumbbell)
                try_update(points_dumbbell)
        except Exception:
            pass

    # --- Phase 4: Final Local Refinement ---
    
    if best_points is not None and time.time() < deadline - 5:
        x0 = best_points.flatten()
        
        # L-BFGS-B on soft min
        try:
            res_lbfgs = minimize(
                lambda x: soft_min_score(x, tau=0.01), 
                x0, 
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            try_update(res_lbfgs.x.reshape((N, 2)))
        except Exception:
            pass
        
        # Nelder-Mead on hard min
        if time.time() < deadline - 3:
            x0 = best_points.flatten()
            try:
                res_nm = minimize(
                    neg_score, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11}
                )
                try_update(res_nm.x.reshape((N, 2)))
            except Exception:
                pass
        
        # Powell on hard min
        if time.time() < deadline - 2:
            x0 = best_points.flatten()
            try:
                res_powell = minimize(
                    neg_score, 
                    x0, 
                    method='Powell',
                    options={'maxiter': 500, 'xtol': 1e-11, 'ftol': 1e-13}
                )
                try_update(res_powell.x.reshape((N, 2)))
            except Exception:
                pass

    if best_points is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
