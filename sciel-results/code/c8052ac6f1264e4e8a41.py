import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import time

N = 14

# Precompute triangle indices for vectorized area calculation
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

def compute_score(points):
    """
    Compute min triangle area / convex hull area, vectorized.
    """
    n = len(points)
    if n < 3:
        return 0.0
    try:
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return 0.0
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

def smooth_min_score(points, temperature=0.05):
    """
    Smooth surrogate using log-sum-exp for the min triangle area.
    This gives smoother gradients for optimization.
    """
    n = len(points)
    if n < 3:
        return 0.0
    try:
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return 0.0
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
    
    # Log-sum-exp for smooth min
    max_area = np.max(areas)
    if max_area < 1e-12:
        return 0.0
    log_sum = np.log(np.sum(np.exp(-(areas - max_area) / temperature))) + (-max_area / temperature)
    smooth_min = -temperature * log_sum
    return smooth_min / hull_area

def neg_score_full(x):
    """
    Objective function for full 28-dimensional optimization (exact).
    """
    points = x.reshape((N, 2))
    return -compute_score(points)

def neg_smooth_score(x, temperature=0.05):
    """
    Objective function using smooth surrogate.
    """
    points = x.reshape((N, 2))
    return -smooth_min_score(points, temperature)

def ring_configuration(params):
    """
    Generate points from ring parameters.
    params: [r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4]
    n1+n2+n3+n4 must be 14.
    """
    r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4 = params
    n1 = int(round(n1))
    n2 = int(round(n2))
    n3 = int(round(n3))
    n4 = int(round(n4))
    
    total = n1 + n2 + n3 + n4
    if total != 14:
        diff = 14 - total
        n4 = max(0, n4 + diff)
        while n1 + n2 + n3 + n4 < 14:
            n4 += 1
        while n1 + n2 + n3 + n4 > 14:
            if n4 > 1: n4 -= 1
            elif n3 > 1: n3 -= 1
            elif n2 > 1: n2 -= 1
            else: n1 -= 1
    
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

def heptagon_plus_seven(r_out=1.0, r_in=0.5, th_out=0.0, th_in=np.pi/7):
    """
    Two concentric heptagons: 7 outer + 7 inner points.
    """
    angles_out = th_out + np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_in = th_in + np.linspace(0, 2*np.pi, 7, endpoint=False)
    
    outer = np.column_stack((r_out*np.cos(angles_out), r_out*np.sin(angles_out)))
    inner = np.column_stack((r_in*np.cos(angles_in), r_in*np.sin(angles_in)))
    
    return np.vstack([outer, inner])

def neg_ring_score(params):
    """
    Objective function for ring parameter optimization.
    """
    try:
        points = ring_configuration(params)
        return -compute_score(points)
    except Exception:
        return 0.0

def neg_heptagon_score(params):
    """
    Objective for heptagon+7 configuration: [r_out, r_in, th_out, th_in]
    """
    try:
        r_out, r_in, th_out, th_in = params
        points = heptagon_plus_seven(r_out, r_in, th_out, th_in)
        return -compute_score(points)
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_points = None
    best_score = -float('inf')
    rng = np.random.default_rng(42)
    
    def try_update(points):
        nonlocal best_points, best_score
        if points is None:
            return
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    # Phase 1a: Try structured initializations
    # Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    try_update(np.column_stack((np.cos(angles), np.sin(angles))))
    
    # Heptagon + 7
    try_update(heptagon_plus_seven(1.0, 0.5, 0.0, np.pi/7))
    try_update(heptagon_plus_seven(1.0, 0.4, 0.0, 0.0))
    try_update(heptagon_plus_seven(1.0, 0.6, 0.0, np.pi/7))
    
    # Two rings: 7+7 with offset
    for r2 in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for offset in [0.0, np.pi/7, np.pi/14]:
            try_update(heptagon_plus_seven(1.0, r2, 0.0, offset))
    
    # Phase 1b: Global Search using DE on Ring Parameters (tighter bounds)
    if time.time() < deadline - 35:
        # Configuration 1: 2 rings (7+7)
        bounds_2ring = [
            (0.9, 1.1),   # r1
            (0.2, 0.8),   # r2
            (0.0, 0.0),   # r3 (unused)
            (0.0, 0.0),   # r4 (unused)
            (0, 2*np.pi), # th1
            (0, 2*np.pi), # th2
            (0, 0),       # th3 (unused)
            (0, 0),       # th4 (unused)
            (7, 7),       # n1
            (7, 7),       # n2
            (0, 0),       # n3
            (0, 0),       # n4
        ]
        
        try:
            de_res = differential_evolution(
                neg_ring_score,
                bounds_2ring,
                seed=42,
                maxiter=30,
                tol=1e-4,
                popsize=12,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            if de_res.fun < 0:
                points = ring_configuration(de_res.x)
                try_update(points)
        except Exception:
            pass
    
    # Phase 1c: DE on heptagon+7 parameters
    if time.time() < deadline - 30:
        bounds_hept = [
            (0.8, 1.2),   # r_out
            (0.1, 0.9),   # r_in
            (0, 2*np.pi), # th_out
            (0, 2*np.pi), # th_in
        ]
        
        try:
            de_res = differential_evolution(
                neg_heptagon_score,
                bounds_hept,
                seed=123,
                maxiter=30,
                tol=1e-4,
                popsize=12,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            if de_res.fun < 0:
                points = heptagon_plus_seven(*de_res.x)
                try_update(points)
        except Exception:
            pass
    
    # Phase 2: Local Refinement with smooth surrogate, then exact
    if best_points is not None and time.time() < deadline - 25:
        x0 = best_points.flatten()
        
        # Step 2a: L-BFGS-B with smooth surrogate for faster convergence
        try:
            res = minimize(
                lambda x: neg_smooth_score(x, 0.05),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < -best_score:
                s = compute_score(res.x.reshape((N, 2)))
                if s > best_score:
                    best_score = s
                    best_points = res.x.reshape((N, 2))
        except Exception:
            pass
        
        # Step 2b: Nelder-Mead with exact objective
        nm_options = {
            'maxiter': 3000,
            'xatol': 1e-9,
            'fatol': 1e-9,
            'disp': False
        }
        
        try:
            res = minimize(neg_score_full, best_points.flatten(), method='Nelder-Mead', options=nm_options)
            if res.fun < -best_score:
                best_score = -res.fun
                best_points = res.x.reshape((N, 2))
        except Exception:
            pass
        
        # Step 2c: Random perturbations to escape local minima
        for trial in range(5):
            if time.time() > deadline - 12:
                break
            
            noise = rng.normal(0, 0.03, size=(N, 2))
            x0_perturbed = (best_points + noise).flatten()
            
            try:
                # Smooth first
                res1 = minimize(
                    lambda x: neg_smooth_score(x, 0.03),
                    x0_perturbed,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10}
                )
                # Then exact
                res2 = minimize(neg_score_full, res1.x, method='Nelder-Mead', options=nm_options)
                if res2.fun < -best_score:
                    best_score = -res2.fun
                    best_points = res2.x.reshape((N, 2))
            except Exception:
                pass

    # Phase 3: Final high-precision polish
    if best_points is not None and time.time() < deadline - 3:
        x0 = best_points.flatten()
        polish_options = {
            'maxiter': 5000,
            'xatol': 1e-11,
            'fatol': 1e-11,
            'disp': False
        }
        try:
            res = minimize(neg_score_full, x0, method='Nelder-Mead', options=polish_options)
            if res.fun < -best_score:
                best_score = -res.fun
                best_points = res.x.reshape((N, 2))
        except Exception:
            pass

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
