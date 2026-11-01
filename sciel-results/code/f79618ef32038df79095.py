import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

# Precompute triangle indices once
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

def compute_score(points):
    """Compute min triangle area / convex hull area, vectorized."""
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

def soft_neg_score(x, tau=0.01):
    """Soft-min objective for smoother landscape."""
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

def _ring_config_3ring(r1, r2, r3, th1, th2, th3, n1, n2, n3):
    """Generate points from 3-ring parameters."""
    n1 = max(1, int(round(n1)))
    n2 = max(0, int(round(n2)))
    n3 = max(0, int(round(n3)))
    
    # Adjust to sum to N
    total = n1 + n2 + n3
    while total > N:
        if n3 > 0:
            n3 -= 1
        elif n2 > 0:
            n2 -= 1
        else:
            n1 -= 1
        total = n1 + n2 + n3
    while total < N:
        n1 += 1
        total = n1 + n2 + n3
    
    pts = []
    if n1 > 0:
        a = th1 + np.linspace(0, 2*np.pi, n1, endpoint=False)
        pts.append(np.column_stack((r1*np.cos(a), r1*np.sin(a))))
    if n2 > 0:
        a = th2 + np.linspace(0, 2*np.pi, n2, endpoint=False)
        pts.append(np.column_stack((r2*np.cos(a), r2*np.sin(a))))
    if n3 > 0:
        a = th3 + np.linspace(0, 2*np.pi, n3, endpoint=False)
        pts.append(np.column_stack((r3*np.cos(a), r3*np.sin(a))))
    
    if not pts:
        return np.zeros((N, 2))
    
    result = np.vstack(pts)
    if len(result) < N:
        result = np.vstack([result, np.zeros((N-len(result), 2))])
    elif len(result) > N:
        result = result[:N]
    return result

def _neg_ring_score_3ring(params):
    try:
        pts = _ring_config_3ring(*params)
        return -compute_score(pts)
    except Exception:
        return 0.0

def _ring_config_4ring(r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4):
    """Generate points from 4-ring parameters."""
    n1 = max(1, int(round(n1)))
    n2 = max(0, int(round(n2)))
    n3 = max(0, int(round(n3)))
    n4 = max(0, int(round(n4)))
    
    total = n1 + n2 + n3 + n4
    while total > N:
        if n4 > 0: n4 -= 1
        elif n3 > 0: n3 -= 1
        elif n2 > 0: n2 -= 1
        else: n1 -= 1
        total = n1 + n2 + n3 + n4
    while total < N:
        n1 += 1
        total = n1 + n2 + n3 + n4
    
    pts = []
    if n1 > 0:
        a = th1 + np.linspace(0, 2*np.pi, n1, endpoint=False)
        pts.append(np.column_stack((r1*np.cos(a), r1*np.sin(a))))
    if n2 > 0:
        a = th2 + np.linspace(0, 2*np.pi, n2, endpoint=False)
        pts.append(np.column_stack((r2*np.cos(a), r2*np.sin(a))))
    if n3 > 0:
        a = th3 + np.linspace(0, 2*np.pi, n3, endpoint=False)
        pts.append(np.column_stack((r3*np.cos(a), r3*np.sin(a))))
    if n4 > 0:
        a = th4 + np.linspace(0, 2*np.pi, n4, endpoint=False)
        pts.append(np.column_stack((r4*np.cos(a), r4*np.sin(a))))
    
    if not pts:
        return np.zeros((N, 2))
    
    result = np.vstack(pts)
    if len(result) < N:
        result = np.vstack([result, np.zeros((N-len(result), 2))])
    elif len(result) > N:
        result = result[:N]
    return result

def _neg_ring_score_4ring(params):
    try:
        pts = _ring_config_4ring(*params)
        return -compute_score(pts)
    except Exception:
        return 0.0

def _optimize_full(points, maxiter=500):
    """Full 2D optimization of all 14 points."""
    x0 = points.flatten()
    try:
        res = minimize(neg_score, x0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'ftol': 1e-10})
        return res.x.reshape((N, 2))
    except Exception:
        return points

def construct():
    start_time = time.time()
    deadline = 142.0  # Safety margin before 150s kill

    best_points = None
    best_score = -float('inf')
    rng = np.random.default_rng(42)

    def try_update(points):
        nonlocal best_points, best_score
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    def remaining():
        return deadline - (time.time() - start_time)

    # --- Phase 1: Differential Evolution on 3-ring parameters (7,5,2 split) ---
    # Params: [r1, r2, r3, th1, th2, th3, n1, n2, n3]
    bounds_3ring = [
        (0.5, 1.5),  # r1 outer
        (0.0, 1.0),  # r2 mid
        (0.0, 0.5),  # r3 inner
        (0, 2*np.pi),  # th1
        (0, 2*np.pi),  # th2
        (0, 2*np.pi),  # th3
        (3, 10),  # n1
        (0, 8),  # n2
        (0, 5),  # n3
    ]
    
    for restart in range(3):
        if remaining() < 35:
            break
        try:
            res = differential_evolution(
                _neg_ring_score_3ring,
                bounds_3ring,
                seed=int(rng.integers(0, 2**31)),
                maxiter=60,
                tol=1e-5,
                popsize=10,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=True
            )
            if res.fun < 0:
                pts = _ring_config_3ring(*res.x)
                try_update(pts)
        except Exception:
            pass
    
    # --- Phase 2: Differential Evolution on 4-ring parameters ---
    bounds_4ring = [
        (0.5, 1.5),  # r1
        (0.0, 1.0),  # r2
        (0.0, 0.7),  # r3
        (0.0, 0.4),  # r4
        (0, 2*np.pi),  # th1
        (0, 2*np.pi),  # th2
        (0, 2*np.pi),  # th3
        (0, 2*np.pi),  # th4
        (2, 12),  # n1
        (0, 8),  # n2
        (0, 5),  # n3
        (0, 4),  # n4
    ]
    
    for restart in range(2):
        if remaining() < 30:
            break
        try:
            res = differential_evolution(
                _neg_ring_score_4ring,
                bounds_4ring,
                seed=int(rng.integers(0, 2**31)),
                maxiter=50,
                tol=1e-5,
                popsize=10,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=True
            )
            if res.fun < 0:
                pts = _ring_config_4ring(*res.x)
                try_update(pts)
        except Exception:
            pass
    
    # --- Phase 3: Diverse initial points ---
    inits = []
    
    # Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    p_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(p_reg)
    
    # Two concentric rings 7+7
    a_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
    for ratio in [0.4, 0.5, 0.6, 0.7]:
        for offset in [0, np.pi/7]:
            a_in = np.linspace(0, 2*np.pi, 7, endpoint=False) + offset
            p_out = np.column_stack((np.cos(a_out), np.sin(a_out)))
            p_in = np.column_stack((ratio*np.cos(a_in), ratio*np.sin(a_in)))
            inits.append(np.vstack((p_out, p_in)))
    
    # Three rings 7+5+2
    a1 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    a2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/7
    a3 = np.linspace(0, 2*np.pi, 2, endpoint=False) + np.pi/5
    for r2, r3 in [(0.6, 0.2), (0.5, 0.3), (0.7, 0.15), (0.55, 0.25)]:
        inits.append(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((r2*np.cos(a2), r2*np.sin(a2))),
            np.column_stack((r3*np.cos(a3), r3*np.sin(a3)))
        )))
    
    # Three rings 6+6+2
    a1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    a2 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    a3 = np.linspace(0, 2*np.pi, 2, endpoint=False) + np.pi/6
    for r2, r3 in [(0.6, 0.2), (0.5, 0.3), (0.7, 0.15)]:
        inits.append(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((r2*np.cos(a2), r2*np.sin(a2))),
            np.column_stack((r3*np.cos(a3), r3*np.sin(a3)))
        )))
    
    # Four rings 4+4+3+3
    a1 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    a2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
    a3 = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
    a4 = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
    inits.append(np.vstack((
        np.column_stack((np.cos(a1), np.sin(a1))),
        np.column_stack((0.65*np.cos(a2), 0.65*np.sin(a2))),
        np.column_stack((0.35*np.cos(a3), 0.35*np.sin(a3))),
        np.column_stack((0.1*np.cos(a4), 0.1*np.sin(a4)))
    )))
    
    # Jittered regular polygon
    for _ in range(3):
        inits.append(p_reg + rng.normal(0, 0.1, (N, 2)))
    
    # Random in disk
    for _ in range(5):
        r = np.sqrt(rng.random(N))
        th = rng.random(N) * 2*np.pi
        inits.append(np.column_stack((r*np.cos(th), r*np.sin(th))))
    
    # Evaluate all inits
    for p in inits:
        try_update(p)
    
    # Sort by score
    scored = [(compute_score(p), p) for p in inits]
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # --- Phase 4: Full 2D Nelder-Mead from top inits ---
    for init_score, init_points in scored[:8]:
        if remaining() < 15:
            break
        new_p = _optimize_full(init_points, maxiter=400)
        try_update(new_p)
    
    # --- Phase 5: Soft-min refinement from best ---
    if best_points is not None and remaining() > 25:
        for tau in [0.02, 0.01, 0.005]:
            if remaining() < 15:
                break
            x0 = best_points.flatten()
            try:
                res = minimize(lambda x: soft_neg_score(x, tau), x0, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-8, 'ftol': 1e-10})
                try_update(res.x.reshape((N, 2)))
            except Exception:
                pass
    
    # --- Phase 6: Basin hopping ---
    if best_points is not None and remaining() > 30:
        for step in [0.3, 0.5, 0.7]:
            if remaining() < 20:
                break
            x0 = best_points.flatten()
            res = minimize(neg_score, x0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-8, 'ftol': 1e-10})
            for _ in range(4):
                if remaining() < 12:
                    break
                perturbation = rng.normal(0, step, (N, 2))
                x_pert = res.x + perturbation.flatten()
                res2 = minimize(neg_score, x_pert, method='Nelder-Mead',
                                options={'maxiter': 150, 'xatol': 1e-8, 'ftol': 1e-10})
                if res2.fun < res.fun:
                    res = res2
            try_update(res.x.reshape((N, 2)))
    
    # --- Phase 7: Tight polish ---
    if best_points is not None and remaining() > 15:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'ftol': 1e-12})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
    
    # --- Phase 8: Final Powell polish ---
    if best_points is not None and remaining() > 8:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Powell',
                           options={'maxiter': 2000, 'xtol': 1e-10, 'ftol': 1e-12})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
    
    if best_points is None:
        return p_reg
    return best_points
