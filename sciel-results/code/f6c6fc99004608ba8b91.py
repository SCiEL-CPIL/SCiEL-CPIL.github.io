import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calc_score(points):
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    pi = points[TRIPLE_INDICES[:, 0]]
    pj = points[TRIPLE_INDICES[:, 1]]
    pk = points[TRIPLE_INDICES[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    min_area = 0.5 * np.min(np.abs(cross))
    return min_area / hull_area

def _soft_score(points, beta=120.0):
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    pi = points[TRIPLE_INDICES[:, 0]]
    pj = TRIPLE_INDICES[:, 1]
    pk = TRIPLE_INDICES[:, 2]
    pj = points[pj]
    pk = points[pk]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    ac = np.abs(cross)
    m = np.max(ac)
    ev = np.exp(-beta * (ac - m))
    sm = m + np.log(np.sum(ev)) / beta
    return 0.5 * sm / hull_area

def _soft_neg(points, beta=120.0):
    return -_soft_score(points, beta)

def _optimize_direct(points0, time_limit, beta=120.0):
    start = time.time()
    x = points0.flatten().copy()
    n = len(x)
    bounds = [(-1.2, 1.2)] * n
    best_x = x.copy()
    best_soft = _soft_neg(x.reshape(N, 2), beta)
    
    # Nelder-Mead with adaptive
    try:
        res = minimize(
            lambda v: _soft_neg(v.reshape(N, 2), beta),
            x,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True}
        )
        if res.x is not None:
            s = _soft_neg(res.x.reshape(N, 2), beta)
            if s < best_soft:
                best_soft = s
                best_x = res.x.copy()
    except Exception:
        pass
    
    if time.time() - start > time_limit:
        return best_x.reshape(N, 2)
    
    # Polish with increasing beta
    for b2 in [200.0, 300.0, 500.0]:
        if time.time() - start > time_limit:
            break
        try:
            res = minimize(
                lambda v: _soft_neg(v.reshape(N, 2), b2),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True}
            )
            if res.x is not None:
                s = _soft_neg(res.x.reshape(N, 2), b2)
                if s < best_soft:
                    best_soft = s
                    best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x.reshape(N, 2)

def _ring_points(n_out, n_in, r, theta):
    ao = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    ai = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
    return np.vstack([
        np.column_stack((np.cos(ao), np.sin(ao))),
        np.column_stack((r*np.cos(ai), r*np.sin(ai)))
    ])

def _three_ring_points(n1, n2, n3, r2, r3, phi2, phi3):
    a1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
    a2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + phi2
    a3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + phi3
    return np.vstack([
        np.column_stack((np.cos(a1), np.sin(a1))),
        np.column_stack((r2*np.cos(a2), r2*np.sin(a2))),
        np.column_stack((r3*np.cos(a3), r3*np.sin(a3)))
    ])

def _optimize_three_ring(n1, n2, n3, time_limit):
    start = time.time()
    best_score = 0.0
    best_pts = None
    
    # Multi-start Nelder-Mead on (r2, r3, phi2, phi3)
    starts = [
        (0.5, 0.25, 0.0, 0.0),
        (0.6, 0.3, 0.3, 0.1),
        (0.4, 0.2, 0.5, 0.2),
        (0.7, 0.35, 0.2, 0.4),
        (0.5, 0.4, 0.0, 0.5),
        (0.65, 0.3, 0.15, 0.25),
    ]
    
    for s in starts:
        if time.time() - start > time_limit:
            break
        def obj(v):
            r2, r3, phi2, phi3 = v
            if r2 < 0.05 or r2 > 0.95 or r3 < 0.05 or r3 > 0.95:
                return 1.0
            if abs(r2 - r3) < 0.05:
                return 1.0
            pts = _three_ring_points(n1, n2, n3, r2, r3, phi2, phi3)
            return -_soft_score(pts, 120.0)
        
        try:
            res = minimize(obj, s, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.x is not None:
                r2, r3, phi2, phi3 = res.x
                if 0.05 < r2 < 0.95 and 0.05 < r3 < 0.95 and abs(r2-r3) > 0.05:
                    pts = _three_ring_points(n1, n2, n3, r2, r3, phi2, phi3)
                    sc = _calc_score(pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = pts
        except Exception:
            pass
    
    # Polish best with direct coordinate optimization
    if best_pts is not None and time.time() - start < time_limit * 0.8:
        remaining = time_limit - (time.time() - start)
        if remaining > 5:
            polished = _optimize_direct(best_pts, remaining * 0.9, beta=120.0)
            sc = _calc_score(polished)
            if sc > best_score:
                best_score = sc
                best_pts = polished
    
    return best_score, best_pts

def _optimize_ring_direct(n_out, n_in, time_limit):
    start = time.time()
    # Start from a reasonable ring config, then do direct coordinate optimization
    r0, theta0 = 0.5, 0.0
    pts0 = _ring_points(n_out, n_in, r0, theta0)
    polished = _optimize_direct(pts0, time_limit, beta=120.0)
    return _calc_score(polished), polished

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_score = 0.0
    best_points = None
    
    # Phase 1: 7+7 ring as initial (fast)
    t1 = min(30.0, deadline - start_time - 5.0)
    if t1 > 5:
        pts77 = _ring_points(7, 7, 0.5, 0.0)
        sc = _calc_score(pts77)
        if sc > best_score:
            best_score = sc
            best_points = pts77
        # Polish with direct optimization
        if time.time() < deadline - 10:
            polished = _optimize_direct(pts77, min(20.0, deadline - time.time() - 5.0), beta=120.0)
            sc2 = _calc_score(polished)
            if sc2 > best_score:
                best_score = sc2
                best_points = polished
    
    # Phase 2: Three-ring families (structural novelty)
    three_ring_configs = [(1, 6, 7), (2, 5, 7), (3, 4, 7), (2, 4, 8)]
    for (n1, n2, n3) in three_ring_configs:
        remaining = deadline - time.time() - 3.0
        if remaining < 8:
            break
        t = min(25.0, remaining - 3.0)
        if t < 5:
            break
        sc, pts = _optimize_three_ring(n1, n2, n3, t)
        if sc > best_score:
            best_score = sc
            best_points = pts
    
    # Phase 3: Multi-start direct optimization from diverse seeds
    remaining = deadline - time.time() - 3.0
    if remaining > 15:
        np.random.seed(42)
        n_starts = min(4, int(remaining // 10))
        for i in range(n_starts):
            if time.time() > deadline - 5:
                break
            # Diverse random starts near circle
            angles = np.sort(np.random.uniform(0, 2*np.pi, N))
            radii = np.ones(N)
            # Perturb some to interior
            interior_idx = np.random.choice(N, size=N//2, replace=False)
            radii[interior_idx] = np.random.uniform(0.3, 0.8, size=N//2)
            pts_seed = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
            t = min(25.0, deadline - time.time() - 5.0)
            if t < 5:
                break
            polished = _optimize_direct(pts_seed, t, beta=100.0)
            sc = _calc_score(polished)
            if sc > best_score:
                best_score = sc
                best_points = polished
    
    # Phase 4: Final polish on best
    if best_points is not None:
        remaining = deadline - time.time() - 2.0
        if remaining > 5:
            polished = _optimize_direct(best_points, remaining, beta=200.0)
            sc = _calc_score(polished)
            if sc > best_score:
                best_score = sc
                best_points = polished
    
    if best_points is None:
        best_points = _ring_points(7, 7, 0.5, 0.0)
    
    return best_points
