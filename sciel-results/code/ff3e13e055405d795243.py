import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

def construct():
    triples = list(itertools.combinations(range(N), 3))
    n_triples = len(triples)
    
    def get_points(x):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2] = [0.0, 1.0]
        for i in range(N - 3):
            pts[i + 3, 0] = x[2 * i]
            pts[i + 3, 1] = x[2 * i + 1]
        return pts
    
    def compute_areas(pts):
        areas = np.empty(n_triples)
        for idx, (i, j, k) in enumerate(triples):
            v1 = pts[j] - pts[i]
            v2 = pts[k] - pts[i]
            det = v1[0] * v2[1] - v1[1] * v2[0]
            areas[idx] = 0.5 * abs(det)
        return areas
    
    def compute_hull_area(pts):
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 1.0
    
    def smooth_min(areas, beta):
        a_max = np.max(areas)
        if a_max < 1e-15:
            return 0.0
        exp_terms = np.exp(-beta * (areas - a_max))
        log_sum = -beta * a_max + np.log(np.sum(exp_terms))
        return -log_sum / beta
    
    def objective(x, beta=100.0):
        pts = get_points(x)
        areas = compute_areas(pts)
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
        smin = smooth_min(areas, beta)
        return -(smin / hull_area)
    
    def raw_objective(x):
        pts = get_points(x)
        areas = compute_areas(pts)
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
        return -(np.min(areas) / hull_area)
    
    def init_circle(offset=0.0, r=1.0):
        x = np.zeros(2 * (N - 3))
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + offset
        for i in range(N - 3):
            idx = i + 3
            x[2 * i] = r * np.cos(angles[idx])
            x[2 * i + 1] = r * np.sin(angles[idx])
        return x
    
    def init_ellipse(rx=1.0, ry=0.7, offset=0.0):
        x = np.zeros(2 * (N - 3))
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + offset
        for i in range(N - 3):
            idx = i + 3
            x[2 * i] = rx * np.cos(angles[idx])
            x[2 * i + 1] = ry * np.sin(angles[idx])
        return x
    
    def init_random(scale=1.0, seed=0):
        rng = np.random.RandomState(seed)
        return rng.uniform(-scale, scale, size=2 * (N - 3))
    
    start_time = time.time()
    time_limit = 115.0
    dim = 2 * (N - 3)
    bounds = [(-3.0, 3.0)] * dim
    
    best_score = -np.inf
    best_x = None
    
    starts = []
    for off in np.linspace(0, 2 * np.pi / N, 8, endpoint=False):
        starts.append(init_circle(off, 1.0))
    for ry in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for off in [0.0, 0.3, 0.7, 1.1]:
            starts.append(init_ellipse(1.0, ry, off))
    for s in range(10):
        starts.append(init_random(1.5, seed=100 + s))
    
    betas = [20, 50, 100, 200, 400, 800, 1600, 3200]
    
    for beta in betas:
        if time.time() - start_time > time_limit - 15.0:
            break
        
        for x0 in starts:
            if time.time() - start_time > time_limit - 15.0:
                break
            
            try:
                res = minimize(
                    objective, x0, args=(beta,),
                    method='SLSQP', bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'eps': 1e-8}
                )
                if res.fun < -best_score:
                    best_score = -res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
            
            try:
                res2 = minimize(
                    objective, x0, args=(beta,),
                    method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res2.fun < -best_score:
                    best_score = -res2.fun
                    best_x = res2.x.copy()
            except Exception:
                pass
        
        if best_x is not None:
            new_starts = [best_x.copy()]
            rng = np.random.RandomState(int(time.time() * 1000) % 10000)
            for _ in range(5):
                jitter = rng.normal(0, 0.1, size=dim)
                new_starts.append(best_x + jitter)
            starts = new_starts + starts[:3]
    
    if time.time() - start_time < time_limit - 5.0 and best_x is not None:
        for beta_f in [5000, 10000]:
            if time.time() - start_time > time_limit - 3.0:
                break
            try:
                res_f = minimize(
                    objective, best_x, args=(beta_f,),
                    method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res_f.fun < -best_score:
                    best_score = -res_f.fun
                    best_x = res_f.x.copy()
            except Exception:
                pass
    
    if time.time() - start_time < time_limit - 2.0 and best_x is not None:
        try:
            res_nm = minimize(
                raw_objective, best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_nm.fun < -best_score:
                best_score = -res_nm.fun
                best_x = res_nm.x.copy()
        except Exception:
            pass
    
    if best_x is None:
        best_x = init_circle(0.0)
    
    return get_points(best_x)

# EVOLVE-BLOCK-END
