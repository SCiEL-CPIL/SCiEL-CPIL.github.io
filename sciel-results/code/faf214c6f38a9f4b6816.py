import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools
import time
import random

N = 14

def construct():
    start_time = time.time()
    time_limit = 115.0

    # Precompute all triangle index triples
    triples = list(itertools.combinations(range(N), 3))
    n_triples = len(triples)

    def get_points(x):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts

    def compute_areas(pts):
        areas = np.zeros(n_triples)
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

    def score(pts):
        areas = compute_areas(pts)
        min_area = np.min(areas)
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 0.0
        return min_area / hull_area

    def neg_score(x):
        pts = get_points(x)
        return -score(pts)

    # Log-sum-exp smoothed objective
    def smooth_obj(x, beta=500.0):
        pts = get_points(x)
        areas = compute_areas(pts)
        a_max = np.max(areas)
        if a_max < 1e-15:
            return 1.0
        exp_terms = np.exp(-beta * (areas - a_max))
        log_sum = -beta * a_max + np.log(np.sum(exp_terms))
        smooth_min = -log_sum / beta
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
        return -(smooth_min / hull_area)

    # Generate diverse initial points
    def init_circle(offset=0.0, r=1.0):
        x = np.zeros(2 * (N - 2))
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + offset
        for i in range(N - 2):
            idx = i + 2
            x[2 * i] = r * np.cos(angles[idx])
            x[2 * i + 1] = r * np.sin(angles[idx])
        return x

    def init_ellipse(rx=1.0, ry=0.7, offset=0.0):
        x = np.zeros(2 * (N - 2))
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + offset
        for i in range(N - 2):
            idx = i + 2
            x[2 * i] = rx * np.cos(angles[idx])
            x[2 * i + 1] = ry * np.sin(angles[idx])
        return x

    def init_spiral(tightness=0.3, r=1.0):
        x = np.zeros(2 * (N - 2))
        for i in range(N - 2):
            idx = i + 2
            t = idx / (N - 2)
            angle = t * 2 * np.pi * 2
            rad = r * (1 - tightness * t)
            x[2 * i] = rad * np.cos(angle)
            x[2 * i + 1] = rad * np.sin(angle)
        return x

    dim = 2 * (N - 2)
    bounds = [(-3.0, 3.0)] * dim

    best_x = None
    best_val = -np.inf

    def try_improve(x_candidate):
        nonlocal best_x, best_val
        v = score(get_points(x_candidate))
        if v > best_val:
            best_val = v
            best_x = x_candidate.copy()

    # Build diverse initial pool
    pool = []
    for off in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        pool.append(init_circle(off, 1.0))
    for ry in [0.5, 0.65, 0.8, 0.9, 0.95]:
        for off in [0.0, 0.4, 0.8, 1.2]:
            pool.append(init_ellipse(1.0, ry, off))
    for t in [0.2, 0.4, 0.6, 0.8]:
        pool.append(init_spiral(t, 1.0))

    # Evaluate pool
    for x0 in pool:
        try_improve(x0)

    # Phase 1: Differential Evolution global search
    if time.time() - start_time < time_limit - 30.0:
        try:
            de_res = differential_evolution(
                neg_score,
                bounds,
                seed=42,
                maxiter=40,
                popsize=18,
                tol=1e-5,
                init='sobol',
                polish=True,
                mutation=(0.5, 1.5),
                recombination=0.7
            )
            try_improve(de_res.x)
        except Exception:
            pass

    # Phase 2: Multi-start L-BFGS with log-sum-exp annealing
    if best_x is not None:
        # Add jittered versions of best
        for _ in range(8):
            jitter = np.random.normal(0, 0.12, size=dim)
            pool.append(best_x + jitter)

    betas = [30.0, 60.0, 120.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]

    for beta in betas:
        if time.time() - start_time > time_limit - 25.0:
            break

        for x0 in pool[:20]:
            if time.time() - start_time > time_limit - 20.0:
                break
            try:
                res = minimize(
                    smooth_obj,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 6000}
                )
                try_improve(res.x)
            except Exception:
                pass

        # Refresh pool with best and small jitters
        if best_x is not None:
            new_pool = [best_x.copy()]
            for _ in range(3):
                new_pool.append(best_x + np.random.normal(0, 0.06, size=dim))
            pool = new_pool + pool[:12]

    # Phase 3: High-precision refinement
    if time.time() - start_time < time_limit - 10.0 and best_x is not None:
        for beta_hi in [5000.0, 10000.0, 20000.0]:
            if time.time() - start_time > time_limit - 8.0:
                break
            try:
                res_hi = minimize(
                    smooth_obj,
                    best_x,
                    args=(beta_hi,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-13}
                )
                try_improve(res_hi.x)
            except Exception:
                pass

    # Phase 4: Nelder-Mead on raw objective
    if time.time() - start_time < time_limit - 4.0 and best_x is not None:
        try:
            res_nm = minimize(
                neg_score,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            try_improve(res_nm.x)
        except Exception:
            pass

    # Phase 5: Final polish
    if time.time() - start_time < time_limit - 1.0 and best_x is not None:
        try:
            res_polish = minimize(
                smooth_obj,
                best_x,
                args=(30000.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 4000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            try_improve(res_polish.x)
        except Exception:
            pass

    if best_x is None:
        best_x = init_circle(0.0)

    return get_points(best_x)
