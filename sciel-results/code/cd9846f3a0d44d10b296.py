import numpy as np
from scipy.optimize import minimize
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def pairwise_dists_sq(pts):
        """Return all pairwise squared distances (upper triangle)."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return d_sq[iu]

    def exact_score(pts):
        d_sq = pairwise_dists_sq(pts)
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        if min_d_sq < 1e-12:
            return 1e10
        return max_d_sq / min_d_sq

    def quantile_objective(pts_flat, k_inner, weight_outer):
        """
        Objective that penalizes:
        1. Variance in the k_inner smallest distances (inner cluster tightness)
        2. The largest distance relative to the inner scale
        
        This encourages a hierarchical structure where inner distances are
        nearly equal, and the outer distance is as small as possible.
        """
        pts = pts_flat.reshape(N, D)
        d_sq = pairwise_dists_sq(pts)
        
        min_d_sq = np.min(d_sq)
        if min_d_sq < 1e-12:
            return 1e10
        
        # Sort distances
        d_sorted = np.sort(d_sq)
        
        # Inner cluster: the k_inner smallest distances
        inner = d_sorted[:k_inner]
        inner_mean = np.mean(inner)
        inner_var = np.var(inner)
        
        # Outer: the largest distance
        outer = d_sorted[-1]
        
        # Normalize by inner scale
        inner_scale = np.sqrt(inner_mean)
        if inner_scale < 1e-8:
            return 1e10
        
        # Penalty: variance in inner distances (relative) + outer distance (relative)
        var_penalty = inner_var / (inner_mean ** 2)  # relative variance
        outer_penalty = outer / inner_mean  # ratio of outer to inner scale
        
        return weight_outer * outer_penalty + (1.0 - weight_outer) * 10.0 * var_penalty

    def annealed_objective(pts_flat, k_inner, weight_outer, temperature):
        """
        Temperature-controlled version: at high temperature, focus on outer penalty;
        at low temperature, focus on inner variance.
        """
        pts = pts_flat.reshape(N, D)
        d_sq = pairwise_dists_sq(pts)
        
        min_d_sq = np.min(d_sq)
        if min_d_sq < 1e-12:
            return 1e10
        
        d_sorted = np.sort(d_sq)
        inner = d_sorted[:k_inner]
        inner_mean = np.mean(inner)
        inner_var = np.var(inner)
        outer = d_sorted[-1]
        
        inner_scale = np.sqrt(inner_mean)
        if inner_scale < 1e-8:
            return 1e10
        
        var_penalty = inner_var / (inner_mean ** 2)
        outer_penalty = outer / inner_mean
        
        # Temperature controls the balance
        t = np.clip(temperature, 0.0, 1.0)
        w_outer = t * weight_outer
        w_inner = (1.0 - t) * (1.0 - weight_outer)
        
        # At high temp: emphasize outer; at low temp: emphasize inner variance
        return w_outer * outer_penalty + w_inner * 10.0 * var_penalty

    # --- Initializers ---
    
    def make_hex_cluster(n):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3) / 2])
        pts = []
        for i in range(-6, 7):
            for j in range(-6, 7):
                pts.append(i * v1 + j * v2)
        pts = np.array(pts)
        pts -= pts.mean(axis=0)
        dists = np.linalg.norm(pts, axis=1)
        order = np.argsort(dists)
        return pts[order[:n]]

    def make_ring_config(n, r_inner, r_outer, n_inner, n_outer):
        pts = [np.array([0.0, 0.0])]
        for k in range(n_inner):
            theta = 2 * np.pi * k / n_inner
            pts.append(r_inner * np.array([np.cos(theta), np.sin(theta)]))
        for k in range(n_outer):
            theta = 2 * np.pi * k / n_outer + np.pi / n_outer
            pts.append(r_outer * np.array([np.cos(theta), np.sin(theta)]))
        return np.array(pts[:n])

    def make_grid(n):
        x, y = np.meshgrid(np.arange(4), np.arange(4))
        pts = np.column_stack([x.ravel(), y.ravel()]).astype(float)
        pts -= pts.mean(axis=0)
        return pts

    def make_random(seed):
        rng = np.random.default_rng(seed)
        pts = rng.random((n, 2)) * 4 - 2
        pts -= pts.mean(axis=0)
        return pts

    candidates = [
        make_hex_cluster(N),
        make_ring_config(N, 1.0, 1.8, 6, 9),
        make_ring_config(N, 0.8, 1.5, 5, 10),
        make_grid(N),
        make_random(1),
        make_random(2),
        make_random(3),
        make_random(4),
    ]

    best_pts = None
    best_score = np.inf

    def update_best(pts, score):
        nonlocal best_pts, best_score
        if score < best_score:
            best_score = score
            best_pts = pts.copy()

    # Evaluate initial candidates
    for c in candidates:
        s = exact_score(c)
        update_best(c, s)

    bounds = [(-6.0, 6.0)] * (N * D)

    # --- Multi-stage optimization with annealing ---
    
    for c in candidates:
        if time.time() > deadline - 15:
            break
        
        current = c.flatten().copy()
        
        # Stage 1: Coarse optimization with high temperature (focus on outer distance)
        if time.time() < deadline - 40:
            try:
                res = minimize(
                    annealed_objective,
                    current,
                    args=(8, 0.7, 0.9),  # k_inner=8, weight_outer=0.7, temp=0.9
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-5}
                )
                if res.fun < 1e9:
                    current = res.x
                    s = exact_score(current.reshape(N, D))
                    update_best(current.reshape(N, D), s)
            except Exception:
                pass
        
        # Stage 2: Medium temperature (balance inner and outer)
        if time.time() < deadline - 20:
            try:
                res = minimize(
                    annealed_objective,
                    current,
                    args=(10, 0.5, 0.5),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-6}
                )
                if res.fun < 1e9:
                    current = res.x
                    s = exact_score(current.reshape(N, D))
                    update_best(current.reshape(N, D), s)
            except Exception:
                pass
        
        # Stage 3: Low temperature (focus on inner variance)
        if time.time() < deadline - 10:
            try:
                res = minimize(
                    annealed_objective,
                    current,
                    args=(12, 0.3, 0.1),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 800, 'ftol': 1e-10, 'gtol': 1e-7}
                )
                if res.fun < 1e9:
                    current = res.x
                    s = exact_score(current.reshape(N, D))
                    update_best(current.reshape(N, D), s)
            except Exception:
                pass
        
        # Stage 4: Direct minimization of exact score for final polish
        if time.time() < deadline - 5:
            try:
                res = minimize(
                    lambda x: exact_score(x.reshape(N, D)),
                    current,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-7}
                )
                if res.fun < 1e9:
                    current = res.x
                    s = exact_score(current.reshape(N, D))
                    update_best(current.reshape(N, D), s)
            except Exception:
                pass

    # Additional random restarts with annealing if time permits
    for i in range(8):
        if time.time() > deadline - 8:
            break
        
        if i % 2 == 0 and best_pts is not None:
            rng = np.random.default_rng(100 + i)
            c = best_pts + rng.normal(0, 0.3, (N, D))
        else:
            c = make_random(200 + i)
        
        current = c.flatten().copy()
        
        # Quick two-stage annealing
        if time.time() < deadline - 15:
            try:
                res1 = minimize(
                    annealed_objective,
                    current,
                    args=(10, 0.6, 0.7),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5}
                )
                if res1.fun < 1e9:
                    current = res1.x
            except Exception:
                pass
        
        if time.time() < deadline - 8:
            try:
                res2 = minimize(
                    annealed_objective,
                    current,
                    args=(12, 0.4, 0.2),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 400, 'ftol': 1e-9, 'gtol': 1e-6}
                )
                if res2.fun < 1e9:
                    current = res2.x
                    s = exact_score(current.reshape(N, D))
                    update_best(current.reshape(N, D), s)
            except Exception:
                pass

    if best_pts is None:
        best_pts = make_hex_cluster(N)

    return best_pts
