import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _score(points):
    """Calculate the packing radius for a set of points."""
    if len(points) != N:
        return -np.inf
    if not np.all(np.isfinite(points)):
        return -np.inf
    radii = np.linalg.norm(points, axis=1)
    boundary_dist = 1.0 - radii
    min_boundary = np.min(boundary_dist)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    return min(min_boundary, min_pair_dist)

def _soft_min(x, beta=20.0):
    """Soft minimum using log-sum-exp trick for stability."""
    y = -beta * x
    y_max = np.max(y)
    sum_exp = np.sum(np.exp(y - y_max))
    if sum_exp == 0:
        return -np.inf
    log_sum_exp = np.log(sum_exp) + y_max
    return -log_sum_exp / beta

def _ring_init(counts, radii_init=None, phases_init=None):
    """Generate points from ring structure."""
    if radii_init is None:
        k = len(counts)
        radii_init = np.linspace(0.05, 0.9, k)
    if phases_init is None:
        phases_init = np.zeros(len(counts))
    
    points = []
    for i, n_i in enumerate(counts):
        r_i = radii_init[i]
        phi_i = phases_init[i]
        if n_i == 1:
            angles = np.array([phi_i])
        else:
            angles = phi_i + np.arange(n_i) * (2 * np.pi / n_i)
        xs = r_i * np.cos(angles)
        ys = r_i * np.sin(angles)
        points.append(np.column_stack((xs, ys)))
    
    points = np.vstack(points)
    return points

def _optimize_rings(counts, beta=15.0, max_iter=300):
    """Optimize ring radii and phases using Nelder-Mead."""
    k = len(counts)
    x0 = np.zeros(2 * k)
    for i in range(k):
        x0[2*i] = 0.1 + 0.8 * i / max(k-1, 1)
        x0[2*i+1] = 0.0
    
    def objective(x):
        radii = np.clip(x[0::2], 0.0, 1.0)
        phases = x[1::2]
        points = _ring_init(counts, radii, phases)
        if len(points) != N:
            return 1e10
        radii_pts = np.linalg.norm(points, axis=1)
        boundary_dists = 1.0 - radii_pts
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        all_dists = np.concatenate([boundary_dists, pair_dists.ravel()])
        sm = _soft_min(all_dists, beta)
        return -sm
    
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': max_iter, 'maxfev': max_iter})
    
    radii_opt = np.clip(res.x[0::2], 0.0, 1.0)
    phases_opt = res.x[1::2]
    points = _ring_init(counts, radii_opt, phases_opt)
    score = _score(points)
    return points, score

def _optimize_free(points0, beta=50.0, max_iter=150):
    """Optimize free points using L-BFGS-B with soft-min objective."""
    def objective_and_grad(x):
        points = x.reshape(-1, 2)
        radii = np.linalg.norm(points, axis=1)
        boundary_dists = 1.0 - radii
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq + 1e-14)
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        all_dists = np.concatenate([boundary_dists, pair_dists.ravel()])
        sm = _soft_min(all_dists, beta)
        obj = -sm
        
        y = -beta * all_dists
        y_max = np.max(y)
        exp_y = np.exp(y - y_max)
        sum_exp = np.sum(exp_y)
        weights = exp_y / sum_exp
        
        grad_total = np.zeros((N, 2))
        
        # Boundary gradient
        w_bound = weights[:N]
        unit_vecs = np.zeros_like(points)
        mask = radii > 1e-10
        unit_vecs[mask] = points[mask] / radii[mask][:, np.newaxis]
        grad_total += -w_bound[:, np.newaxis] * unit_vecs
        
        # Pairwise gradient
        W = np.zeros((N, N))
        flat_w = weights[N:]
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                w = flat_w[idx]
                idx += 1
                W[i, j] = w
                W[j, i] = w
        
        M = np.zeros((N, N))
        mask_d = dists > 1e-10
        M[mask_d] = W[mask_d] / (2.0 * dists[mask_d])
        
        weighted_diff = M[:, :, np.newaxis] * diff
        grad_total += -np.sum(weighted_diff, axis=1)
        
        return obj, grad_total.ravel()
    
    x0 = points0.ravel()
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                   options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50})
    
    points_opt = res.x.reshape(-1, 2)
    score = _score(points_opt)
    return points_opt, score

def _generate_ring_structures():
    """Generate plausible ring count structures summing to N."""
    structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 17, 14],
        [1, 6, 12, 16, 15],
        [1, 6, 11, 18, 14],
        [1, 6, 13, 17, 13],
        [1, 7, 12, 17, 13],
        [1, 6, 12, 19, 12],
        [1, 6, 14, 16, 13],
        [1, 8, 12, 16, 13],
        [1, 6, 12, 15, 16],
    ]
    # Filter valid structures
    valid = []
    seen = set()
    for s in structures:
        if sum(s) == N and all(c > 0 for c in s):
            key = tuple(s)
            if key not in seen:
                seen.add(key)
                valid.append(s)
    return valid

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s limit
    
    structures = _generate_ring_structures()
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Optimize ring structures
    for counts in structures:
        if time.time() > deadline - 60.0:
            break
        try:
            points, score = _optimize_rings(counts, beta=15.0, max_iter=200)
            if score > best_score:
                best_score = score
                best_points = points
        except Exception:
            pass
    
    # If no ring structure worked, use hex lattice fallback
    if best_points is None or best_score < 0.1:
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points, radius = points[mask], radius[mask]
            if len(points) >= N:
                best_points = points[np.argsort(radius, kind='stable')[:N]]
                best_score = _score(best_points)
                break
            spacing *= 0.98
            if spacing < 0.1:
                best_points = np.random.uniform(-0.8, 0.8, (N, 2))
                best_score = _score(best_points)
                break
    
    # Phase 2: Free-point optimization with increasing beta
    betas = [20, 50, 100, 200, 500]
    for beta in betas:
        if time.time() > deadline - 15.0:
            break
        try:
            points_opt, score = _optimize_free(best_points, beta=beta, max_iter=100)
            if score > best_score:
                best_score = score
                best_points = points_opt
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            points_opt, score = _optimize_free(best_points, beta=1000, max_iter=50)
            if score > best_score:
                best_score = score
                best_points = points_opt
        except Exception:
            pass
    
    # Ensure validity
    if best_points is None or not np.all(np.isfinite(best_points)) or len(best_points) != N:
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points, radius = points[mask], radius[mask]
        if len(points) >= N:
            best_points = points[np.argsort(radius, kind='stable')[:N]]
        else:
            best_points = np.random.uniform(-0.8, 0.8, (N, 2))
    
    return best_points
