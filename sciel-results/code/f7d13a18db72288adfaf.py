import numpy as np
import time
import scipy.optimize

N = 300

def _golden_spiral(n):
    points = np.zeros((n, 2))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        r = np.sqrt(i + 1) / np.sqrt(n)
        theta = i * golden_angle
        points[i, 0] = r * np.cos(theta)
        points[i, 1] = r * np.sin(theta)
    return points

def _objective_and_grad(x):
    n = len(x) // 2
    pts = x.reshape(n, 2)
    
    r = np.linalg.norm(pts, axis=1)
    r_safe = np.clip(r, 0.0, 0.999999)
    boundary_cost = np.sum(1.0 / (1.0 - r_safe))
    
    grad_boundary = np.zeros_like(pts)
    for i in range(n):
        if r[i] > 1e-6:
            factor = 1.0 / (r[i] * (1.0 - r[i])**2)
            grad_boundary[i] = pts[i] * factor
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    pair_cost = np.sum(1.0 / dists)
    
    dists_safe = np.where(np.isfinite(dists), dists, 1.0)
    inv_dists_cubed = 1.0 / (dists_safe**3)
    inv_dists_cubed = np.where(np.isfinite(dists), inv_dists_cubed, 0.0)
    
    term = diff * inv_dists_cubed[:, :, np.newaxis]
    grad_pair = -np.sum(term, axis=1)
    
    total_cost = boundary_cost + 10.0 * pair_cost
    total_grad = grad_boundary + 10.0 * grad_pair
    
    return total_cost, total_grad.flatten()

def _calc_score(points):
    n = len(points)
    r = np.linalg.norm(points, axis=1)
    boundary_r = np.min(1.0 - r)
    if boundary_r < 0:
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(boundary_r, min_pair_dist / 2.0)

def construct():
    start_time = time.time()
    deadline = 140.0
    
    x0 = _golden_spiral(N).flatten()
    
    best_points = x0.reshape(N, 2)
    best_score = _calc_score(best_points)
    
    try:
        result = scipy.optimize.minimize(
            _objective_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-9, 'gtol': 1e-9}
        )
        
        if result.fun < _objective_and_grad(x0)[0]:
            new_points = result.x.reshape(N, 2)
            new_score = _calc_score(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        # Try hex lattice
        spacing = 0.1
        pts = []
        for j in range(-20, 21):
            for i in range(-20, 21):
                x = i * spacing
                y = j * spacing * np.sqrt(3) / 2.0
                if j % 2 != 0:
                    x += spacing / 2.0
                if np.hypot(x, y) < 0.9:
                    pts.append([x, y])
        
        if len(pts) >= N:
            pts = np.array(pts)
            radii = np.linalg.norm(pts, axis=1)
            indices = np.argsort(radii)[:N]
            x0_2 = pts[indices].flatten()
            
            try:
                result2 = scipy.optimize.minimize(
                    _objective_and_grad, 
                    x0_2, 
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-9, 'gtol': 1e-9}
                )
                if result2.fun < _objective_and_grad(x0_2)[0]:
                    new_points2 = result2.x.reshape(N, 2)
                    new_score2 = _calc_score(new_points2)
                    if new_score2 > best_score:
                        best_points = new_points2
                        best_score = new_score2
            except Exception:
                pass
    
    return best_points
