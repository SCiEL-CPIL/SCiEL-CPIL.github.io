import numpy as np
import time
from scipy.optimize import minimize
from functools import partial

N = 14
D = 3

def get_pairwise_sq_dists(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-15:
        return np.inf
    return np.max(sq_dists) / np.min(sq_dists)

def objective_smooth(x, tau=50.0):
    """
    Smooth log-ratio objective: log(softmax(sq_dists)) - log(softmin(sq_dists))
    Using log-sum-exp for numerical stability.
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Soft max: log(sum(exp(sq_dists))) - log(N_pairs) ... but we want the actual soft max value
    # soft_max = max + log(sum(exp(x - max))) / tau
    m = np.max(sq_dists)
    log_sum_exp_max = m + np.log(np.sum(np.exp(tau * (sq_dists - m)))) / tau
    
    # Soft min: min - log(sum(exp(-(x - min)))) / tau
    m2 = np.min(sq_dists)
    log_sum_exp_min = m2 - np.log(np.sum(np.exp(-tau * (sq_dists - m2)))) / tau
    
    log_sum_exp_min = np.maximum(log_sum_exp_min, 1e-15)
    
    return np.log(log_sum_exp_max) - np.log(log_sum_exp_min)

def fibonacci_sphere(n_points, radius=1.0):
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_points):
        y = 1.0 - (i / (n_points - 1)) * 2.0
        r = np.sqrt(max(0, 1.0 - y**2))
        theta = golden_angle * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        points.append([x * radius, y * radius, z * radius])
    return np.array(points)

def icosahedron_14():
    """14-point config: 12 icosahedron vertices + 2 at center (slightly offset)"""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico = ico / norms
    
    # Two points near center, slightly offset to avoid coincidence
    center1 = np.array([0.05, 0.0, 0.0])
    center2 = np.array([-0.05, 0.0, 0.0])
    
    pts = np.vstack([ico, center1.reshape(1,3), center2.reshape(1,3)])
    return pts

def rhombic_dodecahedron_14():
    """Rhombic dodecahedron has 14 vertices: 6 axis + 8 face centers"""
    # 6 vertices on axes
    axis_pts = np.array([
        [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]
    ], dtype=float)
    
    # 8 vertices at (±1/2, ±1/2, ±1/2) scaled
    face_pts = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                face_pts.append([sx*0.5, sy*0.5, sz*0.5])
    face_pts = np.array(face_pts)
    
    pts = np.vstack([axis_pts, face_pts])
    # Normalize to unit sphere for consistent scale
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts

def generate_candidates():
    candidates = []
    
    # 1. Icosahedron + 2 centers
    candidates.append(icosahedron_14().flatten())
    
    # 2. Rhombic dodecahedron
    candidates.append(rhombic_dodecahedron_14().flatten())
    
    # 3. Fibonacci sphere
    fib = fibonacci_sphere(N, radius=1.0)
    candidates.append(fib.flatten())
    
    # 4. Random sphere points with various seeds
    for seed in range(20):
        rng = np.random.default_rng(seed)
        pts = rng.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        candidates.append(pts.flatten())
    
    # 5. Random points in ball
    for seed in range(10):
        rng = np.random.default_rng(1000 + seed)
        pts = rng.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        radii = rng.uniform(0.3, 1.0, size=(N, 1))
        pts = pts / norms * radii
        candidates.append(pts.flatten())
    
    # 6. Perturbed icosahedron
    rng = np.random.default_rng(999)
    ico = icosahedron_14()
    for _ in range(5):
        perturbed = ico + rng.normal(size=(N, D)) * 0.05
        candidates.append(perturbed.flatten())
    
    # 7. Perturbed Fibonacci
    for _ in range(5):
        perturbed = fib + rng.normal(size=(N, D)) * 0.05
        candidates.append(perturbed.flatten())
    
    # 8. Perturbed rhombic dodecahedron
    rd = rhombic_dodecahedron_14()
    for _ in range(5):
        perturbed = rd + rng.normal(size=(N, D)) * 0.05
        candidates.append(perturbed.flatten())
    
    # 9. Two-shell sphere: 7 inner + 7 outer
    for seed in range(5):
        rng = np.random.default_rng(2000 + seed)
        inner = rng.normal(size=(7, D))
        inner = inner / np.linalg.norm(inner, axis=1, keepdims=True) * 0.5
        outer = rng.normal(size=(7, D))
        outer = outer / np.linalg.norm(outer, axis=1, keepdims=True) * 1.0
        pts = np.vstack([inner, outer])
        candidates.append(pts.flatten())
    
    return candidates

def optimize_from(x0, time_budget):
    """Run progressive tau optimization from x0 within time_budget."""
    x_curr = x0.copy()
    start = time.time()
    
    # Progressive tau schedule
    tau_schedule = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
    maxiter_schedule = [200, 200, 200, 200, 150, 100]
    
    for idx, tau in enumerate(tau_schedule):
        if time.time() - start > time_budget - 5.0:
            break
        
        obj = partial(objective_smooth, tau=tau)
        try:
            res = minimize(
                obj,
                x_curr,
                method='L-BFGS-B',
                options={'maxiter': maxiter_schedule[idx], 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if res.x is not None and np.isfinite(res.fun):
                x_curr = res.x
        except Exception:
            pass
    
    # Final SLSQP polish with high tau
    if time.time() - start < time_budget - 2.0:
        try:
            obj_final = partial(objective_smooth, tau=5000.0)
            res_polish = minimize(
                obj_final,
                x_curr,
                method='SLSQP',
                options={'maxiter': 100, 'ftol': 1e-16, 'maxls': 30}
            )
            if res_polish.x is not None and np.isfinite(res_polish.fun):
                x_curr = res_polish.x
        except Exception:
            pass
    
    return x_curr

def construct():
    start_time = time.time()
    total_timeout = 140.0  # Safety margin before 150s kill
    
    best_score = np.inf
    best_points = None
    
    candidates = generate_candidates()
    
    # Score all candidates initially
    candidate_scores = []
    for i, cx in enumerate(candidates):
        pts = cx.reshape(N, D)
        s = raw_score(pts)
        candidate_scores.append((s, i, cx))
    
    # Sort by score, keep top 15 for optimization
    candidate_scores.sort(key=lambda x: x[0])
    top_candidates = [cx for _, _, cx in candidate_scores[:15]]
    
    # Also keep the best initial
    if candidate_scores[0][0] < best_score:
        best_score = candidate_scores[0][0]
        best_points = candidate_scores[0][2].reshape(N, D).copy()
    
    # Optimize top candidates with time allocation
    n_opt = len(top_candidates)
    time_per_opt = (total_timeout - 10.0) / max(1, n_opt)
    
    for i, x0 in enumerate(top_candidates):
        if time.time() - start_time > total_timeout - 5.0:
            break
        
        remaining = total_timeout - (time.time() - start_time)
        alloc = min(time_per_opt, remaining - 3.0)
        if alloc < 1.0:
            break
        
        try:
            x_opt = optimize_from(x0, alloc)
            pts = x_opt.reshape(N, D)
            s = raw_score(pts)
            if np.isfinite(s) and s < best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # If we have time, do a few more random restarts with optimization
    if time.time() - start_time < total_timeout - 15.0:
        rng = np.random.default_rng(777)
        for _ in range(5):
            if time.time() - start_time > total_timeout - 10.0:
                break
            pts = rng.normal(size=(N, D))
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            x0 = pts.flatten()
            
            try:
                alloc = min(15.0, total_timeout - (time.time() - start_time) - 5.0)
                if alloc < 2.0:
                    break
                x_opt = optimize_from(x0, alloc)
                pts = x_opt.reshape(N, D)
                s = raw_score(pts)
                if np.isfinite(s) and s < best_score:
                    best_score = s
                    best_points = pts.copy()
            except Exception:
                pass
    
    # Final validation
    if best_points is None:
        best_points = icosahedron_14()
    
    sq_dists = get_pairwise_sq_dists(best_points)
    if np.min(sq_dists) < 1e-8:
        best_points = icosahedron_14()
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    sq_dists = get_pairwise_sq_dists(pts)
    score = (np.max(sq_dists) / np.min(sq_dists))
    print(f"Score: {score}")
