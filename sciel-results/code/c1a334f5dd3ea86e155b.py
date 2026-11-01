import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

def get_pairwise_sq_dists(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

def normalize_points(points):
    points = points - np.mean(points, axis=0)
    mean_norm_sq = np.mean(np.sum(points**2, axis=1))
    if mean_norm_sq > 1e-12:
        points = points / np.sqrt(mean_norm_sq)
    return points

def objective_smooth(x, tau=100.0):
    points = x.reshape(N, D)
    points = normalize_points(points)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    m_max = np.max(sq_dists)
    exp_max = np.exp(tau * (sq_dists - m_max))
    log_sum_exp_max = m_max + np.log(np.sum(exp_max)) / tau
    
    m_min = np.min(sq_dists)
    exp_min = np.exp(-tau * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min)) / tau
    
    return np.log(log_sum_exp_max) - np.log(log_sum_exp_min)

def generate_deformed_ico(delta, scale, deform):
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    ico = ico * scale
    
    # Apply deformation: compress z-axis
    ico[:, 2] *= deform
    
    p1 = np.array([0, 0, delta])
    p2 = np.array([0, 0, -delta])
    
    pts = np.vstack([ico, p1.reshape(1, D), p2.reshape(1, D)])
    return pts

def construct():
    start_time = time.time()
    timeout = 145.0
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        pts = normalize_points(pts)
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    candidates = []
    
    # 1. Standard Ico + 2 points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    for delta in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for scale in [0.8, 0.9, 1.0, 1.1]:
            for deform in [0.8, 0.9, 1.0, 1.1]:
                pts = generate_deformed_ico(delta, scale, deform)
                candidates.append(pts.flatten())
                
    # 2. Random
    for i in range(10):
        pts = rng.normal(size=(N, D))
        candidates.append(pts.flatten())
        
    # 3. Fibonacci
    fib_pts = np.zeros((N, D))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(N):
        z = 1.0 - 2.0 * i / (N - 1)
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * i
        fib_pts[i] = [r * np.cos(theta), r * np.sin(theta), z]
    candidates.append(fib_pts.flatten())

    for i in range(len(candidates)):
        if time.time() - start_time > timeout - 15:
            break
            
        x0 = candidates[i].copy()
        x0 = x0.reshape(N, D)
        x0 = normalize_points(x0)
        x0 = x0.flatten()
        
        try:
            taus = [10.0, 50.0, 200.0, 1000.0]
            current_x = x0
            for tau in taus:
                if time.time() - start_time > timeout - 5:
                    break
                res = minimize(objective_smooth, current_x, method='L-BFGS-B', args=(tau,),
                               options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.x is not None:
                    current_x = res.x
                    update_best(current_x.reshape(N, D))
        except Exception:
            pass

    if best_points is not None and time.time() - start_time < timeout - 10:
        try:
            x_polish = best_points.flatten()
            res = minimize(objective_smooth, x_polish, method='L-BFGS-B', args=(5000.0,),
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.x is not None:
                update_best(res.x.reshape(N, D))
        except Exception:
            pass

    if best_points is None:
        best_points = normalize_points(candidates[0].reshape(N, D))
        
    return best_points
