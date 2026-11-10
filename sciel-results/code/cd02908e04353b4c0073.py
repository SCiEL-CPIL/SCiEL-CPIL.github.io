import numpy as np
import time
from scipy.optimize import differential_evolution, minimize, basinhopping

def construct():
    start_time = time.time()
    deadline = start_time + 147.0  # Leave 3s for safety

    n = 8
    pair_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_indices.append((i, j))
    pair_indices = np.array(pair_indices)
    idx_i = pair_indices[:, 0]
    idx_j = pair_indices[:, 1]
    num_pairs = len(pair_indices)  # 28

    LOG_2 = np.log(2.0)
    LOG_8 = np.log(8.0)
    CONST = 56.0 * LOG_2 - 8.0 * LOG_8

    def objective(x):
        if time.time() > deadline:
            return 1e10
        pts = np.zeros((n, 2))
        pts[1:] = x.reshape(n - 1, 2)
        
        diff = pts[idx_i, :] - pts[idx_j, :]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        
        if np.any(dists < 1e-12):
            return 1e10
        
        D = np.max(dists)
        if D < 1e-12:
            return 1e10
        
        sum_log_d = np.sum(np.log(dists))
        log_score = 2.0 * sum_log_d + CONST - 56.0 * np.log(D)
        return -log_score

    starts = []
    
    # 1. Regular octagon (point 0 at origin)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    oct_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    oct_pts = oct_pts - oct_pts[0]
    starts.append(oct_pts[1:].flatten())
    
    # 2. Two concentric squares (various radii and rotations)
    for r1 in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
        for r2 in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
            for theta in [0.0, np.pi/8, np.pi/4, np.pi/3, np.pi/6]:
                p1 = []
                p2 = []
                for k in range(4):
                    a1 = k * np.pi / 2
                    p1.append([r1 * np.cos(a1), r1 * np.sin(a1)])
                    a2 = theta + k * np.pi / 2
                    p2.append([r2 * np.cos(a2), r2 * np.sin(a2)])
                pts = np.array(p1 + p2)
                pts = pts - pts[0]
                starts.append(pts[1:].flatten())
    
    # 3. Nested squares with rotation
    for r1 in [0.5, 1.0, 1.5]:
        for r2 in [0.5, 1.0, 1.5]:
            for rot in [np.pi/8, np.pi/4, np.pi/6]:
                p1 = []
                p2 = []
                for k in range(4):
                    a1 = rot + k * np.pi / 2
                    p1.append([r1 * np.cos(a1), r1 * np.sin(a1)])
                    a2 = k * np.pi / 2
                    p2.append([r2 * np.cos(a2), r2 * np.sin(a2)])
                pts = np.array(p1 + p2)
                pts = pts - pts[0]
                starts.append(pts[1:].flatten())
    
    # 4. Random perturbations of octagon
    np.random.seed(42)
    oct_flat = starts[0]
    for _ in range(10):
        noise = np.random.normal(0, 0.3, 14)
        starts.append(oct_flat + noise)
    
    # 5. Random starts
    for _ in range(15):
        starts.append(np.random.uniform(-5, 5, 14))
    
    # 6. Points on two circles with varying angles
    for r1 in [0.5, 1.0]:
        for r2 in [1.0, 1.5]:
            for phase in [0, np.pi/4, np.pi/3]:
                pts = []
                for k in range(4):
                    a = phase + k * np.pi / 2
                    pts.append([r1 * np.cos(a), r1 * np.sin(a)])
                for k in range(4):
                    a = -phase + k * np.pi / 2
                    pts.append([r2 * np.cos(a), r2 * np.sin(a)])
                pts = np.array(pts)
                pts = pts - pts[0]
                starts.append(pts[1:].flatten())

    best_x = starts[0]
    best_val = objective(best_x)
    
    for s in starts:
        if time.time() > deadline - 15.0:
            break
        try:
            res = minimize(
                objective,
                s,
                method='Powell',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    if time.time() < deadline - 25.0:
        try:
            bounds = [(-10.0, 10.0)] * 14
            res_de = differential_evolution(
                objective,
                bounds,
                maxiter=80,
                popsize=20,
                tol=1e-8,
                seed=42,
                init='sobol'
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_x = res_de.x
        except Exception:
            pass

    if time.time() < deadline - 10.0:
        try:
            def step(x):
                noise = np.random.normal(0, 0.1, len(x))
                return x + noise
            
            res_bh = basinhopping(
                objective,
                best_x,
                stepfunc=step,
                niter=30,
                minimizer_kwargs={'method': 'Powell', 'options': {'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10}},
                seed=42
            )
            if res_bh.fun < best_val:
                best_val = res_bh.fun
                best_x = res_bh.x
        except Exception:
            pass

    if time.time() < deadline - 3.0:
        try:
            res_l = minimize(
                objective,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if res_l.fun < best_val:
                best_val = res_l.fun
                best_x = res_l.x
        except Exception:
            pass

    if time.time() < deadline - 1.0:
        try:
            res_p = minimize(
                objective,
                best_x,
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            if res_p.fun < best_val:
                best_val = res_p.fun
                best_x = res_p.x
        except Exception:
            pass

    pts = np.zeros((n, 2))
    pts[1:] = best_x.reshape(n - 1, 2)
    
    return [[float(p[0]), float(p[1])] for p in pts]
