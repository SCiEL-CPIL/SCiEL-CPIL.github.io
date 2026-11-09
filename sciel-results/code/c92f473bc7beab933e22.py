import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 600

    def score(pts):
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, d):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x):
        pts = x.reshape((n, d))
        if not np.all(np.isfinite(pts)):
            return 1.0
        s = score(pts)
        return -s

    # Generate a diverse set of initial configurations
    def fibonacci_sphere(n_pts):
        indices = np.arange(0, n_pts, dtype=float) + 0.5
        phi_f = np.arccos(1 - 2 * indices / n_pts)
        theta = np.pi * (1 + 5**0.5) * indices
        x = np.cos(theta) * np.sin(phi_f)
        y = np.sin(theta) * np.sin(phi_f)
        z = np.cos(phi_f)
        return np.column_stack([x, y, z])

    # Icosahedron + 2 extra
    phi = (1 + np.sqrt(5)) / 2
    ico12 = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico12 = ico12 / np.linalg.norm(ico12, axis=1, keepdims=True)

    best_points = None
    best_score = -np.inf

    def update_best(pts):
        nonlocal best_points, best_score
        if pts is None or not np.all(np.isfinite(pts)):
            return
        s = score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # Strategy 1: Differential Evolution (global optimization)
    # Use bounds to keep points in a reasonable range
    bounds = [(-10, 10)] * (n * d)
    
    if time.time() < deadline - 120:
        try:
            result_de = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=300,
                popsize=30,
                tol=1e-10,
                polish=False,  # We'll polish ourselves
                init='sobol'
            )
            pts_de = result_de.x.reshape((n, d))
            if np.all(np.isfinite(pts_de)):
                update_best(pts_de)
        except Exception:
            pass

    # Strategy 2: Multiple Nelder-Mead restarts from diverse initializations
    rng = np.random.RandomState(123)
    
    # Create diverse starting points
    initial_configs = []
    
    # 1. Fibonacci sphere
    fib14 = fibonacci_sphere(14)
    initial_configs.append(fib14)
    
    # 2. Icosahedron + 2 random extra points
    for i in range(5):
        extra = rng.randn(2, 3) * 0.5
        cfg = np.vstack([ico12, extra])
        initial_configs.append(cfg)
    
    # 3. Random points on sphere
    for i in range(10):
        pts = rng.randn(n, d)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        initial_configs.append(pts)
    
    # 4. Perturbed Fibonacci
    for i in range(5):
        pert = fib14 + rng.randn(n, d) * 0.1
        initial_configs.append(pert)
    
    # 5. Cube vertices + 6 more
    cube8 = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ], dtype=float)
    cube8 = cube8 / np.linalg.norm(cube8, axis=1, keepdims=True)
    oct6 = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    initial_configs.append(np.vstack([cube8, oct6]))

    # Run Nelder-Mead on each initial config
    for i, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        current_pts = init_pts.copy()
        
        # First polish with Nelder-Mead
        try:
            result_nm = minimize(
                objective,
                current_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            pts_nm = result_nm.x.reshape((n, d))
            if np.all(np.isfinite(pts_nm)):
                update_best(pts_nm)
                if score(pts_nm) > score(current_pts):
                    current_pts = pts_nm
        except Exception:
            pass
        
        # Try a few random kicks and re-optimize
        if time.time() < deadline - 20:
            for _ in range(3):
                if time.time() > deadline - 20:
                    break
                kick_scale = 0.05 + 0.1 * (i / len(initial_configs))
                kick = rng.randn(n, d) * kick_scale
                kicked_pts = current_pts + kick
                try:
                    result_nm2 = minimize(
                        objective,
                        kicked_pts.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                    )
                    pts_kick = result_nm2.x.reshape((n, d))
                    if np.all(np.isfinite(pts_kick)):
                        update_best(pts_kick)
                        if score(pts_kick) > score(current_pts):
                            current_pts = pts_kick
                except Exception:
                    pass

    # Strategy 3: If we still have time, do a final polish on the best solution
    if best_points is not None and time.time() < deadline - 10:
        try:
            result_final = minimize(
                objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            pts_final = result_final.x.reshape((n, d))
            if np.all(np.isfinite(pts_final)):
                update_best(pts_final)
        except Exception:
            pass

    # Fallback: return best found or a default
    if best_points is None:
        best_points = fibonacci_sphere(14)

    # Normalize to ensure finite and reasonable scale
    best_points = best_points - np.mean(best_points, axis=0)
    norm_max = np.max(np.linalg.norm(best_points, axis=1))
    if norm_max > 0:
        best_points = best_points / norm_max * 100

    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))

    return best_points
