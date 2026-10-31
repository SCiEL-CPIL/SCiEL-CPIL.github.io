import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
import time

N = 16
D = 2

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def objective(pts):
        # pts: (N, 2)
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Mask diagonal
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        d_min = d.min()
        d_max = d.max()
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def make_hexagonal(n):
        # Generate hexagonal lattice points, take first n
        pts = []
        # Use a hexagonal grid
        # Axial coordinates
        for q in range(-10, 11):
            for r in range(-10, 11):
                # Convert axial to cartesian
                # x = size * (q + r/2)
                # y = size * (sqrt(3)/2 * r)
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                pts.append((x, y))
        pts = np.array(pts)
        # Center and sort by distance from origin
        pts -= pts.mean(axis=0)
        dists_to_center = np.sqrt(np.sum(pts**2, axis=1))
        order = np.argsort(dists_to_center)
        return pts[order[:n]]

    def make_circular(n):
        # Points on concentric circles
        pts = []
        # 1 center
        pts.append([0, 0])
        # 6 on first circle
        for i in range(6):
            angle = 2 * np.pi * i / 6
            pts.append([np.cos(angle), np.sin(angle)])
        # 9 on second circle (radius ~2)
        r = 2.0
        for i in range(9):
            angle = 2 * np.pi * i / 9 + np.pi / 9
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        return np.array(pts)

    def make_square_grid(n):
        side = int(np.ceil(n ** 0.5))
        grid = []
        for i in range(side):
            for j in range(side):
                grid.append([i, j])
        return np.array(grid[:n], dtype=float)

    # Initial configurations
    init_configs = [
        make_hexagonal(N),
        make_circular(N),
        make_square_grid(N),
    ]
    
    best_obj = np.inf
    best_pts = init_configs[0].copy()
    
    # Try optimization from each initial config
    for init in init_configs:
        if time.time() > deadline:
            break
        
        # Use Basin Hopping for robustness
        for trial in range(3):
            if time.time() > deadline:
                break
            perturbed = init + np.random.randn(N, D) * 0.1
            
            try:
                # Basin Hopping with Powell as inner minimizer
                # stepsize is important
                result = basinhopping(
                    objective,
                    perturbed,
                    minimizer_kwargs={'method': 'Powell', 'options': {'maxiter': 1000, 'maxfev': 10000}},
                    niter=50,  # Limit iterations to respect deadline
                    stepsize=0.5,
                    T=100.0,
                    seed=trial
                )
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_pts = result.x.copy()
            except Exception:
                pass

    # Final polish with Powell from best
    if time.time() < deadline - 30:
        try:
            result = minimize(
                objective,
                best_pts,
                method='Powell',
                options={'maxiter': 10000, 'maxfev': 100000}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_pts = result.x.copy()
        except Exception:
            pass
    
    return best_pts
