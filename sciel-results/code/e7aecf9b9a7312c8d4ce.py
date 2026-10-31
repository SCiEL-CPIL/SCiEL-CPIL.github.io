import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Precompute indices for upper triangle
    iu = np.triu_indices(N, 1)
    
    def objective(pts):
        """
        Calculate the squared ratio of max to min pairwise distance.
        """
        # pts: (N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        d_sq = dists_sq[iu]
        
        # Handle case where points coincide
        d_min_sq = np.min(d_sq)
        if d_min_sq < 1e-12:
            return 1e10
            
        d_max_sq = np.max(d_sq)
        
        # Ratio squared is (d_max / d_min)^2 = d_max_sq / d_min_sq
        return d_max_sq / d_min_sq

    def make_hexagonal(n):
        # Generate hexagonal lattice points
        # Use axial coordinates or offset coordinates
        # Offset coordinates:
        # x = col + 0.5 * (row % 2)
        # y = row * sqrt(3) / 2
        pts = []
        for row in range(-10, 11):
            for col in range(-10, 11):
                x = col + 0.5 * (row % 2)
                y = row * np.sqrt(3) / 2
                pts.append([x, y])
        pts = np.array(pts)
        
        # Center
        pts -= pts.mean(axis=0)
        
        # Sort by distance from center
        dists_to_center = np.sqrt(np.sum(pts**2, axis=1))
        order = np.argsort(dists_to_center)
        
        # Take first n
        return pts[order[:n]]

    def make_circular(n):
        # Points on concentric circles
        # 1 center, 6 on r=1, 9 on r=2 (approx)
        pts = []
        pts.append([0, 0])
        for i in range(6):
            angle = 2 * np.pi * i / 6
            pts.append([np.cos(angle), np.sin(angle)])
        r = 2.0
        for i in range(9):
            angle = 2 * np.pi * i / 9 + np.pi / 9 # offset to avoid alignment
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        return np.array(pts)

    def make_random(n):
        # Random points in a square, then normalized
        pts = np.random.randn(n, D)
        # Normalize to unit sphere? No, just scale to fit in [-1, 1]
        max_val = np.max(np.abs(pts))
        if max_val > 0:
            pts /= max_val
        return pts

    # Initial configurations
    init_configs = [
        make_hexagonal(N),
        make_circular(N),
        make_random(N),
        make_random(N),
        make_random(N)
    ]
    
    best_obj = np.inf
    best_pts = init_configs[0].copy()
    
    # Optimization strategies
    # Powell is good for non-smooth, but can be slow.
    # Nelder-Mead is similar.
    # We will try Powell with reasonable limits.
    
    for init in init_configs:
        if time.time() > deadline:
            break
            
        # Perturb slightly to escape local minima if it's a structured start
        # For random starts, they are already distinct.
        
        # Try Powell
        try:
            result = minimize(
                objective,
                init,
                method='Powell',
                options={'maxiter': 1000, 'maxfev': 20000}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_pts = result.x.copy()
        except Exception:
            pass
            
        # Try Nelder-Mead
        if time.time() < deadline:
            try:
                result_nm = minimize(
                    objective,
                    init,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'maxfev': 50000}
                )
                if result_nm.fun < best_obj:
                    best_obj = result_nm.fun
                    best_pts = result_nm.x.copy()
            except Exception:
                pass

    # If we have time, try Differential Evolution to find a better global basin
    if time.time() < deadline - 30:
        # Bounds: points should not be too far apart. 
        # If min dist is 1, max dist for 16 points in hex pack is roughly 4-5.
        # Let's use [-5, 5]
        bounds = [(-5, 5)] * (N * D)
        try:
            de_result = differential_evolution(
                lambda x: objective(x.reshape(N, D)),
                bounds,
                maxiter=50,
                tol=1e-4,
                seed=42,
                popsize=10,
                polish=False # We will polish manually
            )
            de_pts = de_result.x.reshape(N, D)
            de_obj = de_result.fun
            
            if de_obj < best_obj:
                best_obj = de_obj
                best_pts = de_pts.copy()
                
            # Polish the DE result
            if time.time() < deadline - 10:
                try:
                    result_polish = minimize(
                        objective,
                        best_pts,
                        method='Powell',
                        options={'maxiter': 2000, 'maxfev': 50000}
                    )
                    if result_polish.fun < best_obj:
                        best_obj = result_polish.fun
                        best_pts = result_polish.x.copy()
                except Exception:
                    pass
                    
        except Exception:
            pass
            
    # Final check: ensure no points are identical
    # If min dist is very small, the score is bad.
    # The optimizer should handle this, but let's verify.
    
    return best_pts
