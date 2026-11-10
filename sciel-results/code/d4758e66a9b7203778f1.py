import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = 550.0  # Leave buffer for finalization

    # Normals for unit hexagon (angle 0)
    # Pointing to midpoints of sides: 30, 90, 150, 210, 270, 330
    hex_normals = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(30, 360, 60)
    ])
    
    # Precompute max projection of a unit hexagon onto these normals
    # For aligned hex, this is sqrt(3)/2 for all
    max_proj_hex = np.full(6, np.sqrt(3)/2)
    
    def calc_side(centers):
        # centers: (11, 2)
        # max_proj_centers: (11, 6) -> (6,)
        proj_centers = centers @ hex_normals.T
        max_proj_centers = np.max(proj_centers, axis=0)
        
        # max extent for outer hex
        # S * sqrt(3)/2 >= max_proj_centers[k] + max_proj_hex[k]
        # S >= (max_proj_centers[k] + max_proj_hex[k]) * 2 / sqrt(3)
        extents = (max_proj_centers + max_proj_hex) * 2.0 / np.sqrt(3.0)
        return np.max(extents)
        
    def overlap_penalty(centers):
        penalty = 0.0
        n = len(centers)
        for i in range(n):
            for j in range(i+1, n):
                d = centers[i] - centers[j]
                projs = np.abs(d @ hex_normals.T)
                max_proj = np.max(projs)
                # Need max_proj >= sqrt(3) + eps
                violation = np.sqrt(3.0) + 1e-8 - max_proj
                if violation > 0:
                    penalty += violation ** 2
        return penalty

    def objective(vars):
        if time.time() > deadline:
            return float('inf')
        centers = vars.reshape(11, 2)
        side = calc_side(centers)
        pen = overlap_penalty(centers)
        return side + 100.0 * pen

    # Initial guesses
    # Guess 1: 7 in center, 4 in corners
    d = np.sqrt(3)
    centers_init = np.zeros((11, 2))
    # 6 around center
    centers_init[1:7] = np.array([
        [d * np.cos(np.deg2rad(a)), d * np.sin(np.deg2rad(a))] for a in range(0, 360, 60)
    ])
    # 4 more at distance 2d, angles 30, 150, 210, 330 (between the 6)
    # Actually, to minimize side, push them out symmetrically
    d2 = 2 * d
    angles_4 = [30, 150, 210, 330]
    for k, a in enumerate(angles_4):
        centers_init[7+k] = [d2 * np.cos(np.deg2rad(a)), d2 * np.sin(np.deg2rad(a))]
        
    guesses = []
    guesses.append(centers_init.flatten())
    
    # Guess 2: Linear chain or other symmetric
    # 3-2-3-2-1 rows?
    # Let's try a few random perturbations of the first guess
    np.random.seed(42)
    for _ in range(10):
        perturbed = centers_init + np.random.normal(0, 0.5, (11, 2))
        guesses.append(perturbed.flatten())
        
    best_side = float('inf')
    best_centers = None
    
    for guess in guesses:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(objective, guess, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_side:
                best_side = res.fun
                best_centers = res.x.reshape(11, 2)
        except Exception as e:
            pass
            
    if best_centers is None:
        # Fallback
        best_centers = centers_init
        best_side = calc_side(best_centers)
        
    # Verify and refine
    # Check if penalty is small
    final_pen = overlap_penalty(best_centers)
    if final_pen > 1e-4:
        # Try a more aggressive optimization or different start
        # For now, return what we have if it's close, or a valid fallback
        # Let's try one more optimization with smaller penalty weight to push out overlaps
        def objective_strict(vars):
            if time.time() > deadline:
                return float('inf')
            centers = vars.reshape(11, 2)
            side = calc_side(centers)
            pen = overlap_penalty(centers)
            return side + 1000.0 * pen

        try:
            res2 = minimize(objective_strict, best_centers, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res2.fun < best_side:
                best_side = res2.fun
                best_centers = res2.x.reshape(11, 2)
        except:
            pass

    final_side = calc_side(best_centers)
    
    # Construct output
    inner = np.column_stack([best_centers, np.zeros(11)])
    outer = np.array([0.0, 0.0, 0.0])
    
    return inner, outer, final_side
