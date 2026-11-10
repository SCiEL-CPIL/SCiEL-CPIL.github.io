"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _fast_objective(x, deadline):
    """
    Fast objective function for 16D vector (8 points).
    Minimizes -log(Score).
    """
    if time.time() > deadline:
        return 1e10
        
    n = 8
    pts = x.reshape(n, 2)
    
    # Precomputed indices for unordered pairs (28 pairs)
    # We compute them once outside if possible, but here we just use a static list
    # Actually, for speed in tight loops, we should hardcode or pass indices.
    # For this scope, calculating them is fast enough, but let's optimize.
    
    # Unordered pairs (i, j) with i < j
    # There are 28 such pairs.
    # We can hardcode the indices for maximum speed in the inner loop if needed, 
    # but numpy slicing is fine.
    
    # Efficient distance calculation
    # diff[i, j] = pts[i] - pts[j]
    # We only need upper triangle.
    
    # Using broadcasting for all 8x8 is easy but computes 64 distances. 
    # 28 is half. 
    # Let's stick to the vectorized approach which is quite fast in numpy.
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    # Mask for upper triangle (i < j)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dists = np.sqrt(dists_sq[mask])
    
    if np.any(dists < 1e-12):
        return 1e10
        
    D = np.max(dists)
    if D < 1e-12:
        return 1e10
        
    # Score = Product(d_ij_scaled) / n^n
    # d_ij_scaled = d_ij * 2 / D
    # log(Score) = sum(log(d_ij) + log(2) - log(D)) - n*log(n)
    # log(Score) = sum(log(d_ij)) + 28*log(2) - 28*log(D) - 8*log(8)
    
    sum_log_d = np.sum(np.log(dists))
    log_score = sum_log_d + 28.0 * np.log(2.0) - 28.0 * np.log(D) - 8.0 * np.log(8.0)
    
    return -log_score

def _param_to_points(p):
    """
    Convert 6D parameter vector to 16D point vector.
    p = [R1, R2, alpha, beta, dx, dy]
    Square 1: radius R1, angle alpha
    Square 2: radius R2, angle beta, offset (dx, dy)
    """
    R1, R2, alpha, beta, dx, dy = p
    
    # Square 1 points
    angles1 = alpha + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    pts1 = np.column_stack([R1 * np.cos(angles1), R1 * np.sin(angles1)])
    
    # Square 2 points
    angles2 = beta + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    pts2 = np.column_stack([R2 * np.cos(angles2), R2 * np.sin(angles2)])
    pts2 = pts2 + np.array([dx, dy])
    
    pts = np.vstack([pts1, pts2])
    return pts.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 8
    
    # 1. Global Search in Reduced Parameter Space (Two Squares Model)
    # Parameters: R1, R2, alpha, beta, dx, dy
    # Bounds:
    # R1, R2 in [0, 10] (arbitrary, scale invariant)
    # alpha, beta in [0, 2*pi]
    # dx, dy in [-5, 5]
    
    bounds = [
        (0.1, 5.0),   # R1
        (0.1, 5.0),   # R2
        (0.0, 2*np.pi), # alpha
        (0.0, 2*np.pi), # beta
        (-2.0, 2.0),  # dx
        (-2.0, 2.0)   # dy
    ]
    
    def param_objective(p):
        if time.time() > deadline:
            return 1e10
        pts = _param_to_points(p)
        return _fast_objective(pts, deadline)
    
    best_x_param = None
    best_val_param = float('inf')
    
    # Run Differential Evolution
    try:
        de_res = differential_evolution(
            param_objective,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-5,
            seed=42,
            polish=False, # We will polish separately
            init='sobol'
        )
        if de_res.fun < best_val_param:
            best_val_param = de_res.fun
            best_x_param = de_res.x
    except Exception:
        pass
    
    # If DE failed or didn't run, fallback to regular octagon params
    if best_x_param is None:
        # Regular octagon can be approximated by two squares of same radius, offset 0, angles 0 and pi/4
        # R1 = R2 = 1/sqrt(2) (radius of octagon with side 1? No, just relative)
        # Let's just use R1=1, R2=1, alpha=0, beta=pi/4, dx=0, dy=0
        best_x_param = np.array([1.0, 1.0, 0.0, np.pi/4, 0.0, 0.0])
        
    # Convert best param to points
    best_pts = _param_to_points(best_x_param)
    
    # 2. Local Refinement in Full 16D Space
    # Use L-BFGS-B for speed
    
    def full_objective(x):
        return _fast_objective(x, deadline)
    
    # Try polishing from the symmetric solution
    try:
        res = minimize(
            full_objective,
            best_pts,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-15}
        )
        if res.fun < best_val_param:
            best_val_param = res.fun
            best_pts = res.x
    except Exception:
        pass
    
    # 3. Try a few other structured initializations in 16D and polish
    # e.g. Regular octagon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x0_oct = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    # Random perturbations of the best found
    rng = np.random.RandomState(42)
    
    for i in range(10):
        if time.time() > deadline - 10:
            break
        
        # Perturb the current best
        noise = rng.randn(16) * 0.1
        x_init = best_pts + noise
        
        try:
            res = minimize(
                full_objective,
                x_init,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-15}
            )
            if res.fun < best_val_param:
                best_val_param = res.fun
                best_pts = res.x
        except Exception:
            pass
            
    # Also try polishing the regular octagon directly in case the 2-square model missed it
    # (Though 2-square model includes octagon)
    try:
        res_oct = minimize(
            full_objective,
            x0_oct,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-15}
        )
        if res_oct.fun < best_val_param:
            best_val_param = res_oct.fun
            best_pts = res_oct.x
    except Exception:
        pass

    # 4. Final Polish with Powell for high precision
    if time.time() < deadline - 2:
        try:
            res_final = minimize(
                full_objective,
                best_pts,
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-15, 'fatol': 1e-15}
            )
            if res_final.fun < best_val_param:
                best_val_param = res_final.fun
                best_pts = res_final.x
        except Exception:
            pass

    best_points = best_pts.reshape(n, 2)
    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
