import time
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Objective function for 8 variables representing 4 complex numbers
    # x = [x1, y1, x2, y2, x3, y3, x4, y4]
    # Points are (xk, yk) and (xk, -yk)
    def get_points(x):
        x1, y1, x2, y2, x3, y3, x4, y4 = x
        return np.array([
            [x1, y1],
            [x1, -y1],
            [x2, y2],
            [x2, -y2],
            [x3, y3],
            [x3, -y3],
            [x4, y4],
            [x4, -y4]
        ])

    def score_from_points(pts):
        # Calculate pairwise distances
        # pts is 8x2
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(8, k=1)
        d = dists[iu]
        
        # Check for coincident points
        if np.min(d) < 1e-9:
            return -np.inf
            
        D = np.max(d)
        if D < 1e-9:
            return -np.inf
            
        # Log score components
        # Score = Delta / 8^8
        # Delta = prod_{i!=j} d'_{ij} where d' is scaled to diameter 2
        # d'_{ij} = d_{ij} * (2/D)
        # Delta = (prod_{i<j} d'_{ij})^2 = (prod_{i<j} (2/D) d_{ij})^2
        # log(Delta) = 2 * sum_{i<j} (log(2) - log(D) + log(d_{ij}))
        #             = 2 * (28*log(2) - 28*log(D) + sum(log(d_{ij})))
        # log(Score) = log(Delta) - 8*log(8)
        
        log_d_sum = np.sum(np.log(d))
        log_delta = 2.0 * (28.0 * math.log(2.0) - 28.0 * math.log(D) + log_d_sum)
        log_score = log_delta - 8.0 * math.log(8.0)
        
        return log_score

    def objective(x):
        pts = get_points(x)
        ls = score_from_points(pts)
        if ls == -np.inf:
            return 1e100
        return -ls # Minimize negative log score

    bounds = [(-10, 10)] * 8

    best_val = np.inf
    best_x = None
    
    # Initial guess: Regular Octagon
    # Regular octagon vertices on unit circle:
    # Angles: 0, 45, 90, 135, 180, 225, 270, 315
    # Conjugate pairs:
    # z1 = 1 (angle 0) -> (1, 0), (1, 0) -- Wait, if y=0, it's on the axis.
    # The representation (x, y), (x, -y) works for y=0 as well (point is duplicated? No, distinct points required).
    # If y=0, the two points are identical. This is invalid (distinct points).
    # So we must ensure y_k != 0 for all k, OR handle the case where points on the axis are treated carefully.
    # The regular octagon has 2 points on the x-axis (1,0) and (-1,0) if aligned, or none if rotated by 22.5 deg.
    # If we rotate the octagon by 22.5 degrees, no points are on the x-axis.
    # Angles: 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5
    # Conjugate pairs:
    # 22.5 and -22.5 (337.5)
    # 67.5 and -67.5 (292.5)
    # 112.5 and -112.5 (247.5)
    # 157.5 and -157.5 (202.5)
    
    angles = np.deg2rad([22.5, 67.5, 112.5, 157.5])
    init_x = []
    for a in angles:
        init_x.append(math.cos(a))
        init_x.append(math.sin(a))
    init_x = np.array(init_x)
    
    val = objective(init_x)
    best_val = val
    best_x = init_x.copy()
    
    # Run Differential Evolution
    # popsize=20, maxiter=100
    # Since 8 variables, this should be fast.
    
    try:
        # Use differential_evolution
        # tol=1e-8, maxiter=200
        # We can run multiple DE instances or just one with large pop.
        
        # To be safe with time, we run a few DE iterations or one long one.
        # 8 variables is small.
        
        # Strategy: Run DE for a fixed number of iterations, then refine.
        
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=20,
            tol=1e-8,
            polish=True  # Uses L-BFGS-B to polish the result
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x.copy()
            
        # Check time
        if time.time() < deadline - 10.0:
             # Try a few more random restarts with L-BFGS-B
             for i in range(5):
                 if time.time() > deadline - 5.0:
                     break
                 # Random start
                 x_rand = np.random.uniform(-5, 5, 8)
                 # Ensure y components are not too close to 0 to avoid degeneracy?
                 # Actually, DE handles this.
                 
                 res_local = minimize(objective, x_rand, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12})
                 if res_local.fun < best_val:
                     best_val = res_local.fun
                     best_x = res_local.x.copy()
                     
                 # Also try perturbing the current best
                 x_perturb = best_x + 0.1 * np.random.randn(8)
                 res_perturb = minimize(objective, x_perturb, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12})
                 if res_perturb.fun < best_val:
                     best_val = res_perturb.fun
                     best_x = res_perturb.x.copy()

        # Final refinement
        if time.time() < deadline - 2.0:
            res_final = minimize(objective, best_x, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-15})
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_x = res_final.x.copy()

    except Exception as e:
        pass

    # Construct final points
    pts = get_points(best_x)
    
    # Verify distinctness
    # If any points are too close, fallback to regular octagon
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(8, k=1)
    d = dists[iu]
    if np.min(d) < 1e-6:
        # Fallback
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return pts.tolist()
