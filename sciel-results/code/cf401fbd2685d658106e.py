import numpy as np
import time
from scipy.optimize import differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    def objective(points_1d):
        """
        Objective function for minimization (we minimize negative log score).
        points_1d: array of shape (16,) representing 8 points in 2D.
        """
        # Check time budget
        if time.time() > deadline:
            return 1e10
            
        points = points_1d.reshape(8, 2)
        
        # Compute pairwise distances
        # Using broadcasting for efficiency
        # diff[i, j] = points[i] - points[j]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # We only need upper triangle (i < j) for the product of unique pairs
        # But the problem statement says "product over ordered pairs".
        # Wait, "product of distances at fixed diameter".
        # Usually "product of distances" in such geometric inequalities refers to unique pairs {i,j} with i!=j.
        # Let's re-read carefully: "Delta is the product over ordered pairs of distances".
        # Ordered pairs (i,j) with i != j. There are 8*7 = 56 such pairs.
        # Distance d_ij = d_ji. So product is (product over i<j d_ij)^2.
        # The regular octagon scores 1.
        # Let's check the seed code:
        # dists = []
        # for i in range(8):
        #   for j in range(i + 1, 8):
        #     dists.append(d)
        # product = np.prod(dists)
        # This calculates product over UNIQUE pairs (i<j).
        # So Delta = prod_{i<j} d_ij.
        # The seed code uses 28 pairs.
        # Let's stick to the seed code's definition: 28 unique pairs.
        
        # Extract upper triangle indices
        i_idx, j_idx = np.triu_indices(8, k=1)
        dists = np.sqrt(dists_sq[i_idx, j_idx])
        
        # Check for zero distances (distinct points required)
        if np.any(dists < 1e-12):
            return 1e10
            
        # Diameter D
        D = np.max(dists)
        
        if D < 1e-12:
            return 1e10
            
        # Score = (Product * (2/D)^28) / 8^8
        # Log Score = sum(log(dists)) + 28*log(2) - 28*log(D) - 8*log(8)
        
        # We want to maximize Log Score, so minimize -Log Score
        log_product = np.sum(np.log(dists))
        log_score = log_product + 28 * np.log(2) - 28 * np.log(D) - 8 * np.log(8)
        
        return -log_score

    # Bounds for coordinates
    # The problem allows |coordinate| < 10^6.
    # However, due to scale invariance, the specific scale doesn't matter for the score,
    # only the relative positions.
    # But DE needs a finite box.
    # If we pick a box too large, the initial population might be sparse.
    # If we pick a box too small, it might constrain the geometry if the optimal shape
    # requires a specific aspect ratio that doesn't fit? No, we can scale.
    # Actually, since the score is scale invariant, we can constrain the search to a box.
    # However, if the optimal configuration has points very close together or far apart,
    # a fixed box might not capture the "shape" well if the box forces a specific scale?
    # No, DE searches the coordinates. If it finds a shape, the scale is part of the coordinates.
    # But since score is invariant, any scale of the same shape yields the same score.
    # So we can restrict the search to a reasonable box, e.g., [-10, 10]^2.
    # If the optimal shape is "large", DE can find it within [-10,10] or scaled down.
    # Let's use a box of [-50, 50] to be safe, but not too huge to avoid numerical issues.
    
    bounds = [(-50.0, 50.0)] * 16
    
    # Initial population
    # Regular octagon on unit circle
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    init_points = np.column_stack((x, y)).flatten()
    
    # Differential Evolution
    # maxiter: number of iterations
    # tol: stopping tolerance
    # popsize: population size multiplier (10*dim = 160 individuals)
    # seed: for reproducibility
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=500, 
            tol=1e-10,
            popsize=20, # 20*16 = 320 individuals. Might be heavy. Let's try 15.
            init='sobol',
            updating='immediate',
            polish=True, # Use L-BFGS-B to polish the result. This is crucial for precision.
            disp=False
        )
        
        best_points_1d = res.x
        best_val = res.fun
        
    except Exception as e:
        # Fallback to regular octagon if DE fails
        best_points_1d = init_points
        best_val = objective(init_points)
        
    # Verify time
    if time.time() > deadline - 5:
        # If we are running out of time, just return what we have
        pass
    else:
        # Try to polish further with local optimization if time permits
        # The 'polish=True' in DE already does this, but we can do a quick Nelder-Mead
        # on the result to ensure we are at a local minimum of the log-score.
        from scipy.optimize import minimize
        
        try:
            res2 = minimize(
                objective, 
                best_points_1d, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if res2.fun < best_val:
                best_points_1d = res2.x
                best_val = res2.fun
        except:
            pass

    # Convert back to list of lists
    points = best_points_1d.reshape(8, 2)
    
    # Ensure distinct points (just in case numerical error made two points identical)
    # If two points are very close, the score is very low (log dist -> -inf).
    # The objective function returns 1e10 for dist < 1e-12, so minimizer should avoid this.
    
    return [[float(points[i][0]), float(points[i][1])] for i in range(8)]

if __name__ == "__main__":
    pts = construct()
    print(pts)
