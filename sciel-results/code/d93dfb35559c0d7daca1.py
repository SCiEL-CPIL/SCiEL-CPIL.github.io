import numpy as np
import scipy.optimize as opt
import itertools
import time

N = 16
D = 2

def score(points):
    # points shape (N, D)
    # Calculate pairwise distances
    # Use broadcasting
    # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    # d_min = min(dists[i,j] for i!=j)
    # d_max = max(dists[i,j] for i!=j)
    # return (d_max / d_min)**2
    
    # Efficient calculation
    # Squared distances
    sq_dists = np.sum(points**2, axis=1)
    dists_sq = sq_dists[:, np.newaxis] + sq_dists[np.newaxis, :] - 2 * points @ points.T
    # Ensure non-negative due to floating point errors
    dists_sq = np.maximum(dists_sq, 0)
    np.fill_diagonal(dists_sq, np.inf)
    
    d_min_sq = np.min(dists_sq)
    d_max_sq = np.max(dists_sq)
    
    if d_min_sq <= 1e-9:
        return 1e10
    return (d_max_sq / d_min_sq)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_score = 1e10
    best_points = None
    
    # Initial guesses
    inits = []
    
    # 1. Grid 4x4
    side = 4
    grid = np.array(list(itertools.product(range(side), repeat=D)), dtype=float)
    # Center it
    grid -= grid.mean(axis=0)
    inits.append(grid.copy())
    
    # 2. Grid 4x4 with random perturbations
    for _ in range(5):
        p = grid.copy()
        p += np.random.normal(0, 0.1, p.shape)
        inits.append(p)
    
    # 3. Fermat's Spiral (Sunflower)
    # r = sqrt(i)
    # theta = i * golden_angle
    golden_angle = np.pi * (3 - np.sqrt(5))
    spiral = []
    for i in range(N):
        r = np.sqrt(i)
        theta = i * golden_angle
        spiral.append([r * np.cos(theta), r * np.sin(theta)])
    spiral = np.array(spiral)
    spiral -= spiral.mean(axis=0)
    inits.append(spiral.copy())
    
    # 4. Random points in a square
    for _ in range(5):
        p = np.random.rand(N, D) - 0.5
        inits.append(p)
        
    # 5. Random points in a circle
    for _ in range(5):
        r = np.sqrt(np.random.rand(N))
        theta = 2 * np.pi * np.random.rand(N)
        p = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(p)

    # Optimization
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Flatten for optimizer
        x0 = init.flatten()
        
        def obj(x):
            p = x.reshape(N, D)
            return score(p)
            
        # Use Nelder-Mead or Powell
        # Powell is often better for these geometric problems as it doesn't assume convexity
        # Nelder-Mead is also good. Let's try Nelder-Mead with default params first, maybe tighter bounds if needed.
        # Since we can scale, there is no bound, but we should prevent points from flying apart or collapsing.
        # The objective function penalizes collapse (d_min -> 0) and expansion (d_max -> inf relative to d_min).
        # Actually, if points collapse, d_min -> 0, ratio -> inf.
        # If points expand, d_max increases, d_min increases, ratio stays similar if uniform scaling.
        # So the objective is scale invariant.
        
        try:
            res = opt.minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.success or res.fun < best_score:
                current_score = res.fun
                if current_score < best_score:
                    best_score = current_score
                    best_points = res.x.reshape(N, D)
                    
        except Exception as e:
            pass
            
        # Check time
        if time.time() > deadline:
            break
            
    # If no optimization succeeded, return the best initial guess
    if best_points is None:
        # Evaluate inits
        for init in inits:
            s = score(init)
            if s < best_score:
                best_score = s
                best_points = init
        if best_points is None:
             best_points = inits[0]
            
    return best_points
