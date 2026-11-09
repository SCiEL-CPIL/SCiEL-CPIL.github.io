import numpy as np
from scipy.spatial.distance import pdist
import time

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    try:
        dists = pdist(points, metric='euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0 or np.isnan(d_min) or np.isnan(d_max) or not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        return (d_min / d_max) ** 2
    except Exception:
        return 0.0

def _nelder_mead(points, objective, max_iter=5000, tol=1e-12, time_limit=None, start_time=None):
    """
    Custom Nelder-Mead implementation for maximizing 'objective'.
    points: (n, d) array.
    objective: function taking (n, d) array and returning scalar score.
    """
    n, d = points.shape
    # Flatten points for optimization
    x0 = points.flatten()
    
    # Initialize simplex
    simplex = []
    fvals = []
    
    # Vertex 0: initial point
    simplex.append(x0.copy())
    f0 = objective(points)
    fvals.append(f0)
    
    # Generate other vertices with small perturbations
    # Scale perturbation relative to magnitude of x0
    scale = np.linalg.norm(x0)
    if scale == 0:
        scale = 1.0
    delta = 0.05 * scale
    
    for i in range(len(x0)):
        new_x = x0.copy()
        new_x[i] += delta
        simplex.append(new_x)
        # Reshape and evaluate
        pts = new_x.reshape(n, d)
        fvals.append(objective(pts))
    
    # Sort vertices by objective value (maximizing, so descending)
    # We want the best (highest score) first
    indices = np.argsort(fvals)[::-1]
    simplex = [simplex[i] for i in indices]
    fvals = [fvals[i] for i in indices]
    
    alpha = 1.0  # Reflection
    gamma = 2.0  # Expansion
    rho = 0.5   # Contraction
    sigma = 0.5 # Shrink
    
    best_val = fvals[0]
    
    for _ in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() > time_limit:
                break
        
        # Check convergence
        if np.std(fvals) < tol:
            break
            
        # Centroid of all points except the worst
        worst_idx = len(fvals) - 1
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[worst_idx])
        pts_r = xr.reshape(n, d)
        fr = objective(pts_r)
        
        if fr > fvals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[worst_idx])
            pts_e = xe.reshape(n, d)
            fe = objective(pts_e)
            
            if fe > fr:
                simplex[worst_idx] = xe
                fvals[worst_idx] = fe
                best_val = max(best_val, fe)
            else:
                simplex[worst_idx] = xr
                fvals[worst_idx] = fr
                best_val = max(best_val, fr)
        elif fr >= fvals[worst_idx - 1]:
            # Accept reflection
            simplex[worst_idx] = xr
            fvals[worst_idx] = fr
            best_val = max(best_val, fr)
        else:
            # Contraction
            # Check if worst is above or below centroid
            if fvals[worst_idx] < fvals[-1]: # This logic is slightly off in standard NM, 
                # Standard NM: if worst is worse than centroid, contract inward
                # Here we maximize. Worst is lowest.
                pass
            
            # If fr < fvals[-2] (second worst)
            # Try contraction toward centroid
            xc = centroid + rho * (simplex[worst_idx] - centroid)
            pts_c = xc.reshape(n, d)
            fc = objective(pts_c)
            
            if fc > fvals[worst_idx]:
                simplex[worst_idx] = xc
                fvals[worst_idx] = fc
                best_val = max(best_val, fc)
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    pts_s = simplex[i].reshape(n, d)
                    fvals[i] = objective(pts_s)
                
                # Re-sort
                indices = np.argsort(fvals)[::-1]
                simplex = [simplex[i] for i in indices]
                fvals = [fvals[i] for i in indices]
                best_val = max(best_val, fvals[0])
        
        # Re-sort after update (except in shrink case where we just did it)
        # To keep it simple and robust, re-sort every step if not just shrunk
        if not (fr < fvals[worst_idx] and fc <= fvals[worst_idx]): # Rough check if shrink happened
             indices = np.argsort(fvals)[::-1]
             simplex = [simplex[i] for i in indices]
             fvals = [fvals[i] for i in indices]
             best_val = max(best_val, fvals[0])

    # Return best point found
    best_idx = np.argmax(fvals)
    best_points = simplex[best_idx].reshape(n, d)
    return best_points, best_val

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Generate initial seeds
    seeds = []
    
    # Seed 1: Icosahedron + 2 points (Z-axis)
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    p1 = np.vstack([ico, np.array([[0, 0, 1], [0, 0, -1]])])
    p1 = p1 / np.linalg.norm(p1, axis=1, keepdims=True)
    seeds.append(p1)
    
    # Seed 2: Random on sphere
    for _ in range(10):
        pts = np.random.randn(n, d)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        # Scale to unit diameter roughly? No, let NM handle scale.
        # Actually, NM works better if scale is consistent.
        # Let's add a few random non-spherical seeds too
        seeds.append(pts)
        
    # Seed 3: Cube + Octahedron vertices? 8+6=14.
    # Cube: (+-1, +-1, +-1)
    cube = np.array(np.meshgrid([-1,1],[-1,1],[-1,1])).T.reshape(-1,3)
    # Octahedron: (+-1,0,0), (0,+-1,0), (0,0,+-1)
    octa = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    p3 = np.vstack([cube, octa])
    # Normalize to unit sphere? Or keep as is?
    # The ratio is scale invariant.
    # Let's normalize to unit sphere for consistency
    p3 = p3 / np.linalg.norm(p3, axis=1, keepdims=True)
    seeds.append(p3)
    
    best_points = seeds[0].copy()
    best_score = _score(best_points)
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        # Run Nelder-Mead
        # We need an objective function that takes a flat vector or (n,d) array.
        # My NM implementation above takes (n,d) in the callback.
        def obj_func(pts):
            return _score(pts)
        
        # NM can be sensitive to initial simplex size.
        # Let's run it.
        try:
            res_points, res_score = _nelder_mead(
                seed, 
                obj_func, 
                max_iter=3000, 
                tol=1e-10,
                time_limit=deadline,
                start_time=start_time
            )
            if res_score > best_score:
                best_score = res_score
                best_points = res_points
        except Exception:
            continue
            
    # Final verification and cleanup
    if not np.all(np.isfinite(best_points)):
        return seeds[0]
        
    return best_points
