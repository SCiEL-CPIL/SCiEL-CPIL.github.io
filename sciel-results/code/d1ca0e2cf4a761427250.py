import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import time

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    def score(pts):
        """Compute the metric (d_min/d_max)^2"""
        try:
            if not np.all(np.isfinite(pts)):
                return 0.0
            dists = pdist(pts)
            if len(dists) == 0:
                return 0.0
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max <= 1e-12 or not np.isfinite(d_min) or not np.isfinite(d_max):
                return 0.0
            r = d_min / d_max
            return float(r * r)
        except Exception:
            return 0.0

    def neg_score(pts_flat):
        """Negative score for minimization"""
        pts = pts_flat.reshape((n, d))
        return -score(pts)

    def soft_max(y, beta):
        # Stable log-sum-exp
        m = np.max(y)
        if m > 100:
            return m + np.log(np.sum(np.exp(beta * (y - m)))) / beta
        return (1.0 / beta) * np.log(np.sum(np.exp(beta * y)))

    def soft_min(y, beta):
        m = np.min(y)
        if m < -100:
            return m - np.log(np.sum(np.exp(-beta * (y - m)))) / beta
        return -(1.0 / beta) * np.log(np.sum(np.exp(-beta * y)))

    def smooth_objective(x, beta):
        if time.time() > deadline:
            return np.inf
            
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        
        try:
            dists = pdist(pts)
        except Exception:
            return np.inf
            
        if np.any(~np.isfinite(dists)) or np.any(dists <= 1e-9):
            return np.inf
            
        log_dists = np.log(dists)
        val = soft_max(log_dists, beta) - soft_min(log_dists, beta)
        return val

    def icosahedron():
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        norms = np.linalg.norm(verts, axis=1)
        return verts / norms[:, np.newaxis]

    def fibonacci_sphere(num_points):
        """Generate points on a sphere using the Fibonacci lattice."""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(num_points):
            z = 1 - (i / (num_points - 1)) * 2
            radius = np.sqrt(1 - z * z)
            theta = np.pi * i / golden_ratio
            x = np.cos(theta) * radius
            y = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    def generate_seeds():
        seeds = []
        ico = icosahedron()
        
        # Seed 1: Icosahedron + 2 points on Z axis (5-fold symmetry)
        # Align ico[0] to Z
        v0 = ico[0]
        v0 = v0 / np.linalg.norm(v0)
        z = np.array([0, 0, 1])
        axis = np.cross(v0, z)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-10:
            axis = axis / axis_norm
            angle = np.arccos(np.dot(v0, z))
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            ico_rot = ico @ R.T
        else:
            ico_rot = ico.copy()
            
        for r in [0.0, 0.5, 0.8, 1.0, 1.2, 1.5]:
            p1 = np.array([0, 0, r])
            p2 = np.array([0, 0, -r])
            seeds.append(np.vstack([ico_rot, p1, p2]))

        # Seed 2: Icosahedron + 2 points on 3-fold axis
        c = (ico[0] + ico[1] + ico[10]) / 3.0
        c = c / np.linalg.norm(c)
        axis = np.cross(c, z)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-10:
            axis = axis / axis_norm
            angle = np.arccos(np.dot(c, z))
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            ico_rot_3 = ico @ R.T
        else:
            ico_rot_3 = ico.copy()
            
        for r in [0.0, 0.5, 0.8, 1.0, 1.2, 1.5]:
            p1 = np.array([0, 0, r])
            p2 = np.array([0, 0, -r])
            seeds.append(np.vstack([ico_rot_3, p1, p2]))
            
        # Seed 3: Fibonacci sphere
        seeds.append(fibonacci_sphere(14))

        # Seed 4: Random points on sphere
        np.random.seed(42)
        for _ in range(5):
            pts = np.random.randn(n, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            pts += np.random.randn(n, 3) * 0.1
            seeds.append(pts)
            
        return seeds

    seeds = generate_seeds()
    
    best_points = None
    best_score_val = 0.0
    
    # Update best if a seed is good
    for s in seeds:
        sc = score(s)
        if sc > best_score_val:
            best_score_val = sc
            best_points = s.copy()

    # Step 1: Differential Evolution for global search
    # We use a bounded box. The optimal configuration likely has points within [-2, 2] or so.
    # Let's try [-1.5, 1.5] initially, maybe expand if needed, but 1.5 is usually enough for normalized configs.
    # Actually, let's just use a box that allows some freedom.
    bounds = [(-1.5, 1.5)] * (n * d)
    
    if time.time() < deadline - 120:
        try:
            # Differential Evolution is stochastic, set seed for reproducibility
            np.random.seed(42)
            de_res = differential_evolution(
                neg_score, 
                bounds, 
                seed=42, 
                maxiter=200, 
                tol=1e-6,
                polish=False, # We will polish later
                init='sobol' # Better initialization
            )
            
            de_pts = de_res.x.reshape((n, d))
            de_score = score(de_pts)
            
            if de_score > best_score_val:
                best_score_val = de_score
                best_points = de_pts.copy()
        except Exception:
            pass

    # Step 2: Local Refinement using L-BFGS-B with smooth objective (annealing)
    # We take the best candidate and try to refine it.
    # We also try refining the top few seeds.
    
    candidates = []
    if best_points is not None:
        candidates.append(best_points)
    
    # Add top seeds
    seed_scores = [(score(s), s) for s in seeds]
    seed_scores.sort(key=lambda x: x[0], reverse=True)
    for sc, s in seed_scores[:5]:
        if not any(np.allclose(s, c) for c in candidates):
            candidates.append(s)

    betas = [10, 50, 100, 200, 500]
    
    for cand in candidates:
        if time.time() > deadline - 20:
            break
            
        current_x = cand.flatten()
        current_smooth_val = np.inf
        
        # Initialize smooth val
        try:
            current_smooth_val = smooth_objective(current_x, 10)
        except:
            pass
        
        for beta in betas:
            if time.time() > deadline - 10:
                break
                
            try:
                res = minimize(
                    smooth_objective, 
                    current_x, 
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if np.isfinite(res.fun) and res.fun < current_smooth_val:
                    current_smooth_val = res.fun
                    current_x = res.x
            except Exception:
                pass
        
        # Check final score
        final_pts = current_x.reshape((n, d))
        final_score = score(final_pts)
        if final_score > best_score_val:
            best_score_val = final_score
            best_points = final_pts.copy()

    # Step 3: Nelder-Mead polish on the absolute best point found so far
    if best_points is not None and time.time() < deadline - 10:
        try:
            nm_res = minimize(
                neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            nm_pts = nm_res.x.reshape((n, d))
            nm_score = score(nm_pts)
            if nm_score > best_score_val:
                best_score_val = nm_score
                best_points = nm_pts.copy()
        except Exception:
            pass

    # Fallback
    if best_points is None:
        best_points = seeds[0]

    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = seeds[0]

    # Center the points? The metric is translation invariant, but centering might help with numerical stability or just be cleaner.
    # The task doesn't require centering, but it's a good practice.
    best_points = best_points - np.mean(best_points, axis=0)

    return best_points
