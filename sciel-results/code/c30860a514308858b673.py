# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 600  # seconds
    safety_margin = 30  # seconds before limit to stop
    
    def get_score(pts):
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_objective(x, beta=0.05):
        """
        Minimizes (d_max / d_min)^2 using soft min/max approximations.
        x: flat array of length 42
        """
        p = x.reshape((n, d))
        
        if not np.all(np.isfinite(p)):
            return 1e10

        dists = pdist(p, metric='euclidean')
        
        a_neg = -dists / beta
        max_a_neg = np.max(a_neg)
        log_sum_neg = max_a_neg + np.log(np.sum(np.exp(a_neg - max_a_neg)))
        soft_min = -beta * log_sum_neg
        
        a_pos = dists / beta
        max_a_pos = np.max(a_pos)
        log_sum_pos = max_a_pos + np.log(np.sum(np.exp(a_pos - max_a_pos)))
        soft_max = beta * log_sum_pos
        
        if soft_min <= 0 or soft_max <= 0:
            return 1e10
            
        ratio_sq = (soft_max / soft_min) ** 2
        return ratio_sq

    def exact_objective(x):
        """Exact (d_max/d_min)^2 for final polish."""
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10
        dists = pdist(p, metric='euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min <= 0 or d_max == 0:
            return 1e10
        return (d_max / d_min) ** 2

    # Initial guess: Icosahedron vertices + 2 poles, normalized
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms_ico = np.linalg.norm(ico_verts, axis=1)
    ico_verts_norm = ico_verts / norms_ico[:, np.newaxis]
    
    # Add two points to make 14. Poles are a natural choice for symmetry.
    points_init = np.vstack([ico_verts_norm, np.array([[0, 0, 1.0], [0, 0, -1.0]])])
    points_init = points_init - np.mean(points_init, axis=0)
    
    best_points = points_init.copy()
    best_score = get_score(best_points)
    
    # 1. Global Optimization using Differential Evolution
    if time.time() - start_time < time_limit - safety_margin:
        try:
            de_deadline = start_time + (time_limit * 0.35)
            
            def de_objective(x):
                if time.time() > de_deadline:
                    return 1e10
                return smooth_objective(x, beta=0.1)
            
            res_de = differential_evolution(
                de_objective, 
                bounds=[(-5, 5)] * (n * d), 
                maxiter=40, 
                popsize=12, 
                tol=1e-5, 
                seed=42,
                init='latinhypercube'
            )
            
            if res_de.fun < 1e9:
                cand_points = res_de.x.reshape((n, d))
                cand_score = get_score(cand_points)
                if cand_score > best_score:
                    best_points = cand_points
                    best_score = cand_score
                    
        except Exception:
            pass

    # 2. Local Refinement using L-BFGS-B with adaptive beta
    candidates_to_polish = [best_points.copy(), points_init.copy()]
    
    # Add random perturbations
    np.random.seed(123)
    for _ in range(8):
        perturb = best_points + np.random.randn(n, d) * 0.15
        candidates_to_polish.append(perturb)
    
    # Fibonacci sphere initial guesses
    def fibonacci_sphere(n_points, seed=0):
        rng = np.random.RandomState(seed)
        points = []
        for i in range(n_points):
            z = 1 - 2 * (i + 0.5) / n_points
            r = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / n_points
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        pts = np.array(points)
        pts = pts - np.mean(pts, axis=0)
        return pts
    
    for s in [1, 2, 3]:
        candidates_to_polish.append(fibonacci_sphere(n, seed=s))
    
    for init_pts in candidates_to_polish:
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        try:
            # First polish with moderate beta
            res_local = minimize(
                lambda x: smooth_objective(x, beta=0.05), 
                init_pts.ravel(), 
                method='L-BFGS-B', 
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res_local.fun < 1e9:
                opt_points = res_local.x.reshape((n, d))
                opt_score = get_score(opt_points)
                
                if opt_score > best_score:
                    best_points = opt_points
                    best_score = opt_score
                    
                    # Second polish with smaller beta
                    if time.time() - start_time < time_limit - safety_margin:
                        res_local2 = minimize(
                            lambda x: smooth_objective(x, beta=0.01), 
                            opt_points.ravel(), 
                            method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                        )
                        if res_local2.fun < 1e9:
                            opt_points2 = res_local2.x.reshape((n, d))
                            opt_score2 = get_score(opt_points2)
                            if opt_score2 > best_score:
                                best_points = opt_points2
                                best_score = opt_score2
        except Exception:
            pass

    # 3. Final high-precision polish with exact objective
    if time.time() - start_time < time_limit - safety_margin:
        try:
            res_final = minimize(
                exact_objective, 
                best_points.ravel(), 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            
            if res_final.fun < 1e9:
                final_points = res_final.x.reshape((n, d))
                final_score = get_score(final_points)
                if final_score > best_score:
                    best_points = final_points
                    best_score = final_score
        except Exception:
            pass

    # 4. Try a few more random restarts if time allows
    if time.time() - start_time < time_limit - 15:
        for seed in [777, 888, 999, 1000, 1111, 2222, 3333]:
            if time.time() - start_time > time_limit - 10:
                break
            try:
                rp = fibonacci_sphere(n, seed=seed)
                res_q = minimize(
                    lambda x: smooth_objective(x, beta=0.03), 
                    rp.ravel(), 
                    method='L-BFGS-B', 
                    options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-10}
                )
                if res_q.fun < 1e9:
                    qp = res_q.x.reshape((n, d))
                    qs = get_score(qp)
                    if qs > best_score:
                        best_points = qp
                        best_score = qs
                        
                        # Quick exact polish
                        if time.time() - start_time < time_limit - 10:
                            res_q2 = minimize(
                                exact_objective, 
                                qp.ravel(), 
                                method='L-BFGS-B', 
                                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
                            )
                            if res_q2.fun < 1e9:
                                qp2 = res_q2.x.reshape((n, d))
                                qs2 = get_score(qp2)
                                if qs2 > best_score:
                                    best_points = qp2
                                    best_score = qs2
            except Exception:
                pass

    # Final validity check
    if not np.all(np.isfinite(best_points)):
        return points_init

    return best_points

# EVOLVE-BLOCK-END
