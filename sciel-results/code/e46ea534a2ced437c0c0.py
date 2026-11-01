import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

def construct():
    # Precompute all triangle index triples
    triples = list(itertools.combinations(range(N), 3))
    n_triples = len(triples)
    
    # Precompute triangle coefficient matrices for vectorized area computation
    # For triangle (i,j,k): area = 0.5 * |det(pts[j]-pts[i], pts[k]-pts[i])|
    # We'll compute signed areas and use abs
    tri_coeffs = []  # Each entry: (i, j, k)
    for (i, j, k) in triples:
        tri_coeffs.append((i, j, k))
    
    def get_points(x):
        """
        x has 2*(N-2) elements: coordinates of points 2..N-1
        Points 0 and 1 are fixed at (0,0) and (1,0)
        """
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts
    
    def compute_signed_areas(pts):
        """Compute signed areas for all triangles"""
        areas = np.zeros(n_triples)
        for idx, (i, j, k) in enumerate(tri_coeffs):
            v1 = pts[j] - pts[i]
            v2 = pts[k] - pts[i]
            det = v1[0] * v2[1] - v1[1] * v2[0]
            areas[idx] = 0.5 * det
        return areas
    
    def compute_abs_areas(pts):
        """Compute absolute areas for all triangles"""
        return np.abs(compute_signed_areas(pts))
    
    def compute_hull_area(pts):
        """Compute convex hull area"""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 1.0
    
    def raw_score(x):
        """Compute the actual score: min_triangle_area / hull_area"""
        pts = get_points(x)
        areas = compute_abs_areas(pts)
        min_area = np.min(areas)
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 0.0
        return min_area / hull_area
    
    def smooth_objective(x, beta, hull_area_target):
        """
        Smoothed objective using log-sum-exp for min of areas.
        We minimize: -logsumexp(-beta * areas) / hull_area_target
        This is a smooth approximation to -min(areas) / hull_area_target
        """
        pts = get_points(x)
        areas = compute_abs_areas(pts)
        
        # Numerically stable log-sum-exp for min
        # min(a_i) ≈ -1/beta * log(sum(exp(-beta * a_i)))
        a_min = np.min(areas)
        if a_min < 1e-15:
            return 1.0
        
        # log(sum(exp(-beta * a))) = -beta * a_min + log(sum(exp(-beta * (a - a_min))))
        exp_terms = np.exp(-beta * (areas - a_min))
        log_sum = -beta * a_min + np.log(np.sum(exp_terms))
        smooth_min = -log_sum / beta
        
        if hull_area_target < 1e-10:
            return 1.0
        
        return -(smooth_min / hull_area_target)
    
    def gradient_smooth_objective(x, beta, hull_area_target):
        """
        Compute gradient of the smoothed objective.
        f(x) = -LSE(-beta * areas(x)) / hull_area_target
        where LSE(z) = log(sum(exp(z)))
        
        areas(x) = |0.5 * det(pts[j]-pts[i], pts[k]-pts[i])|
        For simplicity, we use the fact that near optimal solutions,
        the sign of the determinant is consistent, so we can work with
        signed areas and take abs at the end.
        
        Actually, let's use a different approach: 
        Let s_i = signed area of triangle i
        Then |s_i| = s_i * sign(s_i)
        
        For the gradient, we'll use the subgradient of |s| which is sign(s) * grad(s)
        """
        pts = get_points(x)
        signed_areas = compute_signed_areas(pts)
        abs_areas = np.abs(signed_areas)
        signs = np.sign(signed_areas)
        signs[signs == 0] = 1.0
        
        a_min = np.min(abs_areas)
        if a_min < 1e-15:
            return 1.0, np.zeros_like(x)
        
        # Compute weights for log-sum-exp
        # w_i = exp(-beta * (a_i - a_min)) / sum(exp(-beta * (a_j - a_min)))
        exp_terms = np.exp(-beta * (abs_areas - a_min))
        weights = exp_terms / np.sum(exp_terms)
        
        # Gradient of smooth_min = -1/beta * log(sum(exp(-beta * a_i)))
        # d/dx smooth_min = -1/beta * (-beta) * sum(weights_i * d/dx a_i)
        #                  = sum(weights_i * d/dx a_i)
        # where a_i = |s_i|, so d/dx a_i = signs_i * d/dx s_i
        
        # Compute gradient of signed areas with respect to x
        # x contains coordinates of points 2..N-1
        # s_i = 0.5 * det(pts[j]-pts[i], pts[k]-pts[i])
        #     = 0.5 * ((x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i))
        
        grad = np.zeros_like(x)
        
        for idx, (i, j, k) in enumerate(tri_coeffs):
            w = weights[idx]
            s = signed_areas[idx]
            sign = signs[idx]
            
            # d(s_i)/d(x_p) for p in {i,j,k}
            # s_i = 0.5 * (x_j*y_k - x_j*y_i - x_i*y_k + x_i*y_i + x_k*y_i - x_k*y_j)
            #     = 0.5 * (x_j*(y_k - y_i) + x_i*(y_i - y_k) + x_k*(y_i - y_j))
            
            # d(s_i)/d(x_i) = 0.5 * (y_i - y_k)
            # d(s_i)/d(y_i) = 0.5 * (x_k - x_j)
            # d(s_i)/d(x_j) = 0.5 * (y_k - y_i)
            # d(s_i)/d(y_j) = 0.5 * (x_i - x_k)
            # d(s_i)/d(x_k) = 0.5 * (y_i - y_j)
            # d(s_i)/d(y_k) = 0.5 * (x_j - x_i)
            
            dx_i = 0.5 * (pts[i, 1] - pts[k, 1])
            dy_i = 0.5 * (pts[k, 0] - pts[j, 0])
            dx_j = 0.5 * (pts[k, 1] - pts[i, 1])
            dy_j = 0.5 * (pts[i, 0] - pts[k, 0])
            dx_k = 0.5 * (pts[i, 1] - pts[j, 1])
            dy_k = 0.5 * (pts[j, 0] - pts[i, 0])
            
            # Multiply by sign and weight
            contrib = w * sign
            
            # Add to gradient for points that are in x (points 2..N-1)
            for p, (dx, dy) in [(i, (dx_i, dy_i)), (j, (dx_j, dy_j)), (k, (dx_k, dy_k))]:
                if p >= 2:
                    idx_x = 2 * (p - 2)
                    grad[idx_x] += contrib * dx
                    grad[idx_x + 1] += contrib * dy
        
        # Gradient of objective = -sum(weights_i * d|s_i|/dx) / hull_area_target
        # = -grad / hull_area_target
        if hull_area_target < 1e-10:
            return 1.0, np.zeros_like(x)
        
        obj_val = -(a_min / hull_area_target)  # Approximate value
        grad_obj = -grad / hull_area_target
        
        return obj_val, grad_obj
    
    def initial_circle(seed_offset=0.0, radius=1.0):
        """Generate initial points on a circle"""
        x = np.zeros(2 * (N - 2))
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + seed_offset
        for i in range(N - 2):
            idx = i + 2
            x[2 * i] = radius * np.cos(angles[idx])
            x[2 * i + 1] = radius * np.sin(angles[idx])
        return x
    
    def initial_elliptic(ratio=0.7, seed_offset=0.0, radius=1.0):
        """Generate initial points on an ellipse"""
        x = np.zeros(2 * (N - 2))
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + seed_offset
        for i in range(N - 2):
            idx = i + 2
            x[2 * i] = radius * np.cos(angles[idx])
            x[2 * i + 1] = radius * ratio * np.sin(angles[idx])
        return x

    def initial_spiral(tightness=0.3, radius=1.0):
        """Generate initial points on a spiral"""
        x = np.zeros(2 * (N - 2))
        for i in range(N - 2):
            idx = i + 2
            t = idx / (N - 2)
            angle = t * 2 * np.pi * 2  # Two full turns
            r = radius * (1 - tightness * t)
            x[2 * i] = r * np.cos(angle)
            x[2 * i + 1] = r * np.sin(angle)
        return x

    # Time management
    start_time = time.time()
    time_limit = 115.0
    
    best_score = -np.inf
    best_x = None
    
    # Correct bounds: 2*(N-2) dimensions
    dim = 2 * (N - 2)
    bounds = [(-3.0, 3.0)] * dim
    
    def raw_objective(x):
        """Raw objective for minimization: -score"""
        return -raw_score(x)
    
    # Phase 1: Global search with differential evolution
    try:
        result_de = differential_evolution(
            raw_objective,
            bounds,
            seed=42,
            maxiter=20,
            popsize=12,
            tol=1e-4,
            init='sobol',
            polish=True,
            mutation=(0.5, 1.5),
            recombination=0.7
        )
        if result_de.fun < -best_score:
            best_x = result_de.x.copy()
            best_score = -result_de.fun
    except Exception:
        pass
    
    # Phase 2: Iterative relinearization with log-sum-exp smoothing
    # Key insight: The problem min(areas)/hull_area is non-convex due to hull_area in denominator
    # Strategy: Fix hull_area as a target, solve convex subproblem, update target
    # This is similar to Dinkelbach's algorithm for fractional programming
    
    betas_schedule = [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0]
    
    # Multiple diverse starting points
    start_points = []
    # Regular polygon based
    for angle_offset in [0.0, 0.3, 0.7, 1.1, 1.5, 2.0, 2.5]:
        start_points.append(initial_circle(angle_offset, radius=1.0))
    # Elliptic
    for ratio in [0.5, 0.65, 0.8, 0.9, 0.95]:
        for angle_offset in [0.0, 0.4, 0.8, 1.2]:
            start_points.append(initial_elliptic(ratio, angle_offset, radius=1.0))
    # Spiral
    for tightness in [0.2, 0.4, 0.6]:
        start_points.append(initial_spiral(tightness, radius=1.0))
    
    # Add the DE result if available
    if best_x is not None:
        start_points.append(best_x.copy())
        for _ in range(3):
            jitter = np.random.normal(0, 0.15, size=dim)
            start_points.append(best_x + jitter)
    
    # Iterative relinearization
    for beta in betas_schedule:
        if time.time() - start_time > time_limit - 12.0:
            break
        
        for x0 in start_points[:8]:  # Limit number of starts per beta
            if time.time() - start_time > time_limit - 12.0:
                break
            
            # Get current hull area as target
            pts0 = get_points(x0)
            hull_area_target = compute_hull_area(pts0)
            if hull_area_target < 1e-10:
                continue
            
            # Solve smoothed subproblem
            try:
                res = minimize(
                    smooth_objective,
                    x0,
                    args=(beta, hull_area_target),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 10000}
                )
                
                score = raw_score(res.x)
                if score > best_score:
                    best_x = res.x.copy()
                    best_score = score
            except Exception:
                pass
            
            # Also try with updated hull area target
            if time.time() - start_time > time_limit - 12.0:
                break
            
            pts_res = get_points(res.x)
            hull_area_new = compute_hull_area(pts_res)
            if hull_area_new > 1e-10 and abs(hull_area_new - hull_area_target) > 1e-6:
                try:
                    res2 = minimize(
                        smooth_objective,
                        res.x,
                        args=(beta, hull_area_new),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    score2 = raw_score(res2.x)
                    if score2 > best_score:
                        best_x = res2.x.copy()
                        best_score = score2
                except Exception:
                    pass
        
        # Update start points with best found so far
        if best_x is not None:
            new_starts = [best_x.copy()]
            for _ in range(2):
                jitter = np.random.normal(0, 0.05, size=dim)
                new_starts.append(best_x + jitter)
            start_points = new_starts + start_points[:4]
    
    # Phase 3: Final high-precision refinement with exact hull area
    if time.time() - start_time < time_limit - 10.0 and best_x is not None:
        for beta_final in [4000.0, 8000.0, 16000.0, 32000.0]:
            if time.time() - start_time > time_limit - 5.0:
                break
            
            pts_best = get_points(best_x)
            hull_area_target = compute_hull_area(pts_best)
            
            try:
                res_final = minimize(
                    smooth_objective,
                    best_x,
                    args=(beta_final, hull_area_target),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-13}
                )
                
                score = raw_score(res_final.x)
                if score > best_score:
                    best_x = res_final.x.copy()
                    best_score = score
            except Exception:
                pass
    
    # Phase 4: Nelder-Mead on raw objective for final polish
    if time.time() - start_time < time_limit - 4.0 and best_x is not None:
        try:
            res_nm = minimize(
                raw_objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            score = raw_score(res_nm.x)
            if score > best_score:
                best_x = res_nm.x.copy()
                best_score = score
        except Exception:
            pass
    
    # Phase 5: Very fine final polish
    if time.time() - start_time < time_limit - 1.0 and best_x is not None:
        pts_best = get_points(best_x)
        hull_area_target = compute_hull_area(pts_best)
        
        try:
            res_polish = minimize(
                smooth_objective,
                best_x,
                args=(65536.0, hull_area_target),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            score = raw_score(res_polish.x)
            if score > best_score:
                best_x = res_polish.x.copy()
                best_score = score
        except Exception:
            pass
    
    if best_x is None:
        best_x = initial_circle(0.0)
    
    return get_points(best_x)

# EVOLVE-BLOCK-END
