import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

def construct():
    """
    Maximize minimum triangle area / convex hull area for 14 points.
    Uses multi-scale optimization with log-sum-exp smoothing and direct refinement.
    """
    triples = list(itertools.combinations(range(N), 3))
    n_triples = len(triples)
    
    # Precompute triangle indices
    tri_i = np.array([t[0] for t in triples])
    tri_j = np.array([t[1] for t in triples])
    tri_k = np.array([t[2] for t in triples])
    
    def get_points(x):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts
    
    def compute_areas_vectorized(pts):
        pi = pts[tri_i]
        pj = pts[tri_j]
        pk = pts[tri_k]
        v1 = pj - pi
        v2 = pk - pi
        dets = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(dets)
        return areas
    
    def compute_hull_area(pts):
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 1.0
    
    def objective_smooth(x, beta):
        pts = get_points(x)
        areas = compute_areas_vectorized(pts)
        
        a_min = np.min(areas)
        if a_min < 1e-15:
            return 1.0
        
        # Log-sum-exp for smooth minimum
        a_ref = np.min(areas)
        exp_terms = np.exp(-beta * (areas - a_ref))
        log_sum = -beta * a_ref + np.log(np.sum(exp_terms))
        smooth_min = -log_sum / beta
        
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
        
        return -(smooth_min / hull_area)
    
    def objective_raw(x):
        pts = get_points(x)
        areas = compute_areas_vectorized(pts)
        min_area = np.min(areas)
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
        return -(min_area / hull_area)
    
    def initial_reg_polygon(n_points, center=(0.5, 0.5), radius=0.5, angle_offset=0.0):
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False) + angle_offset
        pts = np.zeros((n_points, 2))
        for i in range(n_points):
            pts[i, 0] = center[0] + radius * np.cos(angles[i])
            pts[i, 1] = center[1] + radius * np.sin(angles[i])
        return pts
    
    def pts_to_x(pts):
        x = np.zeros(2 * (N - 2))
        for i in range(N - 2):
            x[2 * i] = pts[i + 2, 0]
            x[2 * i + 1] = pts[i + 2, 1]
        return x
    
    def score_config(pts):
        """Compute the score for a point configuration"""
        areas = compute_areas_vectorized(pts)
        min_area = np.min(areas)
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 0.0
        return min_area / hull_area
    
    start_time = time.time()
    time_limit = 115.0
    
    best_score = -np.inf
    best_x = None
    
    dim = 2 * (N - 2)
    
    # Generate diverse initial configurations
    def generate_initials():
        initials = []
        
        # Regular polygon with various parameters
        for radius in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            for angle_offset in [0.0, np.pi/14, np.pi/7, 2*np.pi/14, 3*np.pi/14]:
                pts = initial_reg_polygon(N, center=(0.5, 0.5), radius=radius, angle_offset=angle_offset)
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                initials.append(pts_to_x(pts))
        
        # Elliptical distributions
        for rx in [0.35, 0.45, 0.55, 0.65]:
            for ry in [0.25, 0.35, 0.45, 0.55]:
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
                pts = np.zeros((N, 2))
                for i in range(N):
                    pts[i, 0] = 0.5 + rx * np.cos(angles[i])
                    pts[i, 1] = 0.5 + ry * np.sin(angles[i])
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                initials.append(pts_to_x(pts))
        
        # Random perturbations of regular polygon
        base_pts = initial_reg_polygon(N, center=(0.5, 0.5), radius=0.5)
        for sigma in [0.05, 0.1, 0.15]:
            for _ in range(3):
                perturbed = base_pts.copy()
                perturbed += np.random.normal(0, sigma, size=(N, 2))
                perturbed[0] = [0.0, 0.0]
                perturbed[1] = [1.0, 0.0]
                initials.append(pts_to_x(perturbed))
        
        return initials
    
    # Phase 1: Coarse optimization with multiple betas
    betas_coarse = [30.0, 60.0, 120.0, 250.0, 500.0]
    initials = generate_initials()
    
    for beta in betas_coarse:
        if time.time() - start_time > time_limit - 25.0:
            break
        
        for x0 in initials:
            if time.time() - start_time > time_limit - 25.0:
                break
            
            try:
                res = minimize(
                    objective_smooth,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(-1.5, 1.5)] * dim,
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 3000}
                )
                
                if res.fun < -best_score:
                    best_x = res.x.copy()
                    best_score = -res.fun
            except Exception:
                pass
        
        # Update initials with best solution
        if best_x is not None:
            new_initials = [best_x.copy()]
            for sigma in [0.05, 0.1, 0.15, 0.2]:
                for _ in range(2):
                    jitter = np.random.normal(0, sigma, size=dim)
                    new_initials.append(best_x + jitter)
            new_initials.extend(initials[:3])
            initials = new_initials
    
    # Phase 2: Fine optimization
    betas_fine = [1000.0, 2000.0, 4000.0, 8000.0, 15000.0]
    
    if best_x is not None:
        for beta in betas_fine:
            if time.time() - start_time > time_limit - 15.0:
                break
            
            candidates = [best_x.copy()]
            for sigma in [0.02, 0.05, 0.1]:
                for _ in range(4):
                    jitter = np.random.normal(0, sigma, size=dim)
                    candidates.append(best_x + jitter)
            
            for x0 in candidates:
                if time.time() - start_time > time_limit - 15.0:
                    break
                
                try:
                    res = minimize(
                        objective_smooth,
                        x0,
                        args=(beta,),
                        method='L-BFGS-B',
                        bounds=[(-1.5, 1.5)] * dim,
                        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
                    )
                    
                    if res.fun < -best_score:
                        best_x = res.x.copy()
                        best_score = -res.fun
                except Exception:
                    pass
    
    # Phase 3: Ultra-fine with very high beta
    if time.time() - start_time < time_limit - 8.0 and best_x is not None:
        for beta_final in [30000.0, 50000.0, 100000.0]:
            if time.time() - start_time > time_limit - 5.0:
                break
            
            try:
                res_final = minimize(
                    objective_smooth,
                    best_x,
                    args=(beta_final,),
                    method='L-BFGS-B',
                    bounds=[(-1.5, 1.5)] * dim,
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 10000}
                )
                
                if res_final.fun < -best_score:
                    best_x = res_final.x.copy()
                    best_score = -res_final.fun
            except Exception:
                pass
    
    # Phase 4: Direct Nelder-Mead refinement
    if time.time() - start_time < time_limit - 3.0 and best_x is not None:
        try:
            res_nm = minimize(
                objective_raw,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_nm.fun < -best_score:
                best_x = res_nm.x.copy()
                best_score = -res_nm.fun
        except Exception:
            pass
    
    # Phase 5: Final local search with small steps
    if time.time() - start_time < time_limit - 1.0 and best_x is not None:
        for _ in range(50):
            if time.time() - start_time > time_limit - 0.5:
                break
            for sigma in [0.005, 0.01, 0.02]:
                jitter = np.random.normal(0, sigma, size=dim)
                candidate = best_x + jitter
                try:
                    s = -objective_raw(candidate)
                    if s > best_score:
                        best_x = candidate
                        best_score = s
                except Exception:
                    pass
    
    # Fallback
    if best_x is None:
        pts = initial_reg_polygon(N, center=(0.5, 0.5), radius=0.5)
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        best_x = pts_to_x(pts)
    
    return get_points(best_x)
