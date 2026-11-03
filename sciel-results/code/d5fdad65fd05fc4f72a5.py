import numpy as np
from scipy.spatial import ConvexHull
import random
import time
from scipy.optimize import minimize

N = 13

# EVOLVE-BLOCK-START
def construct():
    n = N
    
    # Precompute triangle indices
    tri_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    
    def get_score(pts):
        """Calculate minimum triangle area / convex hull area"""
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area <= 1e-10:
                return 0.0
        except:
            return 0.0
            
        p1 = pts[tri_indices[:, 0]]
        p2 = pts[tri_indices[:, 1]]
        p3 = pts[tri_indices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        return min_area / hull_area

    def objective(pts_flat):
        """Objective function for minimization: -score"""
        pts = pts_flat.reshape(n, 2)
        # Normalize to unit area
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-10:
                return 1e6 # Penalty
            pts = pts / np.sqrt(area)
        except:
            return 1e6
        
        score = get_score(pts)
        return -score

    def generate_initial_points(method='random'):
        if method == 'fermat':
            i = np.arange(n)
            r = np.sqrt(i / n)
            theta = i * 2.399963229728653
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points = np.column_stack((x, y))
        elif method == 'circle':
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            r = np.ones(n)
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            points = np.column_stack((x, y))
        else:
            points = np.random.randn(n, 2)
        
        # Normalize
        try:
            hull = ConvexHull(points)
            area = hull.volume
            if area > 1e-10:
                points /= np.sqrt(area)
        except:
            pass
            
        return points

    start_time = time.time()
    max_time = 110  # Leave buffer
    
    best_score = -np.inf
    best_points = None
    
    # Multi-start Nelder-Mead with refinement
    methods = ['fermat', 'circle', 'random', 'random', 'random']
    
    for idx, method in enumerate(methods):
        if time.time() - start_time > max_time:
            break
            
        pts0 = generate_initial_points(method)
        x0 = pts0.flatten()
        
        # Run Nelder-Mead
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7})
            if result.fun < 0: # Valid solution
                final_pts_flat = result.x
                final_pts = final_pts_flat.reshape(n, 2)
                
                # Normalize
                try:
                    hull = ConvexHull(final_pts)
                    area = hull.volume
                    if area > 1e-10:
                        final_pts /= np.sqrt(area)
                except:
                    pass
                
                score = get_score(final_pts)
                if score > best_score:
                    best_score = score
                    best_points = final_pts.copy()
                    
                    # Refinement: Perturb and run again
                    if time.time() - start_time < max_time - 10:
                        perturbed_pts = final_pts + np.random.randn(n, 2) * 0.05
                        x_pert = perturbed_pts.flatten()
                        try:
                            result2 = minimize(objective, x_pert, method='Nelder-Mead', 
                                              options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7})
                            if result2.fun < 0:
                                final_pts_flat2 = result2.x
                                final_pts2 = final_pts_flat2.reshape(n, 2)
                                try:
                                    hull = ConvexHull(final_pts2)
                                    area = hull.volume
                                    if area > 1e-10:
                                        final_pts2 /= np.sqrt(area)
                                except:
                                    pass
                                score2 = get_score(final_pts2)
                                if score2 > best_score:
                                    best_score = score2
                                    best_points = final_pts2.copy()
                        except:
                            pass
        except:
            pass

    if best_points is None:
        # Fallback
        i = np.arange(n)
        r = np.sqrt(i / n)
        theta = i * 2.399963229728653
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        best_points = np.column_stack((x, y))
        best_points -= np.mean(best_points, axis=0)
        
    return best_points
# EVOLVE-BLOCK-END
