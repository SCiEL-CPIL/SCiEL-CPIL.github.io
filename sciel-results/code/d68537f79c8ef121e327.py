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
    n_tris = len(tri_indices)
    
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
        """Objective function for minimizers (negate score to maximize)"""
        pts = pts_flat.reshape(n, 2)
        # Center to avoid drift
        pts -= np.mean(pts, axis=0)
        score = get_score(pts)
        return -score

    def normalize_points(pts):
        """Center and scale points to unit area if possible"""
        pts = pts.copy()
        pts -= np.mean(pts, axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area > 1e-10:
                pts /= np.sqrt(area)
        except:
            pass
        return pts

    def simulated_annealing(initial_points, max_iter=2000, start_time=0, max_time=60):
        """
        Simulated Annealing to maximize the minimum triangle area ratio.
        """
        pts = initial_points.copy()
        pts -= np.mean(pts, axis=0)
        
        current_score = get_score(pts)
        best_score = current_score
        best_pts = pts.copy()
        
        T = 0.1  # Initial temperature
        T_min = 1e-5
        alpha = 0.999  # Cooling rate
        
        for i in range(max_iter):
            if time.time() - start_time > max_time:
                break
                
            # Generate neighbor: perturb a random point
            new_pts = pts.copy()
            idx = np.random.randint(n)
            # Perturbation magnitude scales with temperature
            step = 0.1 * T
            new_pts[idx] += np.random.randn(2) * step
            
            # Center new points
            new_pts -= np.mean(new_pts, axis=0)
            
            new_score = get_score(new_pts)
            
            delta = new_score - current_score
            
            # Accept if better, or with probability exp(delta/T)
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                pts = new_pts
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
            
            T *= alpha
            if T < T_min:
                T = T_min
                
        return best_pts, best_score

    def refine_with_nelder_mead(initial_points, start_time, max_time):
        """
        Refine a good solution using Nelder-Mead.
        """
        pts = initial_points.copy()
        pts -= np.mean(pts, axis=0)
        
        x0 = pts.flatten()
        
        # Use Nelder-Mead for derivative-free optimization
        # maxiter is limited to respect time
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            
            if res.success or res.fun < 0: # fun is -score, so < 0 means positive score
                refined_pts = res.x.reshape(n, 2)
                refined_pts -= np.mean(refined_pts, axis=0)
                return refine_with_nelder_mead(refined_pts, start_time, max_time) # Recurse? No, just return
                # Actually, just return the result
            else:
                return initial_points, get_score(initial_points)
        except Exception:
            return initial_points, get_score(initial_points)
        
        # Correct implementation without recursion
        # We just return the best of initial and refined
        refined_pts = res.x.reshape(n, 2)
        refined_pts -= np.mean(refined_pts, axis=0)
        refined_score = get_score(refined_pts)
        init_score = get_score(initial_points)
        
        if refined_score > init_score:
            return refined_pts, refined_score
        else:
            return initial_points, init_score

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
        elif method == 'random':
            points = np.random.randn(n, 2)
            points -= points.mean(axis=0)
        else:
            points = np.random.randn(n, 2)
            points -= points.mean(axis=0)
            
        return normalize_points(points)

    start_time = time.time()
    max_time = 110 # Leave 10s buffer
    
    best_points = None
    best_score = -np.inf
    
    # Strategy:
    # 1. Run SA from several diverse starts.
    # 2. Refine the best SA result with Nelder-Mead.
    # 3. If time permits, try another SA restart from the refined point.
    
    num_restarts = 3
    sa_time_per_restart = max_time / (num_restarts * 2) # Leave time for NM
    
    for i in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > max_time * 0.8:
            break
            
        # Generate start
        if i == 0:
            start_pts = generate_initial_points('fermat')
        elif i == 1:
            start_pts = generate_initial_points('circle')
        else:
            start_pts = generate_initial_points('random')
            
        # Run SA
        sa_pts, sa_score = simulated_annealing(start_pts, max_iter=5000, start_time=start_time, max_time=max_time - 30)
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts.copy()
            
    # Refine the best point found by SA
    if best_points is not None:
        elapsed = time.time() - start_time
        if elapsed < max_time - 10:
            # Use Nelder-Mead
            refined_pts, refined_score = refine_with_nelder_mead(best_points, start_time, max_time)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_pts.copy()
                
            # Optional: Another SA run from the refined point if time allows
            elapsed = time.time() - start_time
            if elapsed < max_time - 20:
                final_pts, final_score = simulated_annealing(refined_pts, max_iter=3000, start_time=start_time, max_time=max_time - 5)
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_pts.copy()
    
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
