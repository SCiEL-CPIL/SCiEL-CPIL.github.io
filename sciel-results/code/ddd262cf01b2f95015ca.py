from scipy.optimize import differential_evolution
from scipy.optimize import minimize
import numpy as np
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Define objective
    # We fix p[0] = (0,0) and p[1] = (1,0).
    # Variables are x[0:24] representing p[2]...p[13].
    
    def get_points(x):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = x.reshape(-1, 2)
        return pts

    def min_triangle_area(pts):
        # Calculate min area of all triangles
        # Using cross product: 0.5 * | (b-a) x (c-a) |
        # To avoid sqrt/slow ops, we can just compute the cross product magnitude.
        # Area = 0.5 * |cross|. Minimizing Area is same as minimizing |cross|.
        # We want to MAXIMIZE min Area. So we MINIMIZE -min(|cross|).
        
        min_cross = float('inf')
        # Precompute indices for speed? N=14 is small. 364 triangles.
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    a = pts[i]
                    b = pts[j]
                    c = pts[k]
                    # Cross product z-component
                    cross = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
                    if abs(cross) < min_cross:
                        min_cross = abs(cross)
        return -min_cross

    def hull_area(pts):
        try:
            hull = ConvexHull(pts)
            return hull.volume # In 2D, volume is area
        except Exception:
            return 1e-9 # Degenerate case

    def final_score(pts):
        # For reporting
        areas = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    a = pts[i]
                    b = pts[j]
                    c = pts[k]
                    cross = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
                    areas.append(abs(cross))
        min_area = 0.5 * min(areas)
        h_area = hull_area(pts)
        if h_area == 0:
            return 0
        return min_area / h_area

    # 2. Generate Initial Guesses
    
    def make_regular_polygon():
        # Regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        pts = np.column_stack((np.cos(angles), np.sin(angles)))
        # Translate and rotate to fix p0, p1
        # p0 is currently (1,0). We want p0 at (0,0).
        pts = pts - pts[0]
        # Now p0 is (0,0). p1 is at angle 2pi/14.
        # Rotate so p1 is at (1,0).
        angle_p1 = np.arctan2(pts[1,1], pts[1,0])
        cos_a = np.cos(-angle_p1)
        sin_a = np.sin(-angle_p1)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        pts = pts @ rot.T
        # Scale so distance p0-p1 is 1
        dist = np.linalg.norm(pts[1] - pts[0])
        pts = pts / dist
        # Extract variables
        return pts[2:].flatten()

    def make_two_heptagons(r_ratio=0.5, offset=0):
        # 7 points on inner, 7 on outer
        # Interleaved angles
        angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
        angles_inner = angles_outer + np.pi/7 + offset
        
        pts_outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
        pts_inner = r_ratio * np.column_stack((np.cos(angles_inner), np.sin(angles_inner)))
        
        pts = np.vstack([pts_outer, pts_inner])
        # Sort by angle to ensure convex order? 
        # Actually, for optimization, order doesn't matter for the set of points.
        # But for hull, it's better if they are roughly ordered.
        # The function get_points just takes the array.
        
        # Transform to fixed p0, p1
        # Find closest point to (1,0) and (0,0)? 
        # Simpler: Just apply the same rigid transform logic.
        # Pick a point to be p0 and the next one to be p1.
        # Let's just use the regular polygon transform on this set.
        # This might distort the "two heptagon" structure slightly if the points aren't evenly spaced, 
        # but it's a valid initial guess.
        
        # To make it a valid initial guess for the variable space (p2..p13),
        # we just need to apply the transform.
        
        # Translate
        pts = pts - pts[0]
        # Rotate
        angle_p1 = np.arctan2(pts[1,1], pts[1,0])
        cos_a = np.cos(-angle_p1)
        sin_a = np.sin(-angle_p1)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        pts = pts @ rot.T
        # Scale
        dist = np.linalg.norm(pts[1] - pts[0])
        if dist > 0:
            pts = pts / dist
        else:
            pts = make_regular_polygon().reshape(-1, 2) # Fallback
            
        return pts[2:].flatten()

    def make_random_convex():
        angles = np.sort(np.random.uniform(0, 2*np.pi, N))
        radii = np.random.uniform(0.5, 1.0, N)
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
        # Transform
        pts = pts - pts[0]
        angle_p1 = np.arctan2(pts[1,1], pts[1,0])
        cos_a = np.cos(-angle_p1)
        sin_a = np.sin(-angle_p1)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        pts = pts @ rot.T
        dist = np.linalg.norm(pts[1] - pts[0])
        if dist > 0:
            pts = pts / dist
        else:
            return make_regular_polygon()
            
        return pts[2:].flatten()

    # 3. Optimization Loop
    
    best_x = None
    best_score = 0
    
    initial_guesses = [
        make_regular_polygon(),
        make_two_heptagons(0.5),
        make_two_heptagons(0.7),
        make_two_heptagons(0.8),
        make_random_convex(),
        make_random_convex()
    ]
    
    for x0 in initial_guesses:
        if time.time() > deadline:
            break
            
        try:
            # Nelder-Mead
            res = minimize(min_triangle_area, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.success or res.fun < (best_score if best_x is not None else float('inf')):
                # Note: res.fun is -min_cross. We want to maximize min_cross.
                # So lower res.fun is better.
                if best_x is None or res.fun < best_score_val:
                    best_x = res.x
                    best_score_val = res.fun
        except Exception:
            pass

    # If we have time, try Differential Evolution
    if time.time() < deadline - 20:
        try:
            bounds = [(-2, 2)] * (2*(N-2))
            # DE is slow. Use small pop.
            res_de = differential_evolution(min_triangle_area, bounds, maxiter=50, popsize=10, seed=42)
            if best_x is None or res_de.fun < best_score_val:
                best_x = res_de.x
                best_score_val = res_de.fun
        except Exception:
            pass

    # Final Polish with Nelder-Mead on the best found
    if best_x is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(min_triangle_area, best_x, method='Nelder-Mead', 
                                 options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            best_x = res_final.x
        except Exception:
            pass

    if best_x is None:
        best_x = make_regular_polygon()

    pts = get_points(best_x)
    return pts
