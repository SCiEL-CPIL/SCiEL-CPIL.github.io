import numpy as np
from scipy.spatial import ConvexHull
import random

N = 13

def construct():
    # Initial guess: Regular 13-gon
    theta0 = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = np.column_stack((np.cos(theta0), np.sin(theta0)))
    
    def score(pts):
        # Check convexity of the polygon defined by pts in given order
        # Assume pts are in counter-clockwise order
        # This is true for small perturbations of a convex polygon
        is_convex = True
        first_sign = 0
        for i in range(N):
            p0 = pts[i]
            p1 = pts[(i+1)%N]
            p2 = pts[(i+2)%N]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            if abs(cross) < 1e-9:
                is_convex = False
                break
            sign = 1 if cross > 0 else -1
            if first_sign == 0:
                first_sign = sign
            elif sign != first_sign:
                is_convex = False
                break
        
        if is_convex:
            # Calculate area via shoelace
            area = 0.0
            for i in range(N):
                x0, y0 = pts[i]
                x1, y1 = pts[(i+1)%N]
                area += x0*y1 - x1*y0
            area = 0.5 * abs(area)
            
            if area < 1e-10:
                return -1.0
        else:
            # Fallback to ConvexHull
            try:
                hull = ConvexHull(pts)
                area = hull.volume
                if area < 1e-10:
                    return -1.0
            except Exception:
                return -1.0
                
        # Min triangle area
        min_area = np.inf
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    ax = pts[j, 0] - pts[i, 0]
                    ay = pts[j, 1] - pts[i, 1]
                    bx = pts[k, 0] - pts[i, 0]
                    by = pts[k, 1] - pts[i, 1]
                    area_t = 0.5 * abs(ax * by - ay * bx)
                    if area_t < min_area:
                        min_area = area_t
                        
        return min_area / area

    # Simulated Annealing
    best_pts = pts.copy()
    best_score = score(best_pts)
    curr_pts = pts.copy()
    curr_score = best_score
    
    T = 0.05 # Initial temperature
    T_min = 1e-5
    alpha = 0.9995 # Cooling rate
    max_iter = 50000
    
    step_size = 0.1
    
    for it in range(max_iter):
        # Generate neighbor
        # Randomly perturb 1 or 2 points
        num_perturb = random.randint(1, 2)
        indices = random.sample(range(N), num_perturb)
        new_pts = curr_pts.copy()
        for idx in indices:
            # Gaussian noise
            noise = np.random.normal(0, step_size, 2)
            new_pts[idx] += noise
            
        new_score = score(new_pts)
        
        if new_score > curr_score:
            curr_pts = new_pts
            curr_score = new_score
            if curr_score > best_score:
                best_score = curr_score
                best_pts = curr_pts.copy()
        else:
            # Accept with probability exp((new - curr) / T)
            if new_score > -1.0: # Check validity
                delta = new_score - curr_score
                if delta > 0:
                    accept = True
                else:
                    # Avoid underflow
                    if T > 1e-12:
                        prob = np.exp(delta / T)
                        accept = random.random() < prob
                    else:
                        accept = False
                if accept:
                    curr_pts = new_pts
                    curr_score = new_score
                    
        T *= alpha
        if T < T_min:
            T = T_min
        step_size *= 0.9999 # Slowly reduce step size too

    return best_pts
