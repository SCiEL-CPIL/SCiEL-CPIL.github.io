"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2.0
# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])
TRI_AREA = H / 2.0  # area of the equilateral triangle

def _all_triangle_areas(pts):
    """Compute all C(11,3)=165 triangle areas."""
    n = len(pts)
    areas = np.empty((n*(n-1)*(n-2))//6)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                dx1 = pts[j][0] - pts[i][0]
                dy1 = pts[j][1] - pts[i][1]
                dx2 = pts[k][0] - pts[i][0]
                dy2 = pts[k][1] - pts[i][1]
                areas[idx] = 0.5 * abs(dx1*dy2 - dy1*dx2)
                idx += 1
    return areas

def _min_area(pts):
    """Minimum triangle area among all triples."""
    return _all_triangle_areas(pts).min()

def _project_to_triangle(p):
    """Project point p onto the equilateral triangle."""
    x, y = p
    # Barycentric coordinates
    # p = a*V0 + b*V1 + c*V2, a+b+c=1
    # x = b + 0.5*c, y = H*c
    c = y / H
    b = x - 0.5 * c
    a = 1.0 - b - c
    # Clamp to [0,1]
    if a < 0:
        # Project onto edge V1-V2
        # a=0, b+c=1, x = b + 0.5*c = b + 0.5*(1-b) = 0.5 + 0.5*b => b = 2x - 1
        b = 2*x - 1
        c = 1 - b
        if b < 0:
            b = 0; c = 1
        elif b > 1:
            b = 1; c = 0
        a = 0
    elif b < 0:
        # Project onto edge V0-V2
        b = 0
        # x = 0.5*c => c = 2x, y = H*c => c = y/H
        c = min(1.0, max(0.0, y / H))
        a = 1 - c
    elif c < 0:
        # Project onto edge V0-V1
        c = 0
        b = min(1.0, max(0.0, x))
        a = 1 - b
    # Clamp
    a = max(0.0, min(1.0, a))
    b = max(0.0, min(1.0, b))
    c = max(0.0, min(1.0, c))
    s = a + b + c
    a /= s; b /= s; c /= s
    return a*V0 + b*V1 + c*V2

def _objective(x):
    """Negative of the minimum triangle area (for minimization)."""
    pts = np.array([_project_to_triangle(x[i*2:(i+1)*2]) for i in range(N)])
    return -_min_area(pts)

def _construct_initial(rng, method='grid'):
    """Generate initial point configurations."""
    if method == 'grid':
        # Place points on a grid-like pattern inside the triangle
        points = []
        # Use a few structured patterns
        # Pattern 1: vertices + midpoints + center
        pts = [V0 + 0.05*np.array([rng.random()-0.5, rng.random()-0.5]),
               V1 + 0.05*np.array([rng.random()-0.5, rng.random()-0.5]),
               V2 + 0.05*np.array([rng.random()-0.5, rng.random()-0.5])]
        # Add points along edges and interior
        for i in range(1, 5):
            t = i/5.0
            pts.append(t*V1 + (1-t)*V0 + 0.03*np.array([rng.random()-0.5, rng.random()-0.5]))
        for i in range(1, 4):
            t = i/4.0
            pts.append(t*V2 + (1-t)*V0 + 0.03*np.array([rng.random()-0.5, rng.random()-0.5]))
        for i in range(1, 3):
            t = i/3.0
            pts.append(t*V2 + (1-t)*V1 + 0.03*np.array([rng.random()-0.5, rng.random()-0.5]))
        pts.append(np.array([0.5, H/3.0]) + 0.03*np.array([rng.random()-0.5, rng.random()-0.5]))
        return np.array(pts[:N])
    elif method == 'random':
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 1.0:
                p = u*V1 + v*V2
                points.append(p)
        return np.array(points)
    else:
        # Fibonacci-like spiral
        points = []
        center = (V0 + V1 + V2) / 3.0
        for i in range(N):
            angle = i * 2.399963  # golden angle
            r = 0.35 * np.sqrt(i / N)
            p = center + r * np.array([np.cos(angle), np.sin(angle)])
            p = _project_to_triangle(p)
            points.append(p)
        return np.array(points)

def _local_optimize(pts, rng, max_iter=200):
    """Local optimization using Nelder-Mead on the minimum area."""
    x0 = np.concatenate([pts.flatten()])
    
    best_x = x0.copy()
    best_val = -_min_area(pts)
    
    # Try Nelder-Mead
    try:
        res = minimize(_objective, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except:
        pass
    
    # Gradient-based refinement using critical triangles
    pts_opt = np.array([_project_to_triangle(best_x[i*2:(i+1)*2]) for i in range(N)])
    
    for _ in range(30):
        areas = _all_triangle_areas(pts_opt)
        min_a = areas.min()
        # Find the critical triangles (those with area close to min)
        critical_mask = areas < min_a * 1.05
        crit_idx = np.where(critical_mask)[0]
        
        if len(crit_idx) == 0:
            break
            
        # For each critical triangle, try to move its vertices to increase area
        improved = False
        for ci in crit_idx[:5]:  # Limit to top 5 critical
            # Decode which points form this triangle
            # Reconstruct indices
            i, j, k = _decode_triangle_index(ci, N)
            pi, pj, pk = pts_opt[i], pts_opt[j], pts_opt[k]
            
            # Direction to increase area: move points apart
            # Use centroid-based repulsion
            centroid = (pi + pj + pk) / 3.0
            for idx, p in [(i, pi), (j, pj), (k, pk)]:
                dir_vec = p - centroid
                if np.linalg.norm(dir_vec) > 1e-10:
                    dir_vec = dir_vec / np.linalg.norm(dir_vec)
                    # Try small step
                    new_p = _project_to_triangle(p + 0.02 * dir_vec)
                    old_pts = pts_opt.copy()
                    pts_opt[idx] = new_p
                    new_min = _min_area(pts_opt)
                    if new_min > min_a + 1e-10:
                        min_a = new_min
                        improved = True
                    else:
                        pts_opt = old_pts
        
        if not improved:
            break
    
    return pts_opt, -best_val

def _decode_triangle_index(idx, n):
    """Decode linear index to (i,j,k) for i<j<k."""
    # Precompute or brute force
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if count == idx:
                    return i, j, k
                count += 1
    return 0, 1, 2

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Leave margin
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_score = -np.inf
    
    # Multiple restarts with different initializations
    methods = ['grid', 'random', 'spiral']
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        method = methods[restart % len(methods)]
        pts_init = _construct_initial(rng, method)
        
        # Local optimization
        pts_opt, score = _local_optimize(pts_init, rng, max_iter=100)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
    
    # Final refinement with more iterations if time permits
    if best_pts is not None and time.time() < deadline - 10:
        pts_final, score = _local_optimize(best_pts, rng, max_iter=300)
        if score > best_score:
            best_score = score
            best_pts = pts_final
    
    if best_pts is None:
        # Fallback
        best_pts = _construct_initial(rng, 'grid')
    
    return np.array(best_pts)
