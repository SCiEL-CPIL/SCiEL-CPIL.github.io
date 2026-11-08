import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 115.0

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def build_symmetric(params):
        """12 outer + 1 inner with 12-fold symmetry."""
        r_outer = params[0]
        r_inner = params[1]
        angle_off_outer = params[2]
        angle_off_inner = params[3]
        
        angles_outer = np.linspace(0, 2*np.pi, 12, endpoint=False) + angle_off_outer
        outer = np.column_stack([r_outer * np.cos(angles_outer), 
                                  r_outer * np.sin(angles_outer)])
        
        inner = np.array([[r_inner * np.cos(angle_off_inner), 
                           r_inner * np.sin(angle_off_inner)]])
        
        return np.vstack([outer, inner])

    def build_regular(params):
        """13 points equally spaced on a circle."""
        r = params[0]
        angle_off = params[1]
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False) + angle_off
        return np.column_stack([r * np.cos(angles), r * np.sin(angles)])

    def build_two_ring(params):
        """7 outer + 6 inner."""
        r1, r2 = params[0], params[1]
        off1, off2 = params[2], params[3]
        
        angles1 = np.linspace(0, 2*np.pi, 7, endpoint=False) + off1
        ring1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        
        angles2 = np.linspace(0, 2*np.pi, 6, endpoint=False) + off2
        ring2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        return np.vstack([ring1, ring2])

    def build_three_ring(params):
        """5 + 4 + 4."""
        r1, r2, r3 = params[0], params[1], params[2]
        off1, off2, off3 = params[3], params[4], params[5]
        
        angles1 = np.linspace(0, 2*np.pi, 5, endpoint=False) + off1
        ring1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        
        angles2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + off2
        ring2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        angles3 = np.linspace(0, 2*np.pi, 4, endpoint=False) + off3
        ring3 = np.column_stack([r3 * np.cos(angles3), r3 * np.sin(angles3)])
        
        return np.vstack([ring1, ring2, ring3])

    best_points = None
    best_score = -float('inf')

    families = [
        (build_symmetric, np.array([1.0, 0.3, 0.0, 0.0])),
        (build_regular, np.array([1.0, 0.0])),
        (build_two_ring, np.array([1.0, 0.5, 0.0, np.pi/6])),
        (build_three_ring, np.array([1.0, 0.6, 0.3, 0.0, np.pi/4, np.pi/8])),
    ]

    for builder, x0 in families:
        if time.time() > start_time + time_limit - 5:
            break
        try:
            def obj_sym(params):
                pts = builder(params)
                s = compute_score(pts)
                return -s if s > 0 else 1e6
            
            res = minimize(obj_sym, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_points = builder(res.x)
        except Exception:
            pass

    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    if time.time() < start_time + time_limit - 2:
        try:
            def obj_full(pts_flat):
                pts = pts_flat.reshape(-1, 2)
                s = compute_score(pts)
                return -s if s > 0 else 1e6
            
            res_polish = minimize(obj_full, best_points.flatten(), method='Powell',
                                  options={'maxiter': 100, 'xtol': 1e-12, 'ftol': 1e-12})
            s_before = compute_score(best_points)
            s_after = -res_polish.fun
            if s_after > s_before:
                best_points = res_polish.x.reshape(-1, 2)
        except Exception:
            pass

    return best_points
