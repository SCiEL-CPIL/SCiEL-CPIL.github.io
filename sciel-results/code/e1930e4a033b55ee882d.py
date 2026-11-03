"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 13

# EVOLVE-BLOCK-START

def _min_triangle_area(pts):
    """Compute minimum area over all triples of points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            pj = pts[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = pts[k]
                v2 = pk - pi
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(pts):
    """Compute area of convex hull using Andrew's monotone chain."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.array(pts, dtype=float)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    
    area = 0.0
    m = len(hull)
    for i in range(m):
        j = (i + 1) % m
        area += hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1]
    return abs(area) * 0.5

def _score(pts):
    """Score = min triangle area / convex hull area."""
    min_area = _min_triangle_area(pts)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _ring_configuration(r_inner, theta_inner, r_outer, theta_outer):
    """
    Generate 13 points: 7 inner ring + 6 outer ring.
    Inner ring: 7 points at radius r_inner, rotated by theta_inner.
    Outer ring: 6 points at radius r_outer, rotated by theta_outer.
    """
    angles_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False) + theta_inner
    angles_outer = np.linspace(0, 2 * np.pi, 6, endpoint=False) + theta_outer
    
    inner_pts = np.column_stack((r_inner * np.cos(angles_inner), 
                                  r_inner * np.sin(angles_inner)))
    outer_pts = np.column_stack((r_outer * np.cos(angles_outer), 
                                  r_outer * np.sin(angles_outer)))
    
    return np.vstack([inner_pts, outer_pts])

def _objective(params):
    """Negative score for minimization."""
    r_inner, theta_inner, r_outer, theta_outer = params
    if r_inner < 0.01 or r_outer < 0.1 or r_inner >= r_outer:
        return -1e-10
    pts = _ring_configuration(r_inner, theta_inner, r_outer, theta_outer)
    return -_score(pts)

def _perturb_and_refine(pts, n_steps=500, step_scale=0.01, deadline=None):
    """Local refinement via coordinate descent on individual points."""
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(len(best_pts)):
            for dim in range(2):
                for delta in [step_scale, -step_scale, step_scale * 0.1, -step_scale * 0.1]:
                    trial = best_pts.copy()
                    trial[i, dim] += delta
                    s = _score(trial)
                    if s > best_score:
                        best_pts = trial
                        best_score = s
                        improved = True
                        step_scale *= 0.7
                    else:
                        step_scale *= 0.95
        if not improved:
            step_scale *= 0.8
            if step_scale < 1e-5:
                break
    
    return best_pts, best_score

def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Grid search over ring parameters
    r_inner_vals = np.linspace(0.2, 0.7, 15)
    r_outer_vals = np.linspace(0.8, 1.2, 15)
    theta_inner_vals = np.linspace(0, 2 * np.pi / 7, 8, endpoint=False)
    theta_outer_vals = np.linspace(0, 2 * np.pi / 6, 8, endpoint=False)
    
    for ri in r_inner_vals:
        if time.time() > deadline:
            break
        for ro in r_outer_vals:
            if ro <= ri:
                continue
            for ti in theta_inner_vals:
                if time.time() > deadline:
                    break
                for to in theta_outer_vals:
                    pts = _ring_configuration(ri, ti, ro, to)
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_pts = pts.copy()
    
    # Phase 2: Nelder-Mead refinement on best grid point
    if best_pts is not None:
        # Extract approximate parameters from best grid config
        # We'll just run NM from the best grid point
        r_inner_best = 0.45
        r_outer_best = 1.0
        theta_inner_best = 0.0
        theta_outer_best = 0.0
        
        # Find best grid parameters more carefully
        for ri in r_inner_vals:
            for ro in r_outer_vals:
                if ro <= ri:
                    continue
                for ti in theta_inner_vals:
                    for to in theta_outer_vals:
                        pts = _ring_configuration(ri, ti, ro, to)
                        s = _score(pts)
                        if abs(s - best_score) < 1e-10:
                            r_inner_best = ri
                            r_outer_best = ro
                            theta_inner_best = ti
                            theta_outer_best = to
        
        if time.time() < deadline - 30:
            x0 = [r_inner_best, theta_inner_best, r_outer_best, theta_outer_best]
            result = minimize(_objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < -best_score:
                r_i, th_i, r_o, th_o = result.x
                if r_i > 0.01 and r_o > 0.1 and r_i < r_o:
                    pts = _ring_configuration(r_i, th_i, r_o, th_o)
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_pts = pts.copy()
    
    # Phase 3: Local perturbation refinement on all 13 points
    if best_pts is not None and time.time() < deadline - 10:
        best_pts, best_score = _perturb_and_refine(best_pts, n_steps=200, 
                                                     step_scale=0.02, 
                                                     deadline=deadline - 5)
    
    # Phase 4: Multiple restarts with random perturbations of ring structure
    if best_pts is not None:
        remaining_time = deadline - time.time() - 5
        if remaining_time > 10:
            rng = np.random.RandomState(42)
            n_restarts = min(20, int(remaining_time / 5))
            for restart in range(n_restarts):
                if time.time() > deadline:
                    break
                # Start from best config with small random perturbation
                pts = best_pts.copy()
                pts += rng.normal(0, 0.05, size=pts.shape)
                # Also try different ring ratios
                r_i = 0.3 + rng.random() * 0.3
                r_o = 0.9 + rng.random() * 0.3
                th_i = rng.random() * 2 * np.pi
                th_o = rng.random() * 2 * np.pi
                pts2 = _ring_configuration(r_i, th_i, r_o, th_o)
                
                for candidate in [pts, pts2]:
                    s = _score(candidate)
                    if s > best_score:
                        best_score = s
                        best_pts = candidate.copy()
                        # Quick local refinement
                        if time.time() < deadline - 5:
                            best_pts, best_score = _perturb_and_refine(best_pts, 
                                                                        n_steps=50, 
                                                                        step_scale=0.01,
                                                                        deadline=deadline - 3)
    
    # Normalize: center at centroid, scale to unit area convex hull
    if best_pts is not None:
        centroid = np.mean(best_pts, axis=0)
        best_pts = best_pts - centroid
        hull_area = _convex_hull_area(best_pts)
        if hull_area > 1e-15:
            scale = 1.0 / np.sqrt(hull_area)
            best_pts = best_pts * scale
    
    return best_pts
# EVOLVE-BLOCK-END
