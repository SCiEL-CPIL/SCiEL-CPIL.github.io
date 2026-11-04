import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

# Precompute triple indices for vectorized area computation
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _compute_areas(points):
    """
    Compute all triangle areas efficiently.
    area(i,j,k) = 0.5 * |det(p_j - p_i, p_k - p_i)|
                 = 0.5 * |p_i x p_j + p_j x p_k + p_k x p_i|
    Let C[i,j] = p_i x p_j = p_i[0]*p_j[1] - p_i[1]*p_j[0]
    Then p_i x p_j + p_j x p_k + p_k x p_i = C[i,j] + C[j,k] + C[k,i]
    Note: C[k,i] = -C[i,k]
    So sum = C[i,j] + C[j,k] - C[i,k]
    """
    p = points
    # C is an N x N matrix where C[i,j] = cross(p_i, p_j)
    C = p[:, 0:1] * p[:, 1:2].T - p[:, 1:2] * p[:, 0:1].T
    
    i_idx = TRIPLE_INDICES[:, 0]
    j_idx = TRIPLE_INDICES[:, 1]
    k_idx = TRIPLE_INDICES[:, 2]
    
    # Vectorized calculation for all triples
    vals = C[i_idx, j_idx] + C[j_idx, k_idx] - C[i_idx, k_idx]
    return 0.5 * np.abs(vals)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _calculate_score(points):
    """Exact score: min triangle area / hull area."""
    if points.shape != (N, 2):
        return 0.0
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0
    areas = _compute_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_score(points, temperature=0.01):
    """
    Smooth surrogate score using log-sum-exp for the minimum triangle area.
    This makes the objective differentiable, allowing gradient-based optimization.
    """
    if points.shape != (N, 2):
        return 0.0
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0
    areas = _compute_areas(points)
    
    # Soft min via log-sum-exp
    # soft_min(x) = -T * log(sum(exp(-x/T)))
    # To avoid overflow, we shift by max:
    # soft_min(x) = max(x) - T * log(sum(exp((x - max(x))/T)))
    max_a = np.max(areas)
    if max_a < 1e-12:
        return 0.0
    shifted = areas - max_a
    log_sum = np.log(np.sum(np.exp(shifted / temperature)))
    soft_min_val = max_a - temperature * log_sum
    
    return soft_min_val / hull_area

def _generate_two_ring(n_out, n_in, r_in, phi_rel):
    """Generate points on two concentric rings."""
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    if n_in > 0:
        angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + phi_rel
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        return np.vstack((pts_out, pts_in))
    else:
        return pts_out

def _polish_points(points, time_limit=5.0):
    """
    High-precision polish using smooth surrogate (L-BFGS-B) followed by exact local search (Nelder-Mead).
    """
    start_time = time.time()
    best_pts = points.copy()
    best_score = _calculate_score(points)
    
    # Phase 1: L-BFGS-B with soft-min surrogate (temperature 0.05)
    # This helps escape flat regions where the exact min is non-differentiable.
    temp1 = 0.05
    def neg_smooth_score1(x):
        p = x.reshape(N, 2)
        return -_soft_min_score(p, temp1)
    
    try:
        res1 = minimize(neg_smooth_score1, points.flatten(), method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
        new_pts1 = res1.x.reshape(N, 2)
        score1 = _calculate_score(new_pts1)
        if score1 > best_score:
            best_pts = new_pts1
            best_score = score1
    except Exception:
        pass

    # Phase 2: L-BFGS-B with lower temperature (0.01) for finer adjustment
    if time.time() - start_time < time_limit * 0.5:
        temp2 = 0.01
        def neg_smooth_score2(x):
            p = x.reshape(N, 2)
            return -_soft_min_score(p, temp2)
        
        try:
            res2 = minimize(neg_smooth_score2, best_pts.flatten(), method='L-BFGS-B',
                            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            new_pts2 = res2.x.reshape(N, 2)
            score2 = _calculate_score(new_pts2)
            if score2 > best_score:
                best_pts = new_pts2
                best_score = score2
        except Exception:
            pass

    # Phase 3: Exact local polish using Nelder-Mead on the true objective
    # This snaps the points to the exact local maximum of the min-area function.
    if time.time() - start_time < time_limit * 0.8:
        def neg_exact_score(x):
            p = x.reshape(N, 2)
            return -_calculate_score(p)
        
        try:
            res3 = minimize(neg_exact_score, best_pts.flatten(), method='Nelder-Mead',
                            options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500, 'disp': False})
            new_pts3 = res3.x.reshape(N, 2)
            score3 = _calculate_score(new_pts3)
            if score3 > best_score:
                best_pts = new_pts3
                best_score = score3
        except Exception:
            pass
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    best_score = 0
    best_points = None
    
    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Strategy: Try various ring configurations to find good initial basins
    # Then polish each one thoroughly.
    
    splits = [
        (14, 0),  # Single ring (regular polygon)
        (13, 1),  # 13 outer, 1 inner
        (12, 2),  # 12 outer, 2 inner
        (11, 3),  # 11 outer, 3 inner
        (10, 4),  # 10 outer, 4 inner
        (9, 5),   # 9 outer, 5 inner
        (8, 6),   # 8 outer, 6 inner
        (7, 7),   # 7 outer, 7 inner
    ]
    
    for n_out, n_in in splits:
        if time.time() > deadline - 15.0:
            break
            
        if n_in == 0:
            # Regular polygon
            angles = np.linspace(0, 2*np.pi, n_out, endpoint=False)
            pts = np.column_stack((np.cos(angles), np.sin(angles)))
        else:
            # Optimize ring parameters quickly to find a good starting point
            # We use a coarse grid + Nelder-Mead on the 2D parameter space (r_in, phi_rel)
            # instead of the full 28D space initially.
            
            def ring_objective(params):
                r_in, phi_rel = params
                if r_in < 0.0 or r_in > 1.5:
                    return 1e6
                pts = _generate_two_ring(n_out, n_in, r_in, phi_rel)
                return -_calculate_score(pts)
            
            best_r = 0.5
            best_phi = 0.0
            best_obj = 1e9
            
            # Coarse grid search
            r_starts = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
            phi_starts = [0.0, np.pi/n_in, 2*np.pi/n_in]
            
            for r0 in r_starts:
                for phi0 in phi_starts:
                    if time.time() > deadline - 10.0:
                        break
                    try:
                        res = minimize(ring_objective, [r0, phi0], method='Nelder-Mead',
                                       options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 100})
                        if res.fun < best_obj:
                            best_obj = res.fun
                            best_r = res.x[0]
                            best_phi = res.x[1]
                    except Exception:
                        pass
                
            pts = _generate_two_ring(n_out, n_in, best_r, best_phi)
        
        # Polish the specific configuration
        # Allocate time based on remaining budget, but ensure at least 2s for polish
        remaining = deadline - time.time()
        if remaining > 5.0:
            polish_time = min(8.0, remaining * 0.4)
            polished_pts, polished_score = _polish_points(pts, time_limit=polish_time)
            try_update(polished_score, polished_pts)
    
    # Final intensive polish on the best points found so far
    if best_points is not None and time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 2.0
        final_pts, final_score = _polish_points(best_points, time_limit=remaining)
        try_update(final_score, final_pts)

    # Fallback if nothing worked (should not happen)
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _calculate_score(best_points)

    return best_points
