import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def _all_triangle_areas(points):
    """Vectorized computation of all C(n,3) triangle areas."""
    n = len(points)
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except:
        return 1.0

def _score(points):
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _soft_min_objective(params, beta=100.0):
    """Negative smooth min area / hull area.
    We minimize - (soft_min(areas) / hull_area).
    """
    points = params.reshape(-1, 2)
    areas = _all_triangle_areas(points)
    
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    lse = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min = -lse / beta
    
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 1e10
    
    return -soft_min / hull_area

def _make_multi_shell_points(partitions, radii, phases):
    """Create multi-shell configuration from list of (n_points, radius, phase) tuples."""
    all_points = []
    for n_pts, r, phase in zip(partitions, radii, phases):
        angles = np.linspace(phase, 2 * np.pi + phase, n_pts, endpoint=False)
        pts = np.column_stack((r * np.cos(angles), r * np.sin(angles)))
        all_points.append(pts)
    return np.vstack(all_points)

def _param_to_points(params, partitions):
    """Convert parameter vector to points. params = [r1, r2, ..., r_k, phase1, phase2, ..., phase_k]"""
    k = len(partitions)
    radii = params[:k]
    phases = params[k:]
    return _make_multi_shell_points(partitions, radii, phases)

def _param_objective(params, partitions, beta=50.0):
    """Objective for multi-shell parameter optimization."""
    points = _param_to_points(params, partitions)
    return _soft_min_objective(points.flatten(), beta=beta)

def _grid_search_partition(partitions, r_range, phase_range, n_r, n_phase, beta=50.0, deadline=None, t0=None):
    """Grid search over radii and phases for a given partition."""
    k = len(partitions)
    best_score = 0.0
    best_params = None
    
    # Create grid
    r_vals = np.linspace(r_range[0], r_range[1], n_r)
    ph_vals = np.linspace(phase_range[0], phase_range[1], n_phase)
    
    # For k=2, we can do a 4D grid (2 radii, 2 phases)
    if k == 2:
        for r1 in r_vals:
            for r2 in r_vals:
                for ph1 in ph_vals:
                    for ph2 in ph_vals:
                        if deadline is not None and time.time() > deadline:
                            return best_score, best_params
                        params = np.array([r1, r2, ph1, ph2])
                        points = _param_to_points(params, partitions)
                        s = _score(points)
                        if s > best_score:
                            best_score = s
                            best_params = params.copy()
    elif k == 3:
        # For k=3, do a coarser grid
        for r1 in r_vals:
            for r2 in r_vals:
                for r3 in r_vals:
                    for ph1 in ph_vals:
                        for ph2 in ph_vals:
                            for ph3 in ph_vals:
                                if deadline is not None and time.time() > deadline:
                                    return best_score, best_params
                                params = np.array([r1, r2, r3, ph1, ph2, ph3])
                                points = _param_to_points(params, partitions)
                                s = _score(points)
                                if s > best_score:
                                    best_score = s
                                    best_params = params.copy()
    
    return best_score, best_params

def construct():
    n = N
    t0 = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = 0.0
    
    # Try multiple partition types
    partitions_to_try = [
        [7, 6],
        [8, 5],
        [9, 4],
        [6, 7],  # Same as [7,6] but different role assignment
        [5, 8],
        [4, 9],
        [13, 0],  # Single shell (regular 13-gon)
    ]
    
    # For single shell, handle separately
    for partitions in partitions_to_try:
        if time.time() > deadline:
            break
        
        k = len([p for p in partitions if p > 0])
        if k == 0:
            continue
        
        # Filter out zero partitions
        active_partitions = [p for p in partitions if p > 0]
        
        if len(active_partitions) == 1:
            # Single shell: just regular polygon
            angles = np.linspace(0, 2 * np.pi, active_partitions[0], endpoint=False)
            pts = np.column_stack((np.cos(angles), np.sin(angles)))
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
            continue
        
        if len(active_partitions) == 2:
            # Two-shell: grid search over (r1, r2, ph1, ph2)
            # Use coarser grid for speed
            r_vals = np.linspace(0.3, 1.2, 15)
            ph_vals = np.linspace(-np.pi/13, np.pi/13, 15)
            
            best_s, best_p = _grid_search_partition(active_partitions, (0.3, 1.2), (-np.pi/13, np.pi/13), 15, 15, deadline=deadline, t0=t0)
            
            if best_p is not None and best_s > best_score:
                best_score = best_s
                best_points = _param_to_points(best_p, active_partitions).copy()
    
    # If no points found, fallback to regular 13-gon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _score(best_points)
    
    # Phase 2: Full-coordinate polish from best config
    if time.time() < deadline - 5:
        try:
            x0 = best_points.flatten()
            res1 = minimize(_soft_min_objective, x0, args=(50.0,), method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if time.time() > deadline:
                return best_points
                
            candidate = res1.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate.copy()
        except Exception:
            pass
    
    # Phase 3: Powell method polish
    if time.time() < deadline - 5:
        try:
            x0 = best_points.flatten()
            res2 = minimize(_soft_min_objective, x0, args=(100.0,), method='Powell',
                           options={'maxiter': 150, 'ftol': 1e-12, 'xtol': 1e-12})
            
            if time.time() > deadline:
                return best_points
                
            candidate = res2.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate.copy()
        except Exception:
            pass
            
    # Phase 4: Nelder-Mead high-precision polish
    if time.time() < deadline - 3:
        try:
            x0 = best_points.flatten()
            res3 = minimize(_soft_min_objective, x0, args=(200.0,), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            if time.time() > deadline:
                return best_points
                
            candidate = res3.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate.copy()
        except Exception:
            pass
            
    return best_points
