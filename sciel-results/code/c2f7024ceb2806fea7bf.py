import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13

def _min_triangle_area_and_indices(points):
    """Return (min_area, (i,j,k)) for the minimum-area triangle using vectorized computation."""
    n = len(points)
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n), indexing='ij'
    )
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_f, j_f, k_f = i_idx[mask], j_idx[mask], k_idx[mask]
    
    a = points[i_f]
    b = points[j_f]
    c = points[k_f]
    
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_pos = np.argmin(areas)
    min_area = areas[min_pos]
    min_idx = (i_f[min_pos], j_f[min_pos], k_f[min_pos])
    return min_area, min_idx

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def get_score(points):
    points = np.asarray(points, dtype=float)
    if points.shape != (N, 2):
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area, _ = _min_triangle_area_and_indices(points)
    return min_area / hull_area

def _neg_score(points_flat):
    """Objective for minimization: negative of score."""
    points = points_flat.reshape(N, 2)
    score = get_score(points)
    return -score

def _subgradient_polish(points, deadline, max_iter=100):
    """
    Final subgradient polish to fine-tune the minimum triangle.
    """
    points = points.copy()
    best_points = points.copy()
    best_score = get_score(points)
    
    lr = 0.002
    n_no_improve = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        min_area, (i, j, k) = _min_triangle_area_and_indices(points)
        if min_area < 1e-12:
            break
        
        a = points[i]
        b = points[j]
        c = points[k]
        
        cross_val = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        sign = np.sign(cross_val) if cross_val != 0 else 1.0
        
        u = b - a
        v = c - a
        
        grad_a = np.array([-v[1], v[0]]) * 0.5 * sign
        grad_b = np.array([v[1], -v[0]]) * 0.5 * sign
        grad_c = np.array([-u[1], u[0]]) * 0.5 * sign
        
        new_points = points.copy()
        new_points[i] += lr * grad_a
        new_points[j] += lr * grad_b
        new_points[k] += lr * grad_c
        
        new_score = get_score(new_points)
        if new_score > best_score:
            best_score = new_score
            best_points = new_points.copy()
            n_no_improve = 0
            lr = min(lr * 1.02, 0.01)
        else:
            n_no_improve += 1
            lr *= 0.95
            if n_no_improve > 30:
                break
        
        points = new_points
    
    return best_points

def _generate_initials():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Regular 12-gon + center
    angles12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts12 = np.column_stack([np.cos(angles12), np.sin(angles12)])
    configs.append(np.vstack([pts12, np.array([[0.0, 0.0]])]))
    
    # 3. Regular 11-gon + 2 inner
    angles11 = np.linspace(0, 2*np.pi, 11, endpoint=False)
    pts11 = np.column_stack([np.cos(angles11), np.sin(angles11)])
    configs.append(np.vstack([pts11, np.array([[0.5, 0.0], [-0.5, 0.0]])]))
    
    # 4. Regular 10-gon + 3 inner (triangle at center)
    angles10 = np.linspace(0, 2*np.pi, 10, endpoint=False)
    pts10 = np.column_stack([np.cos(angles10), np.sin(angles10)])
    inner_angles = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
    inner = np.column_stack([0.4*np.cos(inner_angles), 0.4*np.sin(inner_angles)])
    configs.append(np.vstack([pts10, inner]))
    
    # 5. Regular 9-gon + 4 inner (square at center)
    angles9 = np.linspace(0, 2*np.pi, 9, endpoint=False)
    pts9 = np.column_stack([np.cos(angles9), np.sin(angles9)])
    inner_angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
    inner = np.column_stack([0.35*np.cos(inner_angles), 0.35*np.sin(inner_angles)])
    configs.append(np.vstack([pts9, inner]))
    
    # 6. Two concentric rings: 7 outer, 6 inner
    angles7 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer = np.column_stack([np.cos(angles7), np.sin(angles7)])
    angles6 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    inner = np.column_stack([0.5*np.cos(angles6), 0.5*np.sin(angles6)])
    configs.append(np.vstack([outer, inner]))
    
    # 7. 8 outer, 5 inner
    angles8 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    outer = np.column_stack([np.cos(angles8), np.sin(angles8)])
    angles5 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    inner = np.column_stack([0.45*np.cos(angles5), 0.45*np.sin(angles5)])
    configs.append(np.vstack([outer, inner]))
    
    # 8. 6 outer, 7 inner
    angles6 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    outer6 = np.column_stack([np.cos(angles6), np.sin(angles6)])
    angles7 = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    inner7 = np.column_stack([0.55*np.cos(angles7), 0.55*np.sin(angles7)])
    configs.append(np.vstack([outer6, inner7]))
    
    # 9. 5 outer, 8 inner
    angles5 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    outer5 = np.column_stack([np.cos(angles5), np.sin(angles5)])
    angles8 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    inner8 = np.column_stack([0.5*np.cos(angles8), 0.5*np.sin(angles8)])
    configs.append(np.vstack([outer5, inner8]))
    
    # 10. Three concentric rings: 5 outer, 5 middle, 3 inner
    angles5 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    ring1 = np.column_stack([np.cos(angles5), np.sin(angles5)])
    angles5b = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
    ring2 = np.column_stack([0.6*np.cos(angles5b), 0.6*np.sin(angles5b)])
    angles3 = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
    ring3 = np.column_stack([0.3*np.cos(angles3), 0.3*np.sin(angles3)])
    configs.append(np.vstack([ring1, ring2, ring3]))
    
    # 11-16. Random with various seeds
    for seed in [1, 7, 13, 99, 42, 123]:
        rng = np.random.default_rng(seed)
        pts = rng.random((N, 2))
        configs.append(pts)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    configs = _generate_initials()
    
    best_score = -np.inf
    best_points = None
    
    # Phase 1: Nelder-Mead optimization from all configs
    # Use adaptive maxiter based on remaining time
    n_configs = len(configs)
    
    for idx, init_points in enumerate(configs):
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        if remaining < 10.0:
            break
        
        # Allocate time per config: more for early configs, less for later
        # First 6 configs get more budget
        if idx < 6:
            max_iter = 3000
        else:
            max_iter = 1500
        
        try:
            x0 = init_points.flatten()
            
            res = minimize(
                _neg_score,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': max_iter,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            candidate_points = res.x.reshape(N, 2)
            score = get_score(candidate_points)
            
            if score > best_score:
                best_score = score
                best_points = candidate_points.copy()
                
        except Exception:
            pass
    
    # Phase 2: Subgradient polish on the best result
    if best_points is not None and time.time() < deadline - 5.0:
        try:
            polished = _subgradient_polish(best_points, deadline - 2.0, max_iter=200)
            polished_score = get_score(polished)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished
        except Exception:
            pass
    
    # Phase 3: If we have time, try a second Nelder-Mead pass from the best result
    # with different starting perturbation
    if best_points is not None and time.time() < deadline - 15.0:
        try:
            rng = np.random.default_rng(12345)
            perturbed = best_points + rng.normal(0, 0.01, best_points.shape)
            x0 = perturbed.flatten()
            
            res = minimize(
                _neg_score,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            candidate_points = res.x.reshape(N, 2)
            score = get_score(candidate_points)
            
            if score > best_score:
                best_score = score
                best_points = candidate_points.copy()
                
            # Final polish
            if time.time() < deadline - 3.0:
                polished = _subgradient_polish(best_points, deadline - 1.0, max_iter=100)
                polished_score = get_score(polished)
                if polished_score > best_score:
                    best_score = polished_score
                    best_points = polished
        except Exception:
            pass
    
    if best_points is None:
        best_points = configs[0]
    
    return best_points
