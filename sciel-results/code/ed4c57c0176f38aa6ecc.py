# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize, differential_evolution

def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    if points.shape[0] < 2:
        return 0.0
    d = pdist(points)
    if len(d) == 0:
        return 0.0
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    if d_min == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _smooth_objective(points, p_min=10.0, p_max=10.0):
    """
    Smooth surrogate using log-sum-exp based soft min/max.
    """
    d = pdist(points)
    if len(d) == 0:
        return 0.0
    
    d = np.clip(d, 1e-10, 1e10)
    
    # Soft Min using log-sum-exp
    d_neg = -d
    max_d_neg = np.max(d_neg)
    log_sum_exp_neg = max_d_neg + np.log(np.sum(np.exp(d_neg - max_d_neg)))
    soft_min = -log_sum_exp_neg / p_min
    
    # Soft Max using log-sum-exp
    d_pos = d
    max_d_pos = np.max(d_pos)
    log_sum_exp_pos = max_d_pos + np.log(np.sum(np.exp(d_pos - max_d_pos)))
    soft_max = log_sum_exp_pos / p_max
    
    if soft_max == 0:
        return 0.0
        
    ratio = soft_min / soft_max
    return ratio ** 2

def _sqp_objective(flat, d_max_target=1.0):
    """
    SQP objective: maximize min distance subject to max distance constraint.
    We minimize -min_dist where max_dist <= d_max_target.
    """
    points = flat.reshape(-1, 2)
    d = pdist(points)
    if len(d) == 0:
        return 0.0
    d_min = np.min(d)
    d_max = np.max(d)
    
    # If max distance exceeds target, penalize heavily
    if d_max > d_max_target * 1.01:
        penalty = (d_max - d_max_target) / d_max_target
        return -d_min + penalty * 100.0
    
    return -d_min

def _sqp_constraints(flat, d_max_target=1.0):
    """
    Constraints for SQP: max distance should not exceed d_max_target.
    Returns list of constraint dicts.
    """
    points = flat.reshape(-1, 2)
    d = pdist(points)
    if len(d) == 0:
        return []
    d_max = np.max(d)
    
    # Constraint: d_max <= d_max_target
    return [{'type': 'ineq', 'fun': lambda x: d_max_target - np.max(pdist(x.reshape(-1, 2)))}]

def _neg_smooth_objective(flat):
    points = flat.reshape(-1, 2)
    return -_smooth_objective(points)

def _neg_true_objective(flat):
    points = flat.reshape(-1, 2)
    return -_compute_score(points)

def _sqp_refine(points, time_budget=30.0):
    """
    Refine using SQP: maximize min distance subject to max distance constraint.
    This directly optimizes the geometric constraints.
    """
    start = time.time()
    current_points = points.copy()
    current_score = _compute_score(current_points)
    
    # Normalize so max distance = 1
    d = pdist(current_points)
    if len(d) == 0 or np.max(d) == 0:
        return current_points, current_score
    
    d_max = np.max(d)
    normalized_points = current_points / d_max
    
    # Now max distance should be ~1, we want to maximize min distance
    # Use SLSQP with constraint that max distance <= 1
    
    def objective(flat):
        pts = flat.reshape(-1, 2)
        d = pdist(pts)
        if len(d) == 0:
            return 0.0
        d_min = np.min(d)
        d_max_local = np.max(d)
        # Penalize if max distance exceeds 1
        if d_max_local > 1.0:
            penalty = (d_max_local - 1.0) * 1000.0
            return -d_min + penalty
        return -d_min
    
    def constraint_max(flat):
        pts = flat.reshape(-1, 2)
        d = pdist(pts)
        if len(d) == 0:
            return 1.0
        return 1.0 - np.max(d)
    
    try:
        res = minimize(
            objective,
            normalized_points.ravel(),
            method='SLSQP',
            constraints=[{'type': 'ineq', 'fun': constraint_max}],
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        if res.x is not None:
            candidate_points = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate_points)
            if candidate_score > current_score:
                current_points = candidate_points
                current_score = candidate_score
    except Exception:
        pass
    
    # Second pass: try to improve by allowing slight scaling
    if time.time() - start < time_budget * 0.7:
        try:
            # Try multiple scales
            for scale in [0.95, 0.98, 1.0, 1.02, 1.05]:
                scaled_points = current_points * scale
                def objective_scaled(flat):
                    pts = flat.reshape(-1, 2)
                    d = pdist(pts)
                    if len(d) == 0:
                        return 0.0
                    d_min = np.min(d)
                    d_max_local = np.max(d)
                    target = scale
                    if d_max_local > target * 1.01:
                        penalty = (d_max_local - target) / target * 1000.0
                        return -d_min + penalty
                    return -d_min
                
                def constraint_scaled(flat):
                    pts = flat.reshape(-1, 2)
                    d = pdist(pts)
                    if len(d) == 0:
                        return 1.0
                    return scale - np.max(d)
                
                res2 = minimize(
                    objective_scaled,
                    scaled_points.ravel(),
                    method='SLSQP',
                    constraints=[{'type': 'ineq', 'fun': constraint_scaled}],
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                
                if res2.x is not None:
                    candidate_points = res2.x.reshape(-1, 2)
                    candidate_score = _compute_score(candidate_points)
                    if candidate_score > current_score:
                        current_points = candidate_points
                        current_score = candidate_score
                        
                if time.time() - start > time_budget * 0.65:
                    break
        except Exception:
            pass
    
    return current_points, current_score

def _refine(points, time_budget=10.0):
    """
    Refine points using SQP, L-BFGS-B (smooth), and Powell (true).
    """
    start = time.time()
    current_points = points.copy()
    current_score = _compute_score(current_points)
    
    # Step 1: SQP refinement (most effective for this problem)
    if time.time() - start < time_budget * 0.4:
        try:
            sqp_pts, sqp_score = _sqp_refine(current_points, time_budget=min(20, time_budget * 0.4))
            if sqp_score > current_score:
                current_points = sqp_pts
                current_score = sqp_score
        except Exception:
            pass

    # Step 2: L-BFGS-B on smooth surrogate
    if time.time() - start < time_budget * 0.7:
        try:
            res = minimize(_neg_smooth_objective, current_points.ravel(), method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < -_smooth_objective(current_points):
                candidate_points = res.x.reshape(-1, 2)
                candidate_score = _compute_score(candidate_points)
                if candidate_score > current_score:
                    current_points = candidate_points
                    current_score = candidate_score
        except Exception:
            pass

    # Step 3: Powell on true objective
    if time.time() - start < time_budget * 0.9:
        try:
            res = minimize(_neg_true_objective, current_points.ravel(), method='Powell',
                           options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < -current_score:
                current_points = res.x.reshape(-1, 2)
                current_score = -res.fun
        except Exception:
            pass

    # Step 4: Final SQP polish
    if time.time() - start < time_budget * 0.98:
        try:
            final_pts, final_score = _sqp_refine(current_points, time_budget=min(10, time_budget * 0.15))
            if final_score > current_score:
                current_points = final_pts
                current_score = final_score
        except Exception:
            pass

    return current_points, current_score

def _generate_hexagonal_points(n=16):
    """Generate points in a hexagonal lattice pattern."""
    points = []
    q_max = 5
    candidates = []
    for q in range(-q_max, q_max + 1):
        for r in range(-q_max, q_max + 1):
            if abs(q + r) <= q_max:
                x = 1.5 * q
                y = np.sqrt(3) * (r + 0.5 * q)
                candidates.append((x, y))
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    sorted_indices = np.argsort(dists)
    
    selected = candidates[sorted_indices[:n]]
    return selected

def _generate_ring_points(radii, counts, angles_offsets=None):
    """Generate points on concentric rings."""
    points = []
    total = sum(counts)
    if total != 16:
        return None
    
    for i, (r, c) in enumerate(zip(radii, counts)):
        if c == 0:
            continue
        if c == 1:
            points.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2 * np.pi, c, endpoint=False)
            if angles_offsets and i < len(angles_offsets):
                angles += angles_offsets[i]
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            points.extend(zip(x, y))
            
    return np.array(points)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    rng = np.random.default_rng(12345)
    
    best_points = None
    best_score = 0.0
    
    def _update(pts, sc):
        nonlocal best_points, best_score
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()

    # 1. Initializations
    starts = []
    
    # 4x4 Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    starts.append(grid_points)
    
    # Circle
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    circle_points = np.column_stack([np.cos(angles), np.sin(angles)])
    starts.append(circle_points)
    
    # Hexagonal
    hex_points = _generate_hexagonal_points(16)
    starts.append(hex_points)
    
    # Rings: 1 center, 5 inner, 10 outer
    p_rings1 = _generate_ring_points([0.8, 1.8], [1, 5, 10], [0, 0, np.pi/10])
    if p_rings1 is not None:
        starts.append(p_rings1)
        
    # Rings: 1 center, 6 inner, 9 outer
    p_rings2 = _generate_ring_points([0.7, 1.6], [1, 6, 9], [0, 0, np.pi/9])
    if p_rings2 is not None:
        starts.append(p_rings2)
    
    # Rings: 1 center, 7 inner, 8 outer
    p_rings3 = _generate_ring_points([0.6, 1.5], [1, 7, 8], [0, 0, np.pi/8])
    if p_rings3 is not None:
        starts.append(p_rings3)
    
    # Random starts
    for _ in range(3):
        starts.append(rng.uniform(-2, 2, size=(16, 2)))
    
    # Update best with initializations
    for s in starts:
        if s is not None and s.shape == (16, 2):
            sc = _compute_score(s)
            _update(s, sc)
    
    # 2. Differential Evolution (Global Search)
    bounds = [(-3.0, 3.0)] * 32
    de_start = time.time()
    
    try:
        res = differential_evolution(
            _neg_smooth_objective, 
            bounds, 
            seed=12345,
            maxiter=60,
            tol=1e-10,
            popsize=12,
            polish=False,
            init='sobol'
        )
        
        candidate_points = res.x.reshape(-1, 2)
        candidate_score = _compute_score(candidate_points)
        _update(candidate_points, candidate_score)
        
        starts.append(candidate_points)
        
    except Exception:
        pass
        
    # 3. Local Refinement Loop
    if best_points is not None:
        starts.insert(0, best_points)
        
    # Remove duplicates
    unique_starts = []
    for s in starts:
        if s is None or s.shape != (16, 2):
            continue
        is_dup = False
        for u in unique_starts:
            if np.allclose(s, u, atol=1e-3):
                is_dup = True
                break
        if not is_dup:
            unique_starts.append(s)
            
    # Refine the best first with a large budget
    if best_points is not None:
        time_left = deadline - time.time()
        if time_left > 30:
            refined_pts, refined_score = _refine(best_points, time_budget=min(60, time_left - 10))
            _update(refined_pts, refined_score)
            
    # Refine other starts
    for idx, start_pts in enumerate(unique_starts[1:]):
        time_left = deadline - time.time()
        if time_left < 15:
            break
            
        num_remaining = len(unique_starts) - 1 - idx
        per_start_time = (time_left - 5) / max(1, num_remaining)
        per_start_time = min(40, per_start_time)
        
        if per_start_time < 5:
            continue
            
        try:
            refined_pts, refined_score = _refine(start_pts, time_budget=per_start_time)
            _update(refined_pts, refined_score)
        except Exception:
            pass

    # 4. Final Polish with SQP
    if best_points is not None:
        time_left = deadline - time.time()
        if time_left > 10:
            final_pts, final_score = _sqp_refine(best_points, time_budget=time_left - 2)
            _update(final_pts, final_score)
            
            # One more refinement pass
            time_left = deadline - time.time()
            if time_left > 15:
                final_pts2, final_score2 = _refine(best_points, time_budget=time_left - 2)
                _update(final_pts2, final_score2)

    if best_points is None:
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack([xx.ravel(), yy.ravel()])
        
    if not np.all(np.isfinite(best_points)):
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack([xx.ravel(), yy.ravel()])
        
    return best_points

# EVOLVE-BLOCK-END
