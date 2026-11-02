import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13
DEADLINE = 140.0  # Safety margin before 150s kill

# Pre-compute all triples for fast vectorized scoring
# There are C(13, 3) = 286 triples
_triple_i = []
_triple_j = []
_triple_k = []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _triple_i.append(_i)
            _triple_j.append(_j)
            _triple_k.append(_k)

_triple_i = np.array(_triple_i, dtype=np.int32)
_triple_j = np.array(_triple_j, dtype=np.int32)
_triple_k = np.array(_triple_k, dtype=np.int32)

def _compute_areas(pts):
    """Compute all triangle areas for a point set."""
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-12

def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    
    areas = _compute_areas(pts)
    min_area = np.min(areas)
    
    if min_area < 1e-15:
        return 0.0
    
    return min_area / hull_area

def _soft_min_score(pts, beta=100.0):
    """
    Compute a smooth approximation of the min triangle area / hull area.
    Uses log-sum-exp for the min to make it differentiable.
    S = -1/beta * ln(sum(exp(-beta * A_i))) / H
    This is approx -min(A_i) / H for large beta.
    We want to maximize this, so we minimize the negative.
    """
    pts = np.asarray(pts, dtype=np.float64)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return -np.inf
        
    areas = _compute_areas(pts)
    
    vals = -beta * areas
    v_max = np.max(vals)
    if v_max == -np.inf:
        return -np.inf
        
    log_sum = v_max + np.log(np.sum(np.exp(vals - v_max)))
    soft_min_val = -log_sum / beta
    
    return soft_min_val / hull_area

def _neg_soft_score(params, beta):
    """Negative soft score for minimization."""
    pts = params.reshape(N, 2)
    return -_soft_min_score(pts, beta)

def _neg_hard_score(params):
    """Negative hard score for minimization."""
    pts = params.reshape(N, 2)
    return -_score_points(pts)

def _max_min_objective(params, t, active_mask=None, weight_alpha=1.0):
    """
    Max-min fairness objective using epigraph formulation.
    We want to maximize t subject to A_i / H >= t for all i.
    
    This function computes a penalty-based surrogate:
    - Primary: maximize t (via negative t)
    - Secondary: penalize violations where A_i / H < t
    
    If active_mask is provided, give extra weight to active (near-min) triangles.
    """
    pts = params.reshape(N, 2)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 1e10  # Large penalty
    
    areas = _compute_areas(pts)
    normalized_areas = areas / hull_area
    
    # Penalty for violations: sum of max(0, t - A_i/H)^2
    violations = np.maximum(0, t - normalized_areas)
    
    if active_mask is not None:
        # Weight active triangles more heavily
        weights = np.ones(len(normalized_areas))
        weights[active_mask] = 1.0 + weight_alpha
        penalty = np.sum(weights * violations ** 2)
    else:
        penalty = np.sum(violations ** 2)
    
    # Objective: -t + penalty (we minimize this)
    return -t + penalty * 100.0  # Scale penalty to be significant

def _find_active_triangles(pts, t, threshold=0.1):
    """
    Find active (near-minimum) triangles.
    Active if A_i / H is within threshold of t.
    """
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return np.zeros(286, dtype=bool)
    
    areas = _compute_areas(pts)
    normalized_areas = areas / hull_area
    
    # Active if close to the minimum (within threshold of t)
    active = np.abs(normalized_areas - t) < threshold
    return active

def _solve_max_min(pts_init, t_init, max_iters=50, time_budget=30.0):
    """
    Iteratively solve the max-min fairness problem.
    1. Fix t, optimize positions to satisfy A_i/H >= t
    2. Update t to be the new minimum
    3. Repeat
    """
    t_start = time.time()
    pts = pts_init.copy()
    t = t_init
    
    for it in range(max_iters):
        # Time check
        if (time.time() - t_start) > time_budget:
            break
        
        # Find active triangles
        active = _find_active_triangles(pts, t, threshold=0.15)
        
        # Optimize positions to maximize t while satisfying constraints
        def objective(params):
            return _max_min_objective(params, t, active_mask=active, weight_alpha=2.0)
        
        try:
            res = minimize(
                objective,
                params=pts.flatten(),
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 100, 'maxfun': 2000, 'ftol': 1e-10}
            )
            
            if res.success or res.fun < objective(pts.flatten()):
                new_pts = res.x.reshape(N, 2)
                new_hull = _hull_area(new_pts)
                if new_hull > 1e-12:
                    new_areas = _compute_areas(new_pts)
                    new_min_norm = np.min(new_areas) / new_hull
                    
                    # Update t: take the new minimum or slightly above current t
                    # to encourage progress
                    t_new = min(new_min_norm, t * 1.05 + 1e-6)
                    t_new = max(t_new, t * 0.99)  # Don't decrease t too much
                    
                    pts = new_pts
                    t = t_new
        except Exception:
            break
        
        # Check convergence
        if it > 5:
            prev_t = t
            # Simple convergence check based on recent progress
            pass
    
    return pts, t

def _make_initial_guesses():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p1 = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(p1.flatten())
    
    # 2. 1 center + 12 outer
    p2 = np.zeros((N, 2))
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    p2[1:, 0] = np.cos(angles)
    p2[1:, 1] = np.sin(angles)
    inits.append(p2.flatten())
    
    # 3. 1 center + 6 mid + 6 outer
    p3 = np.zeros((N, 2))
    # 6 mid
    angles_m = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    p3[1:7, 0] = 0.5 * np.cos(angles_m)
    p3[1:7, 1] = 0.5 * np.sin(angles_m)
    # 6 outer
    angles_o = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    p3[7:, 0] = np.cos(angles_o)
    p3[7:, 1] = np.sin(angles_o)
    inits.append(p3.flatten())
    
    # 4. Random
    rng = np.random.RandomState(42)
    p4 = rng.uniform(-1, 1, size=(N, 2)).flatten()
    inits.append(p4)
    
    # 5. 1 center + 5 inner + 7 outer (asymmetric layers)
    p5 = np.zeros((N, 2))
    angles_in = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    p5[1:6, 0] = 0.4 * np.cos(angles_in)
    p5[1:6, 1] = 0.4 * np.sin(angles_in)
    angles_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    p5[6:, 0] = np.cos(angles_out)
    p5[6:, 1] = np.sin(angles_out)
    inits.append(p5.flatten())
    
    # 6. 2 center-ish + 11 outer
    p6 = np.zeros((N, 2))
    p6[0] = [0.2, 0]
    p6[1] = [-0.2, 0]
    angles = np.linspace(0, 2 * np.pi, 11, endpoint=False)
    p6[2:, 0] = np.cos(angles)
    p6[2:, 1] = np.sin(angles)
    inits.append(p6.flatten())
    
    return inits

def construct():
    t_start = time.time()
    
    best_params = None
    best_score = -np.inf
    
    # 1. Evaluate initial guesses
    inits = _make_initial_guesses()
    for p in inits:
        pts = p.reshape(N, 2)
        s = _score_points(pts)
        if s > best_score:
            best_score = s
            best_params = p.copy()
            
    # 2. Simulated Annealing with Local Refinement
    T0 = 0.05
    T_end = 1e-5
    cooling_rate = 0.995
    max_iters = 20000
    
    current_params = best_params.copy()
    current_soft_score = _soft_min_score(current_params.reshape(N, 2), beta=50.0)
    
    rng = np.random.RandomState(123)
    step_size = 0.1
    sa_beta = 50.0
    
    for i in range(max_iters):
        if (time.time() - t_start) > (DEADLINE - 40.0):
            break
            
        T = T0 * (cooling_rate ** i)
        
        perturbation = rng.normal(0, step_size, size=(2 * N))
        neighbor_params = current_params + perturbation
        neighbor_params = np.clip(neighbor_params, -1.0, 1.0)
        
        neighbor_pts = neighbor_params.reshape(N, 2)
        neighbor_soft_score = _soft_min_score(neighbor_pts, sa_beta)
        
        if neighbor_soft_score > current_soft_score:
            accept = True
        else:
            delta = current_soft_score - neighbor_soft_score
            if delta < 1e-12:
                accept = True
            else:
                accept = rng.random() < np.exp(-delta / T)
                
        if accept:
            current_params = neighbor_params
            current_soft_score = neighbor_soft_score
            
            if i % 100 == 0 and i > 0:
                try:
                    res = minimize(
                        lambda x: _neg_soft_score(x, sa_beta),
                        current_params,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
                    )
                    if res.fun < _neg_soft_score(current_params, sa_beta):
                        current_params = res.x
                        current_soft_score = _soft_min_score(current_params.reshape(N, 2), sa_beta)
                except Exception:
                    pass
            
            if i % 20 == 0:
                hard_s = _score_points(current_params.reshape(N, 2))
                if hard_s > best_score:
                    best_score = hard_s
                    best_params = current_params.copy()
                    
        step_size = 0.05 + 0.05 * (T / T0)

    # 3. Max-Min Fairness Phase
    # Apply the epigraph-based max-min optimization to the best candidates
    
    candidates = [best_params.copy()]
    
    # Add SA current state if different
    if not np.allclose(best_params, current_params):
        candidates.append(current_params.copy())
    
    # Add top initial guesses
    scored_inits = []
    for p in inits:
        s = _score_points(p.reshape(N, 2))
        scored_inits.append((s, p.copy()))
    scored_inits.sort(key=lambda x: -x[0])
    
    for s, p in scored_inits[:3]:
        is_dup = False
        for c in candidates:
            if np.allclose(c, p, atol=1e-3):
                is_dup = True
                break
        if not is_dup:
            candidates.append(p)
    
    # Run max-min fairness on each candidate
    for cand in candidates:
        if (time.time() - t_start) > (DEADLINE - 15.0):
            break
        
        pts_cand = cand.reshape(N, 2)
        score_cand = _score_points(pts_cand)
        
        # Initialize t slightly below current score to allow progress
        t_init = score_cand * 0.98
        
        try:
            pts_mm, t_mm = _solve_max_min(pts_cand, t_init, max_iters=30, time_budget=15.0)
            
            # Evaluate the result
            score_mm = _score_points(pts_mm)
            
            if score_mm > best_score:
                best_score = score_mm
                best_params = pts_mm.flatten().copy()
        except Exception:
            pass
    
    # 4. Final Polish with High-Beta Soft Min
    # Use the best candidate for final refinement
    
    if (time.time() - t_start) < (DEADLINE - 5.0):
        starts_to_refine = [best_params]
        
        for start in starts_to_refine:
            if (time.time() - t_start) > (DEADLINE - 5.0):
                break
            
            current_refine_params = start.copy()
            
            for beta in [200.0, 500.0, 1000.0]:
                if (time.time() - t_start) > (DEADLINE - 3.0):
                    break
                
                try:
                    res = minimize(
                        lambda x: _neg_soft_score(x, beta),
                        current_refine_params,
                        method='L-BFGS-B',
                        bounds=[(-1.0, 1.0)] * (2 * N),
                        options={'maxiter': 300, 'maxfun': 8000, 'ftol': 1e-14}
                    )
                    if res.success or res.fun < _neg_soft_score(current_refine_params, beta):
                        current_refine_params = res.x
                    
                    score2 = _score_points(current_refine_params.reshape(N, 2))
                    if score2 > best_score:
                        best_score = score2
                        best_params = current_refine_params.copy()
                        
                except Exception:
                    pass
    
    return best_params.reshape(N, 2)
