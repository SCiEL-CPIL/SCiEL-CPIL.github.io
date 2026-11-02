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
    
    # Avoid log(0) or overflow
    # Shift areas to ensure numerical stability? 
    # The areas are typically small (O(1/100) or less).
    # If beta is large, exp(-beta * A) can underflow for large A, which is fine (becomes 0).
    # But if A is very small, exp(-beta*A) is close to 1.
    
    # To avoid underflow issues with very large beta * A, we can scale.
    # However, for N=13, areas are not huge. 
    # Let's use a stable logsumexp calculation.
    # log(sum(exp(-beta * A))) = max(-beta*A) + log(sum(exp(-beta*A - max(-beta*A))))
    
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
    # We use a smooth surrogate for the SA steps and local optimization.
    # The hard score is used for final evaluation and acceptance of "best".
    
    # SA Parameters
    T0 = 0.05  # Initial temperature
    T_end = 1e-5
    cooling_rate = 0.995
    max_iters = 20000  # We will break early based on time
    
    current_params = best_params.copy()
    current_soft_score = _soft_min_score(current_params.reshape(N, 2), beta=50.0)
    
    # We will increase beta over time to sharpen the objective
    # Start with low beta (smooth) to find good basin, increase to refine.
    
    rng = np.random.RandomState(123)
    
    # Adaptive step size
    step_size = 0.1
    
    for i in range(max_iters):
        # Time check
        if (time.time() - t_start) > (DEADLINE - 10.0):
            break
            
        # Update temperature and beta
        T = T0 * (cooling_rate ** i)
        # Beta increases as T decreases. 
        # At high T, we want smooth landscape (low beta).
        # At low T, we want sharp landscape (high beta).
        # Map T to beta: beta = base_beta * (T0 / T) ** 0.5 ? 
        # Or just linearly increase beta.
        # Let's use beta = 20 + 100 * (1 - T/T0)
        beta = 20.0 + 100.0 * (1.0 - T / T0)
        
        # Generate neighbor
        # Gaussian perturbation
        perturbation = rng.normal(0, step_size, size=(2 * N))
        neighbor_params = current_params + perturbation
        neighbor_params = np.clip(neighbor_params, -1.0, 1.0)
        
        neighbor_pts = neighbor_params.reshape(N, 2)
        neighbor_soft_score = _soft_min_score(neighbor_pts, beta)
        
        # Acceptance criterion
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
            
            # Occasionally perform a local refinement on the soft score
            # This helps climb the smooth hills efficiently
            if i % 50 == 0 and i > 0:
                try:
                    res = minimize(
                        lambda x: _neg_soft_score(x, beta),
                        current_params,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
                    )
                    if res.fun < _neg_soft_score(current_params, beta):
                        current_params = res.x
                        current_soft_score = _soft_min_score(current_params.reshape(N, 2), beta)
                except Exception:
                    pass
            
            # Update global best if this is better in terms of HARD score
            # We don't want to waste time calculating hard score every step.
            # But we need to track the best hard score.
            if i % 10 == 0:
                hard_s = _score_points(current_params.reshape(N, 2))
                if hard_s > best_score:
                    best_score = hard_s
                    best_params = current_params.copy()
                    
        else:
            # Reduce step size if rejected? 
            # Or keep constant. 
            # If we are stuck, we might need to restart or jump.
            pass
            
        # Adaptive step size: if we are accepting a lot, increase step?
        # If we are rejecting a lot, decrease step?
        # Simple heuristic:
        # If T is high, step can be larger.
        step_size = 0.05 + 0.05 * (T / T0)

    # 3. Final Local Refinement on Hard Score
    # Use the best soft-scored point as a starting point for hard score optimization.
    # Also try the current best_params from hard score tracking.
    
    starts_to_refine = [best_params]
    
    # Add current SA state
    if not np.allclose(best_params, current_params):
        starts_to_refine.append(current_params)
        
    # Add initial bests
    for p in inits:
        starts_to_refine.append(p)
        
    # Deduplicate
    unique_starts = []
    for s in starts_to_refine:
        is_unique = True
        for u in unique_starts:
            if np.allclose(s, u, atol=1e-3):
                is_unique = False
                break
        if is_unique:
            unique_starts.append(s)
            
    for start in unique_starts:
        if (time.time() - t_start) > (DEADLINE - 5.0):
            break
            
        # Refine using L-BFGS-B on the hard score? 
        # Hard score is non-differentiable. L-BFGS-B uses finite differences.
        # It might work but is slow. 
        # Better: Refine on soft score with very high beta, then evaluate hard.
        
        try:
            # Use very high beta to approximate hard min
            beta_high = 500.0
            res = minimize(
                lambda x: _neg_soft_score(x, beta_high),
                start,
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-12}
            )
            score2 = _score_points(res.x.reshape(N, 2))
            if score2 > best_score:
                best_score = score2
                best_params = res.x.copy()
        except Exception:
            pass
            
    return best_params.reshape(N, 2)
