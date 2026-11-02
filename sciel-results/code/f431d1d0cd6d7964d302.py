import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 13

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
    """Compute min triangle area / convex hull area for given points (Hard Score)."""
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
    This is approx min(A_i) / H for large beta.
    We want to maximize this, so we minimize the negative.
    """
    pts = np.asarray(pts, dtype=np.float64)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return -np.inf
        
    areas = _compute_areas(pts)
    
    # Stable logsumexp calculation
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
    angles_m = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    p3[1:7, 0] = 0.5 * np.cos(angles_m)
    p3[1:7, 1] = 0.5 * np.sin(angles_m)
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
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_params = None
    best_score = -np.inf
    
    # 1. Evaluate initial guesses to find a good starting point
    inits = _make_initial_guesses()
    for p in inits:
        pts = p.reshape(N, 2)
        s = _score_points(pts)
        if s > best_score:
            best_score = s
            best_params = p.copy()
            
    # 2. Simulated Annealing with Local Refinement
    # We use a smooth surrogate for the SA steps to navigate the non-convex landscape.
    
    # SA Parameters
    T0 = 0.05  # Initial temperature
    T_end = 1e-5
    cooling_rate = 0.995
    max_iters = 50000  # We will break early based on time
    
    current_params = best_params.copy()
    
    # We will use a fixed beta for the main SA loop to ensure smoothness.
    # Beta=50 is a good balance between smoothness and fidelity to the min.
    sa_beta = 50.0
    current_soft_score = _soft_min_score(current_params.reshape(N, 2), beta=sa_beta)
    
    rng = np.random.RandomState(123)
    
    # Adaptive step size
    step_size = 0.1
    
    for i in range(max_iters):
        # Time check
        if (time.time() - start_time) > (deadline - 20.0):
            break
            
        # Update temperature
        T = T0 * (cooling_rate ** i)
        
        # Generate neighbor
        # Gaussian perturbation
        perturbation = rng.normal(0, step_size, size=(2 * N))
        neighbor_params = current_params + perturbation
        # Optionally clip to a box to prevent divergence, though not strictly necessary
        # neighbor_params = np.clip(neighbor_params, -10.0, 10.0)
        
        neighbor_pts = neighbor_params.reshape(N, 2)
        neighbor_soft_score = _soft_min_score(neighbor_pts, sa_beta)
        
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
            if i % 100 == 0 and i > 0:
                try:
                    res = minimize(
                        lambda x: _neg_soft_score(x, sa_beta),
                        current_params,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'maxfev': 1000}
                    )
                    if res.fun < _neg_soft_score(current_params, sa_beta):
                        current_params = res.x
                        current_soft_score = -res.fun
                except Exception:
                    pass
            
            # Check if we found a better hard score
            hard_s = _score_points(current_params.reshape(N, 2))
            if hard_s > best_score:
                best_score = hard_s
                best_params = current_params.copy()
                
    # 3. Final Polish with Hard Objective
    # The soft objective is an approximation. We now refine the best candidate
    # using the exact (hard) objective function to squeeze out maximum performance.
    
    if best_params is not None:
        # Try a few local optimizations on the hard score from the best SA result
        # and maybe some perturbed versions if time allows.
        
        candidates = [best_params]
        
        # Add some perturbations to escape potential local minima of the hard landscape
        # that might have been smoothed over by the soft objective
        rng_polish = np.random.RandomState(999)
        for _ in range(5):
            if (time.time() - start_time) > (deadline - 5.0):
                break
            noise = rng_polish.normal(0, 0.05, size=best_params.shape)
            candidates.append(best_params + noise)
        
        for cand in candidates:
            if (time.time() - start_time) > (deadline - 2.0):
                break
            try:
                res = minimize(
                    _neg_hard_score,
                    cand,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9, 'maxfev': 5000}
                )
                hard_s = -res.fun
                if hard_s > best_score:
                    best_score = hard_s
                    best_params = res.x
            except Exception:
                pass

    # 4. Final Verification and Cleanup
    if best_params is None:
        # Fallback to regular polygon if something went wrong
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_params = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
        best_score = _score_points(best_params.reshape(N, 2))
        
    # Center the points to ensure numerical stability and standard position
    pts_out = best_params.reshape(N, 2)
    centroid = np.mean(pts_out, axis=0)
    pts_out = pts_out - centroid
    
    return pts_out
