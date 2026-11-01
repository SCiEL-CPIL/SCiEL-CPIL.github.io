"""Circle packing: 50 equal circles in unit disk.
Strategy: Binary search on radius r with smooth soft-min optimization.
Uses concentric ring initialization to match likely optimal topology.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the packing score: min of pairwise distances/2 and boundary distances."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all points are inside the unit circle
    dists_to_origin = np.linalg.norm(points, axis=1)
    if np.any(dists_to_origin >= 1.0):
        return 0.0
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Get minimum pairwise distance (excluding diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    # Minimum distance to boundary
    min_boundary = 1.0 - np.max(dists_to_origin)
    return min(min_pairwise, min_boundary)

def _soft_min(x, t):
    """
    Compute the soft minimum of a set of values x with temperature t.
    soft_min(x) = -t * log(sum(exp(-x_i/t)))
    This is a smooth approximation to min(x).
    As t -> 0, it approaches min(x).
    """
    # Numerical stability: subtract min
    min_x = np.min(x)
    # If t is very small, just return min
    if t < 1e-8:
        return min_x
    # Clip x to avoid overflow in exp
    # We want exp(-(x_i - min_x)/t). Since x_i >= min_x, the argument is <= 0.
    # Actually, let's use the standard form:
    # exp(-(x_i)/t) can underflow/overflow.
    # exp(-(x_i - min_x)/t) is safe because x_i - min_x >= 0, so exp(...) <= 1.
    # Wait, soft min formula:
    # m(x) = -t * ln( sum_i exp(-x_i/t) )
    # Let x_min = min(x).
    # sum exp(-x_i/t) = exp(-x_min/t) * sum exp(-(x_i - x_min)/t)
    # ln(sum) = -x_min/t + ln( sum exp(-(x_i - x_min)/t) )
    # m(x) = -t * ( -x_min/t + ln( sum exp(-(x_i - x_min)/t) ) )
    # m(x) = x_min - t * ln( sum exp(-(x_i - x_min)/t) )
    
    x_shifted = x - min_x
    # exp(-x_shifted / t)
    # Since x_shifted >= 0, -x_shifted/t <= 0.
    # exp is in (0, 1].
    exp_terms = np.exp(-x_shifted / t)
    sum_exp = np.sum(exp_terms)
    return min_x - t * np.log(sum_exp)

def _objective_and_grad(x, r_target, t):
    """
    Objective: Minimize violation of constraints for radius r_target.
    Constraints:
    1. d_ij >= 2 * r_target
    2. 1 - |p_i| >= r_target
    
    We define a smooth energy function E that is 0 if all constraints are satisfied
    with margin, and positive otherwise.
    Specifically, we want to minimize the "soft minimum" of the margins.
    Margins:
    m_ij = d_ij - 2*r_target
    m_i = (1 - |p_i|) - r_target
    
    We want min(m_ij, m_i) >= 0.
    So we minimize -soft_min(margins, t).
    If min(margins) > 0, soft_min > 0, objective < 0.
    If min(margins) < 0, soft_min < 0, objective > 0.
    
    We want to find the largest r_target such that min(margins) >= 0.
    So we minimize E = -soft_min(margins, t).
    If we can achieve E <= 0, then r_target is feasible.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq + 1e-12)
    
    # Boundary distances
    r_norms = np.linalg.norm(pts, axis=1)
    
    # Margins
    # m_ij = dists - 2*r_target
    # We only need i < j
    iu = np.triu_indices(n, k=1)
    margins_pair = dists[iu] - 2.0 * r_target
    
    # m_i = (1 - r_norms) - r_target
    margins_bound = (1.0 - r_norms) - r_target
    
    # Combine all margins
    # We can just concatenate
    all_margins = np.concatenate([margins_pair, margins_bound])
    
    # Objective: -soft_min(all_margins, t)
    obj_val = -_soft_min(all_margins, t)
    
    # Gradient
    # d/dx (-soft_min) = d/dx (t * log(sum(exp(-m_k/t))))
    # = t * (1/sum) * sum (exp(-m_k/t) * d(-m_k)/dx * (-1/t))
    # = - (1/sum) * sum (exp(-m_k/t) * d(m_k)/dx)
    # Let w_k = exp(-m_k/t) / sum(exp(-m_j/t))
    # grad = - sum_k w_k * d(m_k)/dx
    
    # Calculate weights
    m_shifted = all_margins - np.min(all_margins)
    exp_terms = np.exp(-m_shifted / t)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    # d(m_k)/dx
    # For pair (i,j): m_k = d_ij - 2r
    # d(m_k)/dp_i = (p_i - p_j) / d_ij
    # d(m_k)/dp_j = (p_j - p_i) / d_ij
    # For bound i: m_k = 1 - r_i - r_target
    # d(m_k)/dp_i = - p_i / r_i
    
    grad = np.zeros_like(pts)
    
    # Pairwise contributions
    # We need to map weights back to pairs
    # iu[0], iu[1] contain the indices
    idx_i = iu[0]
    idx_j = iu[1]
    
    # Vectorized update for pairs
    # grad[i] += w_k * (p_i - p_j) / d_ij
    # grad[j] -= w_k * (p_i - p_j) / d_ij
    
    # dists[iu] is the distance for each pair
    d_pairs = dists[iu]
    # Avoid div by zero
    d_safe = np.maximum(d_pairs, 1e-12)
    
    # Unit vectors
    # diff[iu] gives vector from j to i? 
    # diff = pts[:, newaxis] - pts[newaxis]
    # diff[i, j] = p_i - p_j
    vecs = diff[iu] / d_safe[:, np.newaxis]
    
    # Accumulate gradients
    # np.add.at is needed for repeated indices
    np.add.at(grad, idx_i, weights * vecs)
    np.add.at(grad, idx_j, -weights * vecs)
    
    # Boundary contributions
    # weights for bounds are the last len(pts) weights
    w_bound = weights[len(margins_pair):]
    
    # d(m_bound_i)/dp_i = - p_i / r_i
    r_safe = np.maximum(r_norms, 1e-12)
    unit_vecs = pts / r_safe[:, np.newaxis]
    
    # grad[i] += w_bound[i] * (-unit_vecs[i])
    grad -= w_bound[:, np.newaxis] * unit_vecs
    
    return obj_val, grad.flatten()

def _optimize_for_radius(x0, r_target, t_init=0.05, max_iter=200):
    """
    Try to find a configuration where min margin >= 0 for given r_target.
    Returns (success, new_points).
    """
    def obj_and_grad(x):
        return _objective_and_grad(x, r_target, t_init)
    
    try:
        res = minimize(
            obj_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-5, 'maxfun': 1000}
        )
        new_pts = res.x.reshape(-1, 2)
        
        # Check if we actually achieved the radius
        # We need min margin >= -epsilon
        # Let's just check the score
        score = _compute_score(new_pts)
        # If score >= r_target - small_eps, it's a success
        if score >= r_target - 1e-5:
            return True, new_pts
        else:
            return False, new_pts
    except Exception:
        return False, x0

def _ring_init(n, seed=0):
    """
    Initialize points in concentric rings.
    This is a good heuristic for circle packing in a disk.
    """
    rng = np.random.RandomState(seed)
    
    # Try different ring configurations
    best_points = None
    best_score = -1.0
    
    # Possible numbers of rings
    # For N=50, maybe 1 center + rings of 6, 12, 18...
    # Or just random rings
    
    for trial in range(10):
        # Randomize ring radii and counts
        # Start with a center
        pts = np.array([[0.0, 0.0]])
        remaining = n - 1
        
        # Try to distribute remaining into rings
        # Max points in ring k (radius r_k) is approx 2*pi*r_k / (2*r_c)
        # But we don't know r_c yet.
        # Let's just pick random radii and fill them.
        
        # Number of rings
        n_rings = rng.randint(3, 6)
        
        # Radii for rings
        # Spread them out
        radii = np.linspace(0.3, 0.8, n_rings)
        
        # Assign points to rings
        # Weights proportional to circumference
        weights = radii
        weights = weights / np.sum(weights)
        counts = np.floor(weights * remaining).astype(int)
        # Adjust to sum to remaining
        diff = remaining - np.sum(counts)
        # Add to largest rings
        idx = np.argsort(-weights)[:abs(diff)]
        for i in idx:
            counts[i] += np.sign(diff)
        
        # Ensure counts are non-negative
        counts = np.maximum(counts, 0)
        
        current_pts = [pts[0]]
        
        for r, c in zip(radii, counts):
            if c == 0:
                continue
            angles = np.linspace(0, 2*np.pi, c, endpoint=False)
            # Add some jitter
            angles += rng.uniform(-0.1, 0.1, c)
            ring_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
            current_pts.append(ring_pts)
        
        all_pts = np.vstack(current_pts)
        
        # If we have more or less, adjust? 
        # For now, if length is not N, skip or pad
        if len(all_pts) != n:
             # Pad with random points if too few, or truncate if too many
             if len(all_pts) < n:
                 # Add random points
                 extra = n - len(all_pts)
                 theta = rng.uniform(0, 2*np.pi, extra)
                 rad = rng.uniform(0, 1, extra)
                 extra_pts = np.column_stack([rad * np.cos(theta), rad * np.sin(theta)])
                 all_pts = np.vstack([all_pts, extra_pts])
             else:
                 all_pts = all_pts[:n]
        
        # Optimize this initialization slightly to get a decent start
        # Actually, just use it as init for the main optimizer
        score = _compute_score(all_pts)
        if score > best_score:
            best_score = score
            best_points = all_pts.copy()
            
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Get initial guess
    init_pts = _ring_init(N, seed=42)
    
    # Initial score
    current_score = _compute_score(init_pts)
    best_points = init_pts.copy()
    
    # Binary search for r
    # Lower bound: current_score
    # Upper bound: 0.15 (known best is ~0.1258, but maybe we can find better with different topology?)
    # Actually, let's start with a wide range and narrow down.
    r_low = current_score
    r_high = 0.14 # Known best is 0.1258. Let's aim higher.
    
    # If initial score is already high, adjust
    if r_low > r_high:
        r_high = r_low + 0.01
    
    # We will perform a few iterations of binary search
    # For each mid, we try to optimize. If success, r_low = mid. Else r_high = mid.
    
    # To save time, we limit the number of binary search steps.
    max_search_steps = 15
    
    for step in range(max_search_steps):
        if time.time() > deadline - 20.0:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a packing with radius r_mid
        # Use current best points as init, or a perturbed version?
        # Let's use current best points.
        
        # Anneal the temperature t for the soft min
        # Start with larger t for exploration, smaller for exploitation
        t_val = 0.05 * (0.9 ** step)
        
        success, new_pts = _optimize_for_radius(best_points, r_mid, t_init=t_val, max_iter=100)
        
        if success:
            # Update best
            new_score = _compute_score(new_pts)
            if new_score > current_score:
                current_score = new_score
                best_points = new_pts.copy()
            r_low = r_mid
        else:
            r_high = r_mid
            
        # Optional: Perturb best_points to escape local minima if we get stuck
        # If r_high - r_low is small, we are close to the limit for this topology.
        # We might try a new initialization.
        if step == 5 or step == 10:
             # Try a different initialization
             alt_init = _ring_init(N, seed=step*100)
             alt_score = _compute_score(alt_init)
             if alt_score > current_score:
                 best_points = alt_init
                 current_score = alt_score
                 # Reset binary search bounds? 
                 # No, just continue with new points.
                 # The binary search range [r_low, r_high] is for the *current* topology's limit.
                 # If we switch topology, the limit might be different.
                 # Let's widen the range slightly to be safe.
                 r_high = max(r_high, current_score + 0.01)
    
    # Final polish with Nelder-Mead on the raw score to squeeze out last bits
    if time.time() < deadline - 5.0:
        def objective(x):
            pts = x.reshape(-1, 2)
            dists = np.linalg.norm(pts, axis=1)
            if np.any(dists > 1.0):
                return 10.0 + np.sum(dists[dists > 1.0] - 1.0)
            return -_compute_score(pts)
        
        try:
            res = minimize(
                objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            polished_pts = res.x.reshape(-1, 2)
            if np.all(np.linalg.norm(polished_pts, axis=1) < 1.0):
                polished_score = _compute_score(polished_pts)
                if polished_score > current_score:
                    best_points = polished_pts
                    current_score = polished_score
        except Exception:
            pass
            
    return best_points

# EVOLVE-BLOCK-END
