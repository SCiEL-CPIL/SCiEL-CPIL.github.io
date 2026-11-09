# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _normalize(pts):
    centroid = pts.mean(axis=0)
    p = pts - centroid
    d = np.linalg.norm(p, axis=1)
    mean_d = d.mean()
    if mean_d <= 0.01:
        return None, None, None
    p = p / mean_d
    return p, d, mean_d

def _check_strictly_convex(p):
    n = len(p)
    if n < 3:
        return False
    # no repeated points
    for i in range(n):
        if np.linalg.norm(p[i] - p[(i + 1) % n]) < 1e-9:
            return False
    # compute cross products of consecutive edges
    signs = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-9:
            return False
        signs.append(1 if cross > 0 else -1)
    # all same sign
    if signs.count(signs[0]) != n:
        return False
    # total turning = +-2*pi
    angles = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_a = np.clip(cos_a, -1, 1)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        sin_a = cross / (np.linalg.norm(v1) * np.linalg.norm(v2))
        ang = np.arctan2(sin_a, cos_a)
        angles.append(ang)
    total_turn = sum(angles)
    expected = 2 * np.pi if signs[0] > 0 else -2 * np.pi
    if abs(abs(total_turn) - 2 * np.pi) > 1e-4:
        return False
    return True

def _vertex_score(p, v_idx):
    n = len(p)
    v = p[v_idx]
    # distances to all other vertices
    d = np.array([np.linalg.norm(v - p[j]) for j in range(n) if j != v_idx])
    d_sorted = np.sort(d)
    if np.any(d_sorted < 0.05):
        return 0.0
    # find k: first index minimizing d_{k+1} - d_k for k in 0..17
    gaps = d_sorted[1:] - d_sorted[:-1]
    k = int(np.argmin(gaps))  # 0..17
    D = (d_sorted[k] + d_sorted[k + 1]) / 2
    if D < 0.1:
        return 0.0
    # select four distances
    if k == 0:
        selected = d_sorted[0:4]
    elif k == 17:
        selected = d_sorted[15:19]
    else:
        selected = d_sorted[k - 1:k + 3]
    if len(selected) < 4:
        return 0.0
    raw = np.mean([max(di / D, D / di) for di in selected])
    if raw < 1e-12:
        return 0.0
    score = min(1.0, 1.0 / raw)
    return score

def _score_polygon(p):
    """Score a polygon. Returns min vertex score (0 if invalid)."""
    pn, _, _ = _normalize(p)
    if pn is None:
        return 0.0
    if not _check_strictly_convex(pn):
        return 0.0
    min_vs = 1.0
    for i in range(len(pn)):
        vs = _vertex_score(pn, i)
        if vs < min_vs:
            min_vs = vs
        if min_vs < 0.01:  # early exit
            break
    return min_vs

def _smooth_score(radii, angles):
    """
    Smoothed surrogate for the min-vertex-score.
    We approximate the min over vertices by a soft-min (log-sum-exp style)
    to make the objective differentiable (or at least continuous) for Nelder-Mead.
    """
    n = len(radii)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Quick validity check to avoid wasting time on obviously bad shapes
    # (Nelder-Mead might wander into non-convex regions)
    # We'll rely on the penalty being high for invalid shapes.
    
    pn, _, _ = _normalize(pts)
    if pn is None:
        return -100.0 # Heavy penalty
        
    # Check convexity quickly
    if not _check_strictly_convex(pn):
        return -100.0 # Heavy penalty

    # Calculate vertex scores
    scores = []
    for i in range(n):
        vs = _vertex_score(pn, i)
        scores.append(vs)
    
    scores = np.array(scores)
    
    # Use a soft-min to create a smooth landscape.
    # min(x) approx -log(exp(sum(exp(x/tau)))) * tau
    # We want to maximize min(score), so we minimize -min(score).
    # Let's minimize -soft_min(scores).
    # soft_min(scores, tau) = -tau * log(sum(exp(-s/tau)))
    # As tau -> 0, soft_min -> min.
    # We want a balance. Too small tau is non-smooth. Too large is too far from min.
    tau = 0.05
    
    # To avoid overflow in exp:
    # soft_min = -tau * log( sum( exp(-s/tau) ) )
    # = -tau * ( log_sum_exp(-s/tau) )
    # log_sum_exp(x) = max(x) + log(sum(exp(x - max(x))))
    
    neg_scores_over_tau = -scores / tau
    max_neg = np.max(neg_scores_over_tau)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_scores_over_tau - max_neg)))
    soft_min = -tau * log_sum_exp
    
    return -soft_min # We are minimizing the negative of the soft-min score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # 50s safety margin
    
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    best_score = 0.0
    best_radii = np.ones(n)
    
    def radii_to_pts(r):
        return np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    
    def try_candidate(r):
        nonlocal best_score, best_radii
        pts = radii_to_pts(r)
        s = _score_polygon(pts)
        if s > best_score:
            best_score = s
            best_radii = r.copy()

    # Initial seeds
    seeds = []
    seeds.append(np.ones(n)) # Regular
    
    # Add some diverse seeds
    # Alternating
    seeds.append(1.0 + 0.1 * np.array([1, -1] * 10))
    seeds.append(1.0 + 0.05 * np.array([1, -1] * 10))
    
    # Sinusoidal perturbations
    for k in [2, 3, 5, 7]:
        for amp in [0.05, 0.1, 0.2]:
            seeds.append(1.0 + amp * np.sin(k * angles))
            seeds.append(1.0 - amp * np.sin(k * angles))
            
    # Clamp seeds to reasonable range
    valid_seeds = []
    for r in seeds:
        r_clipped = np.clip(r, 0.5, 2.0)
        valid_seeds.append(r_clipped)
        
    # Run Nelder-Mead from multiple seeds
    # We will run for a limited number of iterations per seed to stay within time
    max_iter_per_seed = 2000
    tol = 1e-8
    
    for i, seed in enumerate(valid_seeds):
        if time.time() > deadline:
            break
            
        # Update best with seed
        try_candidate(seed)
        
        # Run optimization
        try:
            res = minimize(
                _smooth_score, 
                seed, 
                args=(angles,),
                method='Nelder-Mead',
                options={'maxiter': max_iter_per_seed, 'xatol': tol, 'fatol': tol}
            )
            
            # Evaluate the result
            if res.success or res.fun < 0: # If it found something valid
                # Note: _smooth_score returns negative of the score.
                # So if res.fun is negative, the score is positive.
                # We need to check the actual score to update best.
                try_candidate(res.x)
            else:
                # Even if not "successful" in scipy's sense, it might have improved
                # but hit a limit. Check the last point.
                try_candidate(res.x)
                
        except Exception as e:
            pass
            
    # Final polish: Do a few random restarts around the best found so far?
    # Or just a fine-grained local search on the best found?
    # Let's do a simple local search with small steps on the best radii.
    if best_score > 0 and time.time() < deadline:
        current = best_radii.copy()
        step = 0.01
        for _ in range(100): # Limit polish steps
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                for delta in [step, -step]:
                    new_r = current.copy()
                    new_r[i] += delta
                    if new_r[i] < 0.5 or new_r[i] > 2.5:
                        continue
                    s_new = _score_polygon(radii_to_pts(new_r))
                    if s_new > best_score:
                        best_score = s_new
                        best_radii = new_r.copy()
                        current = new_r
                        improved = True
            if not improved:
                step *= 0.7
                if step < 0.001:
                    break

    return radii_to_pts(best_radii)

# EVOLVE-BLOCK-END
