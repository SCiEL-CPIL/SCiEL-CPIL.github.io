import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _vertex_score(distances_sorted):
    d = distances_sorted
    n = len(d)
    if n != 19:
        return 0.0
    if np.any(d < 0.05):
        return 0.0
    diffs = d[1:] - d[:-1]
    min_diff = np.min(diffs)
    k = int(np.argmax(diffs == min_diff))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        four_d = d[0:4]
    elif k == 17:
        four_d = d[15:19]
    else:
        four_d = d[k-1:k+3]
    if np.any(four_d <= 0):
        return 0.0
    ratios = np.maximum(four_d / D, D / four_d)
    raw = np.mean(ratios)
    score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
    return float(score)

def _is_valid_convex(p):
    n = len(p)
    if not np.all(np.isfinite(p)):
        return False
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(p[i] - p[j]) < 1e-10:
                return False
    cross_products = np.zeros(n)
    for i in range(n):
        e1 = p[(i+1) % n] - p[i]
        e2 = p[(i+2) % n] - p[(i+1) % n]
        cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
    if np.any(np.abs(cross_products) < 1e-14):
        return False
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return False
    total_turn = 0.0
    for i in range(n):
        v1 = p[(i+1) % n] - p[i]
        v2 = p[(i+2) % n] - p[(i+1) % n]
        ang1 = np.arctan2(v1[1], v1[0])
        ang2 = np.arctan2(v2[1], v2[0])
        turn = ang2 - ang1
        while turn > np.pi:
            turn -= 2 * np.pi
        while turn < -np.pi:
            turn += 2 * np.pi
        total_turn += turn
    if not np.isclose(abs(total_turn), 2 * np.pi, atol=1e-6):
        return False
    return True

def _compute_min_vertex_score(polygon):
    n = len(polygon)
    if n != 20:
        return 0.0
    if not np.all(np.isfinite(polygon)):
        return 0.0
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(p**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist

    if not _is_valid_convex(p):
        return 0.0

    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))

    min_score = 1.0
    for i in range(n):
        d = np.delete(dists[i], i)
        d_sorted = np.sort(d)
        score = _vertex_score(d_sorted)
        min_score = min(min_score, score)
    return min_score

def _regular_polygon():
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)

def _soft_min_vertex_score(polygon, beta=15.0):
    """
    Smooth surrogate for the minimum vertex score.
    Returns a value close to the true min score but differentiable.
    """
    n = len(polygon)
    if n != 20:
        return 0.0
    if not np.all(np.isfinite(polygon)):
        return 0.0
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(p**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist

    if not _is_valid_convex(p):
        return -1000.0 # Heavy penalty for invalid convexity

    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))

    scores = np.zeros(n)
    for i in range(n):
        d = np.delete(dists[i], i)
        d_sorted = np.sort(d)
        scores[i] = _vertex_score(d_sorted)
    
    # Soft min: -1/beta * log(sum(exp(-beta * scores)))
    # If any score is 0, the soft min should be close to 0.
    # To avoid numerical issues with exp, we shift by max score? No, we want min.
    # min(x) approx -1/beta * log(sum(exp(-beta*x)))
    # Let's use a stable log-sum-exp implementation.
    # If all scores are 0, result is 0.
    if np.all(scores == 0):
        return 0.0
    
    # To handle the case where some scores are 0 and others are positive,
    # the min is 0. The soft min will be small.
    # We want to maximize the min score.
    # So we minimize -soft_min.
    
    # Standard log-sum-exp stability:
    # m = max(scores)
    # log_sum_exp(-beta * scores) = -beta * m + log(sum(exp(-beta * (scores - m))))
    # Wait, we want min.
    # min(scores) = - max(-scores)
    # soft_min = -1/beta * log(sum(exp(-beta * scores)))
    
    # If scores are in [0, 1], -beta * scores are in [-beta, 0].
    # exp(-beta * scores) is in [exp(-beta), 1].
    # Sum is in [n*exp(-beta), n].
    # log(Sum) is in [log(n) - beta, log(n)].
    # -1/beta * log(Sum) is in [-log(n)/beta, (beta - log(n))/beta] = [~0, 1 - log(n)/beta].
    # This looks correct. For large beta, it approaches min(scores).
    
    # However, if a score is 0, exp(0)=1. Sum >= 1. log(Sum) >= 0.
    # Result <= 0? No, -1/beta * log(Sum) <= 0.
    # But scores are positive.
    # Let's check: min(0.5, 0.5) = 0.5.
    # soft_min(0.5, 0.5) = -1/beta * log(2 * exp(-0.5*beta)) = -1/beta * (log(2) - 0.5*beta) = 0.5 - log(2)/beta.
    # This is slightly less than 0.5.
    
    # If one score is 0 and others are 1.
    # soft_min(0, 1) = -1/beta * log(exp(0) + exp(-beta)) = -1/beta * log(1 + tiny) ~ 0.
    # This is good.
    
    # We return the negative of this for minimization?
    # The function _objective_soft will return -soft_min.
    
    m = np.max(scores)
    # If m is 0, return 0
    if m == 0:
        return 0.0
    
    # Calculate log(sum(exp(-beta * scores)))
    # To be safe, shift by m? No, -beta*scores are negative.
    # We can just compute directly if beta is not too large.
    # With beta=15, exp(-15) is small.
    exp_terms = np.exp(-beta * scores)
    sum_exp = np.sum(exp_terms)
    if sum_exp == 0:
        return 0.0
    log_sum_exp = np.log(sum_exp)
    soft_min_val = -log_sum_exp / beta
    
    # Clamp to [0, 1]
    return float(np.clip(soft_min_val, 0.0, 1.0))

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0 # Safety margin for 630s limit
    
    n = 20
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)
    
    if time.time() > deadline - 10:
        return best_poly
    
    # Fourier representation:
    # r(theta) = 1 + sum_{k=1}^{K} (a_k cos(k*theta) + b_k sin(k*theta))
    # K=10 -> 20 parameters.
    # This allows complex shapes.
    
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def _fourier_poly(params):
        # params: [a1, b1, a2, b2, ..., a10, b10]
        radii = np.ones(n)
        for k in range(1, 11):
            a_k = params[2*(k-1)]
            b_k = params[2*(k-1)+1]
            radii += a_k * np.cos(k * base_angles)
            radii += b_k * np.sin(k * base_angles)
        
        # Ensure radii are positive to avoid invalid geometry
        # If radii <= 0, the point is at center or invalid.
        # We clip to a small positive value.
        radii = np.maximum(radii, 0.1)
        
        x = radii * np.cos(base_angles)
        y = radii * np.sin(base_angles)
        return np.stack([x, y], axis=1)
    
    def _objective_soft(params):
        poly = _fourier_poly(params)
        s = _soft_min_vertex_score(poly, beta=15.0)
        return -s

    def _objective_exact(params):
        poly = _fourier_poly(params)
        s = _compute_min_vertex_score(poly)
        return -s

    # Phase 1: Differential Evolution on Fourier parameters
    # Bounds: coefficients should be small enough to maintain convexity.
    # Regular polygon has r=1. Perturbations of 0.2 might be safe.
    bounds = [(-0.2, 0.2)] * 20
    
    best_de_score = best_score
    best_de_poly = best_poly.copy()
    
    try:
        # Use sobol init for better coverage
        result_de = differential_evolution(
            _objective_soft,
            bounds,
            seed=42,
            maxiter=50,
            tol=1e-8,
            polish=False,
            init='sobol',
            popsize=20
        )
        if result_de.fun is not None and np.isfinite(result_de.fun):
            de_score = -result_de.fun
            # Note: de_score is based on soft min, so it might be slightly lower than true min.
            # We must evaluate exact score to compare.
            candidate_poly = _fourier_poly(result_de.x)
            exact_candidate_score = _compute_min_vertex_score(candidate_poly)
            if exact_candidate_score > best_de_score:
                best_de_score = exact_candidate_score
                best_de_poly = candidate_poly
    except Exception:
        pass
    
    if best_de_score > best_score:
        best_score = best_de_score
        best_poly = best_de_poly
    
    if time.time() > deadline - 60:
        return best_poly
    
    # Phase 2: Nelder-Mead refinement on Soft Objective
    # Initialize from the best Fourier params found.
    # We need to extract the best params. DE returns result_de.x.
    # If DE failed, we start from zero.
    best_params = np.zeros(20)
    # Re-run DE to get the best x if we didn't store it? 
    # It's better to store best_params during DE or re-derive.
    # Since we stored best_poly, we can try to fit Fourier coeffs? 
    # No, just use the result_de.x if available.
    
    # Let's keep track of best_params in the DE block.
    # Refactoring:
    best_params_de = np.zeros(20)
    # Re-do DE logic to capture params? Or just assume result_de.x is the best.
    # We'll just use result_de.x if the score was good.
    
    # To be safe, we'll run a few NM starts from different initializations.
    starts = [
        np.zeros(20),
        np.array([0.1, 0, 0.05, 0, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ]
    
    # Add the DE result if it was good
    if best_de_score > best_score - 1e-6:
         # We need the params. Let's re-run DE? No, too expensive.
         # We'll just use the stored best_poly to initialize a local search in Cartesian?
         # No, let's stick to Fourier.
         # We can extract approximate Fourier coeffs from best_poly using least squares.
         A = np.zeros((n, 20))
         for k in range(1, 11):
             A[:, 2*(k-1)] = np.cos(k * base_angles)
             A[:, 2*(k-1)+1] = np.sin(k * base_angles)
         # radii = 1 + A @ params
         # radii = norm(best_poly)
         radii_obs = np.linalg.norm(best_poly, axis=1)
         # Solve for params
         params_fit, _, _, _ = np.linalg.lstsq(A, radii_obs - 1, rcond=None)
         starts.append(params_fit)
    
    for init in starts:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(
                _objective_soft,
                init,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun is not None and np.isfinite(res.fun):
                candidate_poly = _fourier_poly(res.x)
                exact_candidate_score = _compute_min_vertex_score(candidate_poly)
                if exact_candidate_score > best_score:
                    best_score = exact_candidate_score
                    best_poly = candidate_poly
        except Exception:
            pass
    
    if time.time() > deadline - 30:
        return best_poly
    
    # Phase 3: Nelder-Mead on Exact Objective
    # Use the best_poly found so far.
    # We need to convert best_poly back to Fourier params for NM in Fourier space.
    # Or we can do NM in Cartesian space directly.
    # Cartesian NM is 40D. Fourier NM is 20D. Fourier is better.
    
    # Extract Fourier params from best_poly
    A = np.zeros((n, 20))
    for k in range(1, 11):
        A[:, 2*(k-1)] = np.cos(k * base_angles)
        A[:, 2*(k-1)+1] = np.sin(k * base_angles)
    radii_obs = np.linalg.norm(best_poly, axis=1)
    params_init, _, _, _ = np.linalg.lstsq(A, radii_obs - 1, rcond=None)
    
    try:
        res = minimize(
            _objective_exact,
            params_init,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if res.fun is not None and np.isfinite(res.fun):
            candidate_poly = _fourier_poly(res.x)
            exact_candidate_score = _compute_min_vertex_score(candidate_poly)
            if exact_candidate_score > best_score:
                best_score = exact_candidate_score
                best_poly = candidate_poly
    except Exception:
        pass
    
    return best_poly
