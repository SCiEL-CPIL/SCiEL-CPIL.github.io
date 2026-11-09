import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

DEADLINE = 620.0

def _raw_score(points):
    """
    Calculate the exact score: (d_min / d_max)^2.
    Returns 0 if invalid.
    """
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    
    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_max < 1e-10:
        return 0.0
    if d_min < 1e-10:
        return 0.0
        
    ratio = d_min / d_max
    return ratio * ratio

def _spherical_to_cartesian(params):
    """
    Convert 28D spherical coordinates (14 points, 2 params each) to (14, 3) cartesian.
    params: (28,) array where params[2i] = theta (polar, 0..pi), params[2i+1] = phi (azimuth, 0..2pi)
    """
    theta = params[0::2]
    phi = params[1::2]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _cartesian_to_spherical(points):
    """
    Convert (14, 3) cartesian points on unit sphere to (28,) spherical params.
    """
    theta = np.arccos(np.clip(points[:, 2], -1.0, 1.0))
    phi = np.arctan2(points[:, 1], points[:, 0])
    phi = (phi + 2 * np.pi) % (2 * np.pi)
    return np.column_stack([theta, phi]).flatten()

def _objective_and_grad(params, lam):
    """
    Compute the smooth surrogate score and its gradient.
    Score = (soft_min / soft_max)^2
    """
    points = _spherical_to_cartesian(params)
    
    # Compute all pairwise distances
    # Using broadcasting for speed
    D = points[:, None, :] - points[None, :, :]
    D2 = np.sum(D * D, axis=2)
    dists_mat = np.sqrt(np.maximum(D2, 1e-30))
    
    # Extract upper triangle distances
    mask = np.triu(np.ones((14, 14), dtype=bool), k=1)
    dists = dists_mat[mask]
    
    if len(dists) == 0:
        return 0.0, np.zeros(28)
    
    # Soft max and soft min
    # To avoid overflow, we can subtract max/min before exp, but with lam ~ 1-10 and dists ~ 0-2, it's safe.
    exp_pos = np.exp(lam * dists)
    sum_exp_pos = np.sum(exp_pos)
    soft_max = np.log(sum_exp_pos) / lam
    
    exp_neg = np.exp(-lam * dists)
    sum_exp_neg = np.sum(exp_neg)
    soft_min = -np.log(sum_exp_neg) / lam
    
    if soft_max < 1e-10 or soft_min < 1e-10:
        return 0.0, np.zeros(28)
        
    score = (soft_min / soft_max) ** 2
    
    # We want to maximize score, so we minimize -score
    val = -score
    
    # Gradients
    # d(soft_max)/d(d_k) = exp(lam*d_k) / sum(exp(lam*d_j))
    w_max = exp_pos / sum_exp_pos
    
    # d(soft_min)/d(d_k) = -exp(-lam*d_k) / sum(exp(-lam*d_j))
    w_min = -exp_neg / sum_exp_neg
    
    # d(score)/d(soft_min) = 2 * soft_min / soft_max^2
    # d(score)/d(soft_max) = -2 * soft_min^2 / soft_max^3
    
    coeff_min = 2.0 * soft_min / (soft_max * soft_max)
    coeff_max = -2.0 * (soft_min * soft_min) / (soft_max * soft_max * soft_max)
    
    # d(score)/d(d_k) = coeff_min * w_min[k] + coeff_max * w_max[k]
    dscore_ddists = coeff_min * w_min + coeff_max * w_max
    
    # Now compute d(score)/d(params)
    # d(score)/d(points[i]) = sum_{j != i} dscore_ddists[idx(i,j)] * (points[i] - points[j]) / d_ij
    # Note: dists_mat[i,j] = d_ij
    
    grad_points = np.zeros((14, 3))
    
    # Indices for upper triangle
    indices_i, indices_j = np.where(mask)
    
    pair_dists = dists_mat[indices_i, indices_j]
    pair_vecs = points[indices_i] - points[indices_j]
    
    # Avoid division by zero
    safe_dists = np.maximum(pair_dists, 1e-15)
    unit_vecs = pair_vecs / safe_dists[:, None]
    
    # contrib_k = dscore_ddists[k] * unit_vec[k]
    contribs = dscore_ddists[:, None] * unit_vecs
    
    # Accumulate gradients
    # For each pair (i, j):
    # grad_points[i] += contrib
    # grad_points[j] -= contrib
    
    # Using np.add.at for efficient accumulation
    np.add.at(grad_points, indices_i, contribs)
    np.add.at(grad_points, indices_j, -contribs)
    
    # Chain rule: d(points)/d(params)
    theta = params[0::2]
    phi = params[1::2]
    
    # d(points[i])/d(theta_i) = [cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)]
    # d(points[i])/d(phi_i) = [-sin(theta)*sin(phi), sin(theta)*cos(phi), 0]
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    
    dpth = np.column_stack([cos_t * cos_p, cos_t * sin_p, -sin_t])
    dphip = np.column_stack([-sin_t * sin_p, sin_t * cos_p, np.zeros(14)])
    
    grad_params = np.zeros(28)
    
    # dot(grad_points[i], dpth[i])
    grad_params[0::2] = np.sum(grad_points * dpth, axis=1)
    # dot(grad_points[i], dphip[i])
    grad_params[1::2] = np.sum(grad_points * dphip, axis=1)
    
    # We are minimizing -score, so gradient is -grad_score
    grad_flat = -grad_params
    
    return val, grad_flat

def _generate_icosahedron():
    phi = (1 + np.sqrt(5)) / 2
    points = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def _generate_fibonacci_sphere(n=14):
    golden_ratio = (1 + np.sqrt(5)) / 2
    points = []
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        radius = np.sqrt(1 - z * z)
        theta = 2 * np.pi * i / golden_ratio
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append([x, y, z])
    return np.array(points)

def _optimize_from(params0, deadline, lam_start=10.0, lam_end=100.0):
    """
    Optimize from initial spherical parameters using annealing on lambda.
    """
    if time.time() > deadline:
        return params0, _raw_score(_spherical_to_cartesian(params0))
    
    params_current = params0.copy()
    best_params = params0.copy()
    best_score = _raw_score(_spherical_to_cartesian(params0))
    
    # Annealing schedule for lambda (sharpness of min/max)
    # Start with soft (small lambda) to escape local minima, end with sharp (large lambda)
    num_stages = 4
    lams = np.geomspace(lam_start, lam_end, num_stages)
    
    for lam in lams:
        if time.time() > deadline - 30:
            break
        
        try:
            res = minimize(
                lambda p: _objective_and_grad(p, lam)[0],
                params_current,
                jac=lambda p: _objective_and_grad(p, lam)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            params_current = res.x
            score = _raw_score(_spherical_to_cartesian(params_current))
            if score > best_score:
                best_score = score
                best_params = params_current.copy()
        except Exception:
            pass
    
    # Final polish with very high lambda (effectively exact min/max)
    if time.time() < deadline - 15:
        try:
            lam_final = 1000.0
            res = minimize(
                lambda p: _objective_and_grad(p, lam_final)[0],
                params_current,
                jac=lambda p: _objective_and_grad(p, lam_final)[1],
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            params_current = res.x
            score = _raw_score(_spherical_to_cartesian(params_current))
            if score > best_score:
                best_score = score
                best_params = params_current.copy()
        except Exception:
            pass
    
    return best_params, best_score

def min_max_dist_dim3_14():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_score = 0.0
    best_params = np.zeros(28)
    
    rng = np.random.default_rng(42)
    
    # Initial guesses in cartesian, then convert to spherical
    guesses_cart = []
    
    # 1. Icosahedron + 2 points (try several placements for the 2 points)
    ico = _generate_icosahedron()
    
    # Add 2 points optimally? Just add random ones and let optimizer fix them.
    # Better: Add points near the "gaps" or just random and let it move.
    for _ in range(5):
        p1 = rng.normal(size=3)
        p1 = p1 / np.linalg.norm(p1)
        p2 = rng.normal(size=3)
        p2 = p2 / np.linalg.norm(p2)
        guess = np.vstack([ico, p1, p2])
        guesses_cart.append(guess)
        
    # 2. Fibonacci sphere
    fib = _generate_fibonacci_sphere(14)
    guesses_cart.append(fib)
    
    # 3. Random points on sphere
    for _ in range(20):
        pts = rng.normal(size=(14, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        guesses_cart.append(pts)
        
    # 4. Cubic points
    cube = np.array([
        [1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
        [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1],
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
    ])
    norms = np.linalg.norm(cube, axis=1, keepdims=True)
    cube = cube / norms
    guesses_cart.append(cube)
    
    # 5. Octahedron + extra points
    octa = np.array([
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
    ])
    norms = np.linalg.norm(octa, axis=1, keepdims=True)
    octa = octa / norms
    for _ in range(4):
        p = rng.normal(size=3)
        p = p / np.linalg.norm(p)
        octa = np.vstack([octa, p])
    guesses_cart.append(octa)
    
    # Sort guesses by initial score to prioritize good starts?
    # Or just iterate. Iterating is fine.
    
    for guess in guesses_cart:
        if time.time() > deadline - 40:
            break
        
        params0 = _cartesian_to_spherical(guess)
        params_opt, score = _optimize_from(params0, deadline)
        if score > best_score:
            best_score = score
            best_params = params_opt
    
    # If we have time, try more random starts
    while time.time() < deadline - 60:
        pts = rng.normal(size=(14, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        params0 = _cartesian_to_spherical(pts)
        params_opt, score = _optimize_from(params0, deadline)
        if score > best_score:
            best_score = score
            best_params = params_opt
    
    # Final polish on best
    if time.time() < deadline - 10 and best_score > 0:
        try:
            params_final, score_final = _optimize_from(
                best_params, deadline, lam_start=50.0, lam_end=500.0
            )
            if score_final > best_score:
                best_score = score_final
                best_params = params_final
        except Exception:
            pass
    
    if best_score == 0.0:
        # Fallback
        best_points = _generate_icosahedron()[:12]
        p1 = rng.normal(size=3)
        p1 = p1 / np.linalg.norm(p1)
        p2 = rng.normal(size=3)
        p2 = p2 / np.linalg.norm(p2)
        best_points = np.vstack([best_points, p1, p2])
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / norms
        return best_points
    
    return _spherical_to_cartesian(best_params)

if __name__ == "__main__":
    pts = min_max_dist_dim3_14()
    print(_raw_score(pts))
