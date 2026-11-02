import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute triangle indices once
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]
_NTRIS = len(_IDX)

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def compute_score(points):
    """Compute min triangle area / convex hull area, vectorized."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull_area = _hull_area(points)
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area

def _smooth_poly_area(points):
    """Smooth proxy for convex hull area: area of polygon formed by
    sorting points by polar angle around their centroid.
    Smooth almost everywhere; equals hull area when points are in convex
    position and centroid is inside."""
    c = points.mean(axis=0)
    rel = points - c
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    order = np.argsort(ang)
    p = points[order]
    # Shoelace
    x = p[:, 0]
    y = p[:, 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

def _soft_min_and_grad(points, beta):
    """Return (soft_min_area, grad_points) where soft_min approximates
    the minimum triangle area via log-sum-exp, and grad is d(soft_min)/d(points)."""
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    max_area = np.max(areas)
    if max_area < 1e-15:
        return 0.0, np.zeros_like(points)

    # shifted so that max -> 0, others negative
    shifted = areas - max_area
    # Use log-sum-exp in a numerically stable way:
    # soft_min = -log(sum(exp(-beta*(A_max - A_i))))/beta
    #          = (A_max - log(sum(exp(-beta*(A_max - A_i)))))/beta  ... wait
    # Let's define: f = min_i A_i approx.
    # soft_min = -(1/beta) * log( sum_i exp(-beta * A_i) )
    # = (1/beta) * ( -log(sum exp(-beta A_i)) )
    # For stability: -log(sum exp(-beta A_i)) = beta*A_max - log(sum exp(-beta(A_i - A_max)))
    # = beta*A_max - log(sum exp(beta*(A_max - A_i)))
    # Note A_max - A_i >= 0, so beta*(A_max - A_i) >= 0 -> exp could overflow for large beta.
    # Use: sum exp(-beta*(A_i - A_max)) where (A_i - A_max) <= 0, so -beta*(A_i - A_max) >= 0.
    # Actually exp(-beta*(A_i - A_max)) = exp(beta*(A_max - A_i)), same thing.
    # To avoid overflow, subtract the max of the exponent. The max exponent is at A_i = min A,
    # which is beta*(A_max - A_min). Let's just compute with the min-shifted version.

    # Exponent e_i = -beta * A_i. Max e_i = -beta * min(A_i) = -beta * A_min.
    # logsumexp(e) = e_max + log(sum exp(e_i - e_max))
    # e_i - e_max = -beta*(A_i - A_min) <= 0, safe.
    e = -beta * areas
    e_max = np.max(e)
    logsum = e_max + np.log(np.sum(np.exp(e - e_max)))
    soft_min = -logsum / beta

    # Gradient: d(soft_min)/dA_i = - (1/beta) * d(logsum)/dA_i
    # d(logsum)/dA_i = d(e_max + log(sum exp(e-e_max)))/dA_i
    # e_i depends on A_i. e_max is the max e, which is a piecewise-constant selection (ignore its derivative, subgradient).
    # d(logsum)/dA_i = (1/sum) * sum_j exp(e_j - e_max) * d(e_j)/dA_i
    # d(e_j)/dA_i = -beta * delta_{ji}
    # So d(logsum)/dA_i = -beta * exp(e_i - e_max) / sum(exp(e_j - e_max))
    # = -beta * w_i  where w_i = exp(e_i - e_max)/sum
    # Therefore d(soft_min)/dA_i = - (1/beta) * (-beta * w_i) = w_i
    # So d(soft_min)/dA_i = w_i (softmax weights over the -beta*A_i).

    w = np.exp(e - e_max)
    w = w / np.sum(w)

    # Now chain rule: A_i = 0.5 * |cross_i|. dA_i/d(cross_i) = 0.5 * sign(cross_i).
    signs = np.sign(cross)
    wA = w * 0.5 * signs  # d(soft_min)/d(cross_i)

    # cross_i = v1_x * v2_y - v1_y * v2_x
    # v1 = p_j - p_i, v2 = p_k - p_i
    # d(cross)/d(p_i) = -d(cross)/d(p_j) - d(cross)/d(p_k)  (since cross depends on p_i via -v1,-v2)
    # Let's compute d(cross)/d(p_j): v1 = p_j - p_i -> dv1/dp_j = I, v2 = p_k - p_i -> dv2/dp_j = 0
    # cross = v1_x v2_y - v1_y v2_x
    # d(cross)/d(p_j_x) = v2_y, d(cross)/d(p_j_y) = -v2_x
    # d(cross)/d(p_k_x) = -v1_y, d(cross)/d(p_k_y) = v1_x
    # d(cross)/d(p_i_x) = -(v2_y) + (v1_y) = v1_y - v2_y
    # d(cross)/d(p_i_y) = -(-v2_x) + (-v1_x) = v2_x - v1_x
    d_cross_di_x = v1[:, 1] - v2[:, 1]
    d_cross_di_y = v2[:, 0] - v1[:, 0]
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]

    grad_points = np.zeros_like(points)
    np.add.at(grad_points, _I, wA[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y]))
    np.add.at(grad_points, _J, wA[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y]))
    np.add.at(grad_points, _K, wA[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y]))

    return soft_min, grad_points

def _smooth_obj_and_grad(x_flat, beta):
    """Objective: -soft_min_area / smooth_poly_area, with gradient."""
    points = x_flat.reshape(N, 2)
    soft_min, grad_sm = _soft_min_and_grad(points, beta)
    poly_area = _smooth_poly_area(points)
    if poly_area < 1e-12:
        return 0.0, np.zeros_like(x_flat)

    score = soft_min / poly_area

    # d(score)/d(points) = (grad_sm * poly_area - soft_min * d(poly_area)/d(points)) / poly_area^2
    # We need d(poly_area)/d(points). The smooth poly area is the shoelace of the angle-sorted points.
    # This is a piecewise-linear function of the points (within a fixed angular ordering), so its
    # gradient is piecewise-constant. Computing it analytically is complex due to the sorting.
    # For a smooth optimization, we approximate d(poly_area)/d(points) = 0 (treat area as constant
    # in the gradient), which is a common approximation when the area is slowly varying relative
    # to the min-area term. This makes the gradient = grad_sm / poly_area.
    # This is valid when the polygon shape (ordering) is stable, which it is near a convex optimum.
    grad_score = grad_sm / poly_area

    return -score, -grad_score.flatten()

def _neg_exact(x_flat):
    points = x_flat.reshape(N, 2)
    return -compute_score(points)

def _regular_polygon(n, radius=1.0, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _two_ring(n1, n2, r1, r2, rot1, rot2):
    return np.vstack([_regular_polygon(n1, r1, rot1), _regular_polygon(n2, r2, rot2)])

def _three_ring(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3):
    return np.vstack([_regular_polygon(n1, r1, rot1),
                      _regular_polygon(n2, r2, rot2),
                      _regular_polygon(n3, r3, rot3)])

def _normalize(points):
    """Normalize so convex hull area = 1 (scale by 1/sqrt(area))."""
    ha = _hull_area(points)
    if ha < 1e-12:
        return points
    return points / np.sqrt(ha)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_points = None
    best_score = -float('inf')
    rng = np.random.default_rng(42)

    def remaining():
        return deadline - time.time()

    def try_update(points):
        nonlocal best_points, best_score
        pts = _normalize(points)
        s = compute_score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
        return s

    def _annealed_smooth_optimize(init_points, betas, maxiter_per_stage):
        """Run L-BFGS-B on the smooth objective, annealing beta."""
        x = _normalize(init_points).flatten()
        for beta in betas:
            if remaining() < 8:
                break
            def _obj_grad(xf, beta_val=beta):
                return _smooth_obj_and_grad(xf, beta_val)
            try:
                res = minimize(_obj_grad, x, jac=True, method='L-BFGS-B',
                               options={'maxiter': maxiter_per_stage, 'gtol': 1e-12, 'ftol': 1e-14})
                x = res.x
                try_update(x.reshape(N, 2))
            except Exception:
                pass
        return x

    def _polish_exact(x_flat, maxiter=500):
        """Final Nelder-Mead on the true score."""
        if remaining() < 5:
            return x_flat
        try:
            res = minimize(_neg_exact, x_flat, method='Nelder-Mead',
                           options={'maxiter': maxiter, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
            try_update(res.x.reshape(N, 2))
            return res.x
        except Exception:
            return x_flat

    # ---- Build diverse initializations ----
    inits = []

    # Regular 14-gon
    inits.append(_regular_polygon(N, 1.0, 0.0))

    # Two-ring configurations
    for n1, n2 in [(7, 7), (6, 8), (8, 6), (5, 9), (9, 5), (4, 10), (10, 4), (3, 11), (11, 3)]:
        for r2 in [0.3, 0.45, 0.6, 0.75]:
            for rot2 in [0.0, np.pi / n2, np.pi / n1]:
                inits.append(_two_ring(n1, n2, 1.0, r2, 0.0, rot2))

    # Three-ring configurations
    for (n1, n2, n3) in [(5, 4, 5), (4, 5, 5), (5, 5, 4), (6, 4, 4), (4, 6, 4), (4, 4, 6)]:
        for r2 in [0.4, 0.55, 0.7]:
            for r3 in [0.2, 0.35]:
                inits.append(_three_ring(n1, n2, n3, 1.0, r2, r3, 0.0, np.pi / n2, np.pi / n3))

    # Jittered regular polygon
    base = _regular_polygon(N, 1.0, 0.0)
    for _ in range(8):
        inits.append(base + rng.normal(0, 0.12, (N, 2)))

    # Random in disk
    for _ in range(12):
        r = np.sqrt(rng.random(N))
        th = rng.random(N) * 2 * np.pi
        inits.append(np.column_stack([r * np.cos(th), r * np.sin(th)]))

    # Evaluate all inits to get a baseline
    for p in inits:
        try_update(p)

    # Sort inits by exact score, descending
    scored_inits = []
    for p in inits:
        pn = _normalize(p)
        scored_inits.append((compute_score(pn), pn))
    scored_inits.sort(key=lambda t: t[0], reverse=True)

    # ---- Annealing schedule ----
    betas = [20.0, 60.0, 150.0, 400.0, 1000.0, 2500.0]

    # ---- Optimize from top inits ----
    top_k = 10
    for sc, init_pts in scored_inits[:top_k]:
        if remaining() < 20:
            break
        x = _annealed_smooth_optimize(init_pts, betas, maxiter_per_stage=150)
        x = _polish_exact(x, maxiter=400)

    # ---- If we still have time, do a second round from the current best with a fresh perturbation ----
    if best_points is not None and remaining() > 40:
        # Perturb the best and re-optimize
        for step in [0.2, 0.4, 0.6]:
            if remaining() < 25:
                break
            x0 = best_points.flatten()
            x_pert = x0 + rng.normal(0, step, (N, 2)).flatten()
            x = _annealed_smooth_optimize(x_pert.reshape(N, 2), betas[2:], maxiter_per_stage=120)
            x = _polish_exact(x, maxiter=300)

    # ---- Final tight polish from best ----
    if best_points is not None and remaining() > 10:
        x0 = best_points.flatten()
        x0 = _polish_exact(x0, maxiter=2000)

    if best_points is None:
        return _regular_polygon(N, 1.0, 0.0)
    return best_points
