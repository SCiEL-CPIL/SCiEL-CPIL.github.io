import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

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

def _compute_score(polygon):
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

    p_next = np.roll(p, -1, axis=0)
    p_next2 = np.roll(p, -2, axis=0)
    e1 = p_next - p
    e2 = p_next2 - p_next
    cross_products = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]

    if np.any(np.abs(cross_products) < 1e-10):
        return 0.0
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return 0.0

    ang1 = np.arctan2(e1[:, 1], e1[:, 0])
    ang2 = np.arctan2(e2[:, 1], e2[:, 0])
    turns = ang2 - ang1
    turns = (turns + np.pi) % (2 * np.pi) - np.pi
    total_turn = np.sum(turns)
    if not np.isclose(abs(total_turn), 2 * np.pi, atol=1e-6):
        return 0.0

    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    idx = np.arange(n)
    dists[idx, idx] = np.inf

    min_score = 1.0
    for i in range(n):
        d = dists[i, :]
        d = d[np.isfinite(d)]
        d_sorted = np.sort(d)
        score = _vertex_score(d_sorted)
        if score < min_score:
            min_score = score
            if min_score == 0.0:
                return 0.0
    return min_score

def _params_to_poly(params, base_theta):
    n = len(base_theta)
    r = np.maximum(params[:n], 0.1)
    dtheta = params[n:]
    theta = base_theta + dtheta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1)

def _soft_min_vertex_score(polygon, temp=1.0):
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

    p_next = np.roll(p, -1, axis=0)
    p_next2 = np.roll(p, -2, axis=0)
    e1 = p_next - p
    e2 = p_next2 - p_next
    cross_products = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]

    if np.any(np.abs(cross_products) < 1e-10):
        return 0.0
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return 0.0

    ang1 = np.arctan2(e1[:, 1], e1[:, 0])
    ang2 = np.arctan2(e2[:, 1], e2[:, 0])
    turns = ang2 - ang1
    turns = (turns + np.pi) % (2 * np.pi) - np.pi
    total_turn = np.sum(turns)
    if not np.isclose(abs(total_turn), 2 * np.pi, atol=1e-6):
        return 0.0

    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    idx = np.arange(n)
    dists[idx, idx] = np.inf

    scores = np.zeros(n)
    for i in range(n):
        d = dists[i, :]
        d = d[np.isfinite(d)]
        d_sorted = np.sort(d)
        scores[i] = _vertex_score(d_sorted)

    if np.any(scores <= 0):
        return 0.0

    neg_scores = -scores / temp
    max_neg = np.max(neg_scores)
    log_sum = np.log(np.sum(np.exp(neg_scores - max_neg))) + max_neg
    soft_min = -log_sum * temp
    return max(0.0, min(1.0, soft_min))

def _regular_polygon():
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0

    best_poly = _regular_polygon()
    best_score = _compute_score(best_poly)

    n = 20
    base_theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    init_r = np.ones(n)
    init_dtheta = np.zeros(n)
    current_params = np.concatenate([init_r, init_dtheta])

    bounds = []
    for _ in range(n):
        bounds.append((0.5, 2.0))
    for _ in range(n):
        bounds.append((-np.pi / 10, np.pi / 10))

    def _smooth_objective(params, temp=1.0):
        poly = _params_to_poly(params, base_theta)
        score = _soft_min_vertex_score(poly, temp)
        return -score

    def _exact_objective(params):
        poly = _params_to_poly(params, base_theta)
        score = _compute_score(poly)
        return -score

    # Phase 1: L-BFGS-B on smooth angular/radial space with decreasing temperature.
    # A smooth log-sum-exp surrogate of the vertex-score minimum gives a differentiable
    # landscape so L-BFGS-B can escape the non-smooth min's local traps and find a
    # better global angular configuration.
    temps = [5.0, 2.0, 1.0, 0.5, 0.25, 0.1]
    for temp in temps:
        if time.time() > deadline - 200:
            break
        try:
            res = minimize(
                _smooth_objective,
                current_params,
                args=(temp,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            cand = _params_to_poly(res.x, base_theta)
            exact = _compute_score(cand)
            if exact > best_score:
                best_score = exact
                best_poly = cand
                current_params = res.x
        except Exception:
            pass

    # Phase 2: DE on the exact metric to explore discrete k-index structure.
    for seed in [123, 456, 789, 101, 202, 303, 404]:
        if time.time() > deadline - 120:
            break
        try:
            res = differential_evolution(
                _exact_objective,
                bounds,
                x0=current_params,
                maxiter=40,
                popsize=12,
                tol=1e-5,
                seed=seed,
                polish=False,
                maxfev=1500
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_poly = _params_to_poly(res.x, base_theta)
                current_params = res.x
        except Exception:
            pass

    # Phase 3: High-precision Nelder-Mead polish in Cartesian space on the exact metric.
    def _objective_cart(flat_params):
        poly = flat_params.reshape(n, 2)
        return -_compute_score(poly)

    try:
        res = minimize(
            _objective_cart,
            best_poly.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True, 'maxls': 200}
        )
        score = -res.fun
        if score > best_score:
            best_score = score
            best_poly = res.x.reshape(n, 2)
    except Exception:
        pass

    # Final tight polish if time remains.
    if time.time() < deadline - 15:
        try:
            res = minimize(
                _objective_cart,
                best_poly.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_poly = res.x.reshape(n, 2)
        except Exception:
            pass

    return best_poly
