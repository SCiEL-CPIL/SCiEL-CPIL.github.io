import numpy as np
import time
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Precomputed base angles (shared by every call)
# ---------------------------------------------------------------------------
_BASE_ANGLES = np.linspace(0, 2 * np.pi, 20, endpoint=False)
_N = 20


def _build_points(radii, angle_deltas=None):
    """Build points from radii and optional angular perturbations."""
    angles = _BASE_ANGLES
    if angle_deltas is not None:
        angles = angles + angle_deltas
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))


def _normalize(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return None
    return points / mean_dist


def _cross_products(points):
    """Signed z-cross of consecutive edges (cyclic)."""
    p_next = np.roll(points, -1, axis=0)
    p_next2 = np.roll(points, -2, axis=0)
    e1 = p_next - points
    e2 = p_next2 - p_next
    return e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]


def _compute_scores(points):
    """Vectorized per-vertex score. Returns array of 20 scores in [0,1]."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))

    dists_self_inf = dists.copy()
    np.fill_diagonal(dists_self_inf, np.inf)
    sorted_d = np.sort(dists_self_inf, axis=1)
    d = sorted_d[:, :n - 1]

    scores = np.zeros(n)

    has_close = np.any(d < 0.05, axis=1)
    gaps = d[:, 1:] - d[:, :-1]
    k_idx = np.argmin(gaps, axis=1)
    D = (d[np.arange(n), k_idx] + d[np.arange(n), k_idx + 1]) / 2.0

    valid = (~has_close) & (D >= 0.1)

    n_minus_3 = n - 3
    selected = np.zeros((n, 4))

    mask_k0 = valid & (k_idx == 0)
    mask_k17 = valid & (k_idx == n_minus_3)
    mask_mid = valid & (k_idx > 0) & (k_idx < n_minus_3)

    for i in np.where(mask_k0)[0]:
        selected[i] = d[i, 0:4]
    for i in np.where(mask_k17)[0]:
        selected[i] = d[i, 15:19]
    for i in np.where(mask_mid)[0]:
        k = k_idx[i]
        selected[i] = d[i, k - 1:k + 3]

    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.maximum(selected / D[:, np.newaxis], D[:, np.newaxis] / selected)
    raw = np.mean(ratios, axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        score_vals = np.minimum(1.0, 1.0 / raw)
    scores = np.where(valid, score_vals, 0.0)
    return scores


def _strictly_convex(points):
    cross = _cross_products(points)
    if np.any(cross == 0):
        return False
    if np.any(cross > 0) and np.any(cross < 0):
        return False
    if np.min(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)) < 1e-9:
        return False
    return True


def _min_vertex_score(params, check_convexity=True):
    """
    params: array of length 2*N (radii + angle_deltas) or just radii (length N).
    Returns min vertex score after normalization. 0 if invalid.
    """
    n = _N
    if len(params) == 2 * n:
        radii = params[:n]
        angle_deltas = params[n:]
    else:
        radii = params
        angle_deltas = None

    radii = np.maximum(radii, 0.05)
    points = _normalize(_build_points(radii, angle_deltas))
    if points is None:
        return 0.0
    if check_convexity and not _strictly_convex(points):
        return 0.0
    return float(np.min(_compute_scores(points)))


def _score_and_penalty(params):
    """Geometric min score + smooth convexity penalty."""
    n = _N
    if len(params) == 2 * n:
        radii = params[:n]
        angle_deltas = params[n:]
    else:
        radii = params
        angle_deltas = None

    radii = np.maximum(radii, 0.05)
    points = _normalize(_build_points(radii, angle_deltas))
    if points is None:
        return 0.0, 100.0
    cross = _cross_products(points)
    pen = float(np.sum(np.maximum(0.0, -cross) ** 2))
    score = float(np.min(_compute_scores(points)))
    return score, pen


def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0

    def objective(params):
        score, pen = _score_and_penalty(params)
        return -score + 50.0 * pen

    # Probe per-evaluation cost
    probe_start = time.time()
    _min_vertex_score(np.ones(_N))
    eval_time = max(time.time() - probe_start, 1e-4)

    best_score = -1.0
    best_params = np.ones(_N)

    def try_optimize(x_init, method, maxiter, xatol, fatol, adaptive=True):
        nonlocal best_score, best_params
        try:
            opts = {'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol}
            if method == 'Nelder-Mead' and adaptive:
                opts['adaptive'] = True
            elif method == 'Powell':
                opts = {'maxiter': maxiter, 'xtol': xatol, 'ftol': fatol}
            res = minimize(objective, x_init, method=method, options=opts)
            cand = res.x
            sc = _min_vertex_score(cand)
            if sc > best_score:
                best_score = sc
                best_params = cand.copy()
        except Exception:
            pass

    a = _BASE_ANGLES
    rng_a = np.random.default_rng(12345)
    rng_b = np.random.default_rng(67890)
    rng_c = np.random.default_rng(24680)
    rng_d = np.random.default_rng(13579)

    # --- Pure radial initializations ---
    inits_radial = [
        np.ones(_N),
        np.maximum(1.0 + 0.15 * np.cos(2 * a), 0.05),
        np.maximum(1.0 + 0.15 * np.cos(4 * a), 0.05),
        np.maximum(1.0 + 0.15 * np.cos(5 * a), 0.05),
        np.maximum(1.0 + 0.10 * np.cos(3 * a), 0.05),
        np.maximum(1.0 + 0.10 * np.cos(6 * a), 0.05),
        np.maximum(1.0 + 0.10 * np.cos(7 * a), 0.05),
        np.maximum(1.0 + 0.25 * np.cos(4 * a), 0.05),
        np.maximum(1.0 + 0.25 * np.cos(5 * a), 0.05),
        np.maximum(1.0 + 0.08 * rng_a.standard_normal(_N), 0.05),
        np.maximum(1.0 + 0.12 * rng_b.standard_normal(_N), 0.05),
        np.maximum(1.0 + 0.08 * rng_c.standard_normal(_N), 0.05),
    ]
    alt = np.ones(_N)
    alt[::2] = 1.1
    alt[1::2] = 0.9
    inits_radial.append(alt)
    alt2 = np.ones(_N)
    alt2[1::2] = 1.1
    alt2[0::2] = 0.9
    inits_radial.append(alt2)

    # --- Joint (radii + angle_deltas) initializations ---
    inits_joint = []
    # Small random angular perturbations
    for seed_rng in [rng_d, np.random.default_rng(99999)]:
        small_angles = 0.03 * seed_rng.standard_normal(_N)
        small_angles -= np.mean(small_angles)  # center
        inits_joint.append(np.concatenate([np.ones(_N), small_angles]))
    # Fourier angular perturbation
    ang_fourier = 0.05 * np.cos(2 * a)
    ang_fourier -= np.mean(ang_fourier)
    inits_joint.append(np.concatenate([np.ones(_N), ang_fourier]))
    ang_fourier2 = 0.05 * np.cos(4 * a)
    ang_fourier2 -= np.mean(ang_fourier2)
    inits_joint.append(np.concatenate([np.ones(_N), ang_fourier2]))
    # Mixed: slight radial + angular
    inits_joint.append(np.concatenate([
        np.maximum(1.0 + 0.1 * np.cos(2 * a), 0.05),
        0.03 * np.cos(3 * a)
    ]))

    # --- Main search: radial first (faster, lower dim), then joint ---
    all_inits = inits_radial + inits_joint

    for x0 in all_inits:
        if time.time() > deadline - 20:
            break
        remaining = deadline - time.time()
        dim = len(x0)
        # Budget scales with dimension
        max_iter = min(5000, max(200, int(remaining * 0.35 / max(eval_time * dim, 1e-3))))
        try_optimize(x0, 'Nelder-Mead', max_iter, 1e-10, 1e-10)

    # --- Polish best candidate ---
    if time.time() < deadline - 25:
        try_optimize(best_params, 'Nelder-Mead', 3000, 1e-12, 1e-12)
    if time.time() < deadline - 20:
        try_optimize(best_params, 'Powell', 2000, 1e-11, 1e-11, adaptive=False)
    if time.time() < deadline - 15:
        try_optimize(best_params, 'Nelder-Mead', 2000, 1e-13, 1e-13)

    # --- Finalize ---
    final = _min_vertex_score(best_params)
    if final <= 0.0:
        best_params = np.ones(_N)

    if len(best_params) == 2 * _N:
        radii = best_params[:_N]
        angle_deltas = best_params[_N:]
    else:
        radii = best_params
        angle_deltas = None

    polygon = _build_points(radii, angle_deltas)
    normalized = _normalize(polygon)
    if normalized is None or not _strictly_convex(normalized):
        reg = _build_points(np.ones(_N))
        normalized = _normalize(reg)

    return normalized
