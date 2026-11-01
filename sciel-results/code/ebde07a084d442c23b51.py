import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 138.0  # 12s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0

    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin > 1.0 + 1e-9):
        return 0.0

    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0

    if N < 2:
        return r_cont

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)

    return min(r_cont, r_pair)


def _soft_min_obj_and_grad(flat, beta):
    """
    Compute the soft-min objective and gradient for L-BFGS-B.
    Objective: minimize f(x) = (1/beta) * ln(sum(exp(-beta * x_k)))
    where x_k are the critical distances (pairwise and boundary).
    """
    pts = flat.reshape(N, 2)

    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)

    iu = np.triu_indices(N, k=1)
    pair_dists = dist_mat[iu]

    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])

    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)

    obj_val = log_sum_exp / beta

    weights = np.exp(y_shifted) / sum_exp

    grad_pts = np.zeros_like(pts)

    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad

    w_pairs = weights[N:]
    i_idx, m_idx = iu
    d_pairs = dist_mat[i_idx, m_idx]
    safe_d_pairs = np.where(d_pairs < 1e-8, 1.0, d_pairs)
    vecs = pts[i_idx] - pts[m_idx]
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]

    grad_contrib_i = -0.5 * w_pairs[:, np.newaxis] * unit_vecs
    grad_contrib_m = 0.5 * w_pairs[:, np.newaxis] * unit_vecs

    np.add.at(grad_pts, i_idx, grad_contrib_i)
    np.add.at(grad_pts, m_idx, grad_contrib_m)

    grad_flat = grad_pts.flatten()

    return obj_val, grad_flat


def _force_directed_relax(pts, r_target, steps=200, lr=0.005):
    """
    Force-directed relaxation: push circles apart and away from boundary.
    Used as a local polishing step before L-BFGS-B.
    """
    pts = pts.copy()
    iu = np.triu_indices(N, k=1)
    i_idx, m_idx = iu

    for step in range(steps):
        if time.time() > _DEADLINE - 10.0:
            break

        dists_origin = np.linalg.norm(pts, axis=1)
        boundary_dists = 1.0 - dists_origin

        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist_mat, np.inf)

        force = np.zeros_like(pts)

        d_pairs = dist_mat[iu]

        # Vectorized pairwise forces
        mask_pair = (d_pairs < 2 * r_target) & (d_pairs > 1e-10)
        if np.any(mask_pair):
            ii = i_idx[mask_pair]
            mm = m_idx[mask_pair]
            d = d_pairs[mask_pair]
            vecs = pts[ii] - pts[mm]
            units = vecs / d[:, np.newaxis]
            strengths = (2 * r_target - d) * 0.5
            np.add.at(force, ii, units * strengths[:, np.newaxis])
            np.add.at(force, mm, -units * strengths[:, np.newaxis])

        # Vectorized boundary forces
        mask_bnd = boundary_dists < r_target
        if np.any(mask_bnd):
            bi = np.where(mask_bnd)[0]
            safe_dists = np.where(dists_origin[bi] < 1e-8, 1.0, dists_origin[bi])
            unit_out = pts[bi] / safe_dists[:, np.newaxis]
            strengths = r_target - boundary_dists[bi]
            force[bi] -= unit_out * strengths[:, np.newaxis]

        pts += lr * force

        # Project back into unit disk
        dists = np.linalg.norm(pts, axis=1)
        mask = dists > 1.0 - r_target
        if np.any(mask):
            pts[mask] = pts[mask] * (1.0 - r_target) / dists[mask, np.newaxis]

    return pts


def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess."""
    best_pts = None
    best_score = -1.0

    for spacing in [0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29]:
        k = int(1.0 / spacing) + 4
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])

        mask = radius <= 1.0
        points = points[mask]
        radius = radius[mask]

        if len(points) >= N:
            idx = np.argsort(radius, kind="stable")[:N]
            pts = points[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts

    if best_pts is not None:
        return best_pts

    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.9, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _ring_init():
    """
    Initialize circles in concentric rings matching the known optimal structure.
    Known best packing uses rings: 1 (center) + 6 + 12 + 18 + 13 = 50
    """
    pts = np.zeros((N, 2))
    
    # Ring 0: center
    pts[0] = [0.0, 0.0]
    
    # Ring 1: 6 circles at radius ~0.25
    r1 = 0.25
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    pts[1:7, 0] = r1 * np.cos(angles1)
    pts[1:7, 1] = r1 * np.sin(angles1)
    
    # Ring 2: 12 circles at radius ~0.5
    r2 = 0.50
    angles2 = np.linspace(0, 2 * np.pi, 12, endpoint=False) + np.pi/12
    pts[7:19, 0] = r2 * np.cos(angles2)
    pts[7:19, 1] = r2 * np.sin(angles2)
    
    # Ring 3: 18 circles at radius ~0.72
    r3 = 0.72
    angles3 = np.linspace(0, 2 * np.pi, 18, endpoint=False)
    pts[19:37, 0] = r3 * np.cos(angles3)
    pts[19:37, 1] = r3 * np.sin(angles3)
    
    # Ring 4: 13 circles at radius ~0.87
    r4 = 0.87
    angles4 = np.linspace(0, 2 * np.pi, 13, endpoint=False) + np.pi/26
    pts[37:50, 0] = r4 * np.cos(angles4)
    pts[37:50, 1] = r4 * np.sin(angles4)
    
    return pts


def _random_init():
    """Random initialization with bias toward uniform area coverage."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.95, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _perturb(pts, scale=0.01):
    """Add small random perturbation to escape local optima."""
    return pts + np.random.normal(0, scale, size=pts.shape)


def _project_to_disk(pts):
    """Project points back into the unit disk."""
    dists = np.linalg.norm(pts, axis=1)
    mask = dists > 1.0
    if np.any(mask):
        pts = pts.copy()
        pts[mask] = pts[mask] / dists[mask, np.newaxis]
    return pts


def _soft_min_optimize(pts, beta_schedule, max_iter_per_beta=300):
    """Run L-BFGS-B with a schedule of increasing beta values."""
    current_pts = pts.copy()
    best_pts = current_pts.copy()
    best_score = _score(current_pts)

    for beta in beta_schedule:
        if time.time() > _DEADLINE - 15.0:
            break

        try:
            # Tighter tolerances for higher beta values
            if beta >= 1000:
                ftol = 1e-13
                gtol = 1e-9
                max_iter = max_iter_per_beta
            else:
                ftol = 1e-12
                gtol = 1e-8
                max_iter = max_iter_per_beta

            res = minimize(
                _soft_min_obj_and_grad,
                current_pts.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
            )

            new_pts = res.x.reshape(N, 2)
            new_pts = _project_to_disk(new_pts)

            new_score = _score(new_pts)

            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                current_pts = new_pts.copy()

        except Exception:
            pass

    return best_pts, best_score


def _binary_search_radius(pts, r_low, r_high, tol=1e-10, max_iter=30):
    """
    Binary search on radius: for a given r, try to find a valid packing.
    Returns the maximum r found.
    """
    best_r = r_low
    best_pts = pts.copy()

    for _ in range(max_iter):
        if time.time() > _DEADLINE - 10.0:
            break
        if r_high - r_low < tol:
            break

        r_mid = (r_low + r_high) / 2.0

        # Force-directed relaxation to try to achieve radius r_mid
        relaxed = _force_directed_relax(pts, r_mid, steps=100, lr=0.003)
        relaxed_score = _score(relaxed)

        if relaxed_score >= r_mid - 1e-8:
            # Achieved at least r_mid, try larger
            best_r = r_mid
            best_pts = relaxed.copy()
            r_low = r_mid
        else:
            r_high = r_mid

    return best_r, best_pts


def _full_optimize(pts):
    """
    Full optimization pipeline:
    1. Force-directed relaxation
    2. Soft-min L-BFGS-B with beta annealing
    3. Binary search radius refinement
    4. Final high-precision polish
    """
    # Step 1: Force-directed relaxation
    initial_score = _score(pts)
    relaxed = _force_directed_relax(pts, initial_score, steps=150, lr=0.005)
    relaxed_score = _score(relaxed)

    if relaxed_score > initial_score:
        pts = relaxed
        initial_score = relaxed_score

    # Step 2: Soft-min L-BFGS-B with beta annealing
    beta_schedule = [20, 100, 500, 2000, 10000, 50000]
    pts, score = _soft_min_optimize(pts, beta_schedule, max_iter_per_beta=400)

    # Step 3: Binary search radius refinement
    r_low = score
    r_high = score * 1.05
    r_best, pts_best = _binary_search_radius(pts, r_low, r_high, tol=1e-10, max_iter=25)

    if r_best > score:
        score = r_best
        pts = pts_best

    # Step 4: Final high-precision polish
    if time.time() < _DEADLINE - 8.0:
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                pts.flatten(),
                args=(200000,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False}
            )
            new_pts = res.x.reshape(N, 2)
            new_pts = _project_to_disk(new_pts)
            new_score = _score(new_pts)
            if new_score > score:
                score = new_score
                pts = new_pts
        except Exception:
            pass

    return pts, score


def construct():
    start_time = time.time()

    # Multiple diverse initializations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_ring_init())
    for _ in range(5):
        inits.append(_random_init())

    best_pts = inits[0]
    best_score = _score(best_pts)

    # Sort inits by initial score to try best ones first
    init_scores = [(i, _score(init_pts)) for i, init_pts in enumerate(inits)]
    init_scores.sort(key=lambda x: x[1], reverse=True)

    for init_idx, (orig_idx, init_score) in enumerate(init_scores):
        if time.time() > _DEADLINE - 25.0:
            break

        init_pts = inits[orig_idx]

        # Add perturbation for non-top inits to escape local minima
        if orig_idx > 0:
            perturbed = _perturb(init_pts, scale=0.012)
        else:
            perturbed = init_pts

        candidate_pts, candidate_score = _full_optimize(perturbed)

        if candidate_score > best_score:
            best_score = candidate_score
            best_pts = candidate_pts.copy()

    # Perturbation-retry loop: kick the best solution and re-optimize
    if time.time() < _DEADLINE - 20.0:
        for kick_iter in range(6):
            if time.time() > _DEADLINE - 20.0:
                break
            # Use different perturbation scales and patterns for diversity
            if kick_iter % 3 == 0:
                perturbed = _perturb(best_pts, scale=0.012)
            elif kick_iter % 3 == 1:
                # Asymmetric perturbation: larger kicks for outer circles
                perturbed = best_pts.copy()
                dists = np.linalg.norm(perturbed, axis=1)
                scales = 0.006 + 0.02 * dists
                perturbed += np.random.normal(0, 1, size=perturbed.shape) * scales[:, np.newaxis]
            else:
                # Directional perturbation
                perturbed = best_pts.copy()
                angle = np.random.uniform(0, 2 * np.pi)
                direction = np.array([np.cos(angle), np.sin(angle)])
                perturbed += np.random.normal(0, 0.008, size=perturbed.shape) + 0.004 * direction[np.newaxis, :]

            candidate_pts, candidate_score = _full_optimize(perturbed)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts.copy()

    # Final high-beta polish on the best result
    if time.time() < _DEADLINE - 8.0:
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                best_pts.flatten(),
                args=(500000,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-13, 'disp': False}
            )
            new_pts = res.x.reshape(N, 2)
            new_pts = _project_to_disk(new_pts)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        except Exception:
            pass

    # Gradient-based refinement as a final polish
    if time.time() < _DEADLINE - 5.0:
        r_est = best_score
        for lr_try in [0.003, 0.002, 0.001]:
            if time.time() > _DEADLINE - 3.0:
                break
            refined = _force_directed_relax(best_pts, r_est, steps=80, lr=lr_try)
            refined_score = _score(refined)
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined.copy()
            else:
                break

    return best_pts
