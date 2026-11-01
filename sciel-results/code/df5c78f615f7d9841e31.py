import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 142.0  # 8s safety margin before 150s kill


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


def _random_init():
    """Random initialization with bias toward uniform area coverage."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.95, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _ring_init():
    """Initialize circles in concentric rings for diverse starting point."""
    pts = np.zeros((N, 2))
    # Place some at center, rest in rings
    n_center = 1
    pts[0] = [0.0, 0.0]
    remaining = N - n_center
    ring_radii = np.linspace(0.2, 0.85, 4)
    per_ring = [remaining // 4] * 4
    per_ring[-1] += remaining - sum(per_ring)
    
    idx = n_center
    for r, count in zip(ring_radii, per_ring):
        if count <= 0:
            continue
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        pts[idx:idx+count, 0] = r * np.cos(angles)
        pts[idx:idx+count, 1] = r * np.sin(angles)
        idx += count
    return pts


def _perturb(pts, scale=0.01):
    """Add small random perturbation to escape local optima."""
    return pts + np.random.normal(0, scale, size=pts.shape)


def _optimize_from_start(pts, beta_schedule, max_iter_per_beta=300):
    """Run L-BFGS-B with a schedule of increasing beta values."""
    current_pts = pts.copy()
    best_pts = current_pts.copy()
    best_score = _score(current_pts)

    for beta in beta_schedule:
        if time.time() > _DEADLINE - 15.0:
            break

        try:
            res = minimize(
                _soft_min_obj_and_grad,
                current_pts.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': max_iter_per_beta, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )

            new_pts = res.x.reshape(N, 2)

            dists = np.linalg.norm(new_pts, axis=1)
            mask = dists > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]

            new_score = _score(new_pts)

            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                current_pts = new_pts.copy()

        except Exception:
            pass

    return best_pts, best_score


def _refine_with_gradient_descent(pts, r_target, steps=200, lr=0.005):
    """
    Vectorized gradient-based refinement: push circles apart and away from boundary.
    Used as a local polishing step.
    """
    pts = pts.copy()
    iu = np.triu_indices(N, k=1)
    i_idx, m_idx = iu

    for step in range(steps):
        if time.time() > _DEADLINE - 10.0:
            break

        dists_origin = np.linalg.norm(pts, axis=1)
        boundary_dists = 1.0 - dists_origin

        # Find critical pairs and boundary violations
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


def construct():
    start_time = time.time()

    # Multiple diverse initializations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_ring_init())
    for _ in range(6):
        inits.append(_random_init())

    best_pts = inits[0]
    best_score = _score(best_pts)

    # Beta schedules for annealing
    beta_schedule_1 = [10, 30, 100, 300, 1000]
    beta_schedule_2 = [50, 200, 1000, 3000, 10000]
    beta_schedule_3 = [20, 60, 200, 600, 2000]

    # Optimize from each initialization
    for init_idx, init_pts in enumerate(inits):
        if time.time() > _DEADLINE - 25.0:
            break

        # Use different beta schedules for different inits to diversify search
        if init_idx % 3 == 0:
            schedule = beta_schedule_1
        elif init_idx % 3 == 1:
            schedule = beta_schedule_2
        else:
            schedule = beta_schedule_3

        # Add perturbation for non-hex inits to escape local minima
        if init_idx > 0:
            perturbed = _perturb(init_pts, scale=0.02)
            candidate_pts, candidate_score = _optimize_from_start(perturbed, schedule, max_iter_per_beta=200)
        else:
            candidate_pts, candidate_score = _optimize_from_start(init_pts, schedule, max_iter_per_beta=400)

        if candidate_score > best_score:
            best_score = candidate_score
            best_pts = candidate_pts.copy()

    # Perturbation-retry loop: kick the best solution and re-optimize
    if time.time() < _DEADLINE - 20.0:
        for kick_iter in range(3):
            if time.time() > _DEADLINE - 20.0:
                break
            perturbed = _perturb(best_pts, scale=0.015)
            schedule = [200, 1000, 5000]
            candidate_pts, candidate_score = _optimize_from_start(perturbed, schedule, max_iter_per_beta=300)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts.copy()

    # Final high-beta polish on the best result
    if time.time() < _DEADLINE - 15.0:
        for beta in [5000, 20000, 50000, 100000]:
            if time.time() > _DEADLINE - 10.0:
                break
            try:
                res = minimize(
                    _soft_min_obj_and_grad,
                    best_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(-1.0, 1.0)] * (2 * N),
                    options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
                )
                new_pts = res.x.reshape(N, 2)
                dists = np.linalg.norm(new_pts, axis=1)
                mask = dists > 1.0
                if np.any(mask):
                    new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]

                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts

            except Exception:
                pass

    # Gradient-based refinement as a final polish
    if time.time() < _DEADLINE - 5.0:
        r_est = best_score
        refined = _refine_with_gradient_descent(best_pts, r_est, steps=150, lr=0.003)
        refined_score = _score(refined)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined

    # One more high-beta polish after refinement if time permits
    if time.time() < _DEADLINE - 3.0:
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                best_pts.flatten(),
                args=(100000,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
            )
            new_pts = res.x.reshape(N, 2)
            dists = np.linalg.norm(new_pts, axis=1)
            mask = dists > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
        except Exception:
            pass

    return best_pts
