import numpy as np
import time
from scipy.optimize import minimize

N = 300


def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    for attempt in range(12):
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * (0.88 + 0.045 * attempt)
        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x, y, r))
        if len(rows) < n_points:
            continue
        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        order = np.argsort(radii, kind="stable")
        selected = pts[order[:n_points]]
        if attempt > 0:
            selected = selected + rng.normal(0, spacing * 0.04, selected.shape)
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0
            selected = selected[mask]
            if len(selected) < n_points:
                continue
            selected = selected[:n_points]
        d = np.linalg.norm(selected[:, np.newaxis, :] - selected[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)
        if r > best_r:
            best_r = r
            best_pts = selected.copy()
    if best_pts is None:
        spacing = 2.0 / (np.sqrt(n_points) + 1.0)
        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
        best_pts = pts[order[:n_points]]
    return best_pts


def _compute_radius(pts):
    """Compute the packing radius for given points."""
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    return min(min_pair, min_bound)


def _soft_min_logsumexp(values, T):
    """Stable soft minimum: -T * logsumexp(-values/T)"""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    neg_vals = -values / T
    max_neg = np.max(neg_vals)
    if max_neg > 700:
        return values.min()
    exp_shifted = np.exp(neg_vals - max_neg)
    log_sum = np.log(np.sum(exp_shifted)) + max_neg
    return -T * log_sum


def _objective_and_grad(x, T):
    """
    Objective: minimize -softmin(constraints) where constraints are:
      - pairwise half-distances d_ij/2 for i<j
      - boundary distances 1 - |c_i|
    """
    pts = x.reshape(-1, 2)
    n = len(pts)

    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms

    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    iu = np.triu_indices(n, k=1)
    pair_half = d[iu] / 2.0

    all_vals = np.concatenate([pair_half, boundary])

    obj = -_soft_min_logsumexp(all_vals, T)

    neg_vals_T = -all_vals / T
    max_nv = np.max(neg_vals_T)
    if max_nv > 700:
        min_idx = np.argmin(all_vals)
        grad_all = np.zeros_like(all_vals)
        grad_all[min_idx] = -1.0
    else:
        exp_shifted = np.exp(neg_vals_T - max_nv)
        weights = exp_shifted / np.sum(exp_shifted)
        grad_all = -weights

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)

    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]

    W = np.zeros((n, n))
    n_pairs = len(pair_half)
    W[iu] = grad_all[:n_pairs]
    W[iu[1], iu[0]] = grad_all[:n_pairs]

    weighted_unit = W[:, :, np.newaxis] * unit_vecs
    grad_pairs = 0.5 * np.sum(weighted_unit, axis=1)

    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_boundary = -pts / safe_norms[:, np.newaxis] * grad_all[n_pairs:][:, np.newaxis]

    grad_pts = grad_pairs + grad_boundary
    grad_obj = grad_pts.ravel()

    return obj, grad_obj


def _dynamics_simulate(pts, dt=0.001, n_steps=3000, damping=0.98, wall_strength=5.0,
                       pair_strength=1.0, min_dist=0.0, deadline=None):
    """
    Rigid-body dynamics simulation with repulsive forces.
    Uses Verlet-like integration with velocity damping.
    """
    n = len(pts)
    pos = pts.copy().astype(np.float64)
    vel = np.zeros_like(pos)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)

    # Precompute which pairs are "close" for force computation
    # Use a simple cutoff: only compute forces for pairs within some distance
    # For 300 particles, full pairwise is 44850 pairs - manageable

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break

        # Compute forces
        forces = np.zeros_like(pos)

        # Pairwise repulsive forces (soft repulsion to avoid singularity)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)

        # Only compute for pairs that are too close (within 2*r + margin)
        # Use a smooth repulsion: F = strength * (r_min - d)^2 / d * direction for d < r_min
        # r_min is the target minimum distance
        target_min_dist = 2.0 * (1.0 / np.sqrt(N)) * 0.95  # approximate target spacing
        mask = dists < target_min_dist * 1.5
        mask = mask & (dists < np.inf)

        if np.any(mask):
            # Compute forces only for masked pairs
            i_idx, j_idx = np.where(mask)
            for ii, jj in zip(i_idx, j_idx):
                d = dists[ii, jj]
                if d < 1e-10:
                    # Random small push for coincident points
                    forces[ii] += np.random.normal(0, 0.001, 2)
                    forces[jj] -= np.random.normal(0, 0.001, 2)
                    continue
                # Repulsive force: stronger when closer
                excess = target_min_dist - d
                if excess > 0:
                    f_mag = pair_strength * excess * excess / d
                    direction = diff[ii, jj] / d
                    forces[ii] += f_mag * direction
                    forces[jj] -= f_mag * direction

        # Wall repulsion: push particles inward if they're too close to boundary
        norms = np.linalg.norm(pos, axis=1)
        wall_margin = 0.02
        for i in range(n):
            if norms[i] > 1.0 - wall_margin:
                excess = norms[i] - (1.0 - wall_margin)
                if excess > 0:
                    f_mag = wall_strength * excess * excess
                    if norms[i] > 1e-10:
                        direction = -pos[i] / norms[i]
                    else:
                        direction = np.array([1.0, 0.0])
                    forces[i] += f_mag * direction

        # Update velocities with damping
        vel = vel * damping + forces * dt
        pos = pos + vel * dt

        # Keep positions inside the unit circle (hard constraint)
        norms = np.linalg.norm(pos, axis=1)
        for i in range(n):
            if norms[i] > 1.0:
                pos[i] = pos[i] / norms[i] * 0.999
                vel[i] = 0.0

        # Check if we improved
        if step % 100 == 0:
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()

    return best_pos, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Phase 1: Multiple hexagonal initializations with dynamics
    best_pts = None
    best_r = 0.0

    for seed in [42, 123, 7, 999, 555]:
        if time.time() > deadline - 30.0:
            break

        pts = _hex_init(N, seed=seed)
        r = _compute_radius(pts)

        # Run dynamics simulation
        dyn_pts, dyn_r = _dynamics_simulate(
            pts,
            dt=0.0005,
            n_steps=2000,
            damping=0.985,
            wall_strength=5.0,
            pair_strength=1.0,
            deadline=deadline - 25.0
        )

        if dyn_r > r:
            pts = dyn_pts
            r = dyn_r

        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Phase 2: Gradient-based refinement with soft-min
    if time.time() < deadline - 10.0 and best_pts is not None:
        temperatures = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
        x0 = best_pts.ravel()

        for T in temperatures:
            if time.time() > deadline - 8.0:
                break
            try:
                result = minimize(
                    lambda x: _objective_and_grad(x, T)[0],
                    x0,
                    jac=lambda x: _objective_and_grad(x, T)[1],
                    method='L-BFGS-B',
                    options={
                        'maxiter': 150,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxls': 50,
                        'maxfun': 400,
                    }
                )
                new_pts = result.x.reshape(-1, 2)
                norms = np.linalg.norm(new_pts, axis=1)
                if np.any(norms >= 1.0) or not np.all(np.isfinite(new_pts)):
                    continue
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
                    x0 = new_pts.ravel()
            except Exception:
                continue

    # Phase 3: Final high-precision polish
    if time.time() < deadline - 3.0:
        try:
            T_final = 0.00005
            result = minimize(
                lambda x: _objective_and_grad(x, T_final)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, T_final)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxls': 100,
                    'maxfun': 800,
                }
            )
            new_pts = result.x.reshape(-1, 2)
            norms = np.linalg.norm(new_pts, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(new_pts)):
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
        except Exception:
            pass

    # Ensure validity
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N, seed=0)

    return best_pts
