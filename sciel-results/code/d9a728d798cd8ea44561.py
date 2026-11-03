import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0

    for attempt in range(10):
        base_spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
        spacing = base_spacing * (0.95 + 0.02 * attempt)

        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                angle = rng.uniform(0, 2 * np.pi)
                x_rot = x * np.cos(angle) - y * np.sin(angle)
                y_rot = x * np.sin(angle) + y * np.cos(angle)
                r = np.hypot(x_rot, y_rot)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x_rot, y_rot, r))

        if len(rows) < n_points:
            continue

        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        order = np.argsort(radii, kind="stable")
        selected = pts[order[:n_points]]

        if attempt > 0:
            selected = selected + rng.normal(0, spacing * 0.05, selected.shape)
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0
            selected = selected[mask]
            if len(selected) < n_points:
                continue
            selected = selected[:n_points]

        d = cdist(selected, selected)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)

        if r > best_r:
            best_r = r
            best_pts = selected.copy()

    if best_pts is None:
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
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
    d = cdist(pts, pts)
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

def _objective_and_grad(x, T, edge_indices=None):
    """
    Objective: minimize -softmin(constraints)
    If edge_indices is provided, only consider those pairs for the pairwise term.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)

    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms

    if edge_indices is None:
        d = cdist(pts, pts)
        iu = np.triu_indices(n, k=1)
        pair_half = d[iu] / 2.0
    else:
        pair_half = np.array([np.linalg.norm(pts[i] - pts[j]) / 2.0 for i, j in edge_indices])

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

    grad_pts = np.zeros_like(pts)

    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_boundary = -pts / safe_norms[:, np.newaxis] * grad_all[len(pair_half):][:, np.newaxis]
    grad_pts += grad_boundary

    if edge_indices is None:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]

        W = np.zeros((n, n))
        n_pairs = len(pair_half)
        iu = np.triu_indices(n, k=1)
        W[iu] = grad_all[:n_pairs]
        W[iu[1], iu[0]] = grad_all[:n_pairs]

        weighted_unit = W[:, :, np.newaxis] * unit_vecs
        grad_pairs = 0.5 * np.sum(weighted_unit, axis=1)
        grad_pts += grad_pairs
    else:
        for k, (i, j) in enumerate(edge_indices):
            w = grad_all[k]
            if w == 0:
                continue
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-10:
                continue
            unit_vec = diff / d
            grad_pts[i] += 0.5 * w * unit_vec
            grad_pts[j] -= 0.5 * w * unit_vec

    return obj, grad_pts.ravel()

def _force_relax(pts, n_steps=2000, deadline=None):
    """Fast force-directed relaxation with Hertzian overlap^4 repulsion."""
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    n = len(pos)

    r_est = best_r
    lr = 0.005
    cooldown = 0.9995

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break

        forces = np.zeros_like(pos)

        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)

        target = 2.0 * r_est
        mask = dists < target
        safe_dists = np.where(mask, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]

        # Hertzian: force ~ (target - d)^4 / d^3 when d < target
        excess = np.where(mask, target - dists, 0.0)
        force_mag = np.where(mask, excess**4 / (safe_dists**3 + 1e-20), 0.0)
        force_mag = np.clip(force_mag, 0, 1.0)

        forces += np.sum(force_mag[:, :, np.newaxis] * unit_vecs, axis=1)

        norms = np.linalg.norm(pos, axis=1)
        wall_margin = 0.03
        for i in range(n):
            if norms[i] > 1.0 - wall_margin:
                excess_b = norms[i] - (1.0 - wall_margin)
                if excess_b > 0:
                    f_mag = 5.0 * excess_b**3
                    if norms[i] > 1e-10:
                        direction = -pos[i] / norms[i]
                    else:
                        direction = np.array([1.0, 0.0])
                    forces[i] += f_mag * direction

        pos = pos + lr * forces

        norms = np.linalg.norm(pos, axis=1)
        mask_out = norms > 1.0
        if np.any(mask_out):
            pos[mask_out] = pos[mask_out] / norms[mask_out, np.newaxis] * 0.999

        if step % 50 == 0:
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
                r_est = r

        lr *= cooldown

    return best_pos, best_r

def _get_active_edges(pts, r, max_edges=None):
    """Get edges from Delaunay triangulation, optionally filtered by distance."""
    try:
        tri = Delaunay(pts)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        edge_list = list(edges)
        if max_edges is not None and len(edge_list) > max_edges:
            # Keep only closest edges
            d = np.array([np.linalg.norm(pts[i] - pts[j]) for i, j in edge_list])
            order = np.argsort(d)
            edge_list = [edge_list[i] for i in order[:max_edges]]
        return edge_list
    except Exception:
        return []

def _local_polish(pts, r, deadline):
    """Iterative local polish with progressively tighter temperatures and active-set edges."""
    best_pts = pts.copy()
    best_r = r
    n = len(pts)

    # Build initial active set
    edge_list = _get_active_edges(best_pts, best_r, max_edges=800)
    if not edge_list:
        return best_pts, best_r

    # Multiple temperature passes
    temperatures = [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001, 5e-6, 2e-6, 1e-6]

    for T in temperatures:
        if time.time() > deadline - 3.0:
            break

        # Rebuild active edges periodically
        if len(temperatures) > 1:
            edge_list = _get_active_edges(best_pts, best_r, max_edges=800)
            if not edge_list:
                break

        x0 = best_pts.ravel()
        try:
            result = minimize(
                lambda x: _objective_and_grad(x, T, edge_list)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, T, edge_list)[1],
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
            continue

    return best_pts, best_r

def _gauss_newton_polish(pts, r, deadline):
    """
    Gauss-Newton method that enforces contact constraints on the Delaunay graph.
    For each edge, if the distance is close to 2r, we enforce d_ij = 2r.
    For boundary points, we enforce |p_i| = 1 - r.
    This treats the configuration as a constrained framework.
    """
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = r
    n = len(pos)

    edge_list = _get_active_edges(pos, r, max_edges=600)
    if not edge_list:
        return best_pos, best_r

    # Identify which edges are "active" (close to contact) and which points are on boundary
    target_dist = 2.0 * r
    tol = 0.005

    for iteration in range(200):
        if time.time() > deadline - 2.0:
            break

        # Build constraint system
        constraints = []  # (type, indices, residual)
        J_rows = []
        b_vals = []

        # Pairwise constraints: for edges where dist is close to target
        for (i, j) in edge_list:
            d = np.linalg.norm(pos[i] - pos[j])
            if abs(d - target_dist) < tol:
                diff = pos[i] - pos[j]
                if d < 1e-10:
                    continue
                unit = diff / d
                # Constraint: d_ij = target_dist
                # Gradient w.r.t. p_i: unit, w.r.t. p_j: -unit
                row = np.zeros(2 * n)
                row[2 * i] = unit[0]
                row[2 * i + 1] = unit[1]
                row[2 * j] = -unit[0]
                row[2 * j + 1] = -unit[1]
                J_rows.append(row)
                b_vals.append(target_dist - d)

        # Boundary constraints: for points near the boundary
        norms = np.linalg.norm(pos, axis=1)
        for i in range(n):
            if abs(norms[i] - (1.0 - r)) < tol:
                if norms[i] < 1e-10:
                    continue
                unit = -pos[i] / norms[i]  # gradient of (1 - |p_i|)
                row = np.zeros(2 * n)
                row[2 * i] = unit[0]
                row[2 * i + 1] = unit[1]
                J_rows.append(row)
                b_vals.append((1.0 - r) - norms[i])

        if not J_rows:
            break

        J = np.array(J_rows)
        b = np.array(b_vals)

        # Solve J @ delta = b (least squares if overdetermined)
        try:
            delta, residuals, rank, sv = np.linalg.lstsq(J, b, rcond=None)
        except Exception:
            break

        # Damped step
        step_size = 0.5
        for _ in range(20):
            new_pos = pos + step_size * delta.reshape(-1, 2)
            # Project back into unit disk
            new_norms = np.linalg.norm(new_pos, axis=1)
            mask_out = new_norms > 1.0
            if np.any(mask_out):
                new_pos[mask_out] = new_pos[mask_out] / new_norms[mask_out, np.newaxis] * 0.999

            new_r = _compute_radius(new_pos)
            if new_r >= best_r - 1e-8:
                pos = new_pos
                if new_r > best_r:
                    best_r = new_r
                    best_pos = pos.copy()
                break
            step_size *= 0.5

        else:
            # No improvement, try smaller step or break
            if step_size < 1e-6:
                break

        # Update r estimate
        r = best_r
        target_dist = 2.0 * r
        tol = max(0.001, 0.005 * (1 - iteration / 200.0))

        # Rebuild active edges
        edge_list = _get_active_edges(pos, r, max_edges=600)
        if not edge_list:
            break

    return best_pos, best_r

def _anneal_restart(pts, r, n_restarts=3, deadline=None):
    """Annealing restarts: perturb the configuration and re-optimize."""
    best_pts = pts.copy()
    best_r = r
    rng = np.random.default_rng(12345)

    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline - 30.0:
            break

        # Perturb
        noise_scale = 0.005 * (1 + 0.5 * restart)
        perturbed = best_pts + rng.normal(0, noise_scale, best_pts.shape)
        # Project back into disk
        norms = np.linalg.norm(perturbed, axis=1)
        mask_out = norms > 1.0
        if np.any(mask_out):
            perturbed[mask_out] = perturbed[mask_out] / norms[mask_out, np.newaxis] * 0.999

        r_pert = _compute_radius(perturbed)
        if r_pert < best_r * 0.95:
            continue

        # Quick force relax
        relaxed_pts, relaxed_r = _force_relax(perturbed, n_steps=500, deadline=deadline)
        if relaxed_r > r_pert:
            perturbed = relaxed_pts
            r_pert = relaxed_r

        # Local polish
        polished_pts, polished_r = _local_polish(perturbed, r_pert, deadline)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()

    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_pts = None
    best_r = 0.0

    # Phase 1: Multi-seed initialization and force relaxation
    for seed in [42, 123, 7, 999, 5555]:
        if time.time() > deadline - 50.0:
            break

        pts = _hex_init(N, seed=seed)
        r = _compute_radius(pts)

        relaxed_pts, relaxed_r = _force_relax(pts, n_steps=1500, deadline=deadline - 30.0)
        if relaxed_r > r:
            pts = relaxed_pts
            r = relaxed_r

        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)

    # Phase 2: Local polish with soft-min and active-set edges
    if time.time() < deadline - 40.0:
        polished_pts, polished_r = _local_polish(best_pts, best_r, deadline)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()

    # Phase 3: Gauss-Newton constraint enforcement
    if time.time() < deadline - 20.0:
        gn_pts, gn_r = _gauss_newton_polish(best_pts, best_r, deadline)
        if gn_r > best_r:
            best_r = gn_r
            best_pts = gn_pts.copy()

    # Phase 4: Annealing restarts
    if time.time() < deadline - 15.0:
        annealed_pts, annealed_r = _anneal_restart(best_pts, best_r, n_restarts=3, deadline=deadline)
        if annealed_r > best_r:
            best_r = annealed_r
            best_pts = annealed_pts.copy()

    # Phase 5: Final high-precision polish
    if time.time() < deadline - 5.0:
        # Final local polish with very tight temperatures
        edge_list = _get_active_edges(best_pts, best_r, max_edges=500)
        if edge_list:
            for T in [1e-7, 5e-8, 1e-8]:
                if time.time() > deadline - 2.0:
                    break
                try:
                    result = minimize(
                        lambda x: _objective_and_grad(x, T, edge_list)[0],
                        best_pts.ravel(),
                        jac=lambda x: _objective_and_grad(x, T, edge_list)[1],
                        method='L-BFGS-B',
                        options={
                            'maxiter': 500,
                            'ftol': 1e-15,
                            'gtol': 1e-13,
                            'maxls': 100,
                            'maxfun': 1000,
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

        # Final Gauss-Newton pass
        if time.time() < deadline - 1.0:
            gn_pts, gn_r = _gauss_newton_polish(best_pts, best_r, deadline)
            if gn_r > best_r:
                best_r = gn_r
                best_pts = gn_pts.copy()

    # Final validity check
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N, seed=0)

    return best_pts
