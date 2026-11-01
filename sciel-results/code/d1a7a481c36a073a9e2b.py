"""Packing 50 equal circles in a unit circle - Delaunay-constrained optimization with soft-min."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay, distance_matrix

N = 50
_START = time.time()
_DEADLINE = _START + 135.0  # 15s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0 + 1e-9):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    return min(r_cont, r_pair)


def _soft_min_obj_and_grad(flat, beta):
    """
    Soft-min objective: minimize (1/beta)*ln(sum(exp(-beta*x_k)))
    where x_k are the critical distances (boundary and pairwise/2).
    """
    pts = flat.reshape(N, 2)

    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)

    iu = np.triu_indices(N, k=1)
    pair_dists = dist_mat[iu]

    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])

    # Log-sum-exp for stability
    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)

    obj_val = log_sum_exp / beta

    weights = np.exp(y_shifted) / sum_exp

    grad_pts = np.zeros_like(pts)

    # Boundary gradient
    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad

    # Pairwise gradient
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


def _delaunay_constraint_obj_and_grad(flat, target_edges, target_lengths, beta, lam):
    """
    Objective combining soft-min packing with Delaunay edge-length variance penalty.
    """
    pts = flat.reshape(N, 2)

    # Part 1: soft-min packing objective
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

    soft_min_val = log_sum_exp / beta
    weights = np.exp(y_shifted) / sum_exp

    grad_pts = np.zeros_like(pts)

    # Boundary gradient
    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad

    # Pairwise gradient
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

    # Part 2: Delaunay edge-length variance penalty
    if len(target_edges) > 0:
        edge_i = target_edges[:, 0]
        edge_j = target_edges[:, 1]
        edge_vecs = pts[edge_i] - pts[edge_j]
        edge_lens = np.linalg.norm(edge_vecs, axis=1)
        safe_lens = np.where(edge_lens < 1e-8, 1.0, edge_lens)

        # Variance of edge lengths relative to target
        residuals = edge_lens - target_lengths
        variance_penalty = 0.5 * np.sum(residuals ** 2)

        # Gradient of variance penalty
        edge_unit_vecs = edge_vecs / safe_lens[:, np.newaxis]
        edge_grad = (residuals / safe_lens)[:, np.newaxis] * edge_unit_vecs

        np.add.at(grad_pts, edge_i, edge_grad)
        np.add.at(grad_pts, edge_j, -edge_grad)

        total_obj = soft_min_val + lam * variance_penalty
        total_grad = grad_pts + lam * 0  # gradient already includes lam via edge_grad scaling
        # Actually need to scale the edge gradient by lam
        # Let me redo this properly
        total_grad = grad_pts.copy()
        # The edge gradient contribution needs to be scaled by lam
        # But we already added it above without lam... let me fix
        # Recompute: total_grad = grad_pts_from_softmin + lam * grad_from_variance
        # We already have grad_pts from softmin. The variance gradient is:
        # d/dpts (0.5 * sum((len - target)^2)) = sum((len - target)/len) * unit_vec * (1 for i, -1 for j)
        # So we need to add lam * that
        # But we already added it above without lam. Let me restructure.

    # Let me restructure the whole thing more carefully
    # Actually, let me just compute both parts separately and combine

    # Recompute cleanly
    grad_softmin = np.zeros_like(pts)

    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_softmin += boundary_grad

    w_pairs = weights[N:]
    i_idx, m_idx = iu
    d_pairs = dist_mat[i_idx, m_idx]
    safe_d_pairs = np.where(d_pairs < 1e-8, 1.0, d_pairs)
    vecs = pts[i_idx] - pts[m_idx]
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]

    grad_contrib_i = -0.5 * w_pairs[:, np.newaxis] * unit_vecs
    grad_contrib_m = 0.5 * w_pairs[:, np.newaxis] * unit_vecs

    np.add.at(grad_softmin, i_idx, grad_contrib_i)
    np.add.at(grad_softmin, m_idx, grad_contrib_m)

    total_obj = soft_min_val
    total_grad = grad_softmin.copy()

    if len(target_edges) > 0:
        edge_i = target_edges[:, 0]
        edge_j = target_edges[:, 1]
        edge_vecs = pts[edge_i] - pts[edge_j]
        edge_lens = np.linalg.norm(edge_vecs, axis=1)
        safe_lens = np.where(edge_lens < 1e-8, 1.0, edge_lens)

        residuals = edge_lens - target_lengths
        variance_penalty = 0.5 * np.sum(residuals ** 2)

        total_obj += lam * variance_penalty

        edge_unit_vecs = edge_vecs / safe_lens[:, np.newaxis]
        edge_grad_coeff = (residuals / safe_lens)[:, np.newaxis] * edge_unit_vecs

        np.add.at(total_grad, edge_i, lam * edge_grad_coeff)
        np.add.at(total_grad, edge_j, -lam * edge_grad_coeff)

    return total_obj, total_grad.flatten()


def _extract_delaunay_edges(pts):
    """Extract Delaunay edges from point cloud."""
    try:
        tri = Delaunay(pts)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        return np.array(sorted(edges), dtype=int)
    except Exception:
        return np.array([], dtype=int).reshape(0, 2)


def _hex_lattice_init(seed=None):
    """Generate initial positions from hexagonal lattice."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_score = -1.0

    for spacing in np.linspace(0.20, 0.32, 30):
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    rows.append((x, y))
        pts = np.array(rows)
        if len(pts) >= N:
            dists = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(dists, kind='stable')[:N]
            pts = pts[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()

    if best_pts is not None:
        return best_pts

    theta = rng.uniform(0, 2 * np.pi, N)
    rho = np.sqrt(rng.uniform(0, 0.95, N))
    return np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])


def _ring_init(seed=None):
    """Initialize with concentric rings matching known optimal structure."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_score = -1.0

    ring_configs = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 17, 14],
        [1, 6, 13, 17, 13],
        [1, 7, 12, 17, 13],
    ]

    for config in ring_configs:
        pts_list = [[0.0, 0.0]]
        for ring_idx, count in enumerate(config[1:], start=1):
            radius = 0.24 * ring_idx
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            if ring_idx % 2 == 0:
                angles += np.pi / count
            for a in angles:
                pts_list.append([radius * np.cos(a), radius * np.sin(a)])

        if len(pts_list) >= N:
            pts = np.array(pts_list[:N])
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()

    if best_pts is not None:
        return best_pts

    return _hex_lattice_init(seed)


def _random_init(seed=None):
    """Random initialization."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, N)
    rho = np.sqrt(rng.uniform(0, 0.95, N))
    return np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])


def _jitter(pts, scale, seed=None):
    """Add random jitter."""
    rng = np.random.default_rng(seed)
    return pts + rng.normal(0, scale, size=pts.shape)


def _optimize_softmin(pts, beta_schedule, max_iter_per_beta=200):
    """Run L-BFGS-B with soft-min objective and beta annealing."""
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
                options={'maxiter': max_iter_per_beta, 'ftol': 1e-14, 'gtol': 1e-9, 'disp': False}
            )

            new_pts = res.x.reshape(N, 2)

            # Project into unit disk
            dists = np.linalg.norm(new_pts, axis=1)
            mask = dists > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]

            new_score = _score(new_pts)

            if new_score > best_score + 1e-15:
                best_score = new_score
                best_pts = new_pts.copy()
                current_pts = new_pts.copy()

        except Exception:
            pass

    return best_pts, best_score


def _optimize_delaunay_constrained(pts, beta_schedule, max_iter_per_beta=200):
    """
    Optimize with Delaunay constraint: extract contact graph, then optimize
    with edge-length variance penalty to enforce the topology.
    """
    current_pts = pts.copy()
    best_pts = current_pts.copy()
    best_score = _score(current_pts)

    # First, do a quick soft-min optimization to get a reasonable starting point
    for beta in beta_schedule[:2]:
        if time.time() > _DEADLINE - 20.0:
            break
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                current_pts.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': max_iter_per_beta, 'ftol': 1e-14, 'gtol': 1e-9, 'disp': False}
            )
            new_pts = res.x.reshape(N, 2)
            dists = np.linalg.norm(new_pts, axis=1)
            mask = dists > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]
            new_score = _score(new_pts)
            if new_score > best_score + 1e-15:
                best_score = new_score
                best_pts = new_pts.copy()
                current_pts = new_pts.copy()
        except Exception:
            pass

    # Extract Delaunay edges from current best
    target_edges = _extract_delaunay_edges(best_pts)

    if len(target_edges) == 0:
        return best_pts, best_score

    # Compute target edge lengths
    edge_i = target_edges[:, 0]
    edge_j = target_edges[:, 1]
    edge_vecs = best_pts[edge_i] - best_pts[edge_j]
    target_lengths = np.linalg.norm(edge_vecs, axis=1)

    # Now optimize with Delaunay constraint
    lam_schedule = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

    for beta, lam in zip(beta_schedule[2:], lam_schedule):
        if time.time() > _DEADLINE - 15.0:
            break

        try:
            res = minimize(
                _delaunay_constraint_obj_and_grad,
                current_pts.flatten(),
                args=(target_edges, target_lengths, beta, lam),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': max_iter_per_beta, 'ftol': 1e-14, 'gtol': 1e-9, 'disp': False}
            )

            new_pts = res.x.reshape(N, 2)
            dists = np.linalg.norm(new_pts, axis=1)
            mask = dists > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]

            new_score = _score(new_pts)

            if new_score > best_score + 1e-15:
                best_score = new_score
                best_pts = new_pts.copy()
                current_pts = new_pts.copy()

        except Exception:
            pass

    return best_pts, best_score


def _force_refine(pts, steps=400, lr=0.003):
    """Force-directed refinement with 1/d^4 repulsion."""
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    iu = np.triu_indices(n, k=1)
    i_idx, j_idx = iu

    best_score = _score(pts)
    best_pts = pts.copy()

    for step in range(steps):
        if time.time() > _DEADLINE - 12.0:
            break

        dx = pts[i_idx, 0] - pts[j_idx, 0]
        dy = pts[i_idx, 1] - pts[j_idx, 1]
        d2 = dx * dx + dy * dy
        d = np.sqrt(d2)
        d = np.maximum(d, 1e-12)

        force_mag = 1.0 / (d2 * d2)
        fx = force_mag * dx / d
        fy = force_mag * dy / d

        Fx = np.zeros(n)
        Fy = np.zeros(n)
        np.add.at(Fx, i_idx, -fx)
        np.add.at(Fx, j_idx, fx)
        np.add.at(Fy, i_idx, -fy)
        np.add.at(Fy, j_idx, fy)

        Fx -= 0.003 * pts[:, 0]
        Fy -= 0.003 * pts[:, 1]

        fmag = np.hypot(Fx, Fy)
        fmag_max = np.max(fmag)
        if fmag_max > 1.0:
            Fx /= fmag_max
            Fy /= fmag_max

        new_pts = pts + lr * np.column_stack([Fx, Fy])

        nr = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = nr > 1.0
        if np.any(mask):
            new_pts[mask] /= nr[mask, None] * (1.0 + 1e-7)

        new_score = _score(new_pts)
        if new_score > best_score + 1e-15:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts.copy()

    return best_pts, best_score


def _local_refine(pts, max_iters=3000):
    """Local refinement using small random perturbations."""
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    best_score = _score(pts)
    best_pts = pts.copy()

    rng = np.random.default_rng(42)

    for it in range(max_iters):
        if time.time() > _DEADLINE - 10.0:
            break

        k = rng.integers(1, 6)
        idx = rng.choice(n, k, replace=False)
        perturbation = rng.normal(0, 0.002, size=(k, 2))
        new_pts = pts.copy()
        new_pts[idx] += perturbation

        nr = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = nr > 1.0
        if np.any(mask):
            new_pts[mask] /= nr[mask, None] * (1.0 + 1e-7)

        new_score = _score(new_pts)
        if new_score > best_score + 1e-15:
            pts = new_pts
            best_score = new_score
            best_pts = pts.copy()

    return best_pts, best_score


def construct():
    """Construct 50 circle centers in unit circle maximizing common radius."""
    best_pts = None
    best_score = 0.0

    beta_schedule = [30.0, 80.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]

    # Generate diverse initializations
    initializations = []

    for seed in range(6):
        initializations.append(_hex_lattice_init(seed=seed))

    for seed in range(6):
        initializations.append(_ring_init(seed=seed))

    for seed in range(4):
        base = _hex_lattice_init(seed=seed)
        initializations.append(_jitter(base, 0.02, seed=seed + 100))

    for seed in range(4):
        base = _ring_init(seed=seed)
        initializations.append(_jitter(base, 0.02, seed=seed + 200))

    for seed in range(4):
        initializations.append(_random_init(seed=seed))

    np.random.shuffle(initializations)

    for init_pts in initializations:
        if time.time() > _DEADLINE - 30.0:
            break

        # Phase 1: Standard soft-min optimization
        optimized_pts, score = _optimize_softmin(init_pts, beta_schedule, max_iter_per_beta=150)

        if score > best_score:
            best_score = score
            best_pts = optimized_pts.copy()

        # Phase 2: Delaunay-constrained optimization
        if time.time() < _DEADLINE - 25.0:
            del_pts, del_score = _optimize_delaunay_constrained(optimized_pts, beta_schedule, max_iter_per_beta=150)
            if del_score > best_score:
                best_score = del_score
                best_pts = del_pts.copy()

        # Phase 3: Force-directed refinement
        if time.time() < _DEADLINE - 18.0:
            force_pts, force_score = _force_refine(best_pts, steps=300, lr=0.002)
            if force_score > best_score:
                best_score = force_score
                best_pts = force_pts.copy()

        # Phase 4: Local refinement
        if time.time() < _DEADLINE - 12.0:
            refined_pts, refined_score = _local_refine(best_pts, max_iters=2000)
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined_pts.copy()

    if best_pts is None:
        best_pts = _hex_lattice_init(seed=0)
        best_score = _score(best_pts)

    # Final validation
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        best_pts[mask] /= r[mask, None] * (1.0 + 1e-7)

    return best_pts
