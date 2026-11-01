"""Packing 50 equal circles in a unit circle - dynamics + gradient polish + ring structure."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if pts is None or len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
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


def _ring_init(n, seed=42):
    """Generate initial positions matching the known optimal ring structure 1,6,12,18,13."""
    rng = np.random.default_rng(seed)
    rings = [1, 6, 12, 18, 13]
    while sum(rings) > n:
        rings[-1] -= 1
        if rings[-1] <= 0:
            rings.pop()
    while sum(rings) < n:
        rings.append(1)

    # Optimal ring radii approximately for r~0.1258
    # Ring 0: center
    # Ring 1: ~2*r
    # Ring 2: ~3.5*r
    # Ring 3: ~5*r
    # Ring 4: ~6.5*r (boundary)
    ring_radii = [0.0, 0.25, 0.44, 0.63, 0.82]

    points = []
    for ring_idx, count in enumerate(rings):
        if count == 0:
            continue
        if ring_idx == 0:
            points.append([0.0, 0.0])
        else:
            radius = ring_radii[min(ring_idx, len(ring_radii) - 1)]
            angle_offset = rng.uniform(0, 2 * np.pi)
            for k in range(count):
                angle = angle_offset + 2 * np.pi * k / count
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                # Small jitter to break symmetry
                r_jitter = rng.normal(0, 0.003)
                x += r_jitter * np.cos(angle)
                y += r_jitter * np.sin(angle)
                points.append([x, y])

    points = np.array(points[:n])
    norms = np.linalg.norm(points, axis=1)
    mask = norms > 0.95
    if np.any(mask):
        points[mask] *= (0.95 / norms[mask])[:, np.newaxis]
    return points


def _hex_lattice_init(n, jitter=0.0, seed=42):
    """Generate initial positions from hexagonal lattice, selecting closest to center."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_score = -1.0
    for spacing in np.linspace(0.14, 0.35, 30):
        k = int(1.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    rows.append((x, y))
        pts = np.array(rows)
        if len(pts) >= n:
            radii = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(radii)[:n]
            pts = pts[idx]
            if jitter > 0:
                pts = pts + rng.randn(n, 2) * jitter
            r = np.hypot(pts[:, 0], pts[:, 1])
            mask = r > 1.0
            if np.any(mask):
                pts[mask] /= r[mask, None] * 1.001
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
    if best_pts is None:
        theta = rng.uniform(0, 2 * np.pi, n)
        rho = np.sqrt(rng.uniform(0, 1, n))
        best_pts = np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])
    return best_pts


def _random_init(n, seed=42):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    rho = np.sqrt(rng.uniform(0, 0.9, n))
    return np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])


def _dynamics_simulate(pts, max_iters=4000, dt=0.002, damping=0.98):
    """Rigid-body dynamics: circles repel on contact, pulled inward by container."""
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    vel = np.zeros_like(pts)
    best_score = _score(pts)
    best_pts = pts.copy()

    dists = np.hypot(pts[:, 0], pts[:, 1])
    r_cont = 1.0 - np.max(dists)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    d_sq = dx * dx + dy * dy
    min_d = np.sqrt(np.min(d_sq))
    r_pair = min_d / 2.0
    r_target = min(r_cont, r_pair) * 1.01

    force_scale = 50.0
    wall_scale = 30.0

    for it in range(max_iters):
        if time.time() > _DEADLINE - 10.0:
            break

        force = np.zeros_like(pts)

        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            dx_ij = pts[i, 0] - pts[j, 0]
            dy_ij = pts[i, 1] - pts[j, 1]
            d = np.sqrt(dx_ij * dx_ij + dy_ij * dy_ij + 1e-15)
            if d < 2 * r_target:
                overlap = 2 * r_target - d
                f_mag = force_scale * overlap
                fx = dx_ij / d * f_mag
                fy = dy_ij / d * f_mag
                force[i] += [fx, fy]
                force[j] -= [fx, fy]

        for i in range(n):
            dist = np.sqrt(pts[i, 0] ** 2 + pts[i, 1] ** 2 + 1e-15)
            if dist + r_target > 1.0:
                excess = dist + r_target - 1.0
                f_mag = wall_scale * excess
                force[i, 0] -= pts[i, 0] / dist * f_mag
                force[i, 1] -= pts[i, 1] / dist * f_mag

        vel = damping * vel + dt * force
        pts = pts + dt * vel

        nr = np.hypot(pts[:, 0], pts[:, 1])
        mask = nr > 1.0
        if np.any(mask):
            pts[mask] /= nr[mask, None] * 1.0001

        current_score = _score(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()

        if it % 100 == 0 and it > 0:
            r_target *= 1.005

    return best_pts, best_score


def _soft_min_obj_and_grad(flat, beta):
    """Soft-min surrogate objective and gradient for maximizing the minimum radius."""
    pts = flat.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e6, np.zeros(2 * N)

    dists = np.hypot(pts[:, 0], pts[:, 1])
    r_cont = 1.0 - np.max(dists)

    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    d_sq = dx * dx + dy * dy
    d = np.sqrt(d_sq + 1e-12)
    half_d = d / 2.0

    all_radii = np.concatenate([[r_cont], half_d])

    x_max = np.max(all_radii)
    exp_terms = np.exp(-beta * (all_radii - x_max))
    soft_min_val = x_max - (1.0 / beta) * np.log(np.sum(exp_terms))

    weights = exp_terms / np.sum(exp_terms)

    grad = np.zeros(2 * N)

    max_idx = np.argmax(dists)
    if dists[max_idx] > 1e-10:
        grad[2 * max_idx] = -weights[0] * pts[max_idx, 0] / dists[max_idx]
        grad[2 * max_idx + 1] = -weights[0] * pts[max_idx, 1] / dists[max_idx]

    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        w = weights[k + 1]
        if d[k] > 1e-10:
            grad[2 * i] += w * (pts[i, 0] - pts[j, 0]) / (2.0 * d[k])
            grad[2 * i + 1] += w * (pts[i, 1] - pts[j, 1]) / (2.0 * d[k])
            grad[2 * j] += w * (pts[j, 0] - pts[i, 0]) / (2.0 * d[k])
            grad[2 * j + 1] += w * (pts[j, 1] - pts[i, 1]) / (2.0 * d[k])

    return -soft_min_val, -grad


def _lbfgs_optimize(pts, beta_start=50.0, beta_end=50000.0, n_phases=8):
    """Optimize using soft-min surrogate with L-BFGS-B, gradually increasing beta."""
    pts = pts.copy().astype(np.float64)
    best_score = _score(pts)
    best_pts = pts.copy()

    betas = np.geomspace(beta_start, beta_end, n_phases)

    for beta in betas:
        if time.time() > _DEADLINE - 8.0:
            break

        def objective_and_grad(flat):
            p = flat.reshape(N, 2)
            val, grad = _soft_min_obj_and_grad(p, beta)
            return val, grad

        flat = pts.flatten()

        try:
            result = minimize(
                objective_and_grad,
                flat,
                jac=True,
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'gtol': 1e-14,
                    'ftol': 1e-16,
                    'maxfun': 5000
                }
            )
            new_pts = result.x.reshape(N, 2)
        except Exception:
            continue

        r = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = r > 1.0
        if np.any(mask):
            new_pts[mask] /= r[mask, None] * 1.0001

        new_score = _score(new_pts)

        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts.copy()

    return best_pts, best_score


def _force_directed_refine(pts, r_current, max_iters=300):
    """Force-directed refinement: push overlapping pairs apart, pull boundary violators in."""
    pts = pts.copy().astype(np.float64)
    best_score = _score(pts)
    best_pts = pts.copy()
    best_r = r_current

    i_idx, j_idx = np.triu_indices(N, k=1)

    for iteration in range(max_iters):
        if time.time() > _DEADLINE - 8.0:
            break

        r_try = best_r * 1.0003 + 0.00003

        dx = pts[i_idx, 0] - pts[j_idx, 0]
        dy = pts[i_idx, 1] - pts[j_idx, 1]
        d_sq = dx * dx + dy * dy
        d = np.sqrt(d_sq + 1e-15)

        force = np.zeros_like(pts)

        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            if d[k] < 2 * r_try:
                overlap = 2 * r_try - d[k]
                if d[k] > 1e-10:
                    fx = (pts[i, 0] - pts[j, 0]) / d[k] * overlap * 0.5
                    fy = (pts[i, 1] - pts[j, 1]) / d[k] * overlap * 0.5
                    force[i] += [fx, fy]
                    force[j] -= [fx, fy]

        dists = np.hypot(pts[:, 0], pts[:, 1])
        for i in range(N):
            if dists[i] + r_try > 1.0:
                if dists[i] > 1e-10:
                    pull = (dists[i] + r_try - 1.0) * 0.5
                    force[i, 0] -= pts[i, 0] / dists[i] * pull
                    force[i, 1] -= pts[i, 1] / dists[i] * pull

        step = 0.05
        new_pts = pts + step * force

        nr = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = nr > 1.0
        if np.any(mask):
            new_pts[mask] /= nr[mask, None] * 1.0001

        new_score = _score(new_pts)

        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            best_r = new_score
            pts = new_pts.copy()
        else:
            step *= 0.5
            if step < 0.0005:
                break

    return best_pts, best_score


def _simulated_annealing(pts, max_iters=3000, T_init=0.06, T_min=1e-5, cooling=0.9997, seed=42):
    """Simulated annealing to escape local minima."""
    rng = np.random.default_rng(seed)
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    T = T_init
    best_score = _score(pts)
    best_pts = pts.copy()
    current_score = best_score

    for it in range(max_iters):
        if time.time() > _DEADLINE - 8.0:
            break

        scale = 0.04 * T / T_init + 0.001
        new_pts = pts + rng.randn(n, 2) * scale

        nr = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = nr > 1.0
        if np.any(mask):
            new_pts[mask] /= nr[mask, None] * 1.0001

        new_score = _score(new_pts)

        if new_score >= current_score:
            pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
        else:
            delta = current_score - new_score
            if delta > 0 and rng.random() < np.exp(-delta / max(T, 1e-10)):
                pts = new_pts
                current_score = new_score

        T = max(T * cooling, T_min)

    return best_pts, best_score


def construct():
    """Construct 50 circle centers in unit circle maximizing common radius."""
    best_pts = None
    best_score = 0.0

    # Multiple initialization strategies, prioritizing the known optimal ring structure
    inits = [
        _ring_init(N, seed=42),
        _ring_init(N, seed=123),
        _hex_lattice_init(N, jitter=0.0, seed=42),
        _hex_lattice_init(N, jitter=0.02, seed=7),
        _hex_lattice_init(N, jitter=0.01, seed=99),
        _random_init(N, seed=55),
        _ring_init(N, seed=777) + np.random.default_rng(1).randn(N, 2) * 0.02,
        _random_init(N, seed=200),
    ]

    for i, pts in enumerate(inits):
        if time.time() > _DEADLINE - 25.0:
            break

        # Project inside container
        r = np.hypot(pts[:, 0], pts[:, 1])
        mask = r > 1.0
        if np.any(mask):
            pts[mask] /= r[mask, None] * 1.001

        # Phase 1: Dynamics simulation to find a good jammed state
        dyn_pts, dyn_score = _dynamics_simulate(pts, max_iters=3000, dt=0.002, damping=0.98)
        if dyn_score > best_score:
            best_score = dyn_score
            best_pts = dyn_pts.copy()

        # Phase 2: L-BFGS-B polish on the dynamics result
        if time.time() < _DEADLINE - 15.0:
            polished_pts, polished_score = _lbfgs_optimize(
                dyn_pts, beta_start=50.0, beta_end=50000.0, n_phases=8
            )
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts.copy()

        # Phase 3: Force-directed refinement for fine-tuning
        if time.time() < _DEADLINE - 10.0:
            fr_pts, fr_score = _force_directed_refine(best_pts, best_score, max_iters=200)
            if fr_score > best_score:
                best_score = fr_score
                best_pts = fr_pts.copy()

        # Phase 4: Simulated annealing to escape local minima (less frequent)
        if time.time() < _DEADLINE - 20.0 and i % 4 == 0:
            sa_pts, sa_score = _simulated_annealing(
                best_pts, max_iters=2000, T_init=0.05, T_min=1e-5, cooling=0.9997, seed=i * 13
            )
            if sa_score > best_score:
                best_score = sa_score
                best_pts = sa_pts.copy()

            # Polish after SA
            if time.time() < _DEADLINE - 10.0:
                pol_pts, pol_score = _lbfgs_optimize(
                    best_pts, beta_start=2000.0, beta_end=50000.0, n_phases=4
                )
                if pol_score > best_score:
                    best_score = pol_score
                    best_pts = pol_pts.copy()

    if best_pts is None:
        best_pts = _ring_init(N, seed=42)
        best_score = _score(best_pts)

    # Final high-precision polish with very high beta
    if time.time() < _DEADLINE - 5.0:
        polished_pts, polished_score = _lbfgs_optimize(
            best_pts, beta_start=10000.0, beta_end=100000.0, n_phases=4
        )
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts

    # Final force-directed micro-refinement
    if time.time() < _DEADLINE - 3.0:
        fr_pts, fr_score = _force_directed_refine(best_pts, best_score, max_iters=100)
        if fr_score > best_score:
            best_score = fr_score
            best_pts = fr_pts

    # Final validation and cleanup
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        best_pts[mask] /= r[mask, None] * 1.0001

    return best_pts
