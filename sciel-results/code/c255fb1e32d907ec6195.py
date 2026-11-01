"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 145.0


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


def _hex_lattice_init(n, jitter=0.0, seed=42):
    """Generate initial positions from hexagonal lattice, selecting closest to center."""
    rng = np.random.RandomState(seed)
    best_pts = None
    best_score = -1.0
    for spacing in np.linspace(0.14, 0.35, 40):
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


def _concentric_ring_init(n):
    """Generate initial positions on concentric rings."""
    pts = [[0.0, 0.0]]
    remaining = n - 1
    ring_radius = 0.2
    while remaining > 0:
        circumference = 2 * np.pi * ring_radius
        n_on_ring = max(1, int(circumference / 0.25))
        n_on_ring = min(n_on_ring, remaining)
        if n_on_ring == 0:
            ring_radius += 0.15
            continue
        angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False)
        for a in angles:
            pts.append([ring_radius * np.cos(a), ring_radius * np.sin(a)])
        remaining -= n_on_ring
        ring_radius += 0.18
    pts = np.array(pts[:n])
    r = np.hypot(pts[:, 0], pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        pts[mask] /= r[mask, None] * 1.001
    return pts


def _random_init(n, seed=42):
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    rho = np.sqrt(rng.uniform(0, 1, n))
    return np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])


def _dynamics_simulate(pts, max_iters=5000, dt=0.002, damping=0.98):
    """
    Rigid-body dynamics simulation: circles repel on contact and are pulled inward
    by container constraint. Uses velocity-Verlet-like integration with damping.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    vel = np.zeros_like(pts)
    best_score = _score(pts)
    best_pts = pts.copy()

    # Estimate target radius from current configuration
    dists = np.hypot(pts[:, 0], pts[:, 1])
    r_cont = 1.0 - np.max(dists)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    d_sq = dx * dx + dy * dy
    min_d = np.sqrt(np.min(d_sq))
    r_pair = min_d / 2.0
    r_target = min(r_cont, r_pair) * 1.01  # slightly larger to create repulsion

    force_scale = 50.0
    wall_scale = 30.0

    for it in range(max_iters):
        if time.time() > _DEADLINE - 10.0:
            break

        force = np.zeros_like(pts)

        # Pairwise repulsion for pairs closer than 2*r_target
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            dx_ij = pts[i, 0] - pts[j, 0]
            dy_ij = pts[i, 1] - pts[j, 1]
            d = np.sqrt(dx_ij * dx_ij + dy_ij * dy_ij + 1e-15)
            if d < 2 * r_target:
                overlap = 2 * r_target - d
                # Strong repulsion, scaled by overlap
                f_mag = force_scale * overlap
                fx = dx_ij / d * f_mag
                fy = dy_ij / d * f_mag
                force[i] += [fx, fy]
                force[j] -= [fx, fy]

        # Container wall: push inward if circle would extend beyond unit circle
        for i in range(n):
            dist = np.sqrt(pts[i, 0] ** 2 + pts[i, 1] ** 2 + 1e-15)
            if dist + r_target > 1.0:
                excess = dist + r_target - 1.0
                f_mag = wall_scale * excess
                force[i, 0] -= pts[i, 0] / dist * f_mag
                force[i, 1] -= pts[i, 1] / dist * f_mag

        # Velocity Verlet-like update with damping
        vel = damping * vel + dt * force
        pts = pts + dt * vel

        # Hard project inside container
        nr = np.hypot(pts[:, 0], pts[:, 1])
        mask = nr > 1.0
        if np.any(mask):
            pts[mask] /= nr[mask, None] * 1.0001

        # Score and track best
        current_score = _score(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()

        # Slowly increase r_target to pack tighter
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
                options={'maxiter': 2000, 'gtol': 1e-8}
            )
            if result.fun < 0:  # negative means improvement
                new_pts = result.x.reshape(N, 2)
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    pts = new_pts
        except Exception:
            pass

    return best_pts, best_score


def _nelder_mead_optimize(pts, beta_start=10.0, n_phases=4):
    """Optimize using Nelder-Mead with soft-min surrogate."""
    pts = pts.copy().astype(np.float64)
    best_score = _score(pts)
    best_pts = pts.copy()

    betas = np.geomspace(beta_start, beta_start * 100, n_phases)

    for beta in betas:
        if time.time() > _DEADLINE - 8.0:
            break

        def objective(flat):
            p = flat.reshape(N, 2)
            val, _ = _soft_min_obj_and_grad(p, beta)
            return val

        flat = pts.flatten()

        try:
            result = minimize(
                objective,
                flat,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            new_pts = result.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                pts = new_pts
        except Exception:
            pass

    return best_pts, best_score


def construct():
    """Construct the best packing found within the time budget."""
    deadline = _DEADLINE
    rng = np.random.RandomState(42)
    
    best_centers = None
    best_score = 0.0
    
    # Generate diverse initial configurations
    initializations = []
    
    # Hexagonal with various rotations and jitters
    for angle in [0, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0]:
        for jitter in [0.0, 0.01, 0.02]:
            centers = _hex_lattice_init(N, jitter=jitter, seed=42 + int(angle * 10) + int(jitter * 100))
            if len(centers) == N:
                c, s = np.cos(angle), np.sin(angle)
                rotated = np.column_stack([
                    centers[:, 0] * c - centers[:, 1] * s,
                    centers[:, 0] * s + centers[:, 1] * c
                ])
                initializations.append(rotated)
    
    # Ring-based initialization
    ring_centers = _concentric_ring_init(N)
    if len(ring_centers) == N:
        initializations.append(ring_centers)
    
    # Random initializations
    for seed in range(5):
        random_centers = _random_init(N, seed=seed)
        initializations.append(random_centers)
    
    # Optimize each initialization with multiple methods
    for i, init_centers in enumerate(initializations):
        if time.time() > deadline - 15:
            break
        
        remaining_time = deadline - time.time() - 2
        if remaining_time < 5:
            break
        
        # Try dynamics first (fast, good for rough packing)
        try:
            dyn_pts, dyn_score = _dynamics_simulate(init_centers, max_iters=2000)
            if dyn_score > best_score:
                best_score = dyn_score
                best_centers = dyn_pts
        except Exception:
            pass
        
        # Then L-BFGS refinement
        if time.time() < deadline - 10:
            try:
                lbfgs_pts, lbfgs_score = _lbfgs_optimize(
                    dyn_pts if dyn_score > best_score else init_centers,
                    beta_start=50.0, beta_end=50000.0, n_phases=5
                )
                if lbfgs_score > best_score:
                    best_score = lbfgs_score
                    best_centers = lbfgs_pts
            except Exception:
                pass
        
        # Finally Nelder-Mead polish
        if time.time() < deadline - 5:
            try:
                nm_pts, nm_score = _nelder_mead_optimize(
                    best_centers if best_centers is not None else init_centers,
                    beta_start=100.0, n_phases=3
                )
                if nm_score > best_score:
                    best_score = nm_score
                    best_centers = nm_pts
            except Exception:
                pass
    
    # Final global polish if time allows
    if best_centers is not None and time.time() < deadline - 3:
        try:
            final_pts, final_score = _lbfgs_optimize(
                best_centers,
                beta_start=1000.0, beta_end=100000.0, n_phases=4
            )
            if final_score > best_score:
                best_score = final_score
                best_centers = final_pts
        except Exception:
            pass
    
    if best_centers is None:
        # Fallback
        best_centers = _hex_lattice_init(N, jitter=0.0, seed=42)
    
    return best_centers
# EVOLVE-BLOCK-END
