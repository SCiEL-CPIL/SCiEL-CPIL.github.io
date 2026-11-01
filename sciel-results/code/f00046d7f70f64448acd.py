import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the largest common radius for the given centers in [-0.5, 0.5]^3."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)

    # Pairwise distances using cKDTree for efficiency
    tree = cKDTree(centers)
    # Query nearest neighbor for each point (excluding self)
    dists, _ = tree.query(centers, k=2)
    min_pair_dist = np.min(dists[:, 1])

    return min(min_wall_dist, min_pair_dist / 2.0)


def _soft_min_obj_grad(x, beta, iu):
    """
    Soft-min objective and gradient for L-BFGS-B.
    Minimizes G(x) = (1/beta) * ln(sum_k exp(-beta * c_k(x)))
    where c_k are the constraint values (wall distances, pair distances/2).
    As beta -> inf, minimizing G maximizes min_k c_k.
    """
    c = x.reshape(-1, 3)

    # Wall constraints: c_w = 0.5 - |c_i|
    wall_vals = 0.5 - np.abs(c)
    wall_vals_flat = wall_vals.flatten()

    # Pair constraints: c_p = dist_ij / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    pair_vals = dists / 2.0
    pair_vals_flat = pair_vals[iu]

    all_vals = np.concatenate([wall_vals_flat, pair_vals_flat])

    # Compute weights w_k = exp(-beta * c_k) / S
    neg_beta_c = -beta * all_vals
    max_neg_beta_c = np.max(neg_beta_c)
    exp_vals = np.exp(neg_beta_c - max_neg_beta_c)
    S = np.sum(exp_vals)
    weights = exp_vals / S

    # Objective value
    obj = (max_neg_beta_c + np.log(S)) / beta

    # Gradient computation
    grad = np.zeros_like(c)

    # Wall gradient: dG/dc[i,k] = w_{i,k} * sign(c[i,k])
    wall_weights = weights[:3 * N].reshape(-1, 3)
    grad += wall_weights * np.sign(c)

    # Pair gradient: dG/dc_i = - sum_j w_{ij} * (c_i - c_j) / dist_ij
    pair_weights = weights[3 * N:]

    mask = dists > 1e-12
    inv_dists = np.zeros_like(dists)
    inv_dists[mask] = 1.0 / dists[mask]
    unit_diff = diff * inv_dists[:, :, np.newaxis]

    w_mat = np.zeros((N, N))
    w_mat[iu] = pair_weights
    w_mat[iu[1], iu[0]] = pair_weights

    grad_pairs = -np.sum(w_mat[:, :, np.newaxis] * unit_diff, axis=1)
    grad += grad_pairs

    return obj, grad.ravel()


def _optimize_soft_min(x0, deadline, beta_start=50.0, beta_end=20000.0, n_steps=8):
    """
    Run L-BFGS-B optimization with increasing beta.
    """
    best_x = x0.copy()
    best_r = _compute_radius(x0.reshape(-1, 3))
    iu = np.triu_indices(N, k=1)
    bounds = [(-0.5, 0.5)] * (N * 3)

    betas = np.geomspace(beta_start, beta_end, n_steps)

    for beta in betas:
        if time.time() > deadline:
            break

        try:
            res = minimize(
                _soft_min_obj_grad,
                best_x,
                args=(beta, iu),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )

            new_x = np.clip(res.x, -0.5, 0.5)
            new_r = _compute_radius(new_x.reshape(-1, 3))

            if new_r > best_r:
                best_r = new_r
                best_x = new_x
        except Exception:
            pass

    return best_x, best_r


def _force_directed_refinement(centers, deadline, n_iters=200):
    """
    Force-directed method: each sphere repels others and is attracted to center.
    Uses cKDTree for efficient neighbor queries.
    """
    c = centers.copy()
    r_current = _compute_radius(c)
    best_c = c.copy()
    best_r = r_current

    for it in range(n_iters):
        if time.time() > deadline:
            break

        # Compute forces using cKDTree for efficiency
        forces = np.zeros_like(c)

        tree = cKDTree(c)
        # Query neighbors within a certain radius
        neighbor_radius = 3.0 * r_current
        pairs = tree.query_pairs(neighbor_radius)

        for i, j in pairs:
            if i > j:
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-12:
                    continue
                # Strong repulsion when close
                force_mag = 1.0 / (dist ** 2 + 1e-8)
                unit_diff = diff / dist
                forces[i] += force_mag * unit_diff
                forces[j] -= force_mag * unit_diff

        # Attraction to center (weak)
        forces -= 0.001 * c

        # Wall repulsion
        wall_dists = 0.5 - np.abs(c)
        wall_mask = wall_dists < r_current
        if np.any(wall_mask):
            wall_force = np.zeros_like(c)
            wall_force[wall_mask] = 0.5 / (wall_dists[wall_mask] + 1e-8)
            forces -= wall_force * np.sign(c)

        # Adaptive step size
        max_force = np.max(np.abs(forces))
        if max_force > 1e-12:
            step = 0.01 / max_force
        else:
            step = 0.01

        c += step * forces
        c = np.clip(c, -0.499, 0.499)

        # Check if improved
        r_new = _compute_radius(c)
        if r_new > best_r:
            best_r = r_new
            best_c = c.copy()

    return best_c, best_r


def _simulated_annealing(x0, deadline, beta=500.0, iu=None):
    """
    Simulated annealing using the soft-min objective.
    Escapes local minima by accepting worse solutions with probability exp(-delta/T).
    """
    if iu is None:
        iu = np.triu_indices(N, k=1)

    c = x0.reshape(-1, 3).copy()
    best_c = c.copy()
    best_r = _compute_radius(c)

    # Compute initial objective
    try:
        best_obj, _ = _soft_min_obj_grad(c.flatten(), beta, iu)
    except Exception:
        best_obj = _soft_min_obj_grad(c.flatten(), beta, iu)[0]

    # Temperature schedule
    T_start = 0.01
    T_end = 0.0001
    n_outer = 10
    n_inner = 50

    rng = np.random.default_rng(42)

    for outer in range(n_outer):
        if time.time() > deadline:
            break

        T = T_start * (T_end / T_start) ** (outer / float(n_outer))

        for it in range(n_inner):
            if time.time() > deadline:
                break

            # Generate perturbation
            perturbation_scale = 0.01 * (1.0 - outer / float(n_outer)) + 0.001
            perturbation = rng.normal(0, perturbation_scale, c.shape)

            c_new = np.clip(c + perturbation, -0.5, 0.5)

            # Compute new objective
            try:
                obj_new, _ = _soft_min_obj_grad(c_new.flatten(), beta, iu)
            except Exception:
                continue

            delta_obj = obj_new - best_obj

            # Accept if better, or with probability exp(-delta/T)
            if delta_obj < 0 or rng.random() < np.exp(-delta_obj / max(T, 1e-10)):
                c = c_new
                best_obj = obj_new

                r_new = _compute_radius(c)
                if r_new > best_r:
                    best_r = r_new
                    best_c = c.copy()

    return best_c.flatten(), best_r


def _init_grid(nx, ny, nz, rng, noise_scale=0.008):
    """Create a perturbed grid initialization."""
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    centers = centers[:N]
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def _init_fcc_like(rng):
    """FCC-like initialization."""
    fcc_offsets = [
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]
    ]
    centers = []
    for ix in range(5):
        for iy in range(5):
            for iz in range(4):
                for off in fcc_offsets:
                    cx = -0.5 + 0.1 + ix * 0.1 + off[0] * 0.1
                    cy = -0.5 + 0.1 + iy * 0.1 + off[1] * 0.1
                    cz = -0.5 + 0.1 + iz * 0.1 + off[2] * 0.1
                    if -0.499 <= cx <= 0.499 and -0.499 <= cy <= 0.499 and -0.499 <= cz <= 0.499:
                        centers.append([cx, cy, cz])
    centers = np.array(centers[:N])
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def _init_bcc_like(rng):
    """BCC-like initialization."""
    bcc_offsets = [
        [0, 0, 0], [0.5, 0.5, 0.5]
    ]
    centers = []
    for ix in range(5):
        for iy in range(5):
            for iz in range(5):
                for off in bcc_offsets:
                    cx = -0.5 + 0.05 + ix * 0.1 + off[0] * 0.1
                    cy = -0.5 + 0.05 + iy * 0.1 + off[1] * 0.1
                    cz = -0.5 + 0.05 + iz * 0.1 + off[2] * 0.1
                    if -0.499 <= cx <= 0.499 and -0.499 <= cy <= 0.499 and -0.499 <= cz <= 0.499:
                        centers.append([cx, cy, cz])
    centers = np.array(centers[:N])
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def construct():
    start_time = time.time()
    # Safety margin: 140s is well within 150s.
    deadline = start_time + 140.0

    best_centers = None
    best_radius = -np.inf

    rng = np.random.default_rng(42)

    # Multiple initialization strategies
    inits = [
        ("grid_554", _init_grid(5, 5, 4, rng, 0.008)),
        ("grid_545", _init_grid(5, 4, 5, rng, 0.008)),
        ("grid_455", _init_grid(4, 5, 5, rng, 0.008)),
        ("fcc", _init_fcc_like(rng)),
        ("bcc", _init_bcc_like(rng)),
    ]

    # Time allocation per initialization
    n_inits = len(inits)
    time_per_init = (deadline - start_time - 10.0) / n_inits

    for init_name, centers in inits:
        if time.time() > deadline - 5.0:
            break

        r_init = _compute_radius(centers)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()

        # Phase 1: Soft-min optimization
        slice_deadline = min(time.time() + time_per_init * 0.4, deadline - 5.0)
        opt_centers, opt_radius = _optimize_soft_min(
            centers, slice_deadline,
            beta_start=50.0, beta_end=5000.0, n_steps=6
        )
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers.copy()

        # Phase 2: Force-directed refinement
        if time.time() < deadline - 5.0:
            slice_deadline = min(time.time() + time_per_init * 0.2, deadline - 5.0)
            fd_centers, fd_radius = _force_directed_refinement(
                best_centers, slice_deadline, n_iters=100
            )
            if fd_radius > best_radius:
                best_radius = fd_radius
                best_centers = fd_centers.copy()

        # Phase 3: Simulated annealing for escape
        if time.time() < deadline - 5.0:
            slice_deadline = min(time.time() + time_per_init * 0.2, deadline - 5.0)
            sa_centers, sa_radius = _simulated_annealing(
                best_centers, slice_deadline, beta=500.0
            )
            if sa_radius > best_radius:
                best_radius = sa_radius
                best_centers = sa_centers.reshape(-1, 3).copy()

    # Final polish on best found
    if best_centers is not None and time.time() < deadline - 2.0:
        c_final, r_final = _optimize_soft_min(
            best_centers, deadline - 0.5,
            beta_start=5000.0, beta_end=20000.0, n_steps=3
        )
        if r_final > best_radius:
            best_radius = r_final
            best_centers = c_final.copy()

    if best_centers is None:
        best_centers = _init_grid(5, 5, 4, rng, 0.01)

    return best_centers

# EVOLVE-BLOCK-END
