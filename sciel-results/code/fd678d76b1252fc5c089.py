"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _exact_radius(C):
    """Compute the exact packing radius."""
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    if n < 2:
        return np.inf
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    min_bnd = np.min(0.5 - np.abs(C))
    return min(min_pair, min_bnd)


def _soft_min_radius(C, beta=50.0):
    """
    Smooth surrogate for the packing radius using log-sum-exp.
    Returns -soft_min so that minimizing it maximizes the radius.
    """
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0
    boundary_terms = 0.5 - np.abs(C)
    all_terms = np.concatenate([pair_terms[np.triu_indices(n, k=1)],
                                boundary_terms.flatten()])
    min_val = np.min(all_terms)
    shifted = -beta * (all_terms - min_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = min_val - (1.0 / beta) * log_sum
    return -soft_min_val


def _grad_soft_min_radius(C, beta=50.0):
    """Gradient of the soft-min radius surrogate (negated for minimization)."""
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-30)
    np.fill_diagonal(dists, 1.0)
    pair_terms = dists / 2.0
    boundary_terms = 0.5 - np.abs(C)
    all_terms = np.concatenate([pair_terms[np.triu_indices(n, k=1)],
                                boundary_terms.flatten()])
    min_val = np.min(all_terms)
    shifted = -beta * (all_terms - min_val)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    grad_all = weights
    grad_C = np.zeros_like(C)
    n_pairs = n * (n - 1) // 2
    pair_grad = grad_all[:n_pairs]
    triu_idx = np.triu_indices(n, k=1)
    for idx in range(n_pairs):
        i, j = triu_idx[0][idx], triu_idx[1][idx]
        d = C[i] - C[j]
        dist = np.sqrt(np.sum(d ** 2) + 1e-30)
        if dist < 1e-15:
            continue
        coeff = pair_grad[idx] / (2.0 * dist)
        grad_C[i] += coeff * d
        grad_C[j] -= coeff * d
    bnd_grad = grad_all[n_pairs:].reshape(n, 3)
    signs = np.sign(C)
    signs[np.abs(C) < 1e-15] = 1.0
    grad_C += -bnd_grad * signs
    return grad_C


def _bcc_lattice_init(n_spheres=50, seed=42):
    """Generate n_spheres centers on a distorted BCC superlattice."""
    n_cells = max(1, int(round(n_spheres / 2.0)))
    m = int(np.ceil(n_cells ** (1.0 / 3.0)))
    while m ** 3 < n_cells:
        m += 1
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                points.append([ix, iy, iz])
                points.append([ix + 0.5, iy + 0.5, iz + 0.5])
    points = np.array(points)
    points = points[:n_spheres]
    span = m - 1 if m > 1 else 1.0
    scale = 0.9 / span if span > 0 else 1.0
    points = points * scale - scale * (m - 1) / 2.0
    rng = np.random.RandomState(seed)
    points += rng.uniform(-0.01, 0.01, size=points.shape)
    points = np.clip(points, -0.45, 0.45)
    return points


def _fcc_lattice_init(n_spheres=50, seed=42):
    """Generate n_spheres centers on a distorted FCC superlattice."""
    m = int(np.ceil((n_spheres / 4.0) ** (1.0 / 3.0)))
    while m ** 3 * 4 < n_spheres:
        m += 1
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                # FCC: corners + face centers
                points.append([ix, iy, iz])
                points.append([ix + 0.5, iy + 0.5, iz])
                points.append([ix + 0.5, iy, iz + 0.5])
                points.append([ix, iy + 0.5, iz + 0.5])
    points = np.array(points)
    points = points[:n_spheres]
    span = m - 1 if m > 1 else 1.0
    scale = 0.9 / span if span > 0 else 1.0
    points = points * scale - scale * (m - 1) / 2.0
    rng = np.random.RandomState(seed)
    points += rng.uniform(-0.01, 0.01, size=points.shape)
    points = np.clip(points, -0.45, 0.45)
    return points


def _jittered_grid_init(n_spheres=50, seed=42):
    """Jittered cubic grid initialization."""
    m = int(np.ceil(n_spheres ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:n_spheres].copy()
    rng = np.random.RandomState(seed)
    jitter = 0.1 / m
    grid += rng.uniform(-jitter, jitter, size=grid.shape)
    grid = np.clip(grid, -0.45, 0.45)
    return grid


def _repulsive_energy_and_grad(C, k=10.0):
    """
    Compute repulsive potential energy and gradient.
    Energy = sum over all pairs of 1/d^k + boundary repulsion.
    Minimizing this spreads points apart.
    """
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-30)
    np.fill_diagonal(dists, 1.0)
    # Pair repulsion
    pair_energy = np.sum(1.0 / (dists ** k))
    # Boundary repulsion
    dist_to_wall = 0.5 - np.abs(C)
    min_wall = np.min(dist_to_wall, axis=1)
    boundary_energy = np.sum(1.0 / (min_wall ** k))
    energy = pair_energy + boundary_energy

    # Gradient
    grad = np.zeros_like(C)
    # Pair gradient
    for i in range(n):
        for j in range(i + 1, n):
            d = C[i] - C[j]
            dist = np.sqrt(np.sum(d ** 2) + 1e-30)
            coeff = k / (dist ** (k + 2))
            grad[i] += coeff * d
            grad[j] -= coeff * d
    # Boundary gradient
    for i in range(n):
        dw = 0.5 - np.abs(C[i])
        min_dw = np.min(dw)
        coeff = k / (min_dw ** (k + 1))
        signs = np.sign(C[i])
        signs[np.abs(C[i]) < 1e-15] = 1.0
        # Only the closest wall contributes
        closest = np.argmin(dw)
        grad[i][closest] -= coeff * signs[closest]

    return energy, grad


def _coordinate_descent_polish(C, max_iter=1000, tol=1e-12, time_budget=5.0):
    """
    Coordinate descent polish: iteratively adjust each coordinate to maximize
    the minimum radius.
    """
    C = np.asarray(C).reshape(-1, 3).copy()
    n = len(C)
    t_start = time.time()

    for iteration in range(max_iter):
        if time.time() - t_start > time_budget:
            break

        improved = False
        for i in range(n):
            for coord in range(3):
                # Try small perturbations in both directions
                best_val = _exact_radius(C)
                for delta in [1e-4, 1e-3, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4]:
                    for sign in [1, -1]:
                        C_test = C.copy()
                        C_test[i, coord] += sign * delta
                        # Check bounds
                        if C_test[i, coord] < -0.5 or C_test[i, coord] > 0.5:
                            continue
                        val = _exact_radius(C_test)
                        if val > best_val + tol:
                            best_val = val
                            C[i, coord] = C_test[i, coord]
                            improved = True

        if not improved:
            break

    return C


def _cubic_grid_init(n_spheres=50):
    """Simple cubic grid initialization."""
    m = int(np.ceil(n_spheres ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n_spheres].copy()


def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # Leave 5s margin for the 120s limit

    # Try multiple initializations and keep the best
    initializations = []
    # BCC lattice
    initializations.append(_bcc_lattice_init(N, seed=42))
    # Cubic grid
    initializations.append(_cubic_grid_init(N))
    # BCC with different seed
    initializations.append(_bcc_lattice_init(N, seed=123))
    # FCC lattice
    initializations.append(_fcc_lattice_init(N, seed=42))
    # Jittered grid
    initializations.append(_jittered_grid_init(N, seed=42))
    # Random with some structure
    rng = np.random.RandomState(7)
    rand_pts = rng.uniform(-0.4, 0.4, size=(N, 3))
    initializations.append(rand_pts)
    # Another random
    rng2 = np.random.RandomState(99)
    rand_pts2 = rng2.uniform(-0.4, 0.4, size=(N, 3))
    initializations.append(rand_pts2)

    best_C = None
    best_r = -np.inf
    for init in initializations:
        r = _exact_radius(init)
        if r > best_r:
            best_r = r
            best_C = init.copy()

    # Multi-beta optimization
    betas = [10.0, 20.0, 50.0, 100.0, 200.0]

    for beta in betas:
        if time.time() > deadline - 40.0:
            break

        def objective(x):
            C = x.reshape(-1, 3)
            return _soft_min_radius(C, beta)

        def gradient(x):
            C = x.reshape(-1, 3)
            g = _grad_soft_min_radius(C, beta)
            return g.flatten()

        x0 = best_C.flatten()

        try:
            res = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 60}
            )
            C_new = res.x.reshape(-1, 3)
            r_new = _exact_radius(C_new)
            if r_new > best_r:
                best_r = r_new
                best_C = C_new.copy()
        except Exception:
            pass

        # Repulsive energy polish after each beta stage
        if time.time() < deadline - 35.0:
            try:
                _, grad_rep = _repulsive_energy_and_grad(best_C, k=8.0)
                x0_rep = best_C.flatten()

                def rep_obj(x):
                    C = x.reshape(-1, 3)
                    e, _ = _repulsive_energy_and_grad(C, k=8.0)
                    return e

                def rep_grad(x):
                    C = x.reshape(-1, 3)
                    _, g = _repulsive_energy_and_grad(C, k=8.0)
                    return g.flatten()

                res_rep = minimize(
                    rep_obj,
                    x0_rep,
                    jac=rep_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40}
                )
                C_rep = res_rep.x.reshape(-1, 3)
                C_rep = np.clip(C_rep, -0.5 + 1e-15, 0.5 - 1e-15)
                r_rep = _exact_radius(C_rep)
                if r_rep > best_r:
                    best_r = r_rep
                    best_C = C_rep.copy()
            except Exception:
                pass

    # Final high-precision polish
    if time.time() < deadline - 15.0:
        beta_final = 500.0

        def objective_final(x):
            C = x.reshape(-1, 3)
            return _soft_min_radius(C, beta_final)

        def gradient_final(x):
            C = x.reshape(-1, 3)
            g = _grad_soft_min_radius(C, beta_final)
            return g.flatten()

        try:
            res = minimize(
                objective_final,
                best_C.flatten(),
                jac=gradient_final,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            C_new = res.x.reshape(-1, 3)
            r_new = _exact_radius(C_new)
            if r_new > best_r:
                best_r = r_new
                best_C = C_new.copy()
        except Exception:
            pass

    # One more polish with even higher beta if time permits
    if time.time() < deadline - 10.0:
        beta_final2 = 1000.0

        def objective_final2(x):
            C = x.reshape(-1, 3)
            return _soft_min_radius(C, beta_final2)

        def gradient_final2(x):
            C = x.reshape(-1, 3)
            g = _grad_soft_min_radius(C, beta_final2)
            return g.flatten()

        try:
            res = minimize(
                objective_final2,
                best_C.flatten(),
                jac=gradient_final2,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100}
            )
            C_new = res.x.reshape(-1, 3)
            r_new = _exact_radius(C_new)
            if r_new > best_r:
                best_r = r_new
                best_C = C_new.copy()
        except Exception:
            pass

    # Coordinate descent polish
    if time.time() < deadline - 3.0:
        time_remaining = deadline - 3.0 - time.time()
        if time_remaining > 2.0:
            try:
                C_cd = _coordinate_descent_polish(best_C, max_iter=500, tol=1e-13, time_budget=min(time_remaining, 8.0))
                r_cd = _exact_radius(C_cd)
                if r_cd > best_r:
                    best_r = r_cd
                    best_C = C_cd.copy()
            except Exception:
                pass

    # Ensure validity: clip to cube and check
    best_C = np.clip(best_C, -0.5 + 1e-15, 0.5 - 1e-15)
    return best_C
# EVOLVE-BLOCK-END
