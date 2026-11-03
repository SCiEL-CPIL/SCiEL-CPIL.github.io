"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
_GLOBAL_START = None
_GLOBAL_DEADLINE = None

# EVOLVE-BLOCK-START

def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0) using log-sum-exp."""
    return np.logaddexp(0, z / eps) * eps

def _softplus_grad(z, eps=1e-4):
    """Derivative of softplus: sigmoid(z/eps)."""
    return 1.0 / (1.0 + np.exp(-z / eps))

def _validate(xyz, tol=1e-7):
    """Check validity."""
    n = xyz.shape[0]
    x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _remaining_time():
    """Get remaining time before deadline."""
    if _GLOBAL_DEADLINE is None:
        return 140.0
    return max(0.0, _GLOBAL_DEADLINE - time.time())

def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
    """
    Objective and analytic gradient using full pairwise overlaps (vectorized).
    Returns (value, gradient).
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    val = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0

    # Boundary violations
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    sp_lx = _softplus(lx, smooth_eps)
    sg_lx = _softplus_grad(lx, smooth_eps)
    grad[0::3] += -2.0 * penalty_weight * sp_lx * sg_lx
    grad[2::3] += 2.0 * penalty_weight * sp_lx * sg_lx

    sp_ux = _softplus(ux, smooth_eps)
    sg_ux = _softplus_grad(ux, smooth_eps)
    grad[0::3] += 2.0 * penalty_weight * sp_ux * sg_ux
    grad[2::3] += 2.0 * penalty_weight * sp_ux * sg_ux

    sp_ly = _softplus(ly, smooth_eps)
    sg_ly = _softplus_grad(ly, smooth_eps)
    grad[1::3] += -2.0 * penalty_weight * sp_ly * sg_ly
    grad[2::3] += 2.0 * penalty_weight * sp_ly * sg_ly

    sp_uy = _softplus(uy, smooth_eps)
    sg_uy = _softplus_grad(uy, smooth_eps)
    grad[1::3] += 2.0 * penalty_weight * sp_uy * sg_uy
    grad[2::3] += 2.0 * penalty_weight * sp_uy * sg_uy

    sp_nr = _softplus(-r, smooth_eps)
    sg_nr = _softplus_grad(-r, smooth_eps)
    grad[2::3] += -2.0 * penalty_weight * sp_nr * sg_nr

    val += penalty_weight * (np.sum(sp_lx ** 2) + np.sum(sp_ux ** 2) +
                             np.sum(sp_ly ** 2) + np.sum(sp_uy ** 2) +
                             np.sum(sp_nr ** 2))

    # Pairwise overlaps (vectorized, full pairs)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    sp_ov = _softplus(overlap, smooth_eps)
    sg_ov = _softplus_grad(overlap, smooth_eps)

    factor = 2.0 * penalty_weight * sp_ov * sg_ov
    inv_dist = 1.0 / dist

    np.add.at(grad[0::3], idx_i, -factor * dx * inv_dist)
    np.add.at(grad[0::3], idx_j, factor * dx * inv_dist)
    np.add.at(grad[1::3], idx_i, -factor * dy * inv_dist)
    np.add.at(grad[1::3], idx_j, factor * dy * inv_dist)
    np.add.at(grad[2::3], idx_i, factor)
    np.add.at(grad[2::3], idx_j, factor)

    val += penalty_weight * np.sum(sp_ov ** 2)

    return val, grad

def _enforce_validity(params):
    """
    Enforce all constraints by shrinking radii and clipping positions.
    Returns a valid configuration.
    """
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    for _ in range(300):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    return p

def _polish_analytic(xyz, time_budget=5.0):
    """
    High-precision polish using analytic gradient L-BFGS-B with multiple penalty phases.
    """
    start_time = time.time()
    params = xyz.flatten().copy()
    best_params = params.copy()
    best_val, _ = _objective_grad(best_params, 1000.0)

    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))

    for penalty_weight in [1000.0, 5000.0, 20000.0, 100000.0]:
        if time.time() - start_time > time_budget:
            break

        def obj(p):
            v, _ = _objective_grad(p, penalty_weight)
            return v

        def grad(p):
            _, g = _objective_grad(p, penalty_weight)
            return g

        try:
            result = minimize(obj, best_params, jac=grad, method='L-BFGS-B',
                              bounds=bounds,
                              options={'maxiter': 2000, 'ftol': 1e-14,
                                       'gtol': 1e-12, 'maxfun': 10000})
            obj_val, _ = _objective_grad(result.x, penalty_weight)
            if obj_val < best_val:
                best_params = result.x
                best_val = obj_val
        except Exception:
            pass

    return best_params.reshape(N, 3)

def _physics_relax(xyz, time_limit=3.0, grow_rate=0.0005):
    """
    Force-directed physics relaxation that also grows radii over time.
    """
    start_time = time.time()
    xyz = xyz.copy().astype(float)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()

    best_score = np.sum(r)
    best_xyz = xyz.copy()

    dt = 0.005

    for iteration in range(5000):
        if time.time() - start_time > time_limit:
            break

        r *= (1.0 + grow_rate)

        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, wall_max)
        r = np.maximum(r, 1e-6)

        fx = np.zeros(N)
        fy = np.zeros(N)

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(max(dist_sq, 1e-12))
                if dist < r[i] + r[j]:
                    overlap = r[i] + r[j] - dist
                    f = 10.0 * overlap / dist
                    fx[i] += f * dx
                    fx[j] -= f * dx
                    fy[i] += f * dy
                    fy[j] -= f * dy

        for i in range(N):
            if x[i] < r[i]:
                fx[i] += 10.0 * (r[i] - x[i])
            if x[i] > 1 - r[i]:
                fx[i] -= 10.0 * (x[i] - (1 - r[i]))
            if y[i] < r[i]:
                fy[i] += 10.0 * (r[i] - y[i])
            if y[i] > 1 - r[i]:
                fy[i] -= 10.0 * (y[i] - (1 - r[i]))

        x += dt * fx
        y += dt * fy

        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)

        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, wall_max)
        r = np.maximum(r, 1e-6)

        current_score = np.sum(r)
        if current_score > best_score:
            best_score = current_score
            best_xyz = np.column_stack([x, y, r])

    return best_xyz

def _fix_boundaries(xyz):
    """Project circles inside unit square."""
    xyz = xyz.copy()
    for i in range(N):
        r = max(0, xyz[i, 2])
        xyz[i, 0] = np.clip(xyz[i, 0], r, 1 - r) if r < 0.5 else 0.5
        xyz[i, 1] = np.clip(xyz[i, 1], r, 1 - r) if r < 0.5 else 0.5
        xyz[i, 2] = r
    return xyz

def _initial_hex():
    """Hexagonal packing initial config."""
    cols = 5
    rows = 6
    positions = []
    r = 0.09
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 1:
                x = (col + 0.5) / cols + 0.5 / cols
            else:
                x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if x > 1 or y > 1:
                continue
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])

def _initial_grid():
    """Grid initial config."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    r = 0.5 / side * 0.9
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])

def _initial_corner_weighted(seed=0):
    """Random config with corner preference."""
    rng = np.random.default_rng(seed)
    positions = []
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
               [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.08])
    for _ in range(N - len(positions)):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        positions.append([x, y, rng.uniform(0.06, 0.10)])
    return np.array(positions[:N])

def _initial_ring(seed=0):
    """Ring-shaped initial config."""
    rng = np.random.default_rng(seed)
    positions = [[0.5, 0.5, 0.08]]
    for ring, n_circles, radius in [(1, 6, 0.10), (2, 10, 0.08), (3, 9, 0.07)]:
        for k in range(n_circles):
            angle = 2 * np.pi * k / n_circles + rng.uniform(-0.1, 0.1)
            dist = ring * 0.14
            x = 0.5 + dist * np.cos(angle)
            y = 0.5 + dist * np.sin(angle)
            positions.append([x, y, radius])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.06])
    return np.array(positions[:N])

def _initial_corner_cluster():
    """Specific initialization placing larger circles in corners and edges."""
    positions = []
    positions.append([0.1, 0.1, 0.12])
    positions.append([0.9, 0.1, 0.12])
    positions.append([0.1, 0.9, 0.12])
    positions.append([0.9, 0.9, 0.12])

    positions.append([0.5, 0.08, 0.10])
    positions.append([0.5, 0.92, 0.10])
    positions.append([0.08, 0.5, 0.10])
    positions.append([0.92, 0.5, 0.10])

    positions.append([0.5, 0.5, 0.12])

    rng = np.random.default_rng(123)
    while len(positions) < N:
        x = rng.uniform(0.15, 0.85)
        y = rng.uniform(0.15, 0.85)
        r = rng.uniform(0.05, 0.08)
        positions.append([x, y, r])

    return np.array(positions[:N])

def _optimize_single(xyz0):
    """Optimize from a single initial config with adaptive penalty and analytic gradients."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])

    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))

    phases = [
        (5.0, 1e-3, 300, 1e-12, 1e-10, 10),
        (50.0, 1e-4, 400, 1e-14, 1e-12, 8),
        (500.0, 1e-5, 400, 1e-15, 1e-13, 5),
        (5000.0, 1e-6, 300, 1e-15, 1e-14, 2),
        (50000.0, 1e-7, 200, 1e-16, 1e-15, 1),
    ]

    for penalty, eps, maxiter, ftol, gtol, min_time in phases:
        remaining = _remaining_time()
        if remaining < min_time:
            continue
        try:
            def obj(p):
                v, _ = _objective_grad(p, penalty, eps)
                return v

            def grad(p):
                _, g = _objective_grad(p, penalty, eps)
                return g

            res = minimize(obj, params, jac=grad,
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
            obj_val, _ = _objective_grad(res.x, penalty, eps)
            obj_cur, _ = _objective_grad(params, penalty, eps)
            if obj_val < obj_cur:
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    return best_params.reshape(N, 3), best_score

def _pt_mc_search(initial_states, temps, steps, rng):
    """
    Parallel Tempering Monte Carlo search.
    """
    n_replicas = len(temps)
    states = [s.copy() for s in initial_states]
    energies = []

    for i, s in enumerate(states):
        x, y, r = s[:, 0], s[:, 1], s[:, 2]

        lx = np.maximum(0, r - x)
        ux = np.maximum(0, x + r - 1)
        ly = np.maximum(0, r - y)
        uy = np.maximum(0, y + r - 1)
        bound_pen = np.sum(lx ** 2 + ux ** 2 + ly ** 2 + uy ** 2)

        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = np.maximum(0, r[idx_i] + r[idx_j] - dist)
        overlap_pen = np.sum(overlap ** 2)

        neg_r_pen = np.sum(np.maximum(0, -r) ** 2)

        penalty = bound_pen + overlap_pen + neg_r_pen
        energy = -np.sum(r) + 100.0 * penalty
        energies.append(energy)

    best_energy = min(energies)
    best_state = states[energies.index(best_energy)].copy()

    for step in range(steps):
        if _remaining_time() < 5.0:
            break

        i = rng.integers(0, n_replicas)

        new_state = states[i].copy()

        scale_pos = 0.01 * temps[i]
        new_state[:, 0] += rng.normal(0, scale_pos, N)
        new_state[:, 1] += rng.normal(0, scale_pos, N)

        scale_r = 0.005 * temps[i]
        new_state[:, 2] += rng.normal(0, scale_r, N)

        new_state[:, 2] = np.maximum(new_state[:, 2], 0.01)
        new_state[:, 0] = np.clip(new_state[:, 0], -0.1, 1.1)
        new_state[:, 1] = np.clip(new_state[:, 1], -0.1, 1.1)

        x, y, r = new_state[:, 0], new_state[:, 1], new_state[:, 2]

        lx = np.maximum(0, r - x)
        ux = np.maximum(0, x + r - 1)
        ly = np.maximum(0, r - y)
        uy = np.maximum(0, y + r - 1)
        bound_pen = np.sum(lx ** 2 + ux ** 2 + ly ** 2 + uy ** 2)

        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = np.maximum(0, r[idx_i] + r[idx_j] - dist)
        overlap_pen = np.sum(overlap ** 2)

        neg_r_pen = np.sum(np.maximum(0, -r) ** 2)

        penalty = bound_pen + overlap_pen + neg_r_pen
        new_energy = -np.sum(r) + 100.0 * penalty

        delta_e = new_energy - energies[i]
        if delta_e < 0 or rng.random() < np.exp(-delta_e / temps[i]):
            states[i] = new_state
            energies[i] = new_energy

            if new_energy < best_energy:
                best_energy = new_energy
                best_state = new_state.copy()

        if n_replicas > 1:
            j = rng.integers(0, n_replicas - 1)
            k = j + 1

            inv_t_j = 1.0 / temps[j]
            inv_t_k = 1.0 / temps[k]

            delta_swap = (energies[j] - energies[k]) * (inv_t_k - inv_t_j)

            if delta_swap < 0 or rng.random() < np.exp(-delta_swap):
                states[j], states[k] = states[k], states[j]
                energies[j], energies[k] = energies[k], energies[j]

                if energies[j] < best_energy:
                    best_energy = energies[j]
                    best_state = states[j].copy()
                if energies[k] < best_energy:
                    best_energy = energies[k]
                    best_state = states[k].copy()

    return best_state

def construct():
    global _GLOBAL_START, _GLOBAL_DEADLINE
    _GLOBAL_START = time.time()
    _GLOBAL_DEADLINE = _GLOBAL_START + 145.0

    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])

    inits = [
        _initial_grid(),
        _initial_hex(),
        _initial_corner_weighted(0),
        _initial_corner_weighted(1),
        _initial_ring(0),
        _initial_ring(1),
        _initial_corner_cluster()
    ]

    for xyz0 in inits:
        remaining = _remaining_time()
        if remaining < 15.0:
            break
        xyz_opt, score = _optimize_single(xyz0)

        if not _validate(xyz_opt):
            xyz_fix = _enforce_validity(xyz_opt.flatten())
            xyz_opt = xyz_fix

        if not _validate(xyz_opt):
            continue

        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt

    # Local refinement from the best solution found
    remaining = _remaining_time()
    if remaining > 15.0:
        rng = np.random.default_rng(42)
        for scale in [0.005, 0.01, 0.02, 0.03]:
            if remaining < 5.0:
                break
            perturbed = best_xyz.copy()
            noise = rng.normal(0, scale, perturbed.shape)
            noise[:, 2] *= 0.1
            perturbed += noise
            perturbed = _fix_boundaries(perturbed)

            xyz_opt, score = _optimize_single(perturbed)

            if not _validate(xyz_opt):
                xyz_fix = _enforce_validity(xyz_opt.flatten())
                xyz_opt = xyz_fix

            if not _validate(xyz_opt):
                continue

            score = np.sum(xyz_opt[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_opt
            remaining = _remaining_time()

    # Physics relaxation from best solution
    remaining = _remaining_time()
    if remaining > 10.0:
        phys_result = _physics_relax(best_xyz, time_limit=min(8.0, remaining - 5.0))
        if _validate(phys_result):
            phys_score = np.sum(phys_result[:, 2])
            if phys_score > best_score:
                best_score = phys_score
                best_xyz = phys_result
        elif _validate(_enforce_validity(phys_result.flatten())):
            phys_fixed = _enforce_validity(phys_result.flatten())
            phys_score = np.sum(phys_fixed[:, 2])
            if phys_score > best_score:
                best_score = phys_score
                best_xyz = phys_fixed

    # High-precision polish
    remaining = _remaining_time()
    if remaining > 5.0:
        polished = _polish_analytic(best_xyz, time_budget=min(4.0, remaining - 2.0))
        if _validate(polished):
            pol_score = np.sum(polished[:, 2])
            if pol_score > best_score:
                best_score = pol_score
                best_xyz = polished
        elif _validate(_enforce_validity(polished.flatten())):
            pol_fixed = _enforce_validity(polished.flatten())
            pol_score = np.sum(pol_fixed[:, 2])
            if pol_score > best_score:
                best_score = pol_score
                best_xyz = pol_fixed

    # Parallel Tempering Monte Carlo refinement
    remaining = _remaining_time()
    if remaining > 20.0:
        rng = np.random.default_rng(99)
        replicas = [best_xyz]
        perturbed = best_xyz.copy()
        perturbed += rng.normal(0, 0.01, perturbed.shape)
        perturbed = _fix_boundaries(perturbed)
        replicas.append(perturbed)

        temps = [1.0, 2.0, 4.0]

        if len(replicas) < 3:
            rep3 = best_xyz.copy()
            rep3 += rng.normal(0, 0.02, rep3.shape)
            rep3 = _fix_boundaries(rep3)
            replicas.append(rep3)

        replicas = replicas[:3]

        steps = 50000

        pt_result = _pt_mc_search(replicas, temps, steps, rng)

        if _validate(pt_result):
            pt_score = np.sum(pt_result[:, 2])
            if pt_score > best_score:
                best_xyz = pt_result
                best_score = pt_score
        elif _validate(_enforce_validity(pt_result.flatten())):
            pt_fixed = _enforce_validity(pt_result.flatten())
            pt_score = np.sum(pt_fixed[:, 2])
            if pt_score > best_score:
                best_xyz = pt_fixed
                best_score = pt_score

    if not _validate(best_xyz):
        best_xyz = _enforce_validity(_initial_grid().flatten())

    return best_xyz

# EVOLVE-BLOCK-END
