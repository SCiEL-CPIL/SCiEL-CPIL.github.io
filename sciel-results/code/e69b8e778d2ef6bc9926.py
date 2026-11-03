"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

_start_time = time.time()
_TIME_LIMIT = 145  # seconds, safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    return np.logaddexp(0, z / eps) * eps


def _sigmoid(z, eps=1e-4):
    """Smooth derivative of softplus."""
    return 1.0 / (1.0 + np.exp(-z / eps))


def _validate(xyz):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -1e-7) or np.any(x + r > 1 + 1e-7):
        return False
    if np.any(y - r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True


def _enforce_validity(params):
    """
    Enforce all constraints by shrinking radii and clipping positions.
    Returns a valid configuration.
    """
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    # Clip radii to fit within walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    # Iteratively resolve overlaps
    iu = np.triu_indices(N, k=1)
    for _ in range(300):
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
            # Shrink the smaller circle preferentially
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
        # Re-clip to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    return p


def _physics_relax(xyz, time_limit=5.0, grow_rate=0.0005):
    """
    Force-directed physics relaxation that also grows radii over time.
    Uses adaptive growth rate and stronger repulsion near convergence.
    """
    start_time = time.time()
    xyz = xyz.copy().astype(float)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()

    # Track best valid configuration
    best_score = np.sum(r)
    best_xyz = xyz.copy()
    last_improvement = 0.0

    dt = 0.005
    velocity_damp = 0.95
    current_grow = grow_rate

    for iteration in range(8000):
        if time.time() - start_time > time_limit:
            break

        # Adaptive growth: slow down if no recent improvement
        if iteration % 100 == 0:
            if _elapsed_since := (time.time() - start_time) - last_improvement > 2.0:
                current_grow *= 0.7
            else:
                current_grow = grow_rate

        # Grow radii slowly to push packing tighter
        r *= (1.0 + current_grow)

        # Clip radii to fit within walls
        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, wall_max)
        r = np.maximum(r, 1e-6)

        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)

        # Pairwise repulsion for overlaps (vectorized)
        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        min_dist = r[idx_i] + r[idx_j]
        overlap = min_dist - dist
        active = overlap > 0

        if np.any(active):
            force = np.zeros(len(active))
            force[active] = overlap[active] * 80.0
            inv_dist = np.where(dist > 1e-10, 1.0 / dist, 0.0)

            np.add.at(fx, idx_i, force * dx * inv_dist)
            np.add.at(fx, idx_j, -force * dx * inv_dist)
            np.add.at(fy, idx_i, force * dy * inv_dist)
            np.add.at(fy, idx_j, -force * dy * inv_dist)

        # Boundary forces - push circles away from walls
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 150.0
            if x[i] > 1 - r[i]:
                fx[i] -= (x[i] - (1 - r[i])) * 150.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 150.0
            if y[i] > 1 - r[i]:
                fy[i] -= (y[i] - (1 - r[i])) * 150.0

        # Update positions with velocity damping
        x += dt * fx
        y += dt * fy

        # Clip to valid positions
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])

        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = r

        # Check validity and track best
        if _validate(xyz):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_xyz = xyz.copy()
                last_improvement = time.time() - start_time

    return best_xyz


def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
    """
    Objective and analytic gradient for soft-plus penalty method.
    Returns (value, gradient).
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    val = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # d(-sum r)/dr

    # Boundary violations
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    # Left x boundary
    sp_lx = _softplus(lx, smooth_eps)
    sg_lx = _sigmoid(lx, smooth_eps)
    grad[0::3] += -2.0 * penalty_weight * sp_lx * sg_lx
    grad[2::3] += 2.0 * penalty_weight * sp_lx * sg_lx

    # Right x boundary
    sp_ux = _softplus(ux, smooth_eps)
    sg_ux = _sigmoid(ux, smooth_eps)
    grad[0::3] += 2.0 * penalty_weight * sp_ux * sg_ux
    grad[2::3] += 2.0 * penalty_weight * sp_ux * sg_ux

    # Left y boundary
    sp_ly = _softplus(ly, smooth_eps)
    sg_ly = _sigmoid(ly, smooth_eps)
    grad[1::3] += -2.0 * penalty_weight * sp_ly * sg_ly
    grad[2::3] += 2.0 * penalty_weight * sp_ly * sg_ly

    # Right y boundary
    sp_uy = _softplus(uy, smooth_eps)
    sg_uy = _sigmoid(uy, smooth_eps)
    grad[1::3] += 2.0 * penalty_weight * sp_uy * sg_uy
    grad[2::3] += 2.0 * penalty_weight * sp_uy * sg_uy

    # Negative radius
    sp_nr = _softplus(-r, smooth_eps)
    sg_nr = _sigmoid(-r, smooth_eps)
    grad[2::3] += -2.0 * penalty_weight * sp_nr * sg_nr

    val += penalty_weight * (np.sum(sp_lx**2) + np.sum(sp_ux**2) +
                             np.sum(sp_ly**2) + np.sum(sp_uy**2) +
                             np.sum(sp_nr**2))

    # Pairwise overlaps (vectorized)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    sp_ov = _softplus(overlap, smooth_eps)
    sg_ov = _sigmoid(overlap, smooth_eps)

    factor = 2.0 * penalty_weight * sp_ov * sg_ov
    inv_dist = 1.0 / dist

    np.add.at(grad[0::3], idx_i, -factor * dx * inv_dist)
    np.add.at(grad[0::3], idx_j, factor * dx * inv_dist)
    np.add.at(grad[1::3], idx_i, -factor * dy * inv_dist)
    np.add.at(grad[1::3], idx_j, factor * dy * inv_dist)
    np.add.at(grad[2::3], idx_i, factor)
    np.add.at(grad[2::3], idx_j, factor)

    val += penalty_weight * np.sum(sp_ov**2)

    return val, grad


def _polish_analytic(xyz, time_budget=5.0):
    """
    High-precision polish using analytic gradient L-BFGS-B with escalating penalty.
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

    # More penalty stages with tighter tolerances
    for penalty_weight in [500.0, 2000.0, 10000.0, 50000.0, 200000.0]:
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
                             options={'maxiter': 5000, 'ftol': 1e-16,
                                      'gtol': 1e-14, 'maxfun': 30000})
            # Validate the result
            result_xyz = result.x.reshape(N, 3)
            if _validate(result_xyz):
                obj_val, _ = _objective_grad(result.x, penalty_weight)
                if obj_val < best_val:
                    best_params = result.x
                    best_val = obj_val
        except Exception:
            pass

    return best_params.reshape(N, 3)


def _get_contact_edges(xyz, tol=1e-5):
    """Get pairs of circles that are in contact (touching)."""
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] + tol:
                edges.append((i, j))
    return edges


def _shake_and_refine(xyz, time_budget=3.0, num_shakes=20):
    """
    Local search: make small random perturbations to positions, then re-optimize
    radii by shrinking to resolve overlaps. This can escape local minima.
    """
    start_time = time.time()
    best_xyz = xyz.copy()
    best_score = np.sum(xyz[:, 2])
    rng = np.random.RandomState(999)

    for shake_idx in range(num_shakes):
        if time.time() - start_time > time_budget:
            break

        # Make a small perturbation
        xyz_shake = best_xyz.copy()
        # Randomly select a subset of circles to perturb
        n_perturb = rng.randint(3, 8)
        perturb_idx = rng.choice(N, size=n_perturb, replace=False)
        xyz_shake[perturb_idx, 0] += rng.uniform(-0.01, 0.01, n_perturb)
        xyz_shake[perturb_idx, 1] += rng.uniform(-0.01, 0.01, n_perturb)

        # Re-optimize: try to grow all radii uniformly while maintaining validity
        # Start from current radii and try to scale up
        scale = 1.0
        for _ in range(50):
            new_r = xyz_shake[:, 2] * (1.0 + 0.002)
            xyz_shake[:, 2] = new_r

            # Check if still valid
            valid = True
            for i in range(N):
                if xyz_shake[i, 0] - new_r[i] < -1e-7 or xyz_shake[i, 0] + new_r[i] > 1 + 1e-7:
                    valid = False
                    break
                if xyz_shake[i, 1] - new_r[i] < -1e-7 or xyz_shake[i, 1] + new_r[i] > 1 + 1e-7:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = xyz_shake[i, 0] - xyz_shake[j, 0]
                        dy = xyz_shake[i, 1] - xyz_shake[j, 1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist < new_r[i] + new_r[j] - 1e-7:
                            valid = False
                            break
                    if not valid:
                        break

            if not valid:
                # Roll back
                xyz_shake[:, 2] = xyz_shake[:, 2] / (1.0 + 0.002)
                break

        # Try a quick polish
        if time.time() - start_time < time_budget - 0.5:
            xyz_opt = _polish_analytic(xyz_shake, time_budget=1.0)
            if _validate(xyz_opt):
                score = np.sum(xyz_opt[:, 2])
                if score > best_score:
                    best_score = score
                    best_xyz = xyz_opt

    return best_xyz


def _initial_config_grid():
    """Simple grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
    return np.array(cells[:N])


def _initial_config_hex():
    """Generate an initial configuration using a hexagonal-like arrangement."""
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            if j % 2 == 1:
                x = (i + 0.5) / side + 0.1 / side
            else:
                x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            positions.append([x, y, r])
    positions.append([0.5, 0.5, 0.5 / side * 0.5])
    return np.array(positions[:N])


def _initial_config_corner():
    """Place circles in corners and along edges for better boundary usage."""
    rng = np.random.RandomState(123)
    circles = np.zeros((N, 3))

    # Corner positions
    corners = [[0.15, 0.15], [0.85, 0.15], [0.15, 0.85], [0.85, 0.85]]
    for i, (cx, cy) in enumerate(corners):
        circles[i, 0] = cx
        circles[i, 1] = cy
        circles[i, 2] = 0.14

    # Edge midpoints
    edges = [[0.5, 0.12], [0.5, 0.88], [0.12, 0.5], [0.88, 0.5]]
    for i, (cx, cy) in enumerate(edges):
        idx = 4 + i
        circles[idx, 0] = cx
        circles[idx, 1] = cy
        circles[idx, 2] = 0.12

    # Center and remaining
    remaining = 12
    for i in range(remaining):
        angle = 2 * np.pi * i / remaining
        cx = 0.5 + 0.25 * np.cos(angle)
        cy = 0.5 + 0.25 * np.sin(angle)
        idx = 8 + i
        circles[idx, 0] = cx
        circles[idx, 1] = cy
        circles[idx, 2] = 0.10

    # Add small random perturbation
    circles[:, 0] += 0.005 * rng.randn(N)
    circles[:, 1] += 0.005 * rng.randn(N)
    circles[:, 0] = np.clip(circles[:, 0], 0.05, 0.95)
    circles[:, 1] = np.clip(circles[:, 1], 0.05, 0.95)

    return circles


def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of radii.
    """
    global_start = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    best_xyz = _initial_config_grid()
    best_score = np.sum(best_xyz[:, 2])

    # Try multiple initial configurations
    initial_configs = []

    # 1. Simple grid
    initial_configs.append(_initial_config_grid())

    # 2. Hexagonal-like
    initial_configs.append(_initial_config_hex())

    # 3. Corner-weighted
    initial_configs.append(_initial_config_corner())

    # 4. Random configurations
    np.random.seed(42)
    for trial in range(4):
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.05, 0.15, N)
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        initial_configs.append(np.column_stack([x, y, r]))

    # 5. Concentric rings pattern
    positions = []
    positions.append([0.5, 0.5, 0.10])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.12 * 1.8 * np.cos(angle)
        y = 0.5 + 0.12 * 1.8 * np.sin(angle)
        positions.append([x, y, 0.12 * 0.9])
    for k in range(10):
        angle = 2 * np.pi * k / 10 + np.pi / 10
        x = 0.5 + 0.10 * 3.5 * np.cos(angle)
        y = 0.5 + 0.10 * 3.5 * np.sin(angle)
        positions.append([x, y, 0.10 * 0.9])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.08 * 4.8 * np.cos(angle)
        y = 0.5 + 0.08 * 4.8 * np.sin(angle)
        positions.append([x, y, 0.08 * 0.9])

    initial_configs.append(np.array(positions[:N]))

    # Optimize from each initial configuration using physics relaxation + analytic polish
    for idx, xyz0 in enumerate(initial_configs):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 12.0:
            break

        # Phase 1: Physics relaxation
        phys_time = min(4.0, remaining * 0.30)
        if phys_time < 1.5:
            break

        try:
            xyz_phys = _physics_relax(xyz0, time_limit=phys_time)

            # Phase 2: Analytic gradient polish
            elapsed = time.time() - global_start
            remaining = deadline - elapsed
            polish_time = min(5.0, remaining * 0.40)
            if polish_time < 1.0:
                xyz_opt = xyz_phys
            else:
                xyz_opt = _polish_analytic(xyz_phys, time_budget=polish_time)

            # Phase 3: Shake and refine
            elapsed = time.time() - global_start
            remaining = deadline - elapsed
            shake_time = min(3.0, remaining * 0.25)
            if shake_time > 1.0:
                xyz_shake = _shake_and_refine(xyz_opt, time_budget=shake_time)
                if np.sum(xyz_shake[:, 2]) > np.sum(xyz_opt[:, 2]):
                    xyz_opt = xyz_shake

            # Validate and fix if necessary
            if not _validate(xyz_opt):
                xyz_fixed = _enforce_validity(xyz_opt)
                xyz_opt = xyz_fixed

            score = np.sum(xyz_opt[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_opt
        except Exception:
            pass

    # Final high-precision polish on the best result
    elapsed = time.time() - global_start
    remaining = deadline - elapsed
    if remaining > 3.0:
        try:
            xyz_final = _polish_analytic(best_xyz, time_budget=min(4.0, remaining * 0.6))
            if _validate(xyz_final) and np.sum(xyz_final[:, 2]) > best_score:
                best_xyz = xyz_final
                best_score = np.sum(best_xyz[:, 2])
        except Exception:
            pass

    # Final check
    if not _validate(best_xyz):
        best_xyz = _enforce_validity(best_xyz)

    if not _validate(best_xyz):
        best_xyz = _initial_config_grid()

    return best_xyz

# EVOLVE-BLOCK-END
