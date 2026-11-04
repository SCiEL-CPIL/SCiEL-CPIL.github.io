import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute pair indices for vectorized constraint evaluation
_PAIR_I, _PAIR_J = np.triu_indices(N, k=1)
_PAIR_COUNT = _PAIR_I.size  # 1225


def _true_score(centers):
    """Compute the true packing radius for a given configuration."""
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    boundary = (0.5 - np.abs(centers)).min()
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist = np.sqrt(dists_sq.min())
    return min(boundary, min_dist / 2.0)


def _soft_min(values, beta=50.0):
    """Soft minimum using log-sum-exp for differentiability."""
    v = np.asarray(values, dtype=np.float64)
    v_min = v.min()
    shifted = v - v_min
    shifted = np.clip(shifted, 0, 100.0 / beta)
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    return v_min - log_sum


def _soft_radius(centers, beta=50.0):
    """Differentiable soft approximation of the packing radius."""
    boundary_dists = 0.5 - np.abs(centers)
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_dists = np.sqrt(dists_sq[iu]) / 2.0
    all_dists = np.concatenate([boundary_dists.ravel(), pair_dists])
    return _soft_min(all_dists, beta)


def _negative_soft_radius(x, beta=50.0):
    """Negative soft radius for minimization."""
    centers = x.reshape(-1, 2)
    return -_soft_radius(centers, beta)


def _init_grid():
    """Grid initialization trying multiple configurations."""
    best_pts = None
    best_r = -1

    for rows, cols in [(7, 8), (8, 7), (5, 10), (10, 5), (6, 9), (9, 6), (7, 7), (5, 5), (4, 13), (13, 4), (8, 8)]:
        total = rows * cols
        if total < N:
            continue

        xs = (np.arange(cols) + 0.5) / cols - 0.5
        ys = (np.arange(rows) + 0.5) / rows - 0.5
        grid = np.array([(x, y) for y in ys for x in xs])

        if total == N:
            pts = grid
        else:
            dists_to_center = np.sqrt((grid ** 2).sum(axis=1))
            corner_dists = np.minimum.reduce([
                np.abs(grid[:, 0] - 0.5) + np.abs(grid[:, 1] - 0.5),
                np.abs(grid[:, 0] + 0.5) + np.abs(grid[:, 1] - 0.5),
                np.abs(grid[:, 0] - 0.5) + np.abs(grid[:, 1] + 0.5),
                np.abs(grid[:, 0] + 0.5) + np.abs(grid[:, 1] + 0.5)
            ])
            score = dists_to_center + 0.5 * corner_dists
            indices = np.argsort(score)[:N]
            pts = grid[indices]

        r = _true_score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    return best_pts if best_pts is not None else (np.random.rand(N, 2) - 0.5) * 0.9


def _init_hexagonal():
    """Improved hexagonal initialization with proper spacing."""
    best_pts = None
    best_r = -1

    for s in [0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21]:
        pts = []
        row_height = s * np.sqrt(3) / 2
        for i in range(-6, 7):
            for j in range(-6, 7):
                x = j * s + (i % 2) * s / 2
                y = i * row_height
                if abs(x) <= 0.49 and abs(y) <= 0.49:
                    pts.append([x, y])

        pts = np.array(pts)
        if len(pts) < N:
            continue

        selected = []
        remaining = pts.copy()
        center_dists = np.linalg.norm(remaining, axis=1)
        idx = np.argmin(center_dists)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)

        for _ in range(N - 1):
            if len(remaining) == 0:
                break
            sel = np.array(selected)
            dists = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
            idx = np.argmax(dists)
            selected.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)

        if len(selected) == N:
            pts_sel = np.array(selected)
            r = _true_score(pts_sel)
            if r > best_r:
                best_r = r
                best_pts = pts_sel

    if best_pts is not None:
        return best_pts
    return _init_grid()


def _force_relax(centers, n_steps=300, lr_init=0.01, lr_decay=0.999):
    """Force-directed relaxation to resolve overlaps and keep in bounds."""
    centers = centers.copy()
    n = len(centers)
    lr = lr_init

    for step in range(n_steps):
        forces = np.zeros_like(centers)

        for k in range(2):
            mask_r = centers[:, k] > 0.45
            forces[mask_r, k] -= (centers[mask_r, k] - 0.45) * 20
            mask_l = centers[:, k] < -0.45
            forces[mask_l, k] += ((-0.45) - centers[mask_l, k]) * 20

        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)

        threshold = 0.12
        mask = dists < threshold
        np.fill_diagonal(mask, False)

        force_mag = np.zeros_like(dists)
        force_mag[mask] = (threshold - dists[mask]) / (dists[mask] + 1e-10) ** 2
        force_mag = np.minimum(force_mag, 50)

        forces += (force_mag[:, :, np.newaxis] * diffs).sum(axis=1)

        centers += forces * lr
        centers = np.clip(centers, -0.5, 0.5)

        lr *= lr_decay

    return centers


def _exact_radius_negative(x):
    """Negative of the exact packing radius for SLSQP."""
    centers = x.reshape(-1, 2)
    boundary = (0.5 - np.abs(centers)).min()
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist = np.sqrt(dists_sq.min())
    return -min(boundary, min_dist / 2.0)


def _slsqp_epigraph_polish(centers, time_budget=30.0):
    """
    SLSQP epigraph formulation: maximize r subject to
    - pairwise distances >= 2r
    - wall distances >= r
    Variables: x = [c_1x, c_1y, ..., c_Nx, c_Ny, r]
    Uses vectorized constraint functions for speed.
    """
    start_time = time.time()
    c = centers.copy()
    r_init = _true_score(c)

    def objective(x):
        return -x[-1]

    def constraint_pairs_vec(x):
        cmat = x[:2*N].reshape(N, 2)
        r = x[-1]
        ci = cmat[_PAIR_I]
        cj = cmat[_PAIR_J]
        diff = ci - cj
        dists = np.sqrt(np.sum(diff * diff, axis=1))
        return dists - 2.0 * r

    def constraint_walls_vec(x):
        cmat = x[:2*N].reshape(N, 2)
        r = x[-1]
        return 0.5 - np.abs(cmat) - r

    constraints = [
        {'type': 'ineq', 'fun': constraint_pairs_vec},
        {'type': 'ineq', 'fun': lambda x: constraint_walls_vec(x).ravel()}
    ]

    x0 = np.concatenate([c.ravel(), [r_init]])
    bounds = [(-0.5, 0.5)] * (2*N) + [(1e-6, 0.5)]

    best_x = x0.copy()
    best_r = r_init

    maxiters_list = [200, 400, 800, 1500]
    for maxiter in maxiters_list:
        if time.time() - start_time > time_budget:
            break
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-13}
            )
            r_new = res.x[-1]
            c_new = np.clip(res.x[:2*N].reshape(N, 2), -0.5, 0.5)
            r_verified = _true_score(c_new)
            if r_verified > best_r:
                best_r = r_verified
                best_x = res.x.copy()
                x0 = res.x.copy()
        except Exception:
            pass

    return best_x[:2*N].reshape(N, 2), best_r


def _optimize_slsqp(centers, time_budget=15.0):
    """Optimize using SLSQP with exact objective and hard constraints."""
    start_time = time.time()
    x0 = centers.ravel()
    best_x = x0.copy()
    best_score = _true_score(centers)

    bounds = [(-0.5, 0.5)] * (2 * N)

    maxiters = [100, 200, 400, 800]

    for maxiter in maxiters:
        elapsed = time.time() - start_time
        if elapsed > time_budget - 2:
            break

        try:
            result = minimize(
                _exact_radius_negative,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-12}
            )

            new_centers = np.clip(result.x.reshape(-1, 2), -0.5, 0.5)
            new_score = _true_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_x = result.x.copy()
            x0 = result.x.copy()
        except Exception:
            pass

    return best_x.reshape(-1, 2), best_score


def _soft_radius_grad(x, beta=50.0):
    """Compute gradient of negative soft radius for L-BFGS-B."""
    centers = x.reshape(N, 2)
    boundary_dists = 0.5 - np.abs(centers)
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_dists = np.sqrt(dists_sq[iu]) / 2.0

    all_dists = np.concatenate([boundary_dists.ravel(), pair_dists])
    v_min = all_dists.min()
    shifted = np.clip(all_dists - v_min, 0, 100.0 / beta)
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp

    grad = np.zeros_like(centers)

    bd_flat = boundary_dists.ravel()
    w_bd = weights[:N*2]
    for i in range(N):
        for k in range(2):
            grad[i, k] -= w_bd[i*2+k] * (-np.sign(centers[i, k]))

    w_pairs = weights[N*2:]
    for idx in range(_PAIR_COUNT):
        i, j = _PAIR_I[idx], _PAIR_J[idx]
        diff = centers[i] - centers[j]
        dist = np.linalg.norm(diff)
        if dist > 1e-12:
            unit = diff / dist
            grad[i] -= w_pairs[idx] * 0.5 * unit
            grad[j] += w_pairs[idx] * 0.5 * unit

    return grad


def _optimize_lbfgsb(centers, time_budget=15.0):
    """Optimize using L-BFGS-B with soft-min surrogate and progressive beta, then SLSQP polish."""
    start_time = time.time()

    x0 = centers.ravel()
    best_x = x0.copy()
    best_score = _true_score(centers)

    betas = [20, 50, 100, 200, 400, 800]

    for beta in betas:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 2:
            break

        def objective(x):
            return _negative_soft_radius(x, beta)

        def grad(x):
            return _soft_radius_grad(x, beta)

        try:
            maxiter = 800 if beta < 100 else 1200
            ftol = 1e-14
            gtol = 1e-12

            result = minimize(
                objective,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
            )

            new_centers = result.x.reshape(-1, 2)
            new_score = _true_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_x = result.x.copy()
            x0 = result.x.copy()
        except Exception:
            pass

    elapsed = time.time() - start_time
    if elapsed < time_budget - 3:
        t_slsqp = min(time_budget - elapsed - 1, 10.0)
        if t_slsqp > 2:
            try:
                centers_slsqp, score_slsqp = _optimize_slsqp(
                    best_x.reshape(-1, 2), time_budget=t_slsqp
                )
                if score_slsqp > best_score:
                    best_score = score_slsqp
                    best_x = centers_slsqp.ravel().copy()
            except Exception:
                pass

    return best_x.reshape(-1, 2), best_score


def _optimize_sa(centers, time_budget=10.0):
    """Simulated annealing refinement with adaptive step size and multi-point moves."""
    start_time = time.time()
    centers = centers.copy()
    best_centers = centers.copy()
    best_score = _true_score(centers)

    T = 0.005
    T_min = 0.00005
    cooling = 0.995
    step_scale = 0.3

    while time.time() - start_time < time_budget and T > T_min:
        n_move = 1
        if np.random.rand() < 0.2:
            n_move = 2
        elif np.random.rand() < 0.05:
            n_move = 3

        idxs = np.random.choice(N, size=n_move, replace=False)
        new_centers = centers.copy()
        for idx in idxs:
            delta = np.random.randn(2) * T * step_scale
            new_centers[idx] = np.clip(centers[idx] + delta, -0.5, 0.5)

        new_score = _true_score(new_centers)
        delta_score = new_score - best_score

        if delta_score > 0 or np.random.rand() < np.exp(delta_score / T):
            centers = new_centers
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()

        T *= cooling

    return best_centers, best_score


def _delaunay_refine(centers, time_budget=10.0):
    """Refine using Delaunay triangulation to identify and fix critical pairs."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _true_score(best_centers)

    for iteration in range(50):
        if time.time() - start_time > time_budget:
            break

        try:
            tri = Delaunay(best_centers)
            simplices = tri.simplices

            min_dist = np.inf
            min_pair = None
            for simplex in simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        d = np.linalg.norm(best_centers[simplex[i]] - best_centers[simplex[j]])
                        if d < min_dist:
                            min_dist = d
                            min_pair = (simplex[i], simplex[j])

            if min_pair is None:
                break

            i, j = min_pair
            direction = best_centers[i] - best_centers[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                break
            direction = direction / norm

            for step in [0.001, 0.0005, 0.0002]:
                for sign in [1, -1]:
                    new_c = best_centers.copy()
                    new_c[i] += sign * direction * step
                    new_c[j] -= sign * direction * step
                    new_c = np.clip(new_c, -0.5, 0.5)
                    s = _true_score(new_c)
                    if s > best_score + 1e-12:
                        best_centers = new_c
                        best_score = s
                        break
        except Exception:
            break

    return best_centers, best_score


def _local_refine(centers, time_budget=20.0):
    """Local refinement focusing on the most constrained points."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _true_score(best_centers)

    step_sizes = [0.001, 0.0005, 0.0002, 0.0001, 0.00005]

    for step in step_sizes:
        if time.time() - start_time > time_budget:
            break

        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False

            boundary_dists = (0.5 - np.abs(best_centers)).min(axis=1)
            diffs = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair_dists = dists.min(axis=1) / 2.0
            constraints = np.minimum(boundary_dists, min_pair_dists)

            order = np.argsort(constraints)

            for i in order[:15]:
                if time.time() - start_time > time_budget:
                    break

                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        new_c = best_centers.copy()
                        new_c[i, 0] += dx
                        new_c[i, 1] += dy
                        if np.any(np.abs(new_c) > 0.5):
                            continue
                        s = _true_score(new_c)
                        if s > best_score + 1e-12:
                            best_centers = new_c
                            best_score = s
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    return best_centers, best_score


def _voronoi_active_set_refine(centers, time_budget=10.0):
    """Voronoi-style active-set refinement: identify kissing graph and nudge circles."""
    start_time = time.time()
    c = centers.copy()
    best_score = _true_score(c)
    best_c = c.copy()
    r_cur = best_score
    
    for iteration in range(30):
        if time.time() - start_time > time_budget:
            break
        if r_cur <= 0:
            break
            
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        tol_d = max(1e-5, 2e-4 * r_cur)
        
        active_pairs = []
        for i in range(N):
            for j in range(i+1, N):
                if d[i,j] < 2*r_cur + tol_d:
                    active_pairs.append((i,j))
                    
        wall_dist = 0.5 - np.abs(c)
        active_walls = []
        for i in range(N):
            for k in range(2):
                if wall_dist[i,k] < r_cur + tol_d:
                    active_walls.append((i,k))
                    
        if not active_pairs and not active_walls:
            break
            
        disp = np.zeros_like(c)
        for i, j in active_pairs:
            dij = c[i] - c[j]
            norm = np.linalg.norm(dij)
            if norm < 1e-12:
                continue
            dij = dij / norm
            disp[i] += dij * 0.5
            disp[j] -= dij * 0.5
            
        for i, k in active_walls:
            if c[i, k] > 0:
                disp[i, k] -= 0.5
            else:
                disp[i, k] += 0.5
                
        disp_norm = np.linalg.norm(disp, axis=1)
        max_norm = disp_norm.max()
        if max_norm > 0:
            disp = disp / max_norm * 0.001
            
        new_c = c + disp
        new_c = np.clip(new_c, -0.5, 0.5)
        new_score = _true_score(new_c)
        
        if new_score > best_score + 1e-12:
            best_score = new_score
            best_c = new_c.copy()
            c = new_c.copy()
            r_cur = new_score
        else:
            new_c = c + disp * 0.1
            new_c = np.clip(new_c, -0.5, 0.5)
            new_score = _true_score(new_c)
            if new_score > best_score + 1e-12:
                best_score = new_score
                best_c = new_c.copy()
                c = new_c.copy()
                r_cur = new_score
            else:
                break
                
    return best_c, best_score


def _slsqp_epigraph_tight(centers, time_budget=20.0):
    """Tighter SLSQP epigraph with more iterations and tighter tolerances."""
    start_time = time.time()
    c = centers.copy()
    r_init = _true_score(c)

    def objective(x):
        return -x[-1]

    def constraint_pairs_vec(x):
        cmat = x[:2*N].reshape(N, 2)
        r = x[-1]
        ci = cmat[_PAIR_I]
        cj = cmat[_PAIR_J]
        diff = ci - cj
        dists = np.sqrt(np.sum(diff * diff, axis=1))
        return dists - 2.0 * r

    def constraint_walls_vec(x):
        cmat = x[:2*N].reshape(N, 2)
        r = x[-1]
        return 0.5 - np.abs(cmat) - r

    constraints = [
        {'type': 'ineq', 'fun': constraint_pairs_vec},
        {'type': 'ineq', 'fun': lambda x: constraint_walls_vec(x).ravel()}
    ]

    x0 = np.concatenate([c.ravel(), [r_init]])
    bounds = [(-0.5, 0.5)] * (2*N) + [(1e-6, 0.5)]

    best_x = x0.copy()
    best_r = r_init

    # More aggressive iteration counts and tighter tolerances
    maxiters_list = [500, 1000, 2000, 3000, 4000]
    for maxiter in maxiters_list:
        if time.time() - start_time > time_budget:
            break
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 50}
            )
            r_new = res.x[-1]
            c_new = np.clip(res.x[:2*N].reshape(N, 2), -0.5, 0.5)
            r_verified = _true_score(c_new)
            if r_verified > best_r:
                best_r = r_verified
                best_x = res.x.copy()
                x0 = res.x.copy()
        except Exception:
            pass

    return best_x[:2*N].reshape(N, 2), best_r


def _active_set_local_search(centers, time_budget=15.0):
    """
    Active-set local search: identify the critical (binding) constraints,
    then perform a focused local optimization on the active set using
    a reduced SLSQP formulation.
    """
    start_time = time.time()
    c = centers.copy()
    best_score = _true_score(c)
    best_c = c.copy()
    r_cur = best_score

    for iteration in range(20):
        if time.time() - start_time > time_budget:
            break
        if r_cur <= 0:
            break

        # Identify active constraints
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        tol_d = max(1e-6, 1e-4 * r_cur)

        active_pairs = []
        for i in range(N):
            for j in range(i+1, N):
                if d[i,j] < 2*r_cur + tol_d:
                    active_pairs.append((i,j))

        wall_dist = 0.5 - np.abs(c)
        active_walls = []
        for i in range(N):
            for k in range(2):
                if wall_dist[i,k] < r_cur + tol_d:
                    active_walls.append((i,k))

        if not active_pairs and not active_walls:
            break

        # Build reduced problem: only optimize over active constraints
        # Use a perturbation approach: nudge active pairs slightly apart
        disp = np.zeros_like(c)
        for i, j in active_pairs:
            dij = c[i] - c[j]
            norm = np.linalg.norm(dij)
            if norm < 1e-12:
                continue
            dij = dij / norm
            disp[i] += dij * 0.3
            disp[j] -= dij * 0.3

        for i, k in active_walls:
            if c[i, k] > 0:
                disp[i, k] -= 0.3
            else:
                disp[i, k] += 0.3

        # Try multiple step sizes
        for step_mult in [1.0, 0.5, 0.2, 0.1, 0.05]:
            new_c = c + disp * step_mult
            new_c = np.clip(new_c, -0.5, 0.5)
            new_score = _true_score(new_c)
            if new_score > best_score + 1e-12:
                best_score = new_score
                best_c = new_c.copy()
                c = new_c.copy()
                r_cur = new_score
                break

        if c is best_c:
            # No improvement, try SLSQP on the current active set
            try:
                centers_epi, score_epi = _slsqp_epigraph_tight(
                    best_c, time_budget=min(5.0, time_budget - (time.time() - start_time))
                )
                if score_epi > best_score:
                    best_score = score_epi
                    best_c = centers_epi.copy()
                    c = centers_epi.copy()
                    r_cur = best_score
            except Exception:
                pass

    return best_c, best_score


def construct():
    """Main entry point combining multiple optimization strategies."""
    start_time = time.time()
    np.random.seed(42)

    # 1. Generate multiple initial configurations
    inits = []
    init_hex = _init_hexagonal()
    inits.append(init_hex)
    inits.append(_init_grid())
    inits.append((np.random.rand(N, 2) - 0.5) * 0.9)
    
    for _ in range(3):
        inits.append((np.random.rand(N, 2) - 0.5) * 0.95)

    best_centers = None
    best_score = -1

    # 2. For each init, do force relax + L-BFGS-B + SA + local refine
    for init_idx, init in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 50:
            break

        relaxed = _force_relax(init, n_steps=200)
        score_relaxed = _true_score(relaxed)
        if score_relaxed > best_score:
            best_score = score_relaxed
            best_centers = relaxed.copy()

        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 30:
            if init_idx < 2:
                t_lbfgs = min(remaining * 0.35, 30.0)
            else:
                t_lbfgs = min(remaining * 0.2, 15.0)
            centers_lbfgs, score_lbfgs = _optimize_lbfgsb(relaxed, time_budget=t_lbfgs)
            if score_lbfgs > best_score:
                best_score = score_lbfgs
                best_centers = centers_lbfgs.copy()

        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 15 and best_centers is not None and init_idx < 3:
            t_sa = min(remaining * 0.08, 6.0)
            centers_sa, score_sa = _optimize_sa(best_centers, time_budget=t_sa)
            if score_sa > best_score:
                best_score = score_sa
                best_centers = centers_sa.copy()

        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 10 and best_centers is not None and init_idx < 3:
            t_del = min(remaining * 0.08, 4.0)
            centers_del, score_del = _delaunay_refine(best_centers, time_budget=t_del)
            if score_del > best_score:
                best_score = score_del
                best_centers = centers_del.copy()

        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 5 and best_centers is not None and init_idx < 3:
            t_lr = min(remaining * 0.1, 5.0)
            centers_lr, score_lr = _local_refine(best_centers, time_budget=t_lr)
            if score_lr > best_score:
                best_score = score_lr
                best_centers = centers_lr.copy()

    # 3. SLSQP Epigraph Polish on best solution
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 25 and best_centers is not None:
        t_epigraph = min(remaining * 0.4, 40.0)
        centers_epi, score_epi = _slsqp_epigraph_polish(
            best_centers, time_budget=t_epigraph
        )
        if score_epi > best_score:
            best_score = score_epi
            best_centers = centers_epi.copy()

    # 4. Active-set local search
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 15 and best_centers is not None:
        t_active = min(remaining * 0.3, 20.0)
        centers_active, score_active = _active_set_local_search(
            best_centers, time_budget=t_active
        )
        if score_active > best_score:
            best_score = score_active
            best_centers = centers_active.copy()

    # 5. Second epigraph pass with tighter tolerances
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 15 and best_centers is not None:
        t_epigraph2 = min(remaining * 0.4, 25.0)
        centers_epi2, score_epi2 = _slsqp_epigraph_tight(
            best_centers, time_budget=t_epigraph2
        )
        if score_epi2 > best_score:
            best_score = score_epi2
            best_centers = centers_epi2.copy()

    # 6. Voronoi active-set refinement
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 10 and best_centers is not None:
        t_voronoi = min(remaining * 0.2, 8.0)
        centers_vor, score_vor = _voronoi_active_set_refine(
            best_centers, time_budget=t_voronoi
        )
        if score_vor > best_score:
            best_score = score_vor
            best_centers = centers_vor.copy()

    # 7. Final Delaunay + local refinement
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 8 and best_centers is not None:
        t_final_del = min(remaining * 0.2, 5.0)
        centers_final_del, score_final_del = _delaunay_refine(best_centers, time_budget=t_final_del)
        if score_final_del > best_score:
            best_score = score_final_del
            best_centers = centers_final_del.copy()

        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 4 and best_centers is not None:
            t_final_lr = min(remaining * 0.4, 6.0)
            centers_final_lr, score_final_lr = _local_refine(best_centers, time_budget=t_final_lr)
            if score_final_lr > best_score:
                best_score = score_final_lr
                best_centers = centers_final_lr.copy()

    # 8. Final tight SLSQP polish with epigraph
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 5 and best_centers is not None:
        try:
            centers_final_epi, score_final_epi = _slsqp_epigraph_tight(
                best_centers, time_budget=min(remaining * 0.5, 15.0)
            )
            if score_final_epi > best_score:
                best_score = score_final_epi
                best_centers = centers_final_epi.copy()
        except Exception:
            pass

    # 9. Ultra-tight final SLSQP on exact objective
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 3 and best_centers is not None:
        try:
            result = minimize(
                _exact_radius_negative,
                best_centers.ravel(),
                method='SLSQP',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 3000, 'ftol': 1e-15, 'maxls': 50}
            )
            new_centers = np.clip(result.x.reshape(-1, 2), -0.5, 0.5)
            new_score = _true_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
        except Exception:
            pass

    # Ensure validity
    if best_centers is None:
        best_centers = _init_grid()
    best_centers = np.clip(best_centers, -0.5, 0.5)

    return best_centers


if __name__ == "__main__":
    t0 = time.time()
    c = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Score: {_true_score(c)}")
    print(f"Shape: {c.shape}")
