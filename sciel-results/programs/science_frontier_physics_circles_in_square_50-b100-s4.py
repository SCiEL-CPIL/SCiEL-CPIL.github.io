import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.spatial import Delaunay

N = 50

def _pairwise_distances(pos):
    """Compute all pairwise distances efficiently."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    return dists

def _compute_radius_fast(positions):
    """Fast radius computation using vectorized operations."""
    min_bound = 0.5 - np.max(np.abs(positions))
    if min_bound <= 0:
        return 0.0
    dists = _pairwise_distances(positions)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_bound, min_dist / 2.0)

def compute_radius(positions):
    """
    Compute the maximum common radius for given circle centers.
    positions: array of shape (N, 2)
    """
    min_bound = 0.5 - np.max(np.abs(positions))
    if min_bound <= 0:
        return 0.0
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_bound, min_dist / 2.0)

def fast_radius_approx(positions):
    """
    Fast approximation of radius using Delaunay triangulation.
    """
    min_bound = 0.5 - np.max(np.abs(positions))
    if min_bound <= 0:
        return 0.0
    try:
        tri = Delaunay(positions)
        edges = tri.simplices
        edge_lengths = []
        for simplex in edges:
            for i in range(3):
                for j in range(i+1, 3):
                    d = np.linalg.norm(positions[simplex[i]] - positions[simplex[j]])
                    edge_lengths.append(d)
        min_edge = np.min(edge_lengths)
        return min(min_bound, min_edge / 2.0)
    except Exception:
        return compute_radius(positions)

def soft_min_radius(x, temperature=0.01):
    """
    Smooth approximation of the min radius using log-sum-exp.
    """
    pos = x.reshape(N, 2)
    bound_terms = np.concatenate([
        0.5 - np.abs(pos[:, 0]),
        0.5 - np.abs(pos[:, 1])
    ])
    dists = cdist(pos, pos)
    np.fill_diagonal(dists, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_terms = dists[iu] / 2.0
    all_terms = np.concatenate([bound_terms, pair_terms])
    if np.any(all_terms < 0):
        return -10.0
    neg_terms = -all_terms / temperature
    soft_min = -temperature * logsumexp(neg_terms)
    return soft_min

def soft_min_radius_grad(x, temperature=0.01):
    """
    Compute value and analytical gradient of soft-min radius.
    """
    pos = x.reshape(N, 2)
    bound_vals = np.concatenate([
        0.5 - np.abs(pos[:, 0]),
        0.5 - np.abs(pos[:, 1])
    ])
    dists = cdist(pos, pos)
    np.fill_diagonal(dists, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_vals = dists[iu] / 2.0
    all_vals = np.concatenate([bound_vals, pair_vals])
    if np.any(all_vals < 0):
        return -10.0, np.zeros_like(x)
    neg_terms = -all_vals / temperature
    log_sum = logsumexp(neg_terms)
    value = -temperature * log_sum
    exp_neg = np.exp(neg_terms - log_sum)
    n_bound = 2 * N
    n_pair = len(pair_vals)
    grad = np.zeros_like(x)
    for i in range(N):
        w = exp_neg[i]
        if pos[i, 0] > 1e-10:
            grad[2*i] += w * (-1.0)
        elif pos[i, 0] < -1e-10:
            grad[2*i] += w * (1.0)
    for i in range(N):
        w = exp_neg[N + i]
        if pos[i, 1] > 1e-10:
            grad[2*i+1] += w * (-1.0)
        elif pos[i, 1] < -1e-10:
            grad[2*i+1] += w * (1.0)
    for k in range(n_pair):
        i, j = iu[0][k], iu[1][k]
        w = exp_neg[n_bound + k]
        diff = pos[i] - pos[j]
        dist = np.linalg.norm(diff)
        if dist > 1e-10:
            factor = w / (2.0 * dist)
            grad[2*i] += factor * diff[0]
            grad[2*i+1] += factor * diff[1]
            grad[2*j] -= factor * diff[0]
            grad[2*j+1] -= factor * diff[1]
    return value, grad

def push_apart_repair(pos, target_r, max_iters=50, tol=1e-8):
    """
    Iteratively push overlapping circles apart until no overlaps remain.
    """
    pos = pos.copy()
    n = len(pos)
    for it in range(max_iters):
        min_bound = 0.5 - np.max(np.abs(pos))
        dists = _pairwise_distances(pos)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        current_r = min(min_bound, min_dist / 2.0)
        if current_r >= target_r - tol:
            break
        if min_bound < min_dist / 2.0:
            for i in range(n):
                for dim in range(2):
                    if pos[i, dim] > 0.5 - target_r:
                        pos[i, dim] = 0.5 - target_r
                    elif pos[i, dim] < -0.5 + target_r:
                        pos[i, dim] = -0.5 + target_r
        else:
            idx_i, idx_j = np.unravel_index(np.argmin(dists), dists.shape)
            if idx_i == idx_j:
                break
            diff = pos[idx_i] - pos[idx_j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                pos[idx_i] += np.random.randn(2) * 1e-3
            else:
                direction = diff / dist
                overlap = target_r * 2 - dist
                if overlap > 0:
                    pos[idx_i] += direction * overlap / 2
                    pos[idx_j] -= direction * overlap / 2
        pos = np.clip(pos, -0.5, 0.5)
    return pos

def constrained_optimize(pos, time_limit=5.0, seed=0):
    """
    Optimize using gradient-based method with analytical soft-min gradient.
    """
    start_time = time.time()
    pos = pos.copy()
    current_r = _compute_radius_fast(pos)
    best_pos = pos.copy()
    best_r = current_r
    target_r = current_r * 1.001

    def objective(x):
        val, _ = soft_min_radius_grad(x, temperature=0.005)
        return -val

    def objective_grad(x):
        _, grad = soft_min_radius_grad(x, temperature=0.005)
        return -grad

    repaired = push_apart_repair(pos, target_r, max_iters=100)
    r_repaired = _compute_radius_fast(repaired)
    if r_repaired > best_r:
        best_r = r_repaired
        best_pos = repaired.copy()

    remaining = time_limit - (time.time() - start_time)
    if remaining > 1.0:
        try:
            bounds = [(-0.5, 0.5)] * (2 * N)
            result = minimize(
                objective,
                best_pos.flatten(),
                jac=objective_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            new_pos = result.x.reshape(N, 2)
            if np.all(np.abs(new_pos) <= 0.5):
                new_r = _compute_radius_fast(new_pos)
                if new_r > best_r:
                    best_r = new_r
                    best_pos = new_pos.copy()
        except Exception:
            pass
    return best_pos

def nelder_mead_optimize(positions, time_limit=5.0, seed=0):
    """
    Optimize positions using Nelder-Mead on the negative radius.
    """
    start_time = time.time()
    best_pos = positions.copy()
    best_r = compute_radius(positions)
    call_count = [0]

    def objective(x):
        pos = x.reshape(N, 2)
        if np.any(np.abs(pos) > 0.5 + 1e-6):
            return 10.0
        r = fast_radius_approx(pos)
        call_count[0] += 1
        if call_count[0] % 10 == 0:
            r = compute_radius(pos)
        return -r

    initial_simplex_size = 0.01
    try:
        result = minimize(
            objective,
            best_pos.flatten(),
            method='Nelder-Mead',
            options={
                'xatol': 1e-7,
                'fatol': 1e-7,
                'maxiter': 5000,
                'maxfev': 10000,
                'initial_simplex_size': initial_simplex_size,
                'adaptive': True
            }
        )
        new_pos = result.x.reshape(N, 2)
        if np.all(np.abs(new_pos) <= 0.5):
            new_r = compute_radius(new_pos)
            if new_r > best_r:
                best_r = new_r
                best_pos = new_pos.copy()
    except Exception:
        pass
    return best_pos, best_r

def repulsion_init(n_iter=100, seed=0):
    """
    Generate an initial configuration using a simple repulsion force model.
    """
    np.random.seed(seed)
    pos = np.random.uniform(-0.4, 0.4, (N, 2))
    strength = 0.01
    for _ in range(n_iter):
        forces = np.zeros_like(pos)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2) + 1e-9
        inv_dist_sq = np.zeros_like(dist_sq)
        mask = dist_sq > 1e-6
        inv_dist_sq[mask] = 1.0 / dist_sq[mask]
        weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
        repulsion = np.sum(weighted_diff, axis=1)
        margin = 0.45
        wall_force = np.zeros_like(pos)
        wall_force[pos[:, 0] < -margin, 0] += 1.0
        wall_force[pos[:, 0] > margin, 0] -= 1.0
        wall_force[pos[:, 1] < -margin, 1] += 1.0
        wall_force[pos[:, 1] > margin, 1] -= 1.0
        forces = repulsion * strength + wall_force * 0.01
        pos += forces
        pos = np.clip(pos, -0.49, 0.49)
    return pos

def polish(positions, time_limit=5.0):
    """
    Local optimization using scipy minimize with annealing temperature and analytical gradient.
    Uses finer temperature schedule for better convergence.
    """
    start_time = time.time()
    positions = positions.copy()
    current_r = compute_radius(positions)
    best_pos = positions.copy()
    best_r = current_r
    bounds = [(-0.5, 0.5)] * (2 * N)
    temps = [0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 0.0005, 0.0002]
    for T in temps:
        if time.time() - start_time > time_limit - 0.5:
            break
        def objective(x):
            if time.time() - start_time > time_limit:
                return 0
            val, _ = soft_min_radius_grad(x, temperature=T)
            return -val
        def objective_grad(x):
            if time.time() - start_time > time_limit:
                return np.zeros_like(x)
            _, grad = soft_min_radius_grad(x, temperature=T)
            return -grad
        maxiter = 200 if T > 0.005 else 300
        try:
            result = minimize(objective, best_pos.flatten(), jac=objective_grad,
                              method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': maxiter, 'maxls': 50, 'ftol': 1e-11, 'gtol': 1e-9})
            new_pos = result.x.reshape(N, 2)
            if np.all(np.abs(new_pos) <= 0.5):
                new_r = compute_radius(new_pos)
                if new_r > best_r:
                    best_r = new_r
                    best_pos = new_pos.copy()
        except Exception:
            pass
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1.0:
        def final_objective(x):
            if time.time() - start_time > time_limit:
                return 0
            val, _ = soft_min_radius_grad(x, temperature=0.0001)
            return -val
        def final_grad(x):
            if time.time() - start_time > time_limit:
                return np.zeros_like(x)
            _, grad = soft_min_radius_grad(x, temperature=0.0001)
            return -grad
        try:
            result = minimize(final_objective, best_pos.flatten(), jac=final_grad,
                              method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 500, 'maxls': 100, 'ftol': 1e-13, 'gtol': 1e-11})
            new_pos = result.x.reshape(N, 2)
            if np.all(np.abs(new_pos) <= 0.5):
                new_r = compute_radius(new_pos)
                if new_r > best_r:
                    best_r = new_r
                    best_pos = new_pos.copy()
        except Exception:
            pass
    return best_pos

def hybrid_polish(positions, time_limit=10.0, seed=0):
    """
    Hybrid polishing: alternate between L-BFGS-B (gradient) and Nelder-Mead (derivative-free).
    """
    start_time = time.time()
    best_pos = positions.copy()
    best_r = compute_radius(positions)
    t1 = min(4.0, time_limit * 0.4)
    best_pos = polish(best_pos, time_limit=t1)
    best_r = compute_radius(best_pos)
    remaining = time_limit - (time.time() - start_time)
    if remaining > 2.0:
        co_pos = constrained_optimize(best_pos, time_limit=remaining * 0.5, seed=seed)
        co_r = compute_radius(co_pos)
        if co_r > best_r:
            best_r = co_r
            best_pos = co_pos.copy()
    remaining = time_limit - (time.time() - start_time)
    if remaining > 2.0:
        nm_pos, nm_r = nelder_mead_optimize(best_pos, time_limit=remaining * 0.6, seed=seed)
        if nm_r > best_r:
            best_r = nm_r
            best_pos = nm_pos.copy()
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1.0:
        best_pos = polish(best_pos, time_limit=remaining)
        best_r = compute_radius(best_pos)
    return best_pos

def generate_grid_init():
    """Generate a hexagonal grid initialization as a strong baseline."""
    positions = []
    sqrt3 = np.sqrt(3.0)
    best_pos = None
    best_r = 0
    for n_rows in range(5, 10):
        n_cols = int(np.ceil(N / n_rows))
        for offset_pattern in [0, 1]:
            s_x = 1.0 / (n_cols + 0.5)
            s_y = s_x * sqrt3 / 2.0
            total_height = (n_rows - 1) * s_y + s_x
            if total_height > 1.0:
                scale = 1.0 / total_height
                s_x *= scale
                s_y *= scale
            pos_list = []
            for row in range(n_rows):
                y = -0.5 + s_x/2 + row * s_y
                if y > 0.5 - s_x/2:
                    continue
                n_in_row = n_cols if row % 2 == offset_pattern else n_cols - 1
                if row % 2 == 1:
                    x_start = -0.5 + s_x
                else:
                    x_start = -0.5 + s_x/2
                for col in range(n_in_row):
                    x = x_start + col * s_x
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                        pos_list.append([x, y])
                    if len(pos_list) >= N:
                        break
                if len(pos_list) >= N:
                    break
            if len(pos_list) >= N:
                pos_arr = np.array(pos_list[:N])
                r = compute_radius(pos_arr)
                if r > best_r:
                    best_r = r
                    best_pos = pos_arr.copy()
    for n_grid in range(5, 8):
        s = 1.0 / (n_grid + 0.5)
        pos_list = []
        for i in range(n_grid):
            for j in range(n_grid):
                x = -0.5 + s/2 + i * s
                y = -0.5 + s/2 + j * s
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    pos_list.append([x, y])
                if len(pos_list) >= N:
                    break
            if len(pos_list) >= N:
                break
        if len(pos_list) >= N:
            pos_arr = np.array(pos_list[:N])
            r = compute_radius(pos_arr)
            if r > best_r:
                best_r = r
                best_pos = pos_arr.copy()
    if best_pos is not None:
        return best_pos
    return repulsion_init(50, seed=1)

def perturbation_search(best_pos, best_r, time_budget, seed=0):
    """
    Try small random perturbations of the best configuration and re-optimize.
    This can escape local minima that pure gradient methods get stuck in.
    """
    start_time = time.time()
    np.random.seed(seed)
    best_pos = best_pos.copy()
    best_r = best_r
    
    n_attempts = 5
    per_attempt = time_budget / n_attempts
    
    for attempt in range(n_attempts):
        if time.time() - start_time > time_budget - 1.0:
            break
        
        # Generate a perturbation: move a few points slightly
        pos_perturbed = best_pos.copy()
        n_perturb = np.random.randint(2, 6)
        indices = np.random.choice(N, n_perturb, replace=False)
        scale = 0.005 + 0.01 * attempt / n_attempts
        for idx in indices:
            pos_perturbed[idx] += np.random.randn(2) * scale
        pos_perturbed = np.clip(pos_perturbed, -0.5, 0.5)
        
        # Quick polish on perturbed config
        try:
            polished = polish(pos_perturbed, time_limit=min(per_attempt * 0.7, 4.0))
            r = compute_radius(polished)
            if r > best_r:
                best_r = r
                best_pos = polished.copy()
        except Exception:
            pass
    
    return best_pos, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_pos = None
    best_r = 0.0
    
    # 1. Start with Grid - use hybrid polish
    pos = generate_grid_init()
    pos = hybrid_polish(pos, time_limit=min(20.0, deadline - time.time()), seed=0)
    r = compute_radius(pos)
    if r > best_r:
        best_r = r
        best_pos = pos.copy()
    
    # 2. Try several Repulsion-Initialized Random Starts
    n_starts = 4
    for i in range(n_starts):
        if time.time() > deadline - 25.0:
            break
        seed = 100 + i
        pos_init = repulsion_init(150, seed=seed)
        current_time = time.time()
        remaining_starts = n_starts - i
        alloc_time = min(18.0, (deadline - 25.0 - current_time) / max(1, remaining_starts))
        if alloc_time < 3.0:
            break
        pos_polished = hybrid_polish(pos_init, time_limit=alloc_time, seed=seed)
        r = compute_radius(pos_polished)
        if r > best_r:
            best_r = r
            best_pos = pos_polished.copy()
    
    # 3. Perturbation search to escape local minima
    if time.time() < deadline - 15.0:
        remaining = min(12.0, deadline - 15.0 - time.time())
        if remaining > 4.0:
            best_pos, best_r = perturbation_search(best_pos, best_r, remaining, seed=77)
    
    # 4. Additional constrained optimization pass on best result
    if time.time() < deadline - 12.0:
        remaining = min(10.0, deadline - 12.0 - time.time())
        if remaining > 3.0:
            co_pos = constrained_optimize(best_pos, time_limit=remaining, seed=99)
            co_r = compute_radius(co_pos)
            if co_r > best_r:
                best_r = co_r
                best_pos = co_pos.copy()
    
    # 5. High-precision final polish with multiple passes at very low temperature
    if time.time() < deadline - 3.0:
        remaining = deadline - 3.0 - time.time()
        if remaining > 5.0:
            best_pos = hybrid_polish(best_pos, time_limit=min(remaining * 0.5, 10.0), seed=42)
            best_r = compute_radius(best_pos)
            
            remaining2 = deadline - 3.0 - time.time()
            if remaining2 > 3.0:
                def high_precision_objective(x):
                    if time.time() > deadline - 2.0:
                        return 0
                    val, _ = soft_min_radius_grad(x, temperature=0.00005)
                    return -val
                def high_precision_grad(x):
                    if time.time() > deadline - 2.0:
                        return np.zeros_like(x)
                    _, grad = soft_min_radius_grad(x, temperature=0.00005)
                    return -grad
                try:
                    bounds = [(-0.5, 0.5)] * (2 * N)
                    result = minimize(high_precision_objective, best_pos.flatten(),
                                     jac=high_precision_grad, method='L-BFGS-B',
                                     bounds=bounds,
                                     options={'maxiter': 500, 'maxls': 100, 'ftol': 1e-14, 'gtol': 1e-12})
                    new_pos = result.x.reshape(N, 2)
                    if np.all(np.abs(new_pos) <= 0.5):
                        new_r = compute_radius(new_pos)
                        if new_r > best_r:
                            best_r = new_r
                            best_pos = new_pos.copy()
                except Exception:
                    pass
                
                remaining3 = deadline - 3.0 - time.time()
                if remaining3 > 2.0:
                    def ultra_precision_objective(x):
                        if time.time() > deadline - 2.0:
                            return 0
                        val, _ = soft_min_radius_grad(x, temperature=0.00001)
                        return -val
                    def ultra_precision_grad(x):
                        if time.time() > deadline - 2.0:
                            return np.zeros_like(x)
                        _, grad = soft_min_radius_grad(x, temperature=0.00001)
                        return -grad
                    try:
                        bounds = [(-0.5, 0.5)] * (2 * N)
                        result = minimize(ultra_precision_objective, best_pos.flatten(),
                                         jac=ultra_precision_grad, method='L-BFGS-B',
                                         bounds=bounds,
                                         options={'maxiter': 300, 'maxls': 100, 'ftol': 1e-14, 'gtol': 1e-13})
                        new_pos = result.x.reshape(N, 2)
                        if np.all(np.abs(new_pos) <= 0.5):
                            new_r = compute_radius(new_pos)
                            if new_r > best_r:
                                best_r = new_r
                                best_pos = new_pos.copy()
                    except Exception:
                        pass
    
    # Ensure valid output
    if best_pos is None:
        best_pos = generate_grid_init()
    best_pos = np.clip(best_pos, -0.5, 0.5)
    return best_pos

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    r = compute_radius(res)
    print(f"Time: {t1-t0:.2f}s, Radius: {r:.10f}")
