"""Combined sphere packing optimizer: gradient-based L-BFGS-B with annealing + multi-init."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

N = 50
T_LIMIT = 145.0  # wall-clock safety margin (evaluator kills at 150s)


def _min_radius(centers):
    """Compute the score: min over all pairs and boundaries."""
    centers = np.asarray(centers).reshape(N, 3)
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary clearance
    boundary = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    tree = cKDTree(centers)
    pairs = tree.query_pairs(1.0)
    if len(pairs) == 0:
        return min_boundary
    dists = np.array([np.linalg.norm(centers[i] - centers[j]) for i, j in pairs])
    min_pair = np.min(dists) / 2.0
    return min(min_pair, min_boundary)


def _soft_min_radius(centers, beta):
    """
    Smooth surrogate for -min_radius using log-sum-exp.
    Returns a scalar to be minimized.
    """
    abs_c = np.abs(centers)
    boundary = 0.5 - np.max(abs_c, axis=1)
    d = pdist(centers, metric='euclidean') / 2.0
    all_vals = np.concatenate([boundary, d])
    
    x_neg = -beta * all_vals
    x_max = np.max(x_neg)
    log_sum_exp = x_max + np.log(np.sum(np.exp(x_neg - x_max)))
    f = (1.0 / beta) * log_sum_exp
    return f


def _gradient_soft_min(centers, beta):
    """Gradient of the soft-min objective w.r.t. centers."""
    abs_c = np.abs(centers)
    boundary = 0.5 - np.max(abs_c, axis=1)
    d = pdist(centers, metric='euclidean') / 2.0
    all_vals = np.concatenate([boundary, d])
    
    x_neg = -beta * all_vals
    x_max = np.max(x_neg)
    exp_vals = np.exp(x_neg - x_max)
    sum_exp = np.sum(exp_vals)
    grad_all = -exp_vals / sum_exp
    
    grad_centers = np.zeros_like(centers)
    
    # Boundary gradient
    for i in range(N):
        k = np.argmax(np.abs(centers[i]))
        if np.abs(centers[i, k]) > 1e-15:
            grad_centers[i, k] += grad_all[i] * (-np.sign(centers[i, k]))
    
    # Pairwise gradient
    pair_idx = 0
    for a in range(N - 1):
        for b in range(a + 1, N):
            w = grad_all[N + pair_idx]
            diff = centers[a] - centers[b]
            dist = np.linalg.norm(diff)
            if dist > 1e-15:
                unit = diff / dist
                grad_centers[a] += w * unit * 0.5
                grad_centers[b] += w * (-unit) * 0.5
            pair_idx += 1
    
    return grad_centers


def _fibonacci_lattice_init(n, seed=42):
    """Generate n points via Fibonacci lattice scaled to [-0.45, 0.45]."""
    rng = np.random.default_rng(seed)
    phi = (1 + np.sqrt(5)) / 2
    points = np.zeros((n, 3))
    
    for i in range(n):
        t = (i + 0.5) / n
        z = 2 * t - 1
        r = np.sqrt(max(0, 1 - z * z))
        theta = 2 * np.pi * i / phi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points[i] = [x, y, z]
    
    points *= 0.45
    points += rng.normal(0, 0.005, size=(n, 3))
    points = np.clip(points, -0.48, 0.48)
    return points


def _grid_init(n, seed=0):
    """Jittered grid initialization."""
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    while m ** 3 < n:
        m += 1
    ticks = (np.arange(m) + 0.5) / m
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.08, 0.08, size=(len(grid), 3))
    grid = grid + jitter
    grid = grid * 0.9 - 0.45
    return grid[:n].copy()


def _random_init(n, seed=123):
    """Random initialization."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))


def _best_first_grid_init(n, seed=0):
    """Best-first grid: place spheres maximizing min distance to already-placed."""
    n_per_dim = 8
    ticks = (np.arange(n_per_dim) + 0.5) / n_per_dim - 0.5
    candidates = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.default_rng(seed)
    # Add jitter to break symmetry
    candidates += rng.normal(0, 0.02, size=candidates.shape)
    
    placed = []
    for _ in range(n):
        if len(candidates) == 0:
            break
        if len(placed) == 0:
            best_idx = np.argmin(np.sum(candidates ** 2, axis=1))
            placed.append(candidates[best_idx].copy())
            candidates = np.delete(candidates, best_idx, axis=0)
        else:
            placed_arr = np.array(placed)
            min_dists = np.min(
                np.sqrt(np.sum((candidates[:, np.newaxis, :] - placed_arr[np.newaxis, :, :]) ** 2, axis=2)),
                axis=1
            )
            wall_dists = np.min(np.abs(0.5 - np.abs(candidates)), axis=1)
            scores = np.minimum(min_dists, wall_dists)
            best_idx = np.argmax(scores)
            placed.append(candidates[best_idx].copy())
            candidates = np.delete(candidates, best_idx, axis=0)
    
    while len(placed) < n:
        placed.append(rng.uniform(-0.45, 0.45, size=3))
    
    return np.array(placed[:n])


def _optimize_from_init(init, time_limit):
    """Run L-BFGS-B with annealing beta from 20 to 1000."""
    best_centers = init.copy()
    best_score = _min_radius(init)
    
    t_start = time.time()
    betas = [20, 50, 100, 200, 500, 1000]
    
    for beta in betas:
        if time.time() - t_start > time_limit:
            break
        
        def obj(x):
            c = x.reshape(-1, 3)
            return _soft_min_radius(c, beta)
        
        def grad(x):
            c = x.reshape(-1, 3)
            g = _gradient_soft_min(c, beta)
            return g.flatten()
        
        x0 = best_centers.flatten()
        
        try:
            result = minimize(
                obj, x0, jac=grad, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
            )
            if result.success or result.fun < obj(x0):
                new_centers = result.x.reshape(-1, 3)
                new_score = _min_radius(new_centers)
                if new_score > best_score:
                    best_centers = new_centers
                    best_score = new_score
        except Exception:
            pass
        
        if time.time() - t_start > time_limit:
            break
    
    return best_centers, best_score


def _local_refinement(centers, time_limit):
    """Nelder-Mead refinement for final polishing."""
    best = centers.copy()
    best_score = _min_radius(best)
    t_start = time.time()
    
    def obj(x):
        c = x.reshape(-1, 3)
        return -_min_radius(c)
    
    try:
        result = minimize(
            obj, best.flatten(), method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        new_centers = result.x.reshape(-1, 3)
        new_score = _min_radius(new_centers)
        if new_score > best_score:
            best = new_centers
            best_score = new_score
    except Exception:
        pass
    
    return best, best_score


def _polish(x_init, max_iter=1500, time_limit=25.0):
    """High-precision polish: focus on active constraints with smaller steps."""
    x = x_init.copy().reshape(N, 3)
    x = np.clip(x, -0.49, 0.49)
    
    best_x = x.copy()
    best_score = _min_radius(x)
    
    t0 = time.time()
    
    for it in range(max_iter):
        if time.time() - t0 > time_limit:
            break
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        wall_dists = np.min(np.abs(0.5 - np.abs(x)), axis=1)
        
        min_pair_dist = np.min(dists)
        min_wall_dist = np.min(wall_dists)
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        threshold = current_r * 1.1
        moved = False
        
        # Move near-active pairs apart
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < 2.0 * threshold:
                    direction = x[i] - x[j]
                    norm = np.linalg.norm(direction)
                    if norm < 1e-15:
                        continue
                    direction /= norm
                    step = 0.002
                    x_new = x.copy()
                    x_new[i] += step * direction
                    x_new[j] -= step * direction
                    x_new = np.clip(x_new, -0.49, 0.49)
                    new_score = _min_radius(x_new)
                    if new_score > current_r + 1e-14:
                        x = x_new
                        moved = True
        
        # Move boundary-violating spheres inward
        for i in range(N):
            if wall_dists[i] < threshold:
                direction = -x[i]
                norm = np.linalg.norm(direction)
                if norm < 1e-15:
                    continue
                direction /= norm
                step = 0.002
                x_new = x.copy()
                x_new[i] += step * direction
                x_new = np.clip(x_new, -0.49, 0.49)
                new_score = _min_radius(x_new)
                if new_score > current_r + 1e-14:
                    x = x_new
                    moved = True
        
        if not moved:
            # Try smaller steps
            for i in range(N):
                for j in range(i + 1, N):
                    d = dists[i, j]
                    if d < 2.0 * threshold:
                        direction = x[i] - x[j]
                        norm = np.linalg.norm(direction)
                        if norm < 1e-15:
                            continue
                        direction /= norm
                        step = 0.0005
                        x_new = x.copy()
                        x_new[i] += step * direction
                        x_new[j] -= step * direction
                        x_new = np.clip(x_new, -0.49, 0.49)
                        new_score = _min_radius(x_new)
                        if new_score > current_r + 1e-15:
                            x = x_new
                            moved = True
            
            for i in range(N):
                if wall_dists[i] < threshold:
                    direction = -x[i]
                    norm = np.linalg.norm(direction)
                    if norm < 1e-15:
                        continue
                    direction /= norm
                    step = 0.0005
                    x_new = x.copy()
                    x_new[i] += step * direction
                    x_new = np.clip(x_new, -0.49, 0.49)
                    new_score = _min_radius(x_new)
                    if new_score > current_r + 1e-15:
                        x = x_new
                        moved = True
            
            if not moved:
                break
        
        current_score = _min_radius(x)
        if current_score > best_score:
            best_score = current_score
            best_x = x.copy()
    
    return best_x, best_score


def construct():
    t0 = time.time()
    deadline = T_LIMIT
    
    best_x = None
    best_score = -np.inf
    
    # Multiple initialization strategies
    inits = []
    
    # 1. Fibonacci lattice
    inits.append(_fibonacci_lattice_init(N, seed=42))
    
    # 2. Jittered grid
    inits.append(_grid_init(N, seed=0))
    
    # 3. Random
    inits.append(_random_init(N, seed=123))
    
    # 4. Best-first grid
    inits.append(_best_first_grid_init(N, seed=0))
    
    # 5. Another random
    inits.append(_random_init(N, seed=777))
    
    # 6. Fibonacci different seed
    inits.append(_fibonacci_lattice_init(N, seed=999))
    
    for init_idx, x_init in enumerate(inits):
        if time.time() > deadline - 40:
            break
        
        x_init = x_init.copy()
        x_init = np.clip(x_init, -0.48, 0.48)
        
        # Phase 1: Gradient-based L-BFGS-B with annealing
        remaining = deadline - time.time() - 15  # reserve 15s for polish
        if remaining < 5:
            continue
        x_refined, score = _optimize_from_init(x_init, time_limit=min(remaining, 25.0))
        
        if score > best_score:
            best_score = score
            best_x = x_refined.copy()
        
        # Phase 2: Local Nelder-Mead refinement
        if time.time() < deadline - 15:
            x_nm, score_nm = _local_refinement(x_refined, time_limit=10.0)
            if score_nm > best_score:
                best_score = score_nm
                best_x = x_nm.copy()
        
        # Phase 3: Polish
        if time.time() < deadline - 8:
            x_polished, score_polished = _polish(best_x, max_iter=1000, time_limit=8.0)
            if score_polished > best_score:
                best_score = score_polished
                best_x = x_polished.copy()
    
    # Final polish on the best result
    if best_x is not None and time.time() < deadline - 3:
        x_final, score_final = _polish(best_x, max_iter=500, time_limit=3.0)
        if score_final > best_score:
            best_score = score_final
            best_x = x_final.copy()
    
    if best_x is None:
        best_x = _grid_init(N)
    
    # Final validity check
    best_x = np.clip(best_x, -0.5, 0.5)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < 1e-10:
                best_x[j] += 1e-5 * np.array([1, 0, 0])
    
    return best_x
