"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _post_process(pos, radii, max_iter=300):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(max_iter):
        valid = True
        for i in range(N):
            if pos[i, 0] < radii[i]:
                radii[i] = max(0.001, pos[i, 0] * 0.99)
                valid = False
            if pos[i, 0] > 1.0 - radii[i]:
                radii[i] = min(radii[i], max(0.001, (1.0 - pos[i, 0]) * 0.99))
                valid = False
            if pos[i, 1] < radii[i]:
                radii[i] = min(radii[i], max(0.001, pos[i, 1] * 0.99))
                valid = False
            if pos[i, 1] > 1.0 - radii[i]:
                radii[i] = min(radii[i], max(0.001, (1.0 - pos[i, 1]) * 0.99))
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    scale = (dist / (min_dist + 1e-12)) * 0.99
                    radii[i] = max(0.001, radii[i] * scale)
                    radii[j] = max(0.001, radii[j] * scale)
                    valid = False
        
        if valid:
            break
    
    return pos, radii


def _compute_max_radii(pos, radii):
    """For each circle, compute the maximum radius it could have given current positions."""
    max_r = np.zeros(N)
    for i in range(N):
        # Boundary constraint
        mr = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        # Neighbor constraint
        for j in range(N):
            if i == j:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            mr = min(mr, dist - radii[j])
        max_r[i] = max(mr, 0.001)
    return max_r


def _voronoi_radius_assignment(pos, radii):
    """
    Compute radius for each circle based on its nearest neighbors (Voronoi-like).
    A circle's max radius is limited by its closest neighbor and boundaries.
    """
    new_radii = np.zeros(N)
    for i in range(N):
        # Distance to boundaries
        d_bound = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        # Distance to nearest neighbor (excluding self)
        min_dist_neighbor = np.inf
        for j in range(N):
            if i == j:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            # The constraint is dist >= r_i + r_j, so r_i <= dist - r_j
            # But we don't know r_j yet, so use a heuristic: assume equal split
            # r_i <= dist / 2 (if neighbors are same size)
            # More precisely, r_i + r_j <= dist, so if we want to maximize sum,
            # we should consider the tightest constraint
            candidate = dist - radii[j]  # current radius of j
            min_dist_neighbor = min(min_dist_neighbor, candidate)
        
        new_radii[i] = min(d_bound, min_dist_neighbor)
        new_radii[i] = max(new_radii[i], 0.001)
    
    return new_radii


def _make_hexagonal(seed=42):
    """Create a hexagonal close-packing initial configuration."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Hexagonal arrangement: rows offset
    r_base = 0.09
    row_spacing = r_base * np.sqrt(3)
    col_spacing = 2 * r_base
    
    idx = 0
    row = 0
    while idx < N:
        y = r_base + row * row_spacing
        if y > 1.0 - r_base:
            break
        if row % 2 == 0:
            x_start = r_base
        else:
            x_start = r_base + col_spacing / 2
        
        col = 0
        while True:
            x = x_start + col * col_spacing
            if x > 1.0 - r_base:
                break
            if idx < N:
                pos[idx] = [x, y]
                radii[idx] = r_base
                idx += 1
            col += 1
        row += 1
    
    # Fill remaining with smaller circles in gaps
    while idx < N:
        # Place in a gap between hexagonal circles
        gap_r = r_base * 0.4
        pos[idx] = [0.25 + rng.uniform(-0.02, 0.02), 0.25 + rng.uniform(-0.02, 0.02)]
        radii[idx] = gap_r
        idx += 1
    
    return pos, radii


def _make_grid(seed=42):
    """Create a grid initial configuration."""
    rng = np.random.RandomState(seed)
    side = 5
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx < N:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                radii[idx] = 0.5 / side - 0.005
    if N > 25:
        pos[25] = [0.5, 0.05]
        radii[25] = 0.04
    return pos, radii


def _make_random(seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(N, 2))
    radii = rng.uniform(0.03, 0.1, size=N)
    return pos, radii


def _smooth_objective(x, beta=80.0, smooth_eps=0.003):
    """Smooth objective using log-sum-exp for soft-max of violations."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    violations = []
    
    # Boundary violations
    for i in range(n):
        for coord in range(2):
            v_left = radii[i] - pos[i, coord]
            v_right = pos[i, coord] - (1.0 - radii[i])
            violations.append(v_left)
            violations.append(v_right)
    
    # Overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = radii[i] + radii[j]
            dist_smooth = np.sqrt(dist_sq + smooth_eps * smooth_eps)
            v = min_dist - dist_smooth + smooth_eps
            violations.append(v)
    
    if violations:
        arr = np.array(violations)
        max_v = np.max(arr)
        if max_v > 1e-10:
            penalty = max_v + smooth_eps * np.log(np.sum(np.exp((arr - max_v) / smooth_eps)))
        else:
            penalty = 0.0
    else:
        penalty = 0.0
    
    return -np.sum(radii) + beta * penalty


def _smooth_objective_grad(x, beta=80.0, smooth_eps=0.003):
    """Smooth objective with gradient."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    violations = []
    # Store indices for gradient computation
    # For each violation, store which variables it depends on and the gradient
    
    # We'll compute the gradient numerically for simplicity but analytically for the main terms
    # Actually, let's do it analytically
    
    # Collect violation values and their gradients
    v_vals = []
    v_grads = []  # each is a 3n-vector
    
    # Boundary violations
    for i in range(n):
        for coord in range(2):
            # v_left = r_i - pos[i, coord]
            g = np.zeros(3*n)
            g[2*n+i] = 1.0
            g[2*i+coord] = -1.0
            v_vals.append(radii[i] - pos[i, coord])
            v_grads.append(g)
            
            # v_right = pos[i, coord] - (1 - r_i)
            g2 = np.zeros(3*n)
            g2[2*n+i] = 1.0
            g2[2*i+coord] = 1.0
            v_vals.append(pos[i, coord] - (1.0 - radii[i]))
            v_grads.append(g2)
    
    # Overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            dist_smooth = np.sqrt(dist_sq + smooth_eps * smooth_eps)
            min_dist = radii[i] + radii[j]
            v = min_dist - dist_smooth + smooth_eps
            
            g = np.zeros(3*n)
            g[2*n+i] = 1.0
            g[2*n+j] = 1.0
            # d/dpos of -dist_smooth
            if dist_smooth > 1e-10:
                g[2*i] = -dx / dist_smooth
                g[2*i+1] = -dy / dist_smooth
                g[2*j] = dx / dist_smooth
                g[2*j+1] = dy / dist_smooth
            
            v_vals.append(v)
            v_grads.append(g)
    
    v_arr = np.array(v_vals)
    v_grad_mat = np.array(v_grads)  # shape (n_v, 3n)
    
    max_v = np.max(v_arr)
    if max_v > 1e-10:
        # log-sum-exp
        exps = np.exp((v_arr - max_v) / smooth_eps)
        sum_exps = np.sum(exps)
        log_sum = np.log(sum_exps)
        
        # Penalty = max_v + smooth_eps * log(sum_exps)
        # dPenalty/dx = smooth_eps * (1/sum_exps) * sum_k (exp((v_k - max_v)/eps) / eps) * dv_k/dx
        #             = (1/sum_exps) * sum_k exp((v_k - max_v)/eps) * dv_k/dx
        
        weights = exps / sum_exps  # shape (n_v,)
        penalty_grad = weights @ v_grad_mat  # shape (3n,)
        penalty = max_v + smooth_eps * log_sum
    else:
        penalty = 0.0
        penalty_grad = np.zeros(3*n)
    
    obj = -np.sum(radii) + beta * penalty
    grad = np.zeros(3*n)
    grad[2*n:] = -1.0
    grad += beta * penalty_grad
    
    return obj, grad


def _run_smooth_optimization(x0, deadline, max_iter=500):
    """Run L-BFGS-B with smooth penalty objective."""
    bounds = []
    for i in range(N):
        bounds.append((0.002, 0.998))
        bounds.append((0.002, 0.998))
    for i in range(N):
        bounds.append((0.002, 0.40))
    
    try:
        result = minimize(
            lambda x: _smooth_objective_grad(x)[0],
            x0,
            jac=lambda x: _smooth_objective_grad(x)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        
        x_opt = result.x
        pos_opt = x_opt[:2*N].reshape(N, 2)
        radii_opt = x_opt[2*N:]
        
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp
    except Exception:
        pass
    
    return None, None


def _local_search(pos, radii, n_restarts=3, deadline=None):
    """Local search with random perturbations and smooth optimization."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    rng = np.random.RandomState(777)
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb positions
        pos_p = pos.copy()
        radii_p = radii.copy()
        
        # Random small perturbation
        scale = 0.03 / (restart + 1)
        pos_p += rng.normal(0, scale, size=pos_p.shape)
        pos_p = np.clip(pos_p, 0.01, 0.99)
        
        # Try to grow radii
        for i in range(N):
            max_r = min(pos_p[i, 0], 1.0 - pos_p[i, 0], pos_p[i, 1], 1.0 - pos_p[i, 1])
            for j in range(N):
                if i == j:
                    continue
                dx = pos_p[i, 0] - pos_p[j, 0]
                dy = pos_p[i, 1] - pos_p[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r = min(max_r, dist - radii_p[j])
            radii_p[i] = max(radii_p[i], max_r * 0.8)
            radii_p[i] = max(radii_p[i], 0.002)
        
        x0 = np.concatenate([pos_p.flatten(), radii_p])
        result_pos, result_radii = _run_smooth_optimization(x0, deadline, max_iter=300)
        
        if result_pos is not None:
            score = np.sum(result_radii)
            if score > best_score:
                best_score = score
                best_pos = result_pos.copy()
                best_radii = result_radii.copy()
    
    return best_pos, best_radii


def _force_directed_anneal(pos, radii, n_iter=1500, deadline=None):
    """Force-directed annealing to explore topology and improve packing."""
    pos = pos.copy()
    radii = radii.copy()
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    vel = np.zeros_like(pos)
    rng = np.random.RandomState(123)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        temp = 0.012 * np.exp(-0.002 * it)
        forces = np.zeros_like(pos)
        
        # Boundary forces
        for i in range(N):
            margin = 0.003
            if pos[i, 0] < radii[i] + margin:
                forces[i, 0] += 15.0 * (radii[i] + margin - pos[i, 0])
            if pos[i, 0] > 1.0 - radii[i] - margin:
                forces[i, 0] -= 15.0 * (pos[i, 0] - (1.0 - radii[i] - margin))
            if pos[i, 1] < radii[i] + margin:
                forces[i, 1] += 15.0 * (radii[i] + margin - pos[i, 1])
            if pos[i, 1] > 1.0 - radii[i] - margin:
                forces[i, 1] -= 15.0 * (pos[i, 1] - (1.0 - radii[i] - margin))
        
        # Overlap forces
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 0.0
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-6:
                    overlap = min_dist - dist
                    force_mag = 20.0 * overlap
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                elif dist < min_dist * 1.5 and dist > 1e-6:
                    force_mag = 3.0 * (min_dist * 1.5 - dist)
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        vel = 0.8 * vel + 0.2 * forces
        pos += vel * 0.006 + rng.normal(0, temp, size=pos.shape)
        
        # Clip positions
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], 0.005, 0.995)
            pos[i, 1] = np.clip(pos[i, 1], 0.005, 0.995)
        
        # Grow radii where possible
        for i in range(N):
            max_grow = min(pos[i, 0] - radii[i], 1.0 - pos[i, 0] - radii[i],
                          pos[i, 1] - radii[i], 1.0 - pos[i, 1] - radii[i])
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                available = dist - radii[j]
                if available < max_grow:
                    max_grow = available
            
            if max_grow > 0.0003:
                radii[i] += max_grow * 0.5
        
        # Ensure minimum radius
        radii = np.maximum(radii, 0.002)
        
        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    return best_pos, best_radii


def _polish(pos, radii, deadline=None):
    """High-precision polish using smooth optimization with tight tolerances."""
    if deadline is None:
        deadline = time.time() + 20
    
    x0 = np.concatenate([pos.flatten(), radii])
    result_pos, result_radii = _run_smooth_optimization(x0, deadline, max_iter=800)
    if result_pos is not None and np.sum(result_radii) > np.sum(radii):
        return result_pos, result_radii
    
    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    strategies = ['hexagonal', 'grid', 'random']
    
    # Phase 1: Multi-start with smooth optimization
    for strategy in strategies:
        if time.time() > deadline - 50:
            break
        
        for seed in range(4):
            if time.time() > deadline - 45:
                break
            
            if strategy == 'hexagonal':
                pos, radii = _make_hexagonal(seed)
            elif strategy == 'grid':
                pos, radii = _make_grid(seed)
            else:
                pos, radii = _make_random(seed)
            
            x0 = np.concatenate([pos.flatten(), radii])
            
            # Try smooth optimization
            result_pos, result_radii = _run_smooth_optimization(x0, deadline, max_iter=400)
            if result_pos is not None:
                score = np.sum(result_radii)
                if score > best_score:
                    best_score = score
                    best_pos = result_pos.copy()
                    best_radii = result_radii.copy()
    
    # Phase 2: Local search from best result
    if best_pos is not None and time.time() < deadline - 40:
        pos_ls, radii_ls = _local_search(best_pos, best_radii, n_restarts=4, deadline=deadline - 35)
        if _is_valid(pos_ls, radii_ls) and np.sum(radii_ls) > best_score:
            best_score = np.sum(radii_ls)
            best_pos = pos_ls.copy()
            best_radii = radii_ls.copy()
    
    # Phase 3: Force-directed annealing
    if best_pos is not None and time.time() < deadline - 30:
        try:
            pos_fd, radii_fd = _force_directed_anneal(best_pos, best_radii, n_iter=1500, deadline=deadline - 25)
            score_fd = np.sum(radii_fd)
            if _is_valid(pos_fd, radii_fd) and score_fd > best_score:
                best_score = score_fd
                best_pos = pos_fd.copy()
                best_radii = radii_fd.copy()
        except Exception:
            pass
    
    # Phase 4: Another local search after annealing
    if best_pos is not None and time.time() < deadline - 20:
        pos_ls2, radii_ls2 = _local_search(best_pos, best_radii, n_restarts=3, deadline=deadline - 15)
        if _is_valid(pos_ls2, radii_ls2) and np.sum(radii_ls2) > best_score:
            best_score = np.sum(radii_ls2)
            best_pos = pos_ls2.copy()
            best_radii = radii_ls2.copy()
    
    # Phase 5: Final high-precision polish
    if best_pos is not None and time.time() < deadline - 8:
        pos_pol, radii_pol = _polish(best_pos, best_radii, deadline=deadline - 3)
        score_pol = np.sum(radii_pol)
        if _is_valid(pos_pol, radii_pol) and score_pol > best_score:
            best_score = score_pol
            best_pos = pos_pol.copy()
            best_radii = radii_pol.copy()
    
    # Fallback
    if best_pos is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.005) for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        return result
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
