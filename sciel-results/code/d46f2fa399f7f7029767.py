import numpy as np
import scipy.optimize as opt
import time

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

def _compute_radius(pts):
    """Compute the maximum radius for given centers."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    d = np.minimum(
        np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0]),
        np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    )
    min_wall = np.min(d)
    return min(min_pair, min_wall)

def _soft_min_obj_and_grad(pts, beta):
    """
    Soft-min objective: f = -softmin(D) where D includes all pairwise and wall distances.
    softmin(D) = -1/beta * log(sum(exp(-beta * D)))
    f = 1/beta * log(sum(exp(-beta * D)))
    We minimize f to maximize the soft-min of distances.
    """
    pts = pts.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    tri_u = np.triu_indices(N, k=1)
    pair_dists = dists[tri_u]
    
    # Wall distances
    d_left = pts[:, 0] + 0.5
    d_right = 0.5 - pts[:, 0]
    d_bottom = pts[:, 1] + 0.5
    d_top = 0.5 - pts[:, 1]
    wall_dists = np.concatenate([d_left, d_right, d_bottom, d_top])
    
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Numerically stable log-sum-exp
    min_all = np.min(all_dists)
    shifted = all_dists - min_all
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp < 1e-300:
        sum_exp = 1e-300
    log_sum = np.log(sum_exp)
    
    # Objective: f = -min_all + (1/beta) * log_sum
    obj_val = -min_all + (1.0 / beta) * log_sum
    
    # Gradient: grad f = -sum(w_k * grad D_k)
    # w_k = exp(-beta * (D_k - min_all)) / sum_exp
    w_all = exp_vals / sum_exp
    w_pairs = w_all[:len(pair_dists)]
    w_walls = w_all[len(pair_dists):]
    
    grad_pts = np.zeros_like(pts)
    
    # Pairwise gradient contributions
    i_idx, j_idx = tri_u
    pair_diff = diff[i_idx, j_idx]
    pair_dist_safe = np.where(pair_dists > 1e-12, pair_dists, 1.0)
    unit_vecs = pair_diff / pair_dist_safe[:, np.newaxis]
    weighted = w_pairs[:, np.newaxis] * unit_vecs
    np.add.at(grad_pts, i_idx, weighted)
    np.add.at(grad_pts, j_idx, -weighted)
    
    # Wall gradient contributions
    grad_pts[:, 0] += w_walls[0:N]       # left wall: grad (1,0)
    grad_pts[:, 0] -= w_walls[N:2*N]     # right wall: grad (-1,0)
    grad_pts[:, 1] += w_walls[2*N:3*N]   # bottom wall: grad (0,1)
    grad_pts[:, 1] -= w_walls[3*N:4*N]   # top wall: grad (0,-1)
    
    # Return gradient of f (we minimize f)
    return obj_val, -grad_pts.flatten()

def _violation(pts, r):
    """Compute max violation: max(0, 2r - dist_ij, 2r - wall_dist_i)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_viol = 2 * r - dists
    np.fill_diagonal(pair_viol, 0)
    
    wall_dists = np.minimum(
        np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0]),
        np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    )
    wall_viol = 2 * r - wall_dists
    
    return np.max(np.maximum(0, np.max(pair_viol), np.max(wall_viol)))

def _soft_violation_obj_and_grad(pts, r, beta):
    """
    Soft violation objective: minimize softmax of violations.
    v_ij = max(0, 2r - dist_ij) for pairs
    v_i = max(0, 2r - wall_dist_i) for walls
    obj = (1/beta) * log(sum(exp(beta * v_k)))
    """
    pts = pts.reshape(N, 2)
    
    # Pairwise violations
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    tri_u = np.triu_indices(N, k=1)
    pair_dists = dists[tri_u]
    pair_viol = np.maximum(0, 2 * r - pair_dists)
    
    # Wall violations
    d_left = pts[:, 0] + 0.5
    d_right = 0.5 - pts[:, 0]
    d_bottom = pts[:, 1] + 0.5
    d_top = 0.5 - pts[:, 1]
    wall_dists = np.concatenate([d_left, d_right, d_bottom, d_top])
    wall_viol = np.maximum(0, 2 * r - wall_dists)
    
    all_viol = np.concatenate([pair_viol, wall_viol])
    
    # Softmax of violations
    max_v = np.max(all_viol)
    if max_v < 1e-15:
        return 0.0, np.zeros(2 * N)
    shifted = all_viol - max_v
    exp_vals = np.exp(beta * shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp < 1e-300:
        sum_exp = 1e-300
    log_sum = np.log(sum_exp)
    obj_val = max_v + (1.0 / beta) * log_sum
    
    # Gradient
    w_all = exp_vals / sum_exp
    w_pairs = w_all[:len(pair_viol)]
    w_walls = w_all[len(pair_viol):]
    
    grad_pts = np.zeros_like(pts)
    
    # Only non-zero violations contribute
    pair_active = pair_viol > 1e-12
    if np.any(pair_active):
        i_idx, j_idx = tri_u
        active_pairs = np.where(pair_active)[0]
        if len(active_pairs) > 0:
            pair_diff = diff[i_idx[active_pairs], j_idx[active_pairs]]
            pair_dist_safe = np.where(pair_dists[active_pairs] > 1e-12, pair_dists[active_pairs], 1.0)
            unit_vecs = pair_diff / pair_dist_safe[:, np.newaxis]
            # d(2r - dist)/d pts[i] = -unit_vec, d(2r - dist)/d pts[j] = unit_vec
            weighted = w_pairs[active_pairs][:, np.newaxis] * (-unit_vecs)
            np.add.at(grad_pts, i_idx[active_pairs], weighted)
            np.add.at(grad_pts, j_idx[active_pairs], -weighted)
    
    # Wall gradients
    # d(2r - d_left)/dx = -1, d(2r - d_right)/dx = +1, etc.
    wall_active = wall_viol > 1e-12
    if np.any(wall_active):
        active_walls = np.where(wall_active)[0]
        if len(active_walls) > 0:
            w_act = w_walls[active_walls]
            # left: 0..N-1, grad_x = -1
            left_act = active_walls < N
            if np.any(left_act):
                idx = active_walls[left_act]
                grad_pts[idx, 0] += -w_act[left_act]
            # right: N..2N-1, grad_x = +1
            right_act = (active_walls >= N) & (active_walls < 2*N)
            if np.any(right_act):
                idx = active_walls[right_act] - N
                grad_pts[idx, 0] += w_act[right_act]
            # bottom: 2N..3N-1, grad_y = -1
            bot_act = (active_walls >= 2*N) & (active_walls < 3*N)
            if np.any(bot_act):
                idx = active_walls[bot_act] - 2*N
                grad_pts[idx, 1] += -w_act[bot_act]
            # top: 3N..4N-1, grad_y = +1
            top_act = active_walls >= 3*N
            if np.any(top_act):
                idx = active_walls[top_act] - 3*N
                grad_pts[idx, 1] += w_act[top_act]
    
    return obj_val, grad_pts.flatten()

def _gradient_polish(pts, r_target, time_budget=5.0):
    """Polish using soft-min with increasing beta."""
    start = time.time()
    current_pts = pts.copy()
    
    for beta in [200, 500, 1000, 2000, 5000]:
        if time.time() - start > time_budget:
            break
        
        def obj(x):
            v, _ = _soft_min_obj_and_grad(x, beta)
            return v
        
        def grad(x):
            _, g = _soft_min_obj_and_grad(x, beta)
            return g
        
        try:
            res = opt.minimize(
                obj, current_pts.flatten(),
                method='L-BFGS-B',
                jac=grad,
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            new_pts = res.x.reshape(N, 2)
            new_r = _compute_radius(new_pts)
            if new_r >= _compute_radius(current_pts) - 1e-12:
                current_pts = new_pts
        except:
            pass
    
    return current_pts

def _sa_search(pts0, r_target, time_budget=30.0):
    """Simulated annealing to find a valid configuration for radius r_target."""
    start = time.time()
    current = pts0.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    T = 0.05
    T_min = 1e-6
    alpha = 0.98
    n_iter = 0
    
    while time.time() - start < time_budget:
        n_iter += 1
        # Pick random point and perturb
        i = np.random.randint(N)
        delta = np.random.normal(0, T, 2)
        new_pt = current[i].copy()
        new_pt += delta
        # Keep in bounds
        new_pt[0] = np.clip(new_pt[0], -0.5, 0.5)
        new_pt[1] = np.clip(new_pt[1], -0.5, 0.5)
        
        new_config = current.copy()
        new_config[i] = new_pt
        new_r = _compute_radius(new_config)
        
        # Acceptance: prefer larger r
        delta_r = new_r - current_r
        if delta_r > 0 or np.random.rand() < np.exp(delta_r / T):
            current = new_config
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
        
        T = max(T * alpha, T_min)
    
    return best, best_r

def _binary_search_radius(pts0, time_budget=60.0):
    """Binary search for maximum radius using violation minimization."""
    start = time.time()
    
    # Initial bracket
    r_lo = 0.05
    r_hi = 0.08
    
    # Check if r_hi is feasible
    def check_feasible(r, pts_init, time_left):
        """Try to find valid config for radius r."""
        if time_left < 5:
            return False, pts_init
        
        # Try gradient-based violation minimization
        beta = 100.0
        current_pts = pts_init.copy()
        
        def obj(x):
            v, _ = _soft_violation_obj_and_grad(x, r, beta)
            return v
        
        def grad(x):
            _, g = _soft_violation_obj_and_grad(x, r, beta)
            return g
        
        try:
            res = opt.minimize(
                obj, current_pts.flatten(),
                method='L-BFGS-B',
                jac=grad,
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            new_pts = res.x.reshape(N, 2)
            viol = _violation(new_pts, r)
            if viol < 1e-8:
                return True, new_pts
        except:
            pass
        
        # Try SA if time permits
        if time.time() - start < time_budget - 10:
            sa_pts, sa_r = _sa_search(current_pts, r, time_budget=10.0)
            if _violation(sa_pts, r) < 1e-8:
                return True, sa_pts
        
        return False, current_pts
    
    feasible, pts_feasible = check_feasible(r_hi, pts0, time_budget)
    if not feasible:
        # Binary search down
        for _ in range(20):
            if time.time() - start > time_budget - 15:
                break
            r_mid = (r_lo + r_hi) / 2
            feasible, pts_mid = check_feasible(r_mid, pts0, time_budget - (time.time() - start))
            if feasible:
                r_lo = r_mid
                pts_feasible = pts_mid
            else:
                r_hi = r_mid
        
        best_r = r_lo
        best_pts = pts_feasible
    else:
        best_r = r_hi
        best_pts = pts_feasible
    
    # Polish the best found
    if time.time() - start < time_budget - 10:
        polished = _gradient_polish(best_pts, best_r, time_budget=time_budget - (time.time() - start) - 5)
        polished_r = _compute_radius(polished)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished
    
    return best_pts, best_r

def construct():
    start_time = time.time()
    
    best_points = None
    best_r = -np.inf
    
    # Multiple initialization strategies
    def init_hexagonal():
        s = 0.15
        pts = []
        dy = s * np.sqrt(3) / 2
        row = 0
        while len(pts) < N + 10:
            y = -0.5 + dy/2 + row * dy
            if y > 0.5:
                break
            x_offset = 0 if row % 2 == 0 else s/2
            col = 0
            while True:
                x = -0.5 + s/2 + x_offset + col * s
                if x > 0.5:
                    break
                pts.append([x, y])
                col += 1
            row += 1
        pts = np.array(pts[:N])
        pts += np.random.uniform(-0.005, 0.005, pts.shape)
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        return pts
    
    def init_random():
        return np.random.uniform(-0.5, 0.5, (N, 2))
    
    def init_grid():
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N].copy()
    
    # Phase 1: Global search with SA and gradient polish
    inits = [init_hexagonal(), init_random(), init_grid(), init_random()]
    
    for idx, pts0 in enumerate(inits):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # SA search
        sa_time = min(15.0, DEADLINE - (time.time() - start_time) - 20)
        if sa_time > 3:
            sa_pts, sa_r = _sa_search(pts0, 0.07, sa_time)
            if sa_r > best_r:
                best_r = sa_r
                best_points = sa_pts.copy()
        
        # Gradient polish from SA result
        if time.time() - start_time < DEADLINE - 10:
            polish_time = min(5.0, DEADLINE - (time.time() - start_time) - 5)
            if polish_time > 1:
                polished = _gradient_polish(sa_pts, sa_r, polish_time)
                pol_r = _compute_radius(polished)
                if pol_r > best_r:
                    best_r = pol_r
                    best_points = polished.copy()
    
    # Phase 2: Binary search on radius with violation minimization
    if time.time() - start_time < DEADLINE - 20 and best_points is not None:
        bs_time = min(40.0, DEADLINE - (time.time() - start_time) - 10)
        bs_pts, bs_r = _binary_search_radius(best_points, bs_time)
        if bs_r > best_r:
            best_r = bs_r
            best_points = bs_pts.copy()
    
    # Phase 3: Final high-precision polish
    if time.time() - start_time < DEADLINE - 5 and best_points is not None:
        final_time = min(8.0, DEADLINE - (time.time() - start_time) - 2)
        if final_time > 1:
            final_pts = _gradient_polish(best_points, best_r, final_time)
            final_r = _compute_radius(final_pts)
            if final_r > best_r:
                best_r = final_r
                best_points = final_pts.copy()
    
    # Fallback
    if best_points is None:
        best_points = init_grid()
    
    # Final safety: ensure valid
    final_r = _compute_radius(best_points)
    if not np.isfinite(final_r) or final_r <= 0:
        best_points = init_grid()
    
    return best_points
