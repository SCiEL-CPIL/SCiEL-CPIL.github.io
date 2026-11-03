"""Packing 50 equal circles in a unit circle using simulated annealing with hexagonal init."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # safety margin before 150s kill


def _pairwise_indices(n):
    """Return array of (i, j) pairs for i < j."""
    iu = np.triu_indices(n, k=1)
    return np.column_stack((iu[0], iu[1]))


def _score_vec(points):
    """Vectorized score computation."""
    points = points.reshape(-1, 2)
    n = len(points)
    radii = 1.0 - np.linalg.norm(points, axis=1)
    r = np.min(radii)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Only consider i < j
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    r = min(r, np.min(pair_dists))
    return r


def _fast_score(points, r_target):
    """Fast O(N) approximate score for SA: only checks violations vs r_target."""
    points = points.reshape(-1, 2)
    n = len(points)
    # Boundary check
    radii = 1.0 - np.linalg.norm(points, axis=1)
    if np.min(radii) < r_target - 1e-12:
        return np.min(radii)
    # Pairwise: only need to check if any pair is too close
    # Use vectorized check
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    return min(np.min(pair_dists), np.min(radii))


def _hexagonal_init(n):
    """Generate a hexagonal lattice initial configuration."""
    best_points = None
    best_r = 0.0
    for spacing in np.linspace(0.05, 0.25, 50):
        s = spacing
        points = []
        max_coord = int(1.0 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - s / 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n]
            cand = points[idx]
            r = _score_vec(cand)
            if r > best_r:
                best_r = r
                best_points = cand.copy()
    if best_points is None:
        best_points = np.random.randn(n, 2) * 0.5
    return best_points


def _local_optimize(c, time_limit=10.0, beta_schedule=None):
    """Local optimization using L-BFGS-B with smooth-min objective."""
    pair_idx = _pairwise_indices(N)
    n_pairs = len(pair_idx)
    
    if beta_schedule is None:
        beta_schedule = [150.0, 300.0, 600.0, 1000.0]
    
    best_c = c.copy()
    best_r = _score_vec(best_c)
    
    t_start = time.time()
    
    for beta in beta_schedule:
        if time.time() - t_start >= time_limit - 2:
            break
        
        def neg_soft_min(x):
            c_local = x.reshape(N, 2)
            ci = c_local[pair_idx[:, 0]]
            cj = c_local[pair_idx[:, 1]]
            d = np.sqrt(np.sum((ci - cj) ** 2, axis=1)) / 2
            b = 1 - np.hypot(c_local[:, 0], c_local[:, 1])
            all_vals = np.concatenate([d, b])
            # Numerically stable log-sum-exp
            max_val = np.max(all_vals)
            return max_val - np.log(np.sum(np.exp(beta * (all_vals - max_val)))) / beta
        
        def callback(x):
            if time.time() - t_start >= time_limit - 1:
                raise Exception("timeout")
        
        try:
            res = minimize(
                neg_soft_min,
                best_c.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-18, 'gtol': 1e-16},
                callback=callback
            )
            if res.success and np.all(np.isfinite(res.x)):
                cand = res.x.reshape(N, 2)
                r = _score_vec(cand)
                if r > best_r:
                    best_r = r
                    best_c = cand.copy()
        except Exception:
            pass
    
    return best_c, best_r


def _anneal(points, t_start, deadline, seed=None):
    """Simulated annealing to escape local minima with fast vectorized cost."""
    if seed is not None:
        np.random.seed(seed)
    
    n = len(points)
    pts = points.copy()
    best_pts = pts.copy()
    best_score = _score_vec(best_pts)
    
    # Precompute pairwise indices for vectorized cost
    iu = np.triu_indices(n, 1)
    
    def cost(p):
        """Vectorized cost: sum of squared violations vs best_score."""
        p = p.reshape(-1, 2)
        r_target = best_score
        # Boundary violations
        boundary = 1.0 - np.linalg.norm(p, axis=1)
        b_viol = np.maximum(0, r_target - boundary)
        # Pairwise violations
        ci = p[iu[0]]
        cj = p[iu[1]]
        d = np.linalg.norm(ci - cj, axis=1)
        p_viol = np.maximum(0, 2 * r_target - d)
        return np.sum(b_viol ** 2) + np.sum(p_viol ** 2)
    
    # SA parameters
    T = 0.03  # initial temperature
    T_min = 0.00005
    cooling = 0.9998
    step_size = 0.015
    
    current_cost = cost(pts)
    
    while time.time() < deadline:
        if T < T_min:
            break
        
        # Propose a move: random perturbation of one or two points
        n_moves = np.random.randint(1, 3)
        new_pts = pts.copy()
        for _ in range(n_moves):
            idx = np.random.randint(n)
            delta = np.random.randn(2) * step_size * (1 + T)
            new_pts[idx] += delta
        
        new_cost = cost(new_pts)
        delta_cost = new_cost - current_cost
        
        # Accept with probability exp(-delta_cost / T) if delta_cost > 0
        if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / T):
            pts = new_pts
            current_cost = new_cost
            # Update best periodically (every ~100 iterations to save time)
            s = _fast_score(pts, best_score)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        
        T *= cooling
    
    return best_pts, best_score


def _compute_r(c):
    """Compute the max common radius for given centers."""
    return _score_vec(c)


def construct():
    t_start = time.time()
    
    # Start with hexagonal initialization
    points = _hexagonal_init(N)
    best_points = points.copy()
    best_score = _score_vec(best_points)
    
    # First pass: local optimization from hex init with multiple betas
    if time.time() < DEADLINE - 25:
        c_opt, r_opt = _local_optimize(points, time_limit=min(15.0, DEADLINE - (time.time() - t_start)))
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
    
    # Simulated annealing to escape local minima
    remaining = DEADLINE - (time.time() - t_start)
    if remaining > 30:
        sa_points, sa_score = _anneal(best_points, time.time(), time.time() + min(remaining - 15, 50), seed=42)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
        
        # Local optimize after SA
        if time.time() < DEADLINE - 15:
            c_opt, r_opt = _local_optimize(best_points, time_limit=min(15.0, DEADLINE - (time.time() - t_start)))
            if r_opt > best_score:
                best_score = r_opt
                best_points = c_opt.copy()
    
    # Multi-start: perturb best and try again
    n_restarts = 0
    while time.time() < DEADLINE - 15:
        perturbation_scale = 0.02 / (1.0 + 0.1 * n_restarts)
        x0 = best_points.flatten() + np.random.randn(2 * N) * perturbation_scale
        
        n_restarts += 1
        
        try:
            c_opt, r_opt = _local_optimize(x0.reshape(N, 2), time_limit=min(10.0, DEADLINE - (time.time() - t_start)))
            if r_opt > best_score:
                best_score = r_opt
                best_points = c_opt.copy()
        except Exception:
            pass
        
        if n_restarts >= 5:
            break
    
    # Final polish with high beta
    if time.time() < DEADLINE - 5:
        c_final, r_final = _local_optimize(best_points, time_limit=min(10.0, DEADLINE - (time.time() - t_start) - 3), beta_schedule=[500.0, 1000.0, 2000.0])
        if r_final > best_score:
            best_points = c_final
            best_score = r_final
    
    # Ultra-final exact penalty-based polish: maximize r subject to constraints
    if time.time() < DEADLINE - 2:
        def _exact_penalty_polish(c_init, r_init, t_limit):
            """Final polish: fix r and optimize positions to satisfy all constraints."""
            t_s = time.time()
            pair_idx = _pairwise_indices(N)
            best_c = c_init.copy()
            best_r = r_init
            
            for r_try in [r_init, r_init + 1e-7, r_init + 5e-7, r_init + 1e-6]:
                if time.time() - t_s > t_limit - 1:
                    break
                
                def penalty_obj(x):
                    c_local = x.reshape(N, 2)
                    # Boundary penalty
                    b = 1 - np.hypot(c_local[:, 0], c_local[:, 1])
                    b_pen = np.sum(np.maximum(0, r_try - b) ** 2)
                    # Pairwise penalty
                    ci = c_local[pair_idx[:, 0]]
                    cj = c_local[pair_idx[:, 1]]
                    d = np.linalg.norm(ci - cj, axis=1)
                    p_pen = np.sum(np.maximum(0, 2 * r_try - d) ** 2)
                    return b_pen + p_pen
                
                def cb(x):
                    if time.time() - t_s > t_limit - 0.5:
                        raise Exception("timeout")
                
                try:
                    res = minimize(
                        penalty_obj,
                        best_c.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-20, 'gtol': 1e-18},
                        callback=cb
                    )
                    if np.all(np.isfinite(res.x)):
                        cand = res.x.reshape(N, 2)
                        r = _score_vec(cand)
                        if r > best_r:
                            best_r = r
                            best_c = cand.copy()
                except Exception:
                    pass
            
            return best_c, best_r
        
        c_polish, r_polish = _exact_penalty_polish(
            best_points, best_score, 
            min(6.0, DEADLINE - (time.time() - t_start) - 2)
        )
        if r_polish > best_score:
            best_points = c_polish
            best_score = r_polish
    
    # Ensure validity: clamp points that might be slightly outside
    c_final = best_points.reshape(N, 2)
    r2 = np.hypot(c_final[:, 0], c_final[:, 1])
    for i in range(N):
        if r2[i] > 1 - best_score + 1e-10:
            scale = (1 - best_score) / r2[i]
            c_final[i] *= scale
    
    # Final validation: recompute score after clamping
    r_check = _score_vec(c_final)
    if r_check < best_score:
        # Clamping reduced score; use the clamped points but note the score
        best_score = r_check
    
    return c_final
