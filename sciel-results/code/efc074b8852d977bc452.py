import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s out of 150s
    
    best_result = None
    best_score = -np.inf
    
    def penalty_function(params):
        """Penalty function to minimize: -sum(r) + overlap_penalty + boundary_penalty"""
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        
        # Boundary penalty
        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r
        boundary_viol = np.sum(np.maximum(0, -d_left)**2) + np.sum(np.maximum(0, -d_right)**2) + \
                        np.sum(np.maximum(0, -d_bottom)**2) + np.sum(np.maximum(0, -d_top)**2)
        
        # Overlap penalty
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        overlap = np.maximum(0, min_dist[iu] - dist[iu])
        overlap_viol = np.sum(overlap**2)
        
        return -np.sum(r) + 100 * boundary_viol + 100 * overlap_viol
    
    def is_valid(xy, r, tol=1e-8):
        n = len(r)
        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False
        
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False
        
        return True
    
    def get_score(xy, r):
        return np.sum(r)
    
    def push_apart(xy, r, max_iters=200):
        """Iteratively push overlapping circles apart."""
        n = len(r)
        xy = xy.copy()
        r = r.copy()
        
        for it in range(max_iters):
            moved = False
            for i in range(n):
                for j in range(i+1, n):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = r[i] + r[j]
                    if dist < min_dist - 1e-10:
                        if dist < 1e-10:
                            dx, dy = 1.0, 0.0
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist
                        overlap = min_dist - dist
                        xy[i, 0] -= 0.5 * overlap * dx
                        xy[i, 1] -= 0.5 * overlap * dy
                        xy[j, 0] += 0.5 * overlap * dx
                        xy[j, 1] += 0.5 * overlap * dy
                        moved = True
            
            # Clamp to boundaries
            for i in range(n):
                xy[i, 0] = np.clip(xy[i, 0], r[i], 1 - r[i])
                xy[i, 1] = np.clip(xy[i, 1], r[i], 1 - r[i])
            
            if not moved:
                break
        
        return xy, r
    
    def refine_solution(xy, r, time_budget=2.0):
        """Refine a valid solution by iteratively growing radii and re-optimizing."""
        xy = xy.copy()
        r = r.copy()
        start = time.time()
        
        # Iteratively grow radii in small steps
        for grow_factor in [1.005, 1.01, 1.015, 1.02, 1.03, 1.04, 1.05]:
            if time.time() - start > time_budget:
                break
            r_trial = r * grow_factor
            if not is_valid(xy, r_trial):
                xy_trial, r_trial = push_apart(xy, r_trial, max_iters=150)
                if not is_valid(xy_trial, r_trial):
                    continue
            else:
                xy_trial = xy.copy()
            
            params = np.concatenate([xy_trial.flatten(), r_trial])
            try:
                result = minimize(penalty_function, params, method='L-BFGS-B',
                                 options={'maxiter': 200, 'ftol': 1e-13})
                xy_new = result.x[:2*N].reshape(N, 2)
                r_new = result.x[2*N:]
                if is_valid(xy_new, r_new):
                    score_new = get_score(xy_new, r_new)
                    score_cur = get_score(xy, r)
                    if score_new > score_cur:
                        xy = xy_new
                        r = r_new
            except Exception:
                continue
        
        return xy, r
    
    def local_search_perturb(xy, r, rng, num_perturbs=5):
        """Try small perturbations to escape local optima."""
        best_xy = xy.copy()
        best_r = r.copy()
        best_score = get_score(xy, r)
        
        for _ in range(num_perturbs):
            xy_pert = xy.copy() + rng.uniform(-0.01, 0.01, size=(N, 2))
            r_pert = r.copy()
            xy_pert[:, 0] = np.clip(xy_pert[:, 0], r_pert, 1 - r_pert)
            xy_pert[:, 1] = np.clip(xy_pert[:, 1], r_pert, 1 - r_pert)
            
            if not is_valid(xy_pert, r_pert):
                xy_pert, r_pert = push_apart(xy_pert, r_pert, max_iters=100)
                if not is_valid(xy_pert, r_pert):
                    continue
            
            params = np.concatenate([xy_pert.flatten(), r_pert])
            try:
                result = minimize(penalty_function, params, method='L-BFGS-B',
                                 options={'maxiter': 100, 'ftol': 1e-12})
                xy_new = result.x[:2*N].reshape(N, 2)
                r_new = result.x[2*N:]
                if is_valid(xy_new, r_new):
                    score_new = get_score(xy_new, r_new)
                    if score_new > best_score:
                        best_xy = xy_new
                        best_r = r_new
                        best_score = score_new
            except Exception:
                continue
        
        return best_xy, best_r
    
    rng = np.random.RandomState(42)
    
    def get_initial_config(restart):
        if restart == 0:
            # 6x6 grid
            side = 6
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
        elif restart == 1:
            # 6x6 grid with jitter
            side = 6
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
            xy += rng.uniform(-0.02, 0.02, size=(N, 2))
        elif restart == 2:
            # Hexagonal packing pattern (rows of 6, 5, 6, 5)
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            row_configs = [6, 5, 6, 5]
            idx = 0
            for row_idx, num_cols in enumerate(row_configs):
                y = (row_idx + 0.5) / 4.0
                for col in range(num_cols):
                    if idx >= N:
                        break
                    x = (col + 0.5) / num_cols + (0.5/num_cols if row_idx % 2 == 1 else 0)
                    xy[idx] = [x, y]
                    r[idx] = 0.08
                    idx += 1
        elif restart == 3:
            # Hexagonal packing pattern (rows of 5, 6, 5, 6)
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            row_configs = [5, 6, 5, 6]
            idx = 0
            for row_idx, num_cols in enumerate(row_configs):
                y = (row_idx + 0.5) / 4.0
                for col in range(num_cols):
                    if idx >= N:
                        break
                    x = (col + 0.5) / num_cols + (0.5/num_cols if row_idx % 2 == 0 else 0)
                    xy[idx] = [x, y]
                    r[idx] = 0.08
                    idx += 1
        elif restart == 4:
            # 4x8 grid
            rows, cols = 4, 8
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // cols
                col = i % cols
                xy[i] = [(col + 0.5)/cols, (row + 0.5)/rows]
                r[i] = 0.5/cols - 0.001
        elif restart == 5:
            # 8x4 grid
            rows, cols = 8, 4
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // cols
                col = i % cols
                xy[i] = [(col + 0.5)/cols, (row + 0.5)/rows]
                r[i] = 0.5/cols - 0.001
        else:
            # Random initialization
            xy = rng.uniform(0.1, 0.9, size=(N, 2))
            r = rng.uniform(0.03, 0.08, size=N)
        return xy, r
    
    max_restarts = 60
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        if restart < 6:
            time_per_restart = min(remaining / (max_restarts - restart), 3.5)
        elif restart < 12:
            time_per_restart = min(remaining / (max_restarts - restart), 2.5)
        else:
            time_per_restart = min(remaining / (max_restarts - restart), 2.0)
        
        maxiter = max(100, int(time_per_restart * 80))
        
        xy, r = get_initial_config(restart % 6)
        params = np.concatenate([xy.flatten(), r])
        
        try:
            result = minimize(penalty_function, params, method='L-BFGS-B', 
                             options={'maxiter': maxiter, 'ftol': 1e-12})
        except Exception:
            continue
        
        xy_opt = result.x[:2*N].reshape(N, 2)
        r_opt = result.x[2*N:]
        
        if is_valid(xy_opt, r_opt):
            score = get_score(xy_opt, r_opt)
            
            # Try local search perturbation if we have time
            if time.time() < deadline - 5 and score >= best_score * 0.95:
                time_left = deadline - time.time() - 3
                if time_left > 1.5:
                    xy_ls, r_ls = local_search_perturb(xy_opt, r_opt, rng, num_perturbs=3)
                    score_ls = get_score(xy_ls, r_ls)
                    if score_ls > score:
                        xy_opt, r_opt, score = xy_ls, r_ls, score_ls
            
            # Refine the best candidate
            if time.time() < deadline - 3 and score > best_score:
                time_left = deadline - time.time() - 2
                if time_left > 1:
                    xy_ref, r_ref = refine_solution(xy_opt, r_opt, time_budget=min(3.0, time_left))
                    score_ref = get_score(xy_ref, r_ref)
                    if score_ref > score:
                        xy_opt, r_opt, score = xy_ref, r_ref, score_ref
            
            if score > best_score:
                best_score = score
                best_result = (xy_opt.copy(), r_opt.copy())
        else:
            xy_rep, r_rep = push_apart(xy_opt, r_opt, max_iters=200)
            if is_valid(xy_rep, r_rep):
                score = get_score(xy_rep, r_rep)
                if score > best_score:
                    best_score = score
                    best_result = (xy_rep.copy(), r_rep.copy())
    
    if best_result is not None:
        xy, r = best_result
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1-r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1-r)
        if is_valid(xy, r):
            return np.column_stack([xy, r])
    
    # Fallback: grid
    side = 6
    cells = []
    for i in range(N):
        row = i // side
        col = i % side
        cells.append([(col + 0.5) / side, (row + 0.5) / side, 0.5 / side - 1e-3])
    return np.array(cells)
