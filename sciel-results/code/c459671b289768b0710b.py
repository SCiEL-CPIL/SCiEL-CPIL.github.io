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
        """Refine a valid solution by growing radii and re-optimizing."""
        xy = xy.copy()
        r = r.copy()
        start = time.time()
        
        for grow_factor in [1.001, 1.002, 1.005, 1.01, 1.02]:
            if time.time() - start > time_budget:
                break
            r_trial = r * grow_factor
            if not is_valid(xy, r_trial):
                xy_trial, r_trial = push_apart(xy, r_trial, max_iters=100)
                if not is_valid(xy_trial, r_trial):
                    continue
            params = np.concatenate([xy_trial.flatten(), r_trial])
            try:
                result = minimize(penalty_function, params, method='L-BFGS-B',
                                 options={'maxiter': 150, 'ftol': 1e-12})
                if is_valid(result.x[:2*N].reshape(N, 2), result.x[2*N:]):
                    score = get_score(result.x[:2*N].reshape(N, 2), result.x[2*N:])
                    if score > get_score(xy, r):
                        xy = result.x[:2*N].reshape(N, 2)
                        r = result.x[2*N:]
            except Exception:
                continue
        
        return xy, r
    
    rng = np.random.RandomState(42)
    
    def get_initial_config(restart):
        if restart == 0:
            side = 6
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
        elif restart == 1:
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
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // 6
                col = i % 6
                x = (col + 0.5)/6 + (0.25 if row % 2 == 1 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.08
        elif restart == 3:
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // 5
                col = i % 5
                x = (col + 0.5)/5 + (0.2 if row % 2 == 1 else 0)
                y = (row + 0.5)/7
                xy[i] = [x, y]
                r[i] = 0.09
        else:
            xy = rng.uniform(0.1, 0.9, size=(N, 2))
            r = rng.uniform(0.03, 0.08, size=N)
        return xy, r
    
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        if restart < 5:
            time_per_restart = min(remaining / (max_restarts - restart), 3.0)
        else:
            time_per_restart = min(remaining / (max_restarts - restart), 2.0)
        
        maxiter = max(100, int(time_per_restart * 80))
        
        xy, r = get_initial_config(restart)
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
            
            if time.time() < deadline - 3 and score > best_score:
                time_left = deadline - time.time() - 2
                if time_left > 1:
                    xy_ref, r_ref = refine_solution(xy_opt, r_opt, time_budget=min(2.0, time_left))
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
