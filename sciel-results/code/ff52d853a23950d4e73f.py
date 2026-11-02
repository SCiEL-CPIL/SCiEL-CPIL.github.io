import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def get_score(xy, r):
        return np.sum(r)

    def is_valid(xy, r, tol=1e-9):
        n = len(r)
        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False
        if n > 1:
            dx = xy[:, 0][:, None] - xy[:, 0][None, :]
            dy = xy[:, 1][:, None] - xy[:, 1][None, :]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = r[:, None] + r[None, :]
            iu = np.triu_indices(n, k=1)
            if np.any(dist[iu] < min_dist[iu] - tol):
                return False
        return True

    def refine_solution(xy, r, max_iter=200):
        """
        Refine a valid packing by maximizing sum of radii.
        """
        n = len(r)
        params = np.concatenate([xy.flatten(), r])
        
        def penalty_function(params):
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
            
            return -np.sum(r) + 1000 * (boundary_viol + overlap_viol)

        try:
            result = minimize(penalty_function, params, method='L-BFGS-B',
                             options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
            new_params = result.x
            new_xy = new_params[:2*n].reshape(n, 2)
            new_r = new_params[2*n:]
            
            # Ensure validity by clipping
            new_r = np.clip(new_r, 0, 0.5)
            new_xy[:, 0] = np.clip(new_xy[:, 0], new_r, 1-new_r)
            new_xy[:, 1] = np.clip(new_xy[:, 1], new_r, 1-new_r)
            
            if is_valid(new_xy, new_r):
                return new_xy, new_r
            else:
                return xy, r # Return original if refinement broke it
        except:
            return xy, r

    def solve_mwis(candidates, n_select):
        """
        Greedy MWIS with local search
        """
        n_cand = len(candidates)
        if n_cand == 0:
            return []
        
        # Sort by radius descending
        sorted_indices = np.argsort([-c[2] for c in candidates])
        
        selected = []
        
        # Greedy selection
        for idx in sorted_indices:
            if len(selected) >= n_select:
                break
            cx, cy, cr = candidates[idx]
            conflict = False
            for sel_idx in selected:
                sx, sy, sr = candidates[sel_idx]
                dist = np.sqrt((cx-sx)**2 + (cy-sy)**2)
                if dist < cr + sr:
                    conflict = True
                    break
            if not conflict:
                selected.append(idx)
        
        # Local Search
        improved = True
        max_iters = 20
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            for i in range(len(selected)):
                sel_idx = selected[i]
                temp_selected = selected[:i] + selected[i+1:]
                temp_selected_set = set(temp_selected)
                
                best_candidate = None
                best_gain = 0
                for u_idx in range(n_cand):
                    if u_idx in temp_selected_set:
                        continue
                    ux, uy, ur = candidates[u_idx]
                    conflict = False
                    for t_idx in temp_selected:
                        tx, ty, tr = candidates[t_idx]
                        dist = np.sqrt((ux-tx)**2 + (uy-ty)**2)
                        if dist < ur + tr:
                            conflict = True
                            break
                    if not conflict:
                        gain = ur - candidates[sel_idx][2]
                        if gain > best_gain:
                            best_gain = gain
                            best_candidate = u_idx
                
                if best_candidate is not None:
                    selected[i] = best_candidate
                    improved = True
                    break
        
        return selected

    # Generate Candidates
    # We use multiple grids to ensure diversity
    best_score = -np.inf
    best_xy = None
    best_r = None
    
    rng = np.random.RandomState(42)
    
    # Try 3 different grid configurations
    for config in range(3):
        if time.time() > deadline - 20:
            break
        
        grid_size = 12 + config * 2 # 12, 14, 16
        offset = rng.uniform(-0.05, 0.05, size=2)
        
        xs = np.linspace(0.1 + offset[0], 0.9 + offset[0], grid_size)
        ys = np.linspace(0.1 + offset[1], 0.9 + offset[1], grid_size)
        radii = np.arange(0.05, 0.26, 0.02)
        
        candidates = []
        for x in xs:
            for y in ys:
                for r in radii:
                    if x - r >= 0 and 1 - x - r >= 0 and y - r >= 0 and 1 - y - r >= 0:
                        candidates.append((x, y, r))
        
        # Solve MWIS
        selected_indices = solve_mwis(candidates, N)
        
        if len(selected_indices) < N:
            continue
        
        xy = np.array([candidates[i][:2] for i in selected_indices])
        r = np.array([candidates[i][2] for i in selected_indices])
        
        # Refine
        xy, r = refine_solution(xy, r, max_iter=300)
        
        if is_valid(xy, r):
            score = get_score(xy, r)
            if score > best_score:
                best_score = score
                best_xy = xy
                best_r = r
                
    # If MWIS didn't find anything good, fallback to simple grid
    if best_xy is None:
        side = int(np.ceil(np.sqrt(N)))
        best_xy = np.zeros((N, 2))
        best_r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            best_xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
            best_r[i] = 0.5/side - 0.001

    # Local search on the best result
    for _ in range(10):
        if time.time() > deadline - 5:
            break
            
        xy_p = best_xy.copy()
        r_p = best_r.copy()
        
        xy_p += rng.uniform(-0.02, 0.02, size=(N, 2))
        r_p += rng.uniform(-0.005, 0.005, size=N)
        r_p = np.clip(r_p, 0.01, 0.4)
        
        # Push apart
        for _ in range(20):
            for i in range(N):
                for j in range(i+1, N):
                    dx = xy_p[j, 0] - xy_p[i, 0]
                    dy = xy_p[j, 1] - xy_p[i, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = r_p[i] + r_p[j]
                    if dist < min_dist and dist > 1e-6:
                        overlap = min_dist - dist
                        push_x = (dx / dist) * overlap * 0.5
                        push_y = (dy / dist) * overlap * 0.5
                        xy_p[i, 0] -= push_x
                        xy_p[i, 1] -= push_y
                        xy_p[j, 0] += push_x
                        xy_p[j, 1] += push_y
            
            for i in range(N):
                xy_p[i, 0] = np.clip(xy_p[i, 0], r_p[i], 1-r_p[i])
                xy_p[i, 1] = np.clip(xy_p[i, 1], r_p[i], 1-r_p[i])
                
        xy_ref, r_ref = refine_solution(xy_p, r_p, max_iter=100)
        
        if is_valid(xy_ref, r_ref):
            score = get_score(xy_ref, r_ref)
            if score > best_score:
                best_score = score
                best_xy = xy_ref
                best_r = r_ref

    return np.column_stack([best_xy, best_r])
