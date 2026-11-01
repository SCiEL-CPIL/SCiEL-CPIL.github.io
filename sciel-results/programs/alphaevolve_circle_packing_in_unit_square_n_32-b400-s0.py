import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s
    
    best_result = None
    best_score = -np.inf
    
    def penalty_function(params):
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
    
    def is_valid(result):
        n = N
        xy = result[:2*n].reshape(n, 2)
        r = result[2*n:]
        
        # Check boundaries
        if np.any(r < -1e-9):
            return False
        if np.any(xy[:, 0] - r < -1e-9) or np.any(1 - xy[:, 0] - r < -1e-9):
            return False
        if np.any(xy[:, 1] - r < -1e-9) or np.any(1 - xy[:, 1] - r < -1e-9):
            return False
        
        # Check overlaps
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - 1e-9):
            return False
        
        return True
    
    def get_score(result):
        n = N
        r = result[2*n:]
        return np.sum(r)
    
    def push_apart(params, n_iters=50):
        """Iteratively push overlapping circles apart and fix boundary violations"""
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        
        for _ in range(n_iters):
            moved = False
            
            # Fix overlaps
            for i in range(n):
                for j in range(i+1, n):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist and dist > 1e-10:
                        # Push apart
                        overlap = min_dist - dist
                        # Split the correction
                        wi = r[i] / (r[i] + r[j])
                        wj = r[j] / (r[i] + r[j])
                        
                        push_x = (dx / dist) * overlap
                        push_y = (dy / dist) * overlap
                        
                        xy[i, 0] -= push_x * wi
                        xy[i, 1] -= push_y * wi
                        xy[j, 0] += push_x * wj
                        xy[j, 1] += push_y * wj
                        moved = True
                    elif dist <= 1e-10:
                        # Random small push
                        angle = np.random.uniform(0, 2*np.pi)
                        xy[i, 0] -= 0.01 * np.cos(angle)
                        xy[i, 1] -= 0.01 * np.sin(angle)
                        xy[j, 0] += 0.01 * np.cos(angle)
                        xy[j, 1] += 0.01 * np.sin(angle)
                        moved = True
            
            # Fix boundaries
            for i in range(n):
                # Left
                if xy[i, 0] < r[i]:
                    xy[i, 0] = r[i]
                    moved = True
                # Right
                if xy[i, 0] > 1 - r[i]:
                    xy[i, 0] = 1 - r[i]
                    moved = True
                # Bottom
                if xy[i, 1] < r[i]:
                    xy[i, 1] = r[i]
                    moved = True
                # Top
                if xy[i, 1] > 1 - r[i]:
                    xy[i, 1] = 1 - r[i]
                    moved = True
            
            if not moved:
                break
        
        return np.concatenate([xy.flatten(), r])
    
    def refine_solution(params, max_iter=200):
        """Use L-BFGS-B to refine a solution"""
        try:
            result = minimize(penalty_function, params, method='L-BFGS-B', 
                             options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
            return result.x
        except Exception:
            return params
    
    def get_initial_config(restart, rng):
        """Generate initial configuration based on restart number"""
        n = N
        side = int(np.ceil(np.sqrt(N)))
        
        if restart == 0:
            # Grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
        elif restart == 1:
            # Perturbed grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
            xy += rng.uniform(-0.02, 0.02, size=(n, 2))
        elif restart == 2:
            # Hexagonal-like
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5)/6 + (0.25 if row % 2 == 1 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.08
        elif restart == 3:
            # Random with larger radii
            xy = rng.uniform(0.1, 0.9, size=(n, 2))
            r = rng.uniform(0.05, 0.1, size=n)
        elif restart == 4:
            # Random with smaller radii
            xy = rng.uniform(0.05, 0.95, size=(n, 2))
            r = rng.uniform(0.03, 0.06, size=n)
        else:
            # Various random configs
            xy = rng.uniform(0.05, 0.95, size=(n, 2))
            r = rng.uniform(0.04, 0.09, size=n)
        
        params = np.concatenate([xy.flatten(), r])
        return params
    
    rng = np.random.RandomState(42)
    
    # Try multiple initial configurations with refinement
    n_restarts = 40
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Get initial configuration
        params = get_initial_config(restart, rng)
        
        # Push apart to get a feasible starting point
        params = push_apart(params, n_iters=30)
        
        # Check if already valid
        if is_valid(params):
            score = get_score(params)
            if score > best_score:
                best_score = score
                best_result = params.copy()
        
        # Refine with L-BFGS-B
        refined = refine_solution(params, max_iter=150)
        
        if is_valid(refined):
            score = get_score(refined)
            if score > best_score:
                best_score = score
                best_result = refined.copy()
        else:
            # Try to repair: push apart then reduce radii
            repaired = push_apart(refined, n_iters=50)
            if is_valid(repaired):
                score = get_score(repaired)
                if score > best_score:
                    best_score = score
                    best_result = repaired.copy()
            else:
                # Reduce radii
                xy = refined[:2*N].reshape(N, 2)
                r = refined[2*N:]
                for scale in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]:
                    r_scaled = r * scale
                    params_test = np.concatenate([xy.flatten(), r_scaled])
                    if is_valid(params_test):
                        score = get_score(params_test)
                        if score > best_score:
                            best_score = score
                            best_result = params_test.copy()
                        break
    
    # If we have a best result, try a final refinement pass
    if best_result is not None and time.time() < deadline - 5:
        final_refined = refine_solution(best_result, max_iter=100)
        if is_valid(final_refined):
            score = get_score(final_refined)
            if score > best_score:
                best_score = score
                best_result = final_refined.copy()
    
    if best_result is not None:
        n = N
        xy = best_result[:2*n].reshape(n, 2)
        r = best_result[2*n:]
        # Clamp to be safe
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1-r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1-r)
        final_params = np.concatenate([xy.flatten(), r])
        if is_valid(final_params):
            return np.column_stack([xy, r])
        else:
            # Return the best_result as-is if clamping broke it
            return np.column_stack([best_result[:2*n].reshape(n, 2), best_result[2*n:]])
    
    # Fallback: grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
