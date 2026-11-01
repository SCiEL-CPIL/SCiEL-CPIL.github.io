import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s out of 150s
    
    best_result = None
    best_score = -np.inf
    
    def smooth_penalty(params, penalty_weight=100.0):
        """Smooth penalty function using softplus-like terms for better gradients"""
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        
        # Boundary violations (soft)
        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r
        # Use smooth approximation: max(0, x) ≈ log(1+exp(x))/log(2) but use tanh-based
        eps = 1e-6
        boundary_viol = np.sum(np.logaddexp(0, -d_left / eps)) + np.sum(np.logaddexp(0, -d_right / eps)) + \
                        np.sum(np.logaddexp(0, -d_bottom / eps)) + np.sum(np.logaddexp(0, -d_top / eps))
        boundary_viol = boundary_viol * eps  # scale back
        
        # Overlap violations (soft)
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        overlap = min_dist[iu] - dist[iu]
        # Smooth max(0, overlap)
        overlap_soft = np.logaddexp(0, overlap / eps) * eps
        overlap_viol = np.sum(overlap_soft)
        
        return -np.sum(r) + penalty_weight * boundary_viol + penalty_weight * overlap_viol
    
    def is_valid(result, tol=1e-9):
        n = N
        xy = result[:2*n].reshape(n, 2)
        r = result[2*n:]
        
        # Check boundaries
        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False
        
        # Check overlaps
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
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
    
    def local_search(params, n_steps=200, step_size=0.01):
        """Gradient-free local search using random perturbations"""
        n = N
        current = params.copy()
        current_score = get_score(current) if is_valid(current) else -np.inf
        
        best = current.copy()
        best_s = current_score
        
        for step in range(n_steps):
            # Random perturbation
            delta = np.random.normal(0, step_size, size=current.shape)
            # Scale position perturbations larger than radius perturbations
            delta[:2*n] *= 2.0
            
            candidate = current + delta
            candidate = push_apart(candidate, n_iters=10)
            
            if is_valid(candidate):
                cand_s = get_score(candidate)
                if cand_s > best_s:
                    best = candidate.copy()
                    best_s = cand_s
                if cand_s >= current_score:
                    current = candidate
                    current_score = cand_s
            else:
                # Try shrinking radii slightly
                xy_c = candidate[:2*n].reshape(n, 2)
                r_c = candidate[2*n:]
                for scale in [0.98, 0.95, 0.9]:
                    r_s = r_c * scale
                    cand2 = np.concatenate([xy_c.flatten(), r_s])
                    if is_valid(cand2):
                        cand2_s = get_score(cand2)
                        if cand2_s > best_s:
                            best = cand2.copy()
                            best_s = cand2_s
                        if cand2_s >= current_score:
                            current = cand2
                            current_score = cand2_s
                        break
            
            # Gradually reduce step size
            step_size *= 0.995
        
        return best
    
    def refine_solution(params, max_iter=200, penalty_weight=100.0):
        """Use L-BFGS-B to refine a solution"""
        try:
            result = minimize(smooth_penalty, params, method='L-BFGS-B', 
                             options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10},
                             args=(penalty_weight,))
            return result.x
        except Exception:
            return params
    
    def get_initial_config(restart, rng):
        """Generate initial configuration based on restart number"""
        n = N
        
        if restart == 0:
            # 6x6 grid
            side = 6
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.002
        elif restart == 1:
            # Perturbed 6x6 grid
            side = 6
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.002
            xy += rng.uniform(-0.03, 0.03, size=(n, 2))
        elif restart == 2:
            # Hexagonal packing approximation
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            # 6 columns, offset rows
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5)/6 + (1.0/12 if row % 2 == 1 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.075
        elif restart == 3:
            # 4x4 grid with larger circles
            side = 4
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.005
        elif restart == 4:
            # 5x5 grid (use first 25, then add 7 more)
            side = 5
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(min(n, 25)):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.003
            # Add 7 more circles in gaps
            gap_positions = [
                [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],
                [0.5, 0.1], [0.1, 0.5], [0.9, 0.5]
            ]
            for i in range(7):
                xy[25+i] = gap_positions[i]
                r[25+i] = 0.04
        elif restart == 5:
            # Random with medium radii
            xy = rng.uniform(0.08, 0.92, size=(n, 2))
            r = rng.uniform(0.06, 0.10, size=n)
        elif restart == 6:
            # Random with varied radii
            xy = rng.uniform(0.05, 0.95, size=(n, 2))
            r = rng.uniform(0.04, 0.09, size=n)
        elif restart == 7:
            # Two-tier: 16 larger + 16 smaller
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            side = 4
            for i in range(16):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.12
            # Place 16 smaller in between
            for i in range(16):
                row = i // 4
                col = i % 4
                xy[16+i] = [(col + 0.5)/4 + 0.125, (row + 0.5)/4 + 0.125]
                r[16+i] = 0.05
        else:
            # Various random configs with different distributions
            xy = rng.uniform(0.05, 0.95, size=(n, 2))
            r = rng.uniform(0.03, 0.10, size=n)
        
        params = np.concatenate([xy.flatten(), r])
        return params
    
    rng = np.random.RandomState(42)
    
    # Try multiple initial configurations with refinement
    n_restarts = 50
    for restart in range(n_restarts):
        if time.time() > deadline - 3:
            break
        
        # Get initial configuration
        params = get_initial_config(restart, rng)
        
        # Push apart to get a feasible starting point
        params = push_apart(params, n_iters=40)
        
        # Check if already valid
        if is_valid(params):
            score = get_score(params)
            if score > best_score:
                best_score = score
                best_result = params.copy()
        
        # Refine with L-BFGS-B using graduated penalty
        refined = refine_solution(params, max_iter=200, penalty_weight=50.0)
        
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
        
        # Local search on the best found so far
        if best_result is not None and time.time() < deadline - 5:
            ls_result = local_search(best_result, n_steps=100, step_size=0.008)
            if is_valid(ls_result):
                ls_score = get_score(ls_result)
                if ls_score > best_score:
                    best_score = ls_score
                    best_result = ls_result.copy()
    
    # Final refinement passes
    if best_result is not None:
        # Second L-BFGS-B pass with higher penalty
        if time.time() < deadline - 10:
            final_refined = refine_solution(best_result, max_iter=150, penalty_weight=200.0)
            if is_valid(final_refined):
                score = get_score(final_refined)
                if score > best_score:
                    best_score = score
                    best_result = final_refined.copy()
        
        # Final local search
        if time.time() < deadline - 5:
            ls_final = local_search(best_result, n_steps=150, step_size=0.005)
            if is_valid(ls_final):
                ls_score = get_score(ls_final)
                if ls_score > best_score:
                    best_score = ls_score
                    best_result = ls_final.copy()
    
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
    side = 6
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
