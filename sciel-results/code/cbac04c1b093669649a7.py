"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Rectangle perimeter is 4, so 2*(w+h) = 4 => w+h = 2.
    # Let w = 2 * t, h = 2 * (1-t), where t in (0, 1).
    # We want to maximize sum of radii of 21 disjoint circles inside [0,w] x [0,h].
    
    def get_rect_dims(t):
        t = np.clip(t, 0.1, 0.9)
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h

    def evaluate(state):
        """
        Returns (score, penalty). Score = sum_r - penalty.
        Higher score is better. Valid solutions have penalty=0.
        """
        t = state[0]
        w, h = get_rect_dims(t)
        
        penalty = 0.0
        
        coords = np.zeros((N, 3))
        for i in range(N):
            coords[i, 0] = state[1 + 3*i]
            coords[i, 1] = state[1 + 3*i+1]
            coords[i, 2] = state[1 + 3*i+2]
        
        radii = coords[:, 2]
        if np.any(radii < 1e-5):
            radii = np.maximum(radii, 1e-5)
            penalty += 0.1 * np.sum(radii[radii < 1e-5])
        
        sum_r = np.sum(radii)
        
        # Boundary violations
        x = coords[:, 0]
        y = coords[:, 1]
        
        viol_left = np.maximum(0, radii - x)
        viol_right = np.maximum(0, x + radii - w)
        viol_bottom = np.maximum(0, radii - y)
        viol_top = np.maximum(0, y + radii - h)
        
        penalty += 500.0 * (np.sum(viol_left) + np.sum(viol_right) + np.sum(viol_bottom) + np.sum(viol_top))
        
        # Overlap violations - vectorized using squared distances
        # For pairs (i,j) with i < j
        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        min_dist = radii[idx_i] + radii[idx_j]
        
        # Only compute sqrt where needed (where dist_sq < min_dist^2)
        overlap_mask = dist_sq < min_dist * min_dist
        if np.any(overlap_mask):
            dist = np.sqrt(dist_sq[overlap_mask])
            overlaps = min_dist[overlap_mask] - dist
            penalty += 1000.0 * np.sum(overlaps)
        
        score = sum_r - penalty
        return score, penalty

    def check_valid(state, eps=1e-7):
        """
        Check if state is valid: all circles inside rectangle, no overlaps.
        """
        t = state[0]
        w, h = get_rect_dims(t)
        
        coords = np.zeros((N, 3))
        for i in range(N):
            coords[i, 0] = state[1 + 3*i]
            coords[i, 1] = state[1 + 3*i+1]
            coords[i, 2] = state[1 + 3*i+2]
        
        radii = coords[:, 2]
        if np.any(radii < 1e-6):
            return False
        
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Boundary check
        if np.any(x - radii < -eps) or np.any(x + radii > w + eps) or \
           np.any(y - radii < -eps) or np.any(y + radii > h + eps):
            return False
        
        # Overlap check
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = radii[i] + radii[j] - eps
                if dist_sq < min_dist * min_dist:
                    return False
        return True

    def make_initial():
        """
        Create an initial state using a grid pattern in a square rectangle (t=0.5).
        """
        t_init = 0.5
        w, h = get_rect_dims(t_init)
        
        state = np.zeros(1 + 3*N)
        state[0] = t_init
        
        # Use a 5x5 grid, fill 21 cells
        side = 5
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                gx = (i + 0.5) / side * w
                gy = (j + 0.5) / side * h
                # Radius that fits in cell with small margin
                r = 0.5 / side * min(w, h) * 0.95
                state[1 + 3*idx] = gx
                state[1 + 3*idx + 1] = gy
                state[1 + 3*idx + 2] = r
                idx += 1
        
        return state

    def polish_gradient(state, time_limit, max_iter=5000):
        """
        Gradient-based polish using smooth penalty formulation.
        """
        from scipy.optimize import minimize
        
        def objective(s):
            t = s[0]
            w, h = get_rect_dims(t)
            coords = np.zeros((N, 3))
            for i in range(N):
                coords[i, 0] = s[1 + 3*i]
                coords[i, 1] = s[1 + 3*i+1]
                coords[i, 2] = s[1 + 3*i+2]
            
            radii = coords[:, 2]
            x = coords[:, 0]
            y = coords[:, 1]
            
            # Negative sum of radii (we minimize)
            obj = -np.sum(radii)
            
            # Smooth boundary penalties using softplus-like approximation
            viol_l = np.maximum(0, radii - x)
            viol_r = np.maximum(0, x + radii - w)
            viol_b = np.maximum(0, radii - y)
            viol_t = np.maximum(0, y + radii - h)
            
            obj += 1000.0 * (np.sum(viol_l**2) + np.sum(viol_r**2) + np.sum(viol_b**2) + np.sum(viol_t**2))
            
            # Overlap penalty using smooth approximation
            idx_i, idx_j = np.triu_indices(N, k=1)
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[idx_i] + radii[idx_j]
            
            # Smooth penalty: use max(0, min_dist^2 - dist_sq) with smoothing
            overlap_amount = min_dist**2 - dist_sq
            # Use a smooth approximation of max(0, x) = 0.5*(x + sqrt(x^2 + eps^2))
            smooth_overlap = 0.5 * (overlap_amount + np.sqrt(overlap_amount**2 + 1e-8))
            obj += 5000.0 * np.sum(smooth_overlap**2)
            
            return obj
        
        def gradient(s):
            t = s[0]
            w, h = get_rect_dims(t)
            coords = np.zeros((N, 3))
            for i in range(N):
                coords[i, 0] = s[1 + 3*i]
                coords[i, 1] = s[1 + 3*i+1]
                coords[i, 2] = s[1 + 3*i+2]
            
            radii = coords[:, 2]
            x = coords[:, 0]
            y = coords[:, 1]
            
            grad = np.zeros_like(s)
            
            # d(-sum_r)/dr_i = -1
            for i in range(N):
                grad[1 + 3*i + 2] += -1.0
            
            # Boundary gradients
            viol_l = np.maximum(0, radii - x)
            for i in range(N):
                if viol_l[i] > 0:
                    grad[1 + 3*i] += -2.0 * 1000.0 * viol_l[i]
                    grad[1 + 3*i + 2] += 2.0 * 1000.0 * viol_l[i]
            
            viol_r = np.maximum(0, x + radii - w)
            for i in range(N):
                if viol_r[i] > 0:
                    grad[1 + 3*i] += 2.0 * 1000.0 * viol_r[i]
                    grad[1 + 3*i + 2] += 2.0 * 1000.0 * viol_r[i]
            
            viol_b = np.maximum(0, radii - y)
            for i in range(N):
                if viol_b[i] > 0:
                    grad[1 + 3*i + 1] += -2.0 * 1000.0 * viol_b[i]
                    grad[1 + 3*i + 2] += 2.0 * 1000.0 * viol_b[i]
            
            viol_t = np.maximum(0, y + radii - h)
            for i in range(N):
                if viol_t[i] > 0:
                    grad[1 + 3*i + 1] += 2.0 * 1000.0 * viol_t[i]
                    grad[1 + 3*i + 2] += 2.0 * 1000.0 * viol_t[i]
            
            # Overlap gradients
            idx_i, idx_j = np.triu_indices(N, k=1)
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[idx_i] + radii[idx_j]
            overlap_amount = min_dist**2 - dist_sq
            
            # Compute smooth_overlap for gradient
            smooth_overlap = 0.5 * (overlap_amount + np.sqrt(overlap_amount**2 + 1e-8))
            # Gradient of smooth_overlap = 0.5 * (1 + overlap_amount / sqrt(overlap_amount^2 + 1e-8))
            smooth_grad = 0.5 * (1.0 + overlap_amount / np.sqrt(overlap_amount**2 + 1e-8))
            
            # Penalty is 5000 * sum(smooth_overlap^2)
            # d/dx_i: 5000 * 2 * smooth_overlap * (-2*dx) * smooth_grad
            # d/dr_i: 5000 * 2 * smooth_overlap * (2*min_dist) * smooth_grad
            
            for k in range(len(idx_i)):
                i = idx_i[k]
                j = idx_j[k]
                so = smooth_overlap[k]
                sg = smooth_grad[k]
                
                # x_i
                grad[1 + 3*i] += 5000.0 * 2.0 * so * (-2.0 * dx[k]) * sg
                # x_j
                grad[1 + 3*j] += 5000.0 * 2.0 * so * (2.0 * dx[k]) * sg
                # y_i
                grad[1 + 3*i + 1] += 5000.0 * 2.0 * so * (-2.0 * dy[k]) * sg
                # y_j
                grad[1 + 3*j + 1] += 5000.0 * 2.0 * so * (2.0 * dy[k]) * sg
                # r_i
                grad[1 + 3*i + 2] += 5000.0 * 2.0 * so * (2.0 * min_dist[k]) * sg
                # r_j
                grad[1 + 3*j + 2] += 5000.0 * 2.0 * so * (2.0 * min_dist[k]) * sg
            
            return grad
        
        # Use scipy minimize with L-BFGS-B
        bounds = [(0.1, 0.9)]
        for i in range(N):
            bounds.append((0, 2.0))  # x
            bounds.append((0, 2.0))  # y
            bounds.append((1e-4, 0.5))  # r
        
        try:
            result = minimize(
                objective,
                state,
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 40}
            )
            if result.success or result.fun < objective(state):
                return result.x
        except Exception:
            pass
        
        return state
    
    def local_optimize(state, time_limit):
        """
        Run simulated annealing to improve state.
        """
        best_state = state.copy()
        best_score, _ = evaluate(best_state)
        
        T_start = 0.3
        T_end = 1e-6
        alpha = 0.9997
        
        T = T_start
        no_improve_count = 0
        stuck_iters = 0
        
        while time.time() < time_limit:
            # Check time
            if time.time() > time_limit - 2:
                break
            
            # Generate neighbor - pick 2 random variables to perturb for better exploration
            new_state = best_state.copy()
            
            # Perturb 1-2 variables
            n_perturb = random.randint(1, 2)
            perturbed = set()
            
            for _ in range(n_perturb):
                var_idx = random.randint(0, len(new_state)-1)
                if var_idx in perturbed:
                    continue
                perturbed.add(var_idx)
                
                if var_idx == 0:
                    # Perturb aspect ratio
                    new_state[0] += random.gauss(0, 0.015)
                    new_state[0] = np.clip(new_state[0], 0.05, 0.95)
                else:
                    circle_idx = (var_idx - 1) // 3
                    comp = (var_idx - 1) % 3
                    
                    # Scale perturbation by current temperature
                    scale = T / T_start * 0.08 + 0.005
                    
                    if comp == 0:  # x
                        new_state[var_idx] += random.gauss(0, scale)
                    elif comp == 1:  # y
                        new_state[var_idx] += random.gauss(0, scale)
                    else:  # r
                        new_state[var_idx] += random.gauss(0, scale * 0.25)
                        if new_state[var_idx] < 1e-4:
                            new_state[var_idx] = 1e-4
                        w, h = get_rect_dims(new_state[0])
                        max_r = 0.5 * min(w, h)
                        if new_state[var_idx] > max_r:
                            new_state[var_idx] = max_r
            
            new_score, new_penalty = evaluate(new_state)
            
            # Acceptance criterion
            if new_score > best_score:
                best_state = new_state
                best_score = new_score
                no_improve_count = 0
                stuck_iters = 0
            elif random.random() < np.exp((new_score - best_score) / T):
                best_state = new_state
                no_improve_count += 1
            else:
                no_improve_count += 1
            
            # If stuck for a while, boost temperature slightly
            if no_improve_count > 500:
                T = max(T * 1.1, T_start * 0.3)
                no_improve_count = 0
            
            T *= alpha
            if T < T_end:
                T = T_end
        
        return best_state, best_score

    # Main algorithm
    best_state = None
    best_score = -np.inf
    
    # Multiple restarts
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline - 50:
            break
        
        # Create initial state with some randomization
        if restart == 0:
            state = make_initial()
        else:
            state = make_initial()
            # Add random perturbation
            w, h = get_rect_dims(state[0])
            for i in range(N):
                state[1 + 3*i] += random.uniform(-0.05, 0.05) * min(w, h)
                state[1 + 3*i + 1] += random.uniform(-0.05, 0.05) * min(w, h)
                state[1 + 3*i + 2] *= random.uniform(0.8, 1.0)
            state[0] += random.uniform(-0.1, 0.1)
        
        # Time budget for this restart
        remaining = deadline - 50 - time.time()
        if remaining < 15:
            break
        
        # Allocate time for SA (leave room for polish)
        sa_time = max(8, remaining * 0.45)
        polish_time = max(10, remaining * 0.5)
        sa_deadline = time.time() + sa_time
        
        state, score = local_optimize(state, sa_deadline)
        
        # Gradient polish phase - run multiple times for better convergence
        if time.time() < deadline - 20:
            for polish_iter in range(4):
                if time.time() > deadline - 15:
                    break
                polish_deadline = time.time() + min(12, (deadline - 15 - time.time()) / (4 - polish_iter))
                polished = polish_gradient(state, polish_deadline)
                pol_score, pol_pen = evaluate(polished)
                if pol_score > score:
                    state = polished
                    score = pol_score
                else:
                    break
        
        if score > best_score:
            best_score = score
            best_state = state.copy()
        
        # If we found a valid solution, try to refine it
        if check_valid(best_state):
            # Try to increase radii slightly and re-optimize
            refined = best_state.copy()
            w, h = get_rect_dims(refined[0])
            for i in range(N):
                r_idx = 1 + 3*i + 2
                # Increase radius by a small amount if it doesn't violate boundaries
                refined[r_idx] *= 1.03
            # Quick local optimization
            quick_deadline = min(time.time() + 10, deadline - 15)
            refined, ref_score = local_optimize(refined, quick_deadline)
            # Polish the refined solution
            if time.time() < deadline - 15:
                polished_ref = polish_gradient(refined, min(time.time() + 8, deadline - 10))
                pr_score, _ = evaluate(polished_ref)
                if pr_score > ref_score:
                    refined = polished_ref
                    ref_score = pr_score
            if ref_score > best_score and check_valid(refined):
                best_score = ref_score
                best_state = refined.copy()
    
    # Final high-precision polish on the best solution
    if best_state is not None and time.time() < deadline - 10:
        final_polish_deadline = deadline - 5
        final_polished = polish_gradient(best_state, final_polish_deadline)
        fp_score, fp_pen = evaluate(final_polished)
        if fp_score > best_score:
            best_state = final_polished
            best_score = fp_score
        
        # One more polish pass if time allows
        if time.time() < deadline - 8:
            second_polish = polish_gradient(best_state, deadline - 3)
            sp_score, sp_pen = evaluate(second_polish)
            if sp_score > best_score:
                best_state = second_polish
                best_score = sp_score
        
        # Final micro-refinement: try to squeeze more radius
        if time.time() < deadline - 6:
            refined_final = best_state.copy()
            w_f, h_f = get_rect_dims(refined_final[0])
            for i in range(N):
                r_idx = 1 + 3*i + 2
                # Try small radius increase
                refined_final[r_idx] *= 1.01
            # Quick polish to fix any violations
            micro_deadline = min(time.time() + 5, deadline - 4)
            micro_polished = polish_gradient(refined_final, micro_deadline)
            mp_score, mp_pen = evaluate(micro_polished)
            if mp_score > best_score and check_valid(micro_polished):
                best_state = micro_polished
                best_score = mp_score
    
    # Final validity check and repair if needed
    if best_state is None:
        # Fallback: return a simple valid grid
        best_state = make_initial()
    elif not check_valid(best_state):
        # Try to repair by shrinking radii
        repair = best_state.copy()
        for _ in range(50):
            if check_valid(repair):
                break
            for i in range(N):
                repair[1 + 3*i + 2] *= 0.98
        if check_valid(repair):
            best_state = repair
        else:
            # Last resort: use the initial grid
            best_state = make_initial()
    
    # Extract result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_state[1 + 3*i]
        result[i, 1] = best_state[1 + 3*i+1]
        result[i, 2] = best_state[1 + 3*i+2]
    
    # Final clamp to ensure validity
    t_final = best_state[0]
    w_final, h_final = get_rect_dims(t_final)
    for i in range(N):
        x = result[i, 0]
        y = result[i, 1]
        r = result[i, 2]
        
        # Clamp x
        if x - r < 0:
            x = r
        if x + r > w_final:
            x = w_final - r
        result[i, 0] = max(0, min(w_final, x))
        
        # Clamp y
        if y - r < 0:
            y = r
        if y + r > h_final:
            y = h_final - r
        result[i, 1] = max(0, min(h_final, y))
        
        # Clamp radius
        max_r = 0.5 * min(w_final, h_final)
        result[i, 2] = max(1e-6, min(max_r, r))
    
    return result

# EVOLVE-BLOCK-END
