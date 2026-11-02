"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    # Rectangle perimeter is 4, so 2*(w+h) = 4 => w+h = 2.
    # Let w = 2 * t, h = 2 * (1-t), where t in (0, 1).
    
    # Precompute pair indices for vectorized operations
    idx_i, idx_j = np.triu_indices(N, k=1)
    n_pairs = len(idx_i)

    def get_rect_dims(t):
        t = np.clip(t, 0.1, 0.9)
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h

    def evaluate(state):
        """
        Vectorized evaluation of score and penalty.
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
        
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Boundary violations
        viol_left = np.maximum(0, radii - x)
        viol_right = np.maximum(0, x + radii - w)
        viol_bottom = np.maximum(0, radii - y)
        viol_top = np.maximum(0, y + radii - h)
        
        penalty += 500.0 * (np.sum(viol_left) + np.sum(viol_right) + np.sum(viol_bottom) + np.sum(viol_top))
        
        # Overlap violations - vectorized
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        min_dist = radii[idx_i] + radii[idx_j]
        
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
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        min_dist = radii[idx_i] + radii[idx_j] - eps
        if np.any(dist_sq < min_dist * min_dist):
            return False
        return True

    def make_initial_hexagonal(t_init=0.5, jitter=0.0):
        """
        Create an initial state using a hexagonal lattice pattern.
        """
        w, h = get_rect_dims(t_init)
        
        state = np.zeros(1 + 3*N)
        state[0] = t_init
        
        # Estimate radius
        r_est = np.sqrt(0.9069 * w * h / (N * np.pi))
        
        positions = []
        r_current = r_est
        
        for attempt in range(20):
            positions = []
            y = r_current
            row_idx = 0
            while y <= h - r_current + 1e-9:
                if row_idx % 2 == 0:
                    x_start = r_current
                else:
                    x_start = r_current + r_current
                
                x = x_start
                while x <= w - r_current + 1e-9:
                    positions.append((x, y, r_current))
                    x += 2.0 * r_current
                y += np.sqrt(3) * r_current
                row_idx += 1
            
            if len(positions) >= N:
                break
            else:
                r_current *= 0.95
        
        if len(positions) < N:
            side = 5
            idx = 0
            positions = []
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    gx = (i + 0.5) / side * w
                    gy = (j + 0.5) / side * h
                    r = 0.5 / side * min(w, h) * 0.95
                    positions.append((gx, gy, r))
                    idx += 1
        
        for i in range(N):
            if i < len(positions):
                x, y, r = positions[i]
                if jitter > 0:
                    x += random.uniform(-jitter, jitter)
                    y += random.uniform(-jitter, jitter)
                state[1 + 3*i] = x
                state[1 + 3*i + 1] = y
                state[1 + 3*i + 2] = r
            else:
                state[1 + 3*i] = w/2
                state[1 + 3*i + 1] = h/2
                state[1 + 3*i + 2] = 0.01
        
        return state

    def polish_gradient(state, time_limit, max_iter=2000):
        """
        Vectorized gradient-based polish using smooth softplus penalties.
        """
        from scipy.optimize import minimize
        
        def objective_and_grad(s):
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
            
            # Equal-radius variance penalty: encourage uniform radii for denser packing
            mean_r = np.mean(radii)
            var_r = np.var(radii)
            obj += 50.0 * var_r  # Penalty for radius variance
            
            # Smooth boundary penalties using softplus
            beta_b = 0.005
            viol_l = radii - x
            viol_r = x + radii - w
            viol_b = radii - y
            viol_t = y + radii - h
            
            # Softplus: log(1 + exp(x/beta)) * beta
            # To avoid overflow, use np.logaddexp(0, x/beta) * beta
            sl = beta_b * np.logaddexp(0, viol_l / beta_b)
            sr = beta_b * np.logaddexp(0, viol_r / beta_b)
            sb = beta_b * np.logaddexp(0, viol_b / beta_b)
            st = beta_b * np.logaddexp(0, viol_t / beta_b)
            
            obj += 1000.0 * (np.sum(sl**2) + np.sum(sr**2) + np.sum(sb**2) + np.sum(st**2))
            
            # Overlap penalty - smooth
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[idx_i] + radii[idx_j]
            
            # Use distance-based overlap for better conditioning
            dist = np.sqrt(dist_sq + 1e-12)
            overlap = min_dist - dist
            beta_o = 0.01
            so = beta_o * np.logaddexp(0, overlap / beta_o)
            obj += 5000.0 * np.sum(so**2)
            
            # Gradient calculation (Vectorized)
            grad = np.zeros_like(s)
            
            # d(-sum_r)/dr_i = -1
            grad[2::3] -= 1.0
            
            # Boundary gradients
            # d(sl)/dv = sigmoid(v/beta)
            sig_l = 1.0 / (1.0 + np.exp(-viol_l / beta_b))
            sig_r = 1.0 / (1.0 + np.exp(-viol_r / beta_b))
            sig_b = 1.0 / (1.0 + np.exp(-viol_b / beta_b))
            sig_t = 1.0 / (1.0 + np.exp(-viol_t / beta_b))
            
            # d(obj)/dx_i = 2000 * sl * sig_l * (-1) + 2000 * sr * sig_r * (1)
            # d(obj)/dy_i = 2000 * sb * sig_b * (-1) + 2000 * st * sig_t * (1)
            # d(obj)/dr_i = 2000 * (sl*sig_l + sr*sig_r + sb*sig_b + st*sig_t)
            
            grad[0::3] += 2000.0 * (-sl * sig_l + sr * sig_r)
            grad[1::3] += 2000.0 * (-sb * sig_b + st * sig_t)
            grad[2::3] += 2000.0 * (sl * sig_l + sr * sig_r + sb * sig_b + st * sig_t)
            
            # Overlap gradients
            # d(so)/d(overlap) = sigmoid(overlap/beta_o)
            sig_o = 1.0 / (1.0 + np.exp(-overlap / beta_o))
            
            # factor = 10000 * so * sig_o
            factor = 10000.0 * so * sig_o
            
            # d(overlap)/dx_i = -dx/dist, d(overlap)/dx_j = dx/dist
            # d(overlap)/dy_i = -dy/dist, d(overlap)/dy_j = dy/dist
            # d(overlap)/dr_i = 1, d(overlap)/dr_j = 1
            
            # Accumulate gradients for x, y, r using np.add.at
            grad_x_contrib = np.zeros(N)
            grad_y_contrib = np.zeros(N)
            grad_r_contrib = np.zeros(N)
            
            np.add.at(grad_x_contrib, idx_i, -dx / dist * factor)
            np.add.at(grad_x_contrib, idx_j, dx / dist * factor)
            
            np.add.at(grad_y_contrib, idx_i, -dy / dist * factor)
            np.add.at(grad_y_contrib, idx_j, dy / dist * factor)
            
            np.add.at(grad_r_contrib, idx_i, factor)
            np.add.at(grad_r_contrib, idx_j, factor)
            
            grad[0::3] += grad_x_contrib
            grad[1::3] += grad_y_contrib
            grad[2::3] += grad_r_contrib
            
            return obj, grad
        
        def objective(s):
            return objective_and_grad(s)[0]
        
        def gradient(s):
            return objective_and_grad(s)[1]
        
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
                options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 80}
            )
            if result.success or result.fun < objective(state):
                return result.x
        except Exception:
            pass
        
        return state

    def force_relax(state, time_limit, max_iter=5000):
        """
        Vectorized force-directed relaxation.
        """
        best_state = state.copy()
        best_score, _ = evaluate(best_state)
        
        step_size = 0.005
        patience = 0
        max_patience = 200
        
        for it in range(max_iter):
            if time.time() > time_limit:
                break
            
            s = best_state.copy()
            t = s[0]
            w, h = get_rect_dims(t)
            
            coords = np.zeros((N, 3))
            for i in range(N):
                coords[i, 0] = s[1 + 3*i]
                coords[i, 1] = s[1 + 3*i+1]
                coords[i, 2] = s[1 + 3*i+2]
            
            radii = coords[:, 2]
            x = coords[:, 0].copy()
            y = coords[:, 1].copy()
            
            fx = np.zeros(N)
            fy = np.zeros(N)
            fr = np.zeros(N)
            
            # Boundary forces
            # Left
            viol_l = np.maximum(0, radii - x)
            fx += viol_l * 10.0
            # Right
            viol_r = np.maximum(0, x + radii - w)
            fx -= viol_r * 10.0
            # Bottom
            viol_b = np.maximum(0, radii - y)
            fy += viol_b * 10.0
            # Top
            viol_t = np.maximum(0, y + radii - h)
            fy -= viol_t * 10.0
            
            # Pair forces
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq + 1e-12)
            target = radii[idx_i] + radii[idx_j]
            
            diff = dist - target
            
            # Spring force
            k_repel = 50.0
            k_attract = 2.0
            
            # Force magnitude: if diff < 0 (overlap), push apart. If diff > 0, pull together.
            # F = k * diff. Direction is normalized (dx, dy).
            # For i: force is F * (dx/dist). If diff < 0, F is negative, so force is towards j?
            # Wait. If i is at 0, j is at 1. dx = -1. dist = 1.
            # If overlapping, dist < target. diff < 0.
            # We want to push i away from j. Direction from j to i is -dx/dist.
            # Let's define force vector on i as F_vec_i.
            # If dist < target, we want F_vec_i to be in direction (x_i - x_j).
            # (x_i - x_j) / dist is (dx, dy)/dist.
            # So F_vec_i = k * (dist - target) * (dx/dist)?
            # If dist < target, dist-target is negative.
            # (dx/dist) points from j to i.
            # So negative * (dx/dist) points from i to j. That's wrong.
            
            # Correct logic:
            # Vector from j to i: v = (x_i - x_j, y_i - y_j) = (dx, dy).
            # Unit vector u = v / dist.
            # If dist < target (overlap), we want to push i in direction u.
            # So force on i should be positive * u.
            # Let's use force_scalar = k * (dist - target).
            # If dist < target, force_scalar is negative.
            # Force on i = force_scalar * u? No, that pushes towards j.
            # Force on i = -force_scalar * u?
            # Let's just use:
            # If dist < target: repel. Force on i is in direction (x_i - x_j).
            # If dist > target: attract. Force on i is in direction (x_j - x_i).
            
            # Let's stick to the spring model: F = -k * (dist - target) * (x_i - x_j) / dist
            # If dist < target, (dist - target) < 0. -k * negative = positive.
            # Direction (x_i - x_j) is away from j. So force is away from j. Correct.
            # If dist > target, (dist - target) > 0. -k * positive = negative.
            # Direction (x_i - x_j) is away from j. Negative sign makes it towards j. Correct.
            
            # We need different k for repel and attract.
            # Let's compute force_vector for each pair.
            
            k_eff = np.where(diff < 0, k_repel, k_attract)
            # Force magnitude on i (scalar, positive means repel)
            # F_mag = k_eff * (dist - target)
            # Vector on i = F_mag * (dx/dist, dy/dist)
            # Wait, if diff < 0, F_mag is negative.
            # Vector = negative * (dx/dist).
            # dx is x_i - x_j.
            # So vector points opposite to (x_i - x_j), i.e., towards j.
            # This is attraction. We want repulsion.
            
            # Let's flip the sign logic.
            # We want:
            # Overlap (dist < target): Push apart. Force on i is in direction (x_i - x_j).
            # No overlap (dist > target): Pull together. Force on i is in direction (x_j - x_i).
            
            # Let's define:
            # dir_x = dx / dist
            # dir_y = dy / dist
            # This points from j to i.
            
            # If dist < target: Force on i = +k_repel * (target - dist) * dir
            # If dist > target: Force on i = -k_attract * (dist - target) * dir
            
            force_mag = np.where(diff < 0, k_repel * (-diff), -k_attract * diff)
            # force_mag is positive for repel, negative for attract.
            
            dir_x = dx / dist
            dir_y = dy / dist
            
            f_x_pair = force_mag * dir_x
            f_y_pair = force_mag * dir_y
            
            np.add.at(fx, idx_i, f_x_pair)
            np.add.at(fx, idx_j, -f_x_pair)
            np.add.at(fy, idx_i, f_y_pair)
            np.add.at(fy, idx_j, -f_y_pair)
            
            # Radii force
            fr[idx_i] += 0.1 * np.where(diff > 0.01, 1.0, 0.0)
            fr[idx_j] += 0.1 * np.where(diff > 0.01, 1.0, 0.0)
            fr[idx_i] -= 0.5 * np.where(diff < -0.01, 1.0, 0.0)
            fr[idx_j] -= 0.5 * np.where(diff < -0.01, 1.0, 0.0)
            
            # Update
            x_new = x + step_size * fx
            y_new = y + step_size * fy
            r_new = radii + step_size * 0.5 * fr
            
            # Clamp
            for i in range(N):
                x_new[i] = max(radii[i], min(w - radii[i], x_new[i]))
                y_new[i] = max(radii[i], min(h - radii[i], y_new[i]))
                r_new[i] = max(1e-4, min(0.5 * min(w, h), r_new[i]))
            
            new_state = s.copy()
            for i in range(N):
                new_state[1 + 3*i] = x_new[i]
                new_state[1 + 3*i + 1] = y_new[i]
                new_state[1 + 3*i + 2] = r_new[i]
            
            new_score, new_penalty = evaluate(new_state)
            
            if new_score > best_score:
                best_state = new_state
                best_score = new_score
                patience = 0
                step_size = min(0.01, step_size * 1.02)
            else:
                patience += 1
                step_size = max(0.001, step_size * 0.95)
            
            if patience > max_patience:
                break
        
        return best_state

    def local_optimize(state, time_limit):
        """
        Simulated annealing with vectorized evaluation.
        """
        best_state = state.copy()
        best_score, _ = evaluate(best_state)
        
        T_start = 0.5
        T_end = 1e-5
        alpha = 0.9995
        
        T = T_start
        
        while time.time() < time_limit:
            if time.time() > time_limit - 2:
                break
            
            new_state = best_state.copy()
            var_idx = random.randint(0, len(new_state)-1)
            
            if var_idx == 0:
                new_state[0] += random.gauss(0, 0.02)
                if new_state[0] < 0.05 or new_state[0] > 0.95:
                    T *= alpha
                    if T < T_end:
                        T = T_end
                    continue
            else:
                circle_idx = (var_idx - 1) // 3
                comp = (var_idx - 1) % 3
                
                scale = T / T_start * 0.1 + 0.01
                
                if comp == 0:
                    new_state[var_idx] += random.gauss(0, scale)
                elif comp == 1:
                    new_state[var_idx] += random.gauss(0, scale)
                else:
                    new_state[var_idx] += random.gauss(0, scale * 0.3)
                    if new_state[var_idx] < 1e-4:
                        new_state[var_idx] = 1e-4
                    w, h = get_rect_dims(new_state[0])
                    max_r = 0.5 * min(w, h)
                    if new_state[var_idx] > max_r:
                        new_state[var_idx] = max_r
            
            new_score, new_penalty = evaluate(new_state)
            
            if new_score > best_score or random.random() < np.exp((new_score - best_score) / T):
                best_state = new_state
                best_score = new_score
            
            T *= alpha
            if T < T_end:
                T = T_end
        
        return best_state, best_score

    # Main algorithm
    best_state = None
    best_score = -np.inf
    
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline - 60:
            break
        
        # Try different aspect ratios for variety
        t_candidates = [0.4, 0.45, 0.5, 0.55, 0.6]
        t_init = t_candidates[restart % len(t_candidates)] + random.uniform(-0.03, 0.03)
        state = make_initial_hexagonal(t_init, jitter=0.015)
        
        remaining = deadline - 60 - time.time()
        if remaining < 15:
            break
        
        # Phase 1: Simulated annealing
        sa_time = max(6, remaining * 0.35)
        sa_deadline = time.time() + sa_time
        state, score = local_optimize(state, sa_deadline)
        
        # Phase 2: Force relaxation
        if time.time() < deadline - 40:
            force_deadline = time.time() + min(8, (deadline - 40 - time.time()) * 0.3)
            force_state = force_relax(state, force_deadline)
            force_score, _ = evaluate(force_state)
            if force_score > score:
                state = force_state
                score = force_score
        
        # Phase 3: Gradient polish (multiple passes)
        if time.time() < deadline - 25:
            for polish_iter in range(3):
                if time.time() > deadline - 20:
                    break
                polish_deadline = time.time() + min(10, (deadline - 20 - time.time()) / (3 - polish_iter))
                polished = polish_gradient(state, polish_deadline, max_iter=1500)
                pol_score, pol_pen = evaluate(polished)
                if pol_score > score:
                    state = polished
                    score = pol_score
                else:
                    break
        
        # Phase 4: Short force relax + polish if time permits
        if time.time() < deadline - 15:
            force_deadline2 = time.time() + min(4, (deadline - 15 - time.time()) * 0.4)
            force_state2 = force_relax(state, force_deadline2)
            force_score2, _ = evaluate(force_state2)
            if force_score2 > score:
                state = force_state2
                score = force_score2
                if time.time() < deadline - 12:
                    polished2 = polish_gradient(state, time.time() + 5, max_iter=1000)
                    pol_score2, _ = evaluate(polished2)
                    if pol_score2 > score:
                        state = polished2
                        score = pol_score2
        
        if score > best_score:
            best_score = score
            best_state = state.copy()
    
    # Final high-precision polish phase
    if best_state is not None and time.time() < deadline - 10:
        # Force relax to clean up any issues
        if time.time() < deadline - 12:
            force_final = force_relax(best_state, time.time() + 5)
            ff_score, _ = evaluate(force_final)
            if ff_score > best_score:
                best_state = force_final
                best_score = ff_score
        
        # Multiple high-precision polish passes
        for polish_pass in range(3):
            if time.time() > deadline - 5:
                break
            remaining_polish = deadline - 5 - time.time()
            if remaining_polish < 3:
                break
            final_polished = polish_gradient(best_state, time.time() + min(remaining_polish, 15), max_iter=5000)
            fp_score, fp_pen = evaluate(final_polished)
            if fp_score > best_score:
                best_state = final_polished
                best_score = fp_score
            else:
                break
    
    if best_state is None:
        best_state = make_initial_hexagonal(0.5)
    elif not check_valid(best_state):
        repair = best_state.copy()
        repaired = False
        for shrink in range(40):
            if check_valid(repair):
                repaired = True
                break
            for i in range(N):
                repair[1 + 3*i + 2] *= 0.95
            t_r = repair[0]
            w_r, h_r = get_rect_dims(t_r)
            for i in range(N):
                r_r = repair[1 + 3*i + 2]
                x_r = repair[1 + 3*i]
                y_r = repair[1 + 3*i + 1]
                x_r = max(r_r, min(w_r - r_r, x_r))
                y_r = max(r_r, min(h_r - r_r, y_r))
                repair[1 + 3*i] = x_r
                repair[1 + 3*i + 1] = y_r
        if repaired:
            best_state = repair
        else:
            best_state = make_initial_hexagonal(0.5)
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_state[1 + 3*i]
        result[i, 1] = best_state[1 + 3*i+1]
        result[i, 2] = best_state[1 + 3*i+2]
    
    t_final = best_state[0]
    w_final, h_final = get_rect_dims(t_final)
    for i in range(N):
        x = result[i, 0]
        y = result[i, 1]
        r = result[i, 2]
        
        if x - r < 0:
            x = r
        if x + r > w_final:
            x = w_final - r
        result[i, 0] = max(0, min(w_final, x))
        
        if y - r < 0:
            y = r
        if y + r > h_final:
            y = h_final - r
        result[i, 1] = max(0, min(h_final, y))
        
        max_r = 0.5 * min(w_final, h_final)
        result[i, 2] = max(1e-6, min(max_r, r))
    
    return result

# EVOLVE-BLOCK-END
