"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    def get_rect_dims(t):
        t = np.clip(t, 0.1, 0.9)
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h

    def extract_coords(state):
        """Extract (x, y, r) arrays from state vector."""
        x = state[1::3]
        y = state[2::3]
        r = state[3::3]
        return x, y, r

    def evaluate(state):
        """
        Returns (score, penalty). Score = sum_r - penalty.
        Higher score is better. Valid solutions have penalty=0.
        """
        t = state[0]
        w, h = get_rect_dims(t)
        
        x, y, r = extract_coords(state)
        
        penalty = 0.0
        
        sum_r = np.sum(r)
        
        # Boundary violations
        viol_left = np.maximum(0, r - x)
        viol_right = np.maximum(0, x + r - w)
        viol_bottom = np.maximum(0, r - y)
        viol_top = np.maximum(0, y + r - h)
        
        penalty += 500.0 * (np.sum(viol_left) + np.sum(viol_right) + np.sum(viol_bottom) + np.sum(viol_top))
        
        # Overlap violations - vectorized
        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        min_dist = r[idx_i] + r[idx_j]
        
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
        
        x, y, r = extract_coords(state)
        
        if np.any(r < 1e-6):
            return False
        
        # Boundary check
        if np.any(x - r < -eps) or np.any(x + r > w + eps) or \
           np.any(y - r < -eps) or np.any(y + r > h + eps):
            return False
        
        # Overlap check
        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        min_dist = r[idx_i] + r[idx_j] - eps
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
        
        r_est = np.sqrt(0.9069 * w * h / (N * np.pi))
        
        positions = []
        r_current = r_est
        
        for attempt in range(20):
            positions = []
            y_pos = r_current
            row_idx = 0
            while y_pos <= h - r_current + 1e-9:
                if row_idx % 2 == 0:
                    x_start = r_current
                else:
                    x_start = r_current + r_current
                
                x_pos = x_start
                while x_pos <= w - r_current + 1e-9:
                    positions.append((x_pos, y_pos, r_current))
                    x_pos += 2.0 * r_current
                y_pos += np.sqrt(3) * r_current
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

    def make_initial_grid(t_init=0.5, jitter=0.0):
        """
        Create an initial state using a grid pattern.
        """
        w, h = get_rect_dims(t_init)
        
        state = np.zeros(1 + 3*N)
        state[0] = t_init
        
        # Try different grid configurations
        best_positions = None
        best_score = -np.inf
        
        for rows in range(3, 8):
            for cols in range(3, 8):
                if rows * cols < N:
                    continue
                if rows * cols > 2 * N:
                    continue
                
                # Calculate radius that fits
                r = min(w / (cols * 2.2), h / (rows * 2.2))
                r = max(r, 0.01)
                
                positions = []
                for i in range(rows):
                    for j in range(cols):
                        if len(positions) >= N:
                            break
                        x = (j + 0.5) / cols * w
                        y = (i + 0.5) / rows * h
                        positions.append((x, y, r))
                
                if len(positions) >= N:
                    score = sum(r for _, _, r in positions[:N])
                    if score > best_score:
                        best_score = score
                        best_positions = positions[:N]
        
        if best_positions is None:
            # Fallback
            side = 5
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) >= N:
                        break
                    gx = (i + 0.5) / side * w
                    gy = (j + 0.5) / side * h
                    r = 0.5 / side * min(w, h) * 0.95
                    positions.append((gx, gy, r))
            best_positions = positions
        
        for i in range(N):
            if i < len(best_positions):
                x, y, r = best_positions[i]
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

    def make_initial_random(t_init=0.5):
        """
        Create an initial state with random positions.
        """
        w, h = get_rect_dims(t_init)
        
        state = np.zeros(1 + 3*N)
        state[0] = t_init
        
        # Estimate radius
        r_est = np.sqrt(0.9069 * w * h / (N * np.pi)) * 0.8
        
        for i in range(N):
            r = r_est * random.uniform(0.7, 1.3)
            r = max(0.01, min(r, 0.4 * min(w, h)))
            x = random.uniform(r, w - r)
            y = random.uniform(r, h - r)
            state[1 + 3*i] = x
            state[1 + 3*i + 1] = y
            state[1 + 3*i + 2] = r
        
        return state

    def polish_gradient(state, max_iter=2000):
        """
        Gradient-based polish using smooth penalty formulation.
        """
        idx_i, idx_j = np.triu_indices(N, k=1)
        n_pairs = len(idx_i)
        
        def objective_and_grad(s):
            t = s[0]
            w, h = get_rect_dims(t)
            
            x, y, r = extract_coords(s)
            
            # Negative sum of radii (we minimize)
            obj = -np.sum(r)
            
            # Smooth boundary penalties using softplus
            beta_b = 0.01
            viol_l = r - x
            viol_r = x + r - w
            viol_b = r - y
            viol_t = y + r - h
            
            # Use quadratic penalty on softplus for smoothness
            sl = beta_b * np.log1p(np.exp(viol_l / beta_b))
            sr = beta_b * np.log1p(np.exp(viol_r / beta_b))
            sb = beta_b * np.log1p(np.exp(viol_b / beta_b))
            st = beta_b * np.log1p(np.exp(viol_t / beta_b))
            
            obj += 1000.0 * (np.sum(sl**2) + np.sum(sr**2) + np.sum(sb**2) + np.sum(st**2))
            
            # Overlap penalty - smooth
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[idx_i] + r[idx_j]
            
            # A = min_dist^2 - dist_sq (positive when overlapping)
            overlap_amount = min_dist**2 - dist_sq
            beta_o = 0.001
            so = beta_o * np.log1p(np.exp(overlap_amount / beta_o))
            obj += 5000.0 * np.sum(so**2)
            
            # Gradient calculation
            grad = np.zeros_like(s)
            
            # d(-sum_r)/dr_i = -1
            grad[2::3] += -1.0
            
            # Boundary gradients
            sig_l = 1.0 / (1.0 + np.exp(-viol_l / beta_b))
            sig_r = 1.0 / (1.0 + np.exp(-viol_r / beta_b))
            sig_b = 1.0 / (1.0 + np.exp(-viol_b / beta_b))
            sig_t = 1.0 / (1.0 + np.exp(-viol_t / beta_b))
            
            # d(obj)/dx_i
            grad[1::3] += 2000.0 * sl * sig_l * (-1.0)
            grad[1::3] += 2000.0 * sr * sig_r * (1.0)
            # d(obj)/dy_i
            grad[2::3] += 2000.0 * sb * sig_b * (-1.0)
            grad[2::3] += 2000.0 * st * sig_t * (1.0)
            # d(obj)/dr_i
            grad[3::3] += 2000.0 * sl * sig_l * (1.0)
            grad[3::3] += 2000.0 * sr * sig_r * (1.0)
            grad[3::3] += 2000.0 * sb * sig_b * (1.0)
            grad[3::3] += 2000.0 * st * sig_t * (1.0)
            
            # Overlap gradients
            sig_o = 1.0 / (1.0 + np.exp(-overlap_amount / beta_o))
            factor = 10000.0 * so * sig_o
            
            for k in range(n_pairs):
                i = idx_i[k]
                j = idx_j[k]
                
                grad[1 + 3*i] += factor[k] * (-2.0 * dx[k])
                grad[1 + 3*j] += factor[k] * (2.0 * dx[k])
                grad[2 + 3*i] += factor[k] * (-2.0 * dy[k])
                grad[2 + 3*j] += factor[k] * (2.0 * dy[k])
                grad[3 + 3*i] += factor[k] * (2.0 * min_dist[k])
                grad[3 + 3*j] += factor[k] * (2.0 * min_dist[k])
            
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
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if result.success or result.fun < objective(state):
                return result.x
        except Exception:
            pass
        
        return state

    def force_relax(state, max_iter=5000):
        """
        Force-directed relaxation: treat circles as particles with repulsive
        forces when too close and attractive forces when too far (toward contact).
        """
        best_state = state.copy()
        best_score, _ = evaluate(best_state)
        
        idx_i, idx_j = np.triu_indices(N, k=1)
        
        # Adaptive step size
        step_size = 0.005
        patience = 0
        max_patience = 200
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            s = best_state.copy()
            t = s[0]
            w, h = get_rect_dims(t)
            
            x, y, r = extract_coords(s)
            x = x.copy()
            y = y.copy()
            
            # Forces on positions
            fx = np.zeros(N)
            fy = np.zeros(N)
            fr = np.zeros(N)
            
            # Boundary forces (push circles inward)
            for i in range(N):
                if x[i] < r[i]:
                    push = (r[i] - x[i]) * 10.0
                    fx[i] += push
                if x[i] + r[i] > w:
                    push = (x[i] + r[i] - w) * 10.0
                    fx[i] -= push
                if y[i] < r[i]:
                    push = (r[i] - y[i]) * 10.0
                    fy[i] += push
                if y[i] + r[i] > h:
                    push = (y[i] + r[i] - h) * 10.0
                    fy[i] -= push
            
            # Pair forces: repel when overlapping, attract toward contact
            for k in range(len(idx_i)):
                i = idx_i[k]
                j = idx_j[k]
                
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq + 1e-12)
                target = r[i] + r[j]
                
                diff = dist - target
                if diff < 0:
                    k_spring = 50.0
                    force = k_spring * diff
                else:
                    k_spring = 2.0
                    force = k_spring * diff
                
                if dist > 1e-10:
                    fx[i] += force * (dx / dist)
                    fy[i] += force * (dy / dist)
                    fx[j] -= force * (dx / dist)
                    fy[j] -= force * (dy / dist)
                
                if diff > 0.01:
                    fr[i] += 0.1
                    fr[j] += 0.1
                elif diff < -0.01:
                    fr[i] -= 0.5
                    fr[j] -= 0.5
            
            # Update positions with adaptive step
            x_new = x + step_size * fx
            y_new = y + step_size * fy
            r_new = r + step_size * 0.5 * fr
            
            # Clamp
            for i in range(N):
                x_new[i] = max(r[i], min(w - r[i], x_new[i]))
                y_new[i] = max(r[i], min(h - r[i], y_new[i]))
                r_new[i] = max(1e-4, min(0.5 * min(w, h), r_new[i]))
            
            # Build new state
            new_state = s.copy()
            for i in range(N):
                new_state[1 + 3*i] = x_new[i]
                new_state[2 + 3*i] = y_new[i]
                new_state[3 + 3*i] = r_new[i]
            
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
        Run simulated annealing to improve state.
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
            
            # Generate neighbor
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
    
    # Multiple restarts with different aspect ratios and initialization strategies
    num_restarts = 15
    
    init_strategies = ['hex', 'grid', 'random']
    
    for restart in range(num_restarts):
        if time.time() > deadline - 40:
            break
        
        # Vary aspect ratio for diversity
        t_init = 0.5 + random.uniform(-0.25, 0.25)
        
        # Choose initialization strategy
        strategy = init_strategies[restart % len(init_strategies)]
        
        if strategy == 'hex':
            state = make_initial_hexagonal(t_init, jitter=0.02)
        elif strategy == 'grid':
            state = make_initial_grid(t_init, jitter=0.02)
        else:
            state = make_initial_random(t_init)
        
        # Time budget for this restart
        remaining = deadline - 40 - time.time()
        if remaining < 10:
            break
        
        # Allocate time for SA, force relax, and polish
        sa_time = max(5, remaining * 0.4)
        sa_deadline = time.time() + sa_time
        
        state, score = local_optimize(state, sa_deadline)
        
        # Force-directed relaxation phase
        if time.time() < deadline - 25:
            force_deadline = time.time() + min(8, (deadline - 25 - time.time()) * 0.4)
            force_state = force_relax(state, max_iter=3000)
            force_score, _ = evaluate(force_state)
            if force_score > score:
                state = force_state
                score = force_score
        
        # Gradient polish phase
        if time.time() < deadline - 15:
            polish_iter = 0
            for polish_iter in range(3):
                if time.time() > deadline - 12:
                    break
                polished = polish_gradient(state, max_iter=1500)
                pol_score, pol_pen = evaluate(polished)
                if pol_score > score:
                    state = polished
                    score = pol_score
                else:
                    break
        
        if score > best_score:
            best_score = score
            best_state = state.copy()
    
    # Final high-precision polish on the best solution
    if best_state is not None and time.time() < deadline - 8:
        # Force relaxation first
        if time.time() < deadline - 10:
            force_final = force_relax(best_state, max_iter=2000)
            ff_score, _ = evaluate(force_final)
            if ff_score > best_score:
                best_state = force_final
                best_score = ff_score
        
        final_polished = polish_gradient(best_state, max_iter=3000)
        fp_score, fp_pen = evaluate(final_polished)
        if fp_score > best_score:
            best_state = final_polished
            best_score = fp_score
        
        # One more polish pass if time allows
        if time.time() < deadline - 5:
            second_polish = polish_gradient(best_state, max_iter=2000)
            sp_score, sp_pen = evaluate(second_polish)
            if sp_score > best_score:
                best_state = second_polish
                best_score = sp_score
    
    # Final validity check and repair if needed
    if best_state is None:
        best_state = make_initial_hexagonal(0.5)
    elif not check_valid(best_state):
        # Try to repair by shrinking radii and re-polishing
        repair = best_state.copy()
        repaired = False
        for shrink in range(30):
            if check_valid(repair):
                repaired = True
                break
            for i in range(N):
                repair[1 + 3*i + 2] *= 0.97
            t_r = repair[0]
            w_r, h_r = get_rect_dims(t_r)
            for i in range(N):
                r_r = repair[1 + 3*i + 2]
                x_r = repair[1 + 3*i]
                y_r = repair[1 + 3*i + 1]
                x_r = max(r_r, min(w_r - r_r, x_r))
                y_r = max(r_r, min(h_r - r_r, y_r))
                repair[1 + 3*i] = x_r
                repair[2 + 3*i] = y_r
        if repaired:
            best_state = repair
        else:
            best_state = make_initial_hexagonal(0.5)
    
    # Extract result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_state[1 + 3*i]
        result[i, 1] = best_state[2 + 3*i]
        result[i, 2] = best_state[3 + 3*i]
    
    # Final clamp to ensure validity
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
