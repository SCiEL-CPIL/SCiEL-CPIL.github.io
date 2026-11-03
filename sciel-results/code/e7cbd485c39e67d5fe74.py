"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

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
        
        # Overlap violations
        idx_i, idx_j = np.triu_indices(N, k=1)
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
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = radii[i] + radii[j] - eps
                if dist_sq < min_dist * min_dist:
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
        r_current = r_est
        positions = []
        
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

    def make_initial_grid(t_init=0.5, jitter=0.0):
        """
        Create an initial state using a square grid pattern.
        """
        w, h = get_rect_dims(t_init)
        state = np.zeros(1 + 3*N)
        state[0] = t_init
        
        best_positions = None
        best_mean_r = 0
        for rows in range(3, 8):
            cols = int(np.ceil(N / rows))
            if rows * cols < N:
                continue
            positions = []
            for i in range(rows):
                for j in range(cols):
                    if len(positions) >= N:
                        break
                    gx = (j + 0.5) / cols * w
                    gy = (i + 0.5) / rows * h
                    r = 0.5 * min(w / cols, h / rows) * 0.9
                    positions.append((gx, gy, r))
            if len(positions) >= N:
                mean_r = np.mean([p[2] for p in positions])
                if mean_r > best_mean_r:
                    best_mean_r = mean_r
                    best_positions = positions
        
        if best_positions is None:
            side = 5
            best_positions = []
            for i in range(side):
                for j in range(side):
                    if len(best_positions) >= N:
                        break
                    gx = (i + 0.5) / side * w
                    gy = (j + 0.5) / side * h
                    r = 0.5 / side * min(w, h) * 0.95
                    best_positions.append((gx, gy, r))
        
        for i in range(N):
            x, y, r = best_positions[i]
            if jitter > 0:
                x += random.uniform(-jitter, jitter)
                y += random.uniform(-jitter, jitter)
            state[1 + 3*i] = x
            state[1 + 3*i + 1] = y
            state[1 + 3*i + 2] = r
        return state

    def make_initial_random(t_init=0.5):
        """
        Create a random initial state with non-overlapping circles.
        """
        w, h = get_rect_dims(t_init)
        state = np.zeros(1 + 3*N)
        state[0] = t_init
        
        positions = []
        max_attempts = 1000
        r_est = np.sqrt(0.9069 * w * h / (N * np.pi))
        
        for i in range(N):
            placed = False
            for _ in range(max_attempts):
                r = r_est * random.uniform(0.7, 1.0)
                x = random.uniform(r, w - r)
                y = random.uniform(r, h - r)
                
                # Check overlap with existing
                ok = True
                for (px, py, pr) in positions:
                    if (x - px)**2 + (y - py)**2 < (r + pr)**2:
                        ok = False
                        break
                if ok:
                    positions.append((x, y, r))
                    placed = True
                    break
            
            if not placed:
                # Fallback: place with smaller radius
                r = r_est * 0.5
                x = random.uniform(r, w - r)
                y = random.uniform(r, h - r)
                positions.append((x, y, r))
        
        for i in range(N):
            x, y, r = positions[i]
            state[1 + 3*i] = x
            state[1 + 3*i + 1] = y
            state[1 + 3*i + 2] = r
        return state

    def polish_gradient(state, time_limit, max_iter=2000):
        """
        Gradient-based polish using squared-distance constraints.
        Minimizes penalty (negative of score) using simple gradient descent.
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
            
            # Add penalty for violations using smooth approximations
            # Boundary
            obj += 100.0 * np.sum(np.maximum(0, radii - x)**2)
            obj += 100.0 * np.sum(np.maximum(0, x + radii - w)**2)
            obj += 100.0 * np.sum(np.maximum(0, radii - y)**2)
            obj += 100.0 * np.sum(np.maximum(0, y + radii - h)**2)
            
            # Overlap using squared distance
            idx_i, idx_j = np.triu_indices(N, k=1)
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            min_dist_sq = (radii[idx_i] + radii[idx_j])**2
            # Penalty for overlap: (min_dist_sq - dist_sq)^2 where positive
            overlap_pen = np.maximum(0, min_dist_sq - dist_sq)
            obj += 1000.0 * np.sum(overlap_pen**2)
            
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
                    grad[1 + 3*i] += -2.0 * 100.0 * viol_l[i]
                    grad[1 + 3*i + 2] += 2.0 * 100.0 * viol_l[i]
            
            viol_r = np.maximum(0, x + radii - w)
            for i in range(N):
                if viol_r[i] > 0:
                    grad[1 + 3*i] += 2.0 * 100.0 * viol_r[i]
                    grad[1 + 3*i + 2] += 2.0 * 100.0 * viol_r[i]
            
            viol_b = np.maximum(0, radii - y)
            for i in range(N):
                if viol_b[i] > 0:
                    grad[1 + 3*i + 1] += -2.0 * 100.0 * viol_b[i]
                    grad[1 + 3*i + 2] += 2.0 * 100.0 * viol_b[i]
            
            viol_t = np.maximum(0, y + radii - h)
            for i in range(N):
                if viol_t[i] > 0:
                    grad[1 + 3*i + 1] += 2.0 * 100.0 * viol_t[i]
                    grad[1 + 3*i + 2] += 2.0 * 100.0 * viol_t[i]
            
            # Overlap gradients (squared distance formulation)
            idx_i, idx_j = np.triu_indices(N, k=1)
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[idx_i] + radii[idx_j]
            min_dist_sq = min_dist**2
            
            pen_val = np.maximum(0, min_dist_sq - dist_sq)
            mask = pen_val > 0
            
            if np.any(mask):
                for k in range(np.sum(mask)):
                    i = idx_i[mask][k]
                    j = idx_j[mask][k]
                    dv = 2.0 * 1000.0 * pen_val[mask][k]
                    ddx = -2.0 * dx[mask][k]
                    ddy = -2.0 * dy[mask][k]
                    grad[1 + 3*i] += dv * ddx
                    grad[1 + 3*j] += -dv * ddx
                    grad[1 + 3*i + 1] += dv * ddy
                    grad[1 + 3*j + 1] += -dv * ddy
                    grad[1 + 3*i + 2] += 2.0 * 1000.0 * 2.0 * min_dist[mask][k] * pen_val[mask][k]
                    grad[1 + 3*j + 2] += 2.0 * 1000.0 * 2.0 * min_dist[mask][k] * pen_val[mask][k]
            
            return grad
        
        bounds = [(0.1, 0.9)]
        for i in range(N):
            bounds.append((0, 2.0))
            bounds.append((0, 2.0))
            bounds.append((1e-4, 0.5))
        
        try:
            result = minimize(
                objective,
                state,
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
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
        
        T_start = 0.5
        T_end = 1e-5
        alpha = 0.9995
        
        T = T_start
        no_improve_count = 0
        
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
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            T *= alpha
            if T < T_end:
                T = T_end
        
        return best_state, best_score

    # Main algorithm
    best_state = None
    best_score = -np.inf
    
    # Multiple restarts
    num_restarts = 6
    
    for restart in range(num_restarts):
        if time.time() > deadline - 40:
            break
        
        # Create initial state
        if restart == 0:
            state = make_initial_hexagonal(0.5, 0.0)
        elif restart == 1:
            state = make_initial_grid(0.5, 0.0)
        elif restart == 2:
            state = make_initial_random(0.5)
        elif restart == 3:
            state = make_initial_hexagonal(0.4, 0.05)
        elif restart == 4:
            state = make_initial_grid(0.6, 0.05)
        else:
            state = make_initial_random(0.5)
            w, h = get_rect_dims(state[0])
            for i in range(N):
                state[1 + 3*i] += random.uniform(-0.05, 0.05) * min(w, h)
                state[1 + 3*i + 1] += random.uniform(-0.05, 0.05) * min(w, h)
                state[1 + 3*i + 2] *= random.uniform(0.8, 1.0)
            state[0] += random.uniform(-0.1, 0.1)
        
        remaining = deadline - 40 - time.time()
        if remaining < 10:
            break
        
        sa_time = max(5, remaining * 0.6)
        polish_time = max(5, remaining * 0.35)
        sa_deadline = time.time() + sa_time
        
        state, score = local_optimize(state, sa_deadline)
        
        if time.time() < deadline - 15:
            polish_deadline = time.time() + polish_time
            polished = polish_gradient(state, polish_deadline)
            pol_score, pol_pen = evaluate(polished)
            if pol_score > score:
                state = polished
                score = pol_score
        
        if score > best_score:
            best_score = score
            best_state = state.copy()
        
        if check_valid(best_state):
            refined = best_state.copy()
            w, h = get_rect_dims(refined[0])
            for i in range(N):
                r_idx = 1 + 3*i + 2
                refined[r_idx] *= 1.05
            quick_deadline = min(time.time() + 15, deadline - 10)
            refined, ref_score = local_optimize(refined, quick_deadline)
            if time.time() < deadline - 10:
                polished_ref = polish_gradient(refined, min(time.time() + 10, deadline - 5))
                pr_score, _ = evaluate(polished_ref)
                if pr_score > ref_score:
                    refined = polished_ref
                    ref_score = pr_score
            if ref_score > best_score and check_valid(refined):
                best_score = ref_score
                best_state = refined.copy()
    
    # Final polish
    if best_state is not None and time.time() < deadline - 10:
        final_polish_deadline = deadline - 5
        final_polished = polish_gradient(best_state, final_polish_deadline)
        fp_score, fp_pen = evaluate(final_polished)
        if fp_score > best_score:
            best_state = final_polished
            best_score = fp_score
    
    # Final validity check
    if best_state is None:
        best_state = make_initial_hexagonal(0.5, 0.0)
    elif not check_valid(best_state):
        repair = best_state.copy()
        for _ in range(50):
            if check_valid(repair):
                break
            for i in range(N):
                repair[1 + 3*i + 2] *= 0.98
        if check_valid(repair):
            best_state = repair
        else:
            best_state = make_initial_hexagonal(0.5, 0.0)
    
    # Extract result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_state[1 + 3*i]
        result[i, 1] = best_state[1 + 3*i+1]
        result[i, 2] = best_state[1 + 3*i+2]
    
    return result

# EVOLVE-BLOCK-END
