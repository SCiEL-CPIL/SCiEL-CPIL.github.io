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

    def make_initial_hexagonal(t_init=0.5, jitter=0.0):
        """
        Create an initial state using a hexagonal lattice pattern.
        This provides a much better starting point than square grids.
        """
        w, h = get_rect_dims(t_init)
        
        state = np.zeros(1 + 3*N)
        state[0] = t_init
        
        # For hexagonal packing, we want rows offset by half a diameter.
        # Let's try to fit as many circles as possible in a hexagonal pattern.
        # We will place circles row by row.
        
        # Estimate radius. In a hexagonal packing of N circles in a rectangle,
        # the radius is roughly related to area.
        # Area = N * pi * r^2. Rectangle area = w*h.
        # Packing density ~ 0.9069.
        # N * pi * r^2 ~ 0.9069 * w * h
        # r ~ sqrt(0.9069 * w * h / (N * pi))
        
        r_est = np.sqrt(0.9069 * w * h / (N * np.pi))
        
        # Hexagonal spacing:
        # Horizontal distance between centers in same row: d = 2r
        # Vertical distance between rows: dy = sqrt(3) * r
        # Odd rows shifted by r horizontally.
        
        # Let's start with a guess for r and adjust if needed.
        # We will place circles and check if they fit.
        
        rows = []
        r_current = r_est
        
        # We need to place N circles.
        # Let's simulate the placement.
        
        # The width available for centers is [r, w-r].
        # The height available for centers is [r, h-r].
        
        # Let's try to fit circles.
        # If r is too big, they won't fit. If too small, we have space.
        
        # We will just place them in a grid-like hex pattern and then let the optimizer adjust.
        # To ensure a good start, we pick r such that the hex lattice fits.
        
        # Number of circles per row depends on row index.
        # Row 0: x = r + i * 2r
        # Row 1: x = r + r + i * 2r = r(2+i)
        
        # Let's just generate positions based on r_current and see how many fit.
        # We'll iterate to find a good r.
        
        # Simple approach: start with r_est, place circles, if any go out of bounds, shrink r.
        
        positions = []
        
        # Try to fill the rectangle
        # Max rows = floor((h - 2r) / (sqrt(3)*r)) + 1
        # Max cols in row 0 = floor((w - 2r) / (2r)) + 1
        
        # We'll just generate a dense hex grid and take the first N that fit.
        # If they don't fit, we shrink r.
        
        for attempt in range(20):
            positions = []
            y = r_current
            row_idx = 0
            while y <= h - r_current + 1e-9:
                # Determine x start
                if row_idx % 2 == 0:
                    x_start = r_current
                else:
                    x_start = r_current + r_current # offset by r
                
                x = x_start
                while x <= w - r_current + 1e-9:
                    positions.append((x, y, r_current))
                    x += 2.0 * r_current
                y += np.sqrt(3) * r_current
                row_idx += 1
            
            if len(positions) >= N:
                # We have enough positions. Take the first N.
                # But we might want to center them or pick the best N.
                # Taking the first N (top-left) is fine for initialization.
                break
            else:
                # Shrink radius
                r_current *= 0.95
        
        # If we still don't have enough, just use a smaller r
        if len(positions) < N:
            # Fallback to square grid
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
        
        # Fill state
        for i in range(N):
            if i < len(positions):
                x, y, r = positions[i]
                # Add jitter
                if jitter > 0:
                    x += random.uniform(-jitter, jitter)
                    y += random.uniform(-jitter, jitter)
                state[1 + 3*i] = x
                state[1 + 3*i + 1] = y
                state[1 + 3*i + 2] = r
            else:
                # Should not happen
                state[1 + 3*i] = w/2
                state[1 + 3*i + 1] = h/2
                state[1 + 3*i + 2] = 0.01
        
        return state

    def polish_gradient(state, time_limit, max_iter=2000):
        """
        Gradient-based polish using smooth penalty formulation with proper gradients.
        """
        from scipy.optimize import minimize
        
        # Precompute pair indices
        idx_i, idx_j = np.triu_indices(N, k=1)
        
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
            
            # Smooth boundary penalties
            viol_l = np.maximum(0, radii - x)
            viol_r = np.maximum(0, x + radii - w)
            viol_b = np.maximum(0, radii - y)
            viol_t = np.maximum(0, y + radii - h)
            
            # Quadratic penalty for smoothness
            obj += 1000.0 * (np.sum(viol_l**2) + np.sum(viol_r**2) + np.sum(viol_b**2) + np.sum(viol_t**2))
            
            # Overlap penalty
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[idx_i] + radii[idx_j]
            
            # Smooth penalty: max(0, min_dist^2 - dist_sq)
            overlap_amount = min_dist**2 - dist_sq
            smooth_overlap = 0.5 * (overlap_amount + np.sqrt(overlap_amount**2 + 1e-8))
            obj += 5000.0 * np.sum(smooth_overlap**2)
            
            # Gradient calculation
            grad = np.zeros_like(s)
            
            # d(-sum_r)/dr_i = -1
            for i in range(N):
                grad[1 + 3*i + 2] += -1.0
            
            # Boundary gradients
            # d/dx_i of viol_l^2: if viol_l > 0, d/dx (r-x)^2 = -2(r-x)
            # d/dr_i of viol_l^2: if viol_l > 0, d/dr (r-x)^2 = 2(r-x)
            
            for i in range(N):
                if viol_l[i] > 0:
                    grad[1 + 3*i] += -2.0 * 1000.0 * viol_l[i]
                    grad[1 + 3*i + 2] += 2.0 * 1000.0 * viol_l[i]
                
                if viol_r[i] > 0:
                    grad[1 + 3*i] += 2.0 * 1000.0 * viol_r[i]
                    grad[1 + 3*i + 2] += 2.0 * 1000.0 * viol_r[i]
                
                if viol_b[i] > 0:
                    grad[1 + 3*i + 1] += -2.0 * 1000.0 * viol_b[i]
                    grad[1 + 3*i + 2] += 2.0 * 1000.0 * viol_b[i]
                
                if viol_t[i] > 0:
                    grad[1 + 3*i + 1] += 2.0 * 1000.0 * viol_t[i]
                    grad[1 + 3*i + 2] += 2.0 * 1000.0 * viol_t[i]
            
            # Overlap gradients
            # smooth_overlap = 0.5 * (A + sqrt(A^2 + eps)) where A = min_dist^2 - dist_sq
            # d(smooth_overlap)/dA = 0.5 * (1 + A/sqrt(A^2+eps))
            # dA/dx_i = -2*dx
            # dA/dx_j = 2*dx
            # dA/dy_i = -2*dy
            # dA/dy_j = 2*dy
            # dA/dr_i = 2*min_dist
            # dA/dr_j = 2*min_dist
            
            # obj += 5000 * sum(so^2)
            # d(obj)/dx_i = 5000 * 2 * so * d(so)/dA * dA/dx_i
            
            so = smooth_overlap
            d_so_dA = 0.5 * (1.0 + overlap_amount / np.sqrt(overlap_amount**2 + 1e-8))
            
            # Factor: 5000 * 2 * so * d_so_dA
            factor = 10000.0 * so * d_so_dA
            
            for k in range(len(idx_i)):
                i = idx_i[k]
                j = idx_j[k]
                
                # x_i
                grad[1 + 3*i] += factor[k] * (-2.0 * dx[k])
                # x_j
                grad[1 + 3*j] += factor[k] * (2.0 * dx[k])
                # y_i
                grad[1 + 3*i + 1] += factor[k] * (-2.0 * dy[k])
                # y_j
                grad[1 + 3*j + 1] += factor[k] * (2.0 * dy[k])
                # r_i
                grad[1 + 3*i + 2] += factor[k] * (2.0 * min_dist[k])
                # r_j
                grad[1 + 3*j + 2] += factor[k] * (2.0 * min_dist[k])
            
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
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
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
            else:
                pass
            
            T *= alpha
            if T < T_end:
                T = T_end
        
        return best_state, best_score

    # Main algorithm
    best_state = None
    best_score = -np.inf
    
    # Multiple restarts with different aspect ratios and initializations
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline - 50:
            break
        
        # Vary aspect ratio for diversity
        t_init = 0.5 + random.uniform(-0.2, 0.2)
        
        # Use hexagonal initialization
        state = make_initial_hexagonal(t_init, jitter=0.02)
        
        # Time budget for this restart
        remaining = deadline - 50 - time.time()
        if remaining < 15:
            break
        
        # Allocate time for SA and polish
        sa_time = max(8, remaining * 0.5)
        polish_time = max(10, remaining * 0.45)
        sa_deadline = time.time() + sa_time
        
        state, score = local_optimize(state, sa_deadline)
        
        # Gradient polish phase
        if time.time() < deadline - 20:
            for polish_iter in range(3):
                if time.time() > deadline - 15:
                    break
                polish_deadline = time.time() + min(15, (deadline - 15 - time.time()) / (3 - polish_iter))
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
    
    # Final validity check and repair if needed
    if best_state is None:
        best_state = make_initial_hexagonal(0.5)
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
            best_state = make_initial_hexagonal(0.5)
    
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
