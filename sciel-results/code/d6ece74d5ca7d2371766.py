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

    def make_hex_initial(t_init=0.5, shear=0.0):
        """
        Create a hexagonal lattice initial state.
        """
        w, h = get_rect_dims(t_init)
        state = np.zeros(1 + 3*N)
        state[0] = t_init

        # Hexagonal lattice parameters
        # For hexagonal packing, the unit cell has width 2*r and height sqrt(3)*r
        # Let's try to fit a hexagonal grid.
        # Number of columns and rows to get ~21 circles.
        # Let's try a 5x5 grid with hexagonal offset.
        # Or just generate a dense hexagonal grid and pick the first 21 that fit.

        # Let's estimate radius.
        # Area of rect = w*h. Area of 21 circles = 21 * pi * r^2.
        # Packing fraction ~ 0.9 for hexagonal.
        # 21 * pi * r^2 ~ 0.9 * w * h
        # r ~ sqrt(0.9 * w * h / (21 * pi))
        est_r = np.sqrt(0.9 * w * h / (21 * np.pi))

        # Hexagonal lattice vectors
        a = 2.0 * est_r * 1.02  # slight expansion to avoid overlap
        b = np.sqrt(3) * est_r * 1.02

        # Generate points
        points = []
        # Try a few rows
        for row in range(10):
            for col in range(10):
                x = (col + 0.5) * a + (row % 2) * (a / 2.0) + shear * row * a
                y = (row + 0.5) * b
                if x < w and y < h and x > 0 and y > 0:
                    points.append([x, y, est_r])

        # Sort by distance from center? Or just take first N?
        # Let's take the ones closest to the center to maximize fit.
        cx, cy = w/2, h/2
        points.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)

        if len(points) < N:
            # Fallback to grid
            return make_grid_initial(t_init)

        for i in range(N):
            state[1 + 3*i] = points[i][0]
            state[1 + 3*i + 1] = points[i][1]
            state[1 + 3*i + 2] = points[i][2]

        return state

    def make_grid_initial(t_init=0.5):
        """
        Create an initial state using a grid pattern.
        """
        w, h = get_rect_dims(t_init)
        state = np.zeros(1 + 3*N)
        state[0] = t_init

        side = 5
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                gx = (i + 0.5) / side * w
                gy = (j + 0.5) / side * h
                r = 0.5 / side * min(w, h) * 0.95
                state[1 + 3*idx] = gx
                state[1 + 3*idx + 1] = gy
                state[1 + 3*idx + 2] = r
                idx += 1

        return state

    def polish_gradient(state, time_limit, max_iter=3000):
        """
        Gradient-based polish using smooth penalty formulation with proper gradients.
        """
        from scipy.optimize import minimize

        # Precompute pair indices
        idx_i, idx_j = np.triu_indices(N, k=1)

        def objective_and_grad(s):
            t = s[0]
            w, h = get_rect_dims(t)

            x = s[1:1+3*N:3]
            y = s[2:2+3*N:3]
            r = s[3:3+3*N:3]

            # Negative sum of radii (we minimize)
            obj = -np.sum(r)
            grad = np.zeros_like(s)
            grad[3:3+3*N:3] = -1.0  # d(-sum_r)/dr_i = -1

            # Smooth boundary penalties using softplus
            eps_b = 1e-3
            def softplus(v):
                # Stable softplus
                return np.where(v > 0, v + eps_b * np.log1p(np.exp(-v/eps_b)), eps_b * np.log1p(np.exp(v/eps_b)))

            def sigmoid(v):
                return np.where(v > 0, 1/(1+np.exp(-v/eps_b)), np.exp(v/eps_b)/(1+np.exp(v/eps_b)))

            # Left: violation = r_i - x_i
            v_l = r - x
            obj += 1000.0 * np.sum(softplus(v_l))
            sig_l = sigmoid(v_l)
            grad[1:1+3*N:3] += -1000.0 * sig_l
            grad[3:3+3*N:3] += 1000.0 * sig_l

            # Right: violation = x_i + r_i - w
            v_r = x + r - w
            obj += 1000.0 * np.sum(softplus(v_r))
            sig_r = sigmoid(v_r)
            grad[1:1+3*N:3] += 1000.0 * sig_r
            grad[3:3+3*N:3] += 1000.0 * sig_r

            # Bottom: violation = r_i - y_i
            v_b = r - y
            obj += 1000.0 * np.sum(softplus(v_b))
            sig_b = sigmoid(v_b)
            grad[2:2+3*N:3] += -1000.0 * sig_b
            grad[3:3+3*N:3] += 1000.0 * sig_b

            # Top: violation = y_i + r_i - h
            v_t = y + r - h
            obj += 1000.0 * np.sum(softplus(v_t))
            sig_t = sigmoid(v_t)
            grad[2:2+3*N:3] += 1000.0 * sig_t
            grad[3:3+3*N:3] += 1000.0 * sig_t

            # Overlap penalty using smooth approximation
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[idx_i] + r[idx_j]

            # Smooth penalty: g = min_dist - sqrt(dist_sq + eps^2)
            eps_o = 1e-3
            dist_smooth = np.sqrt(dist_sq + eps_o**2)
            g = min_dist - dist_smooth

            sp = eps_o * np.log1p(np.exp(g / eps_o))
            obj += 5000.0 * np.sum(sp)

            sig_o = 1.0 / (1.0 + np.exp(-g / eps_o))

            # d(obj)/dx_i = 5000 * sig_o * dx / dist_smooth
            # d(obj)/dx_j = -5000 * sig_o * dx / dist_smooth
            # d(obj)/dy_i = 5000 * sig_o * dy / dist_smooth
            # d(obj)/dy_j = -5000 * sig_o * dy / dist_smooth
            # d(obj)/dr_i = 5000 * sig_o
            # d(obj)/dr_j = 5000 * sig_o

            factor = 5000.0 * sig_o / dist_smooth

            # Accumulate gradients for x and y
            # We need to scatter back to the main grad array
            # This is tricky with slices. Let's use np.add.at or explicit loops if N is small.
            # N=21, pairs=210. Loops are fine.

            for k in range(len(idx_i)):
                i = idx_i[k]
                j = idx_j[k]
                f = factor[k]
                grad[1 + 3*i] += f * dx[k]
                grad[1 + 3*j] -= f * dx[k]
                grad[2 + 3*i] += f * dy[k]
                grad[2 + 3*j] -= f * dy[k]
                grad[3 + 3*i] += 5000.0 * sig_o[k]
                grad[3 + 3*j] += 5000.0 * sig_o[k]

            return obj, grad

        def objective(s):
            return objective_and_grad(s)[0]

        def gradient(s):
            return objective_and_grad(s)[1]

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
        best_score = np.sum(best_state[3:3+3*N:3]) # Quick score estimate

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

            # Quick evaluation: sum of radii minus rough penalty
            # Just use sum of radii for speed, rely on polish to fix validity
            new_score = np.sum(new_state[3:3+3*N:3])

            if new_score > best_score or random.random() < np.exp((new_score - best_score) / T):
                best_state = new_state
                best_score = new_score
                no_improve_count = 0
            else:
                no_improve_count += 1

            T *= alpha
            if T < T_end:
                T = T_end

        return best_state

    # Main algorithm
    best_state = None
    best_score = -np.inf

    # Multiple restarts
    num_restarts = 5

    for restart in range(num_restarts):
        if time.time() > deadline - 40:
            break

        if restart == 0:
            state = make_hex_initial(0.5)
        elif restart == 1:
            state = make_hex_initial(0.6)
        elif restart == 2:
            state = make_hex_initial(0.4)
        elif restart == 3:
            state = make_grid_initial(0.5)
        else:
            state = make_hex_initial(random.uniform(0.3, 0.7))

        remaining = deadline - 40 - time.time()
        if remaining < 10:
            break

        sa_time = max(5, remaining * 0.5)
        polish_time = max(5, remaining * 0.4)
        sa_deadline = time.time() + sa_time

        state = local_optimize(state, sa_deadline)

        if time.time() < deadline - 15:
            polish_deadline = time.time() + polish_time
            polished = polish_gradient(state, polish_deadline)

            # Check validity and score
            if check_valid(polished):
                score = np.sum(polished[3:3+3*N:3])
                if score > best_score:
                    best_score = score
                    best_state = polished.copy()

    # Final polish on the best solution
    if best_state is not None and time.time() < deadline - 10:
        final_polish_deadline = deadline - 5
        final_polished = polish_gradient(best_state, final_polish_deadline)
        if check_valid(final_polished):
            score = np.sum(final_polished[3:3+3*N:3])
            if score > best_score:
                best_state = final_polished
                best_score = score

    # Fallback
    if best_state is None or not check_valid(best_state):
        # Try to repair or use grid
        fallback = make_grid_initial(0.5)
        if check_valid(fallback):
            best_state = fallback
        else:
            # Shrink radii
            for _ in range(50):
                if check_valid(fallback):
                    break
                for i in range(N):
                    fallback[1 + 3*i + 2] *= 0.95
            best_state = fallback

    # Extract result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_state[1 + 3*i]
        result[i, 1] = best_state[1 + 3*i+1]
        result[i, 2] = best_state[1 + 3*i+2]

    return result

# EVOLVE-BLOCK-END
