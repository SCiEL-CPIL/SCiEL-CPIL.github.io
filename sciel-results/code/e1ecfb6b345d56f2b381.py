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
        t = np.clip(t, 0.15, 0.85)
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h

    def pack_cost(state):
        """
        Cost function for SLSQP: minimize negative sum of radii.
        state: array of shape (1 + 3*N,)
        Returns: -sum_r (to minimize)
        """
        t = state[0]
        w, h = get_rect_dims(t)

        x = state[1:1+3*N:3]
        y = state[2:2+3*N:3]
        r = state[3:3+3*N:3]

        # Ensure positive radii
        r = np.maximum(r, 1e-6)

        cost = -np.sum(r)

        # Constraints
        constraints = []

        # Boundary constraints: x - r >= 0, x + r <= w, y - r >= 0, y + r <= h
        for i in range(N):
            # x_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda s, i=i: s[1+3*i] - s[3+3*i]})
            # x_i + r_i <= w
            constraints.append({'type': 'ineq', 'fun': lambda s, i=i: get_rect_dims(s[0])[0] - s[1+3*i] - s[3+3*i]})
            # y_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda s, i=i: s[2+3*i] - s[3+3*i]})
            # y_i + r_i <= h
            constraints.append({'type': 'ineq', 'fun': lambda s, i=i: get_rect_dims(s[0])[1] - s[2+3*i] - s[3+3*i]})

        # Overlap constraints: dist(i,j) >= r_i + r_j
        for i in range(N):
            for j in range(i+1, N):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda s, i=i, j=j: np.sqrt((s[1+3*i]-s[1+3*j])**2 + (s[2+3*i]-s[2+3*j])**2) - s[3+3*i] - s[3+3*j]
                })

        return cost, constraints

    def check_valid(state, eps=1e-7):
        """
        Check if state is valid: all circles inside rectangle, no overlaps.
        """
        t = state[0]
        w, h = get_rect_dims(t)

        x = state[1:1+3*N:3]
        y = state[2:2+3*N:3]
        r = state[3:3+3*N:3]

        if np.any(r < 1e-6):
            return False

        # Boundary check
        if np.any(x - r < -eps) or np.any(x + r > w + eps) or \
           np.any(y - r < -eps) or np.any(y + r > h + eps):
            return False

        # Overlap check
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j] - eps
                if dist_sq < min_dist * min_dist:
                    return False
        return True

    def make_initial_grid(t_val):
        """
        Create an initial state using a grid pattern.
        """
        w, h = get_rect_dims(t_val)
        state = np.zeros(1 + 3*N)
        state[0] = t_val

        # Determine grid dimensions
        # Try to find a good grid for 21 circles
        # 21 = 3*7, so 3 rows x 7 cols or 7 rows x 3 cols
        # Or 5x5 grid with 21 filled

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
                r = 0.5 / side * min(w, h) * 0.9
                state[1 + 3*idx] = gx
                state[2 + 3*idx] = gy
                state[3 + 3*idx] = r
                idx += 1

        return state

    def make_initial_hex(t_val):
        """
        Create an initial state using a hexagonal packing pattern.
        """
        w, h = get_rect_dims(t_val)
        state = np.zeros(1 + 3*N)
        state[0] = t_val

        # Hexagonal packing: rows offset
        # Try to fit circles in hex pattern
        r_base = min(w, h) / 6.0 * 0.8
        dx = 2.0 * r_base
        dy = np.sqrt(3) * r_base

        idx = 0
        row = 0
        while idx < N:
            offset = (dx / 2) if (row % 2 == 1) else 0
            col = 0
            while idx < N:
                x = offset + col * dx + r_base
                y = row * dy + r_base
                if x + r_base <= w and y + r_base <= h:
                    state[1 + 3*idx] = x
                    state[2 + 3*idx] = y
                    state[3 + 3*idx] = r_base
                    idx += 1
                col += 1
            row += 1

        return state

    def make_initial_random(t_val):
        """
        Create a random initial state.
        """
        w, h = get_rect_dims(t_val)
        state = np.zeros(1 + 3*N)
        state[0] = t_val

        r_base = min(w, h) / 5.0 * 0.7
        for i in range(N):
            x = r_base + random.random() * (w - 2*r_base)
            y = r_base + random.random() * (h - 2*r_base)
            r = r_base * random.uniform(0.7, 1.0)
            state[1 + 3*i] = x
            state[2 + 3*i] = y
            state[3 + 3*i] = r

        return state

    def local_optimize_slsqp(state, time_limit):
        """
        Run SLSQP optimization to improve state.
        """
        t = state[0]
        w, h = get_rect_dims(t)

        # Define objective: minimize -sum(r)
        def objective(s):
            r = s[3:3+3*N:3]
            r = np.maximum(r, 1e-6)
            return -np.sum(r)

        # Define constraints as a list of dictionaries
        # To avoid lambda closure issues, we'll use a different approach
        # We'll define constraints using a helper that captures i,j

        cons = []

        # Boundary constraints
        for i in range(N):
            # x_i - r_i >= 0
            def bound_left(s, i=i):
                return s[1+3*i] - s[3+3*i]
            cons.append({'type': 'ineq', 'fun': bound_left})

            # x_i + r_i <= w(s[0])
            def bound_right(s, i=i):
                ww, _ = get_rect_dims(s[0])
                return ww - s[1+3*i] - s[3+3*i]
            cons.append({'type': 'ineq', 'fun': bound_right})

            # y_i - r_i >= 0
            def bound_bottom(s, i=i):
                return s[2+3*i] - s[3+3*i]
            cons.append({'type': 'ineq', 'fun': bound_bottom})

            # y_i + r_i <= h(s[0])
            def bound_top(s, i=i):
                _, hh = get_rect_dims(s[0])
                return hh - s[2+3*i] - s[3+3*i]
            cons.append({'type': 'ineq', 'fun': bound_top})

        # Overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                def overlap(s, i=i, j=j):
                    dx = s[1+3*i] - s[1+3*j]
                    dy = s[2+3*i] - s[2+3*j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    return dist - s[3+3*i] - s[3+3*j]
                cons.append({'type': 'ineq', 'fun': overlap})

        # Bounds
        bounds = []
        bounds.append((0.15, 0.85))  # t
        max_r = 0.5 * min(w, h)
        for i in range(N):
            bounds.append((1e-6, max_r * 2))  # x_i
            bounds.append((1e-6, max_r * 2))  # y_i
            bounds.append((1e-6, max_r))  # r_i

        # Run SLSQP
        try:
            result = minimize(
                objective,
                state,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            if result.success or result.fun < objective(state):
                return result.x
        except Exception:
            pass

        return state

    def polish(state):
        """
        Final high-precision polish pass.
        """
        # Run SLSQP with tighter tolerance
        t = state[0]
        w, h = get_rect_dims(t)

        def objective(s):
            r = s[3:3+3*N:3]
            r = np.maximum(r, 1e-6)
            return -np.sum(r)

        cons = []
        for i in range(N):
            def bound_left(s, i=i):
                return s[1+3*i] - s[3+3*i]
            cons.append({'type': 'ineq', 'fun': bound_left})

            def bound_right(s, i=i):
                ww, _ = get_rect_dims(s[0])
                return ww - s[1+3*i] - s[3+3*i]
            cons.append({'type': 'ineq', 'fun': bound_right})

            def bound_bottom(s, i=i):
                return s[2+3*i] - s[3+3*i]
            cons.append({'type': 'ineq', 'fun': bound_bottom})

            def bound_top(s, i=i):
                _, hh = get_rect_dims(s[0])
                return hh - s[2+3*i] - s[3+3*i]
            cons.append({'type': 'ineq', 'fun': bound_top})

        for i in range(N):
            for j in range(i+1, N):
                def overlap(s, i=i, j=j):
                    dx = s[1+3*i] - s[1+3*j]
                    dy = s[2+3*i] - s[2+3*j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    return dist - s[3+3*i] - s[3+3*j]
                cons.append({'type': 'ineq', 'fun': overlap})

        bounds = []
        bounds.append((0.15, 0.85))
        max_r = 0.5 * min(w, h)
        for i in range(N):
            bounds.append((1e-6, max_r * 2))
            bounds.append((1e-6, max_r * 2))
            bounds.append((1e-6, max_r))

        try:
            result = minimize(
                objective,
                state,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            if result.success or result.fun < objective(state):
                return result.x
        except Exception:
            pass

        return state

    # Main algorithm
    best_state = None
    best_sum_r = -np.inf

    # Multiple restarts with different initializations and aspect ratios
    t_values = [0.5, 0.45, 0.55, 0.4, 0.6, 0.5, 0.5, 0.48, 0.52, 0.5]

    for restart in range(len(t_values)):
        if time.time() > deadline - 15:
            break

        t_val = t_values[restart]

        # Choose initialization method
        if restart % 3 == 0:
            state = make_initial_grid(t_val)
        elif restart % 3 == 1:
            state = make_initial_hex(t_val)
        else:
            state = make_initial_random(t_val)

        # SLSQP optimization
        remaining = deadline - 15 - time.time()
        if remaining < 5:
            break

        state = local_optimize_slsqp(state, remaining)

        # Check validity and compute sum
        if check_valid(state):
            r = state[3:3+3*N:3]
            sum_r = np.sum(r)
            if sum_r > best_sum_r:
                best_sum_r = sum_r
                best_state = state.copy()

    # Polish the best solution if time allows
    if best_state is not None and time.time() < deadline - 10:
        polished = polish(best_state)
        if check_valid(polished):
            r = polished[3:3+3*N:3]
            sum_r = np.sum(r)
            if sum_r > best_sum_r:
                best_sum_r = sum_r
                best_state = polished.copy()

    # Final validity check and repair if needed
    if best_state is None:
        # Fallback: return a simple valid grid
        best_state = make_initial_grid(0.5)
    elif not check_valid(best_state):
        # Try to repair by shrinking radii
        repair = best_state.copy()
        for _ in range(100):
            if check_valid(repair):
                break
            for i in range(N):
                repair[3 + 3*i] *= 0.98
        if check_valid(repair):
            best_state = repair
        else:
            best_state = make_initial_grid(0.5)

    # Extract result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_state[1 + 3*i]
        result[i, 1] = best_state[2 + 3*i]
        result[i, 2] = best_state[3 + 3*i]

    return result

# EVOLVE-BLOCK-END
