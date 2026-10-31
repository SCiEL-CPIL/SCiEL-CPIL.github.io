import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    timeout = 110.0

    def objective(vars):
        W = vars[0]
        H = 2.0 - W
        if W <= 0.01 or H <= 0.01:
            return 1e10

        circles = vars[1:].reshape(N, 3)
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]

        # Maximize sum of radii
        sum_r = np.sum(rs)

        # Boundary penalties (quadratic)
        # x - r >= 0  =>  max(0, r - x)^2
        # x + r <= W  =>  max(0, x + r - W)^2
        # y - r >= 0  =>  max(0, r - y)^2
        # y + r <= H  =>  max(0, y + r - H)^2
        
        pen_x_left = np.maximum(0, rs - xs)
        pen_x_right = np.maximum(0, xs + rs - W)
        pen_y_bot = np.maximum(0, rs - ys)
        pen_y_top = np.maximum(0, ys + rs - H)

        boundary_penalty = np.sum(pen_x_left**2 + pen_x_right**2 + pen_y_bot**2 + pen_y_top**2)

        # Overlap penalties (quadratic)
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    overlap_penalty += (min_dist - dist)**2

        # We want to minimize -sum_r + penalties
        # Weight the penalties heavily
        return -sum_r + 100.0 * (boundary_penalty + overlap_penalty)

    def initial_guess():
        side = 5
        W_init = 1.0
        H_init = 1.0
        r_init = 0.1
        xs = []
        ys = []
        rs = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (i + 0.5) * (1.0 / side)
                y = (j + 0.5) * (1.0 / side)
                xs.append(x)
                ys.append(y)
                rs.append(r_init)
                count += 1
            if count >= N:
                break
        
        vars_init = [W_init]
        for i in range(N):
            vars_init.extend([xs[i], ys[i], rs[i]])
        return np.array(vars_init)

    bounds = [(0.1, 1.9)]  # W
    for _ in range(N):
        bounds.append((0.0, 2.0))  # x
        bounds.append((0.0, 2.0))  # y
        bounds.append((0.0, 1.0))  # r

    best_result = None
    best_score = -np.inf

    x0 = initial_guess()

    options = {'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}

    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options=options)
        if res.success:
            score = -res.fun
            if score > best_score:
                best_score = score
                best_result = res.x
    except Exception:
        pass

    # Restart with perturbations
    if time.time() - start_time < timeout:
        for restart in range(10):
            if time.time() - start_time > timeout:
                break
            x0_perturbed = x0 + np.random.normal(0, 0.1, size=x0.shape)
            x0_perturbed[0] = np.clip(x0_perturbed[0], 0.1, 1.9)
            # Clip radii to be positive
            for i in range(N):
                idx = 1 + i*3 + 2
                x0_perturbed[idx] = np.clip(x0_perturbed[idx], 0.0, 1.0)
            
            try:
                res = minimize(objective, x0_perturbed, method='L-BFGS-B', bounds=bounds, options=options)
                if res.success:
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_result = res.x
            except Exception:
                pass

    if best_result is None:
        best_result = x0

    W = best_result[0]
    H = 2.0 - W
    circles = best_result[1:].reshape(N, 3)

    # Final cleanup: ensure no negative radii
    circles[:, 2] = np.maximum(0, circles[:, 2])

    output = np.zeros((N, 3))
    for i in range(N):
        output[i] = circles[i]

    return output
