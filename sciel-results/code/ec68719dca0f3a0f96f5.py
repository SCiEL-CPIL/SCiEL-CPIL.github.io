import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    def initial_grid(n):
        """Create initial grid placement."""
        side = int(np.ceil(np.sqrt(n)))
        xs = np.zeros(n)
        ys = np.zeros(n)
        rs = np.zeros(n)
        for i in range(n):
            row = i // side
            col = i % side
            xs[i] = (col + 0.5) / side
            ys[i] = (row + 0.5) / side
            rs[i] = 0.5 / side - 1e-4
        return xs, ys, rs

    def check_validity(xs, ys, rs, tol=1e-9):
        """Check if circles are valid (inside square, non-overlapping)."""
        for i in range(N):
            if xs[i] < rs[i] - tol or xs[i] > 1.0 - rs[i] + tol:
                return False
            if ys[i] < rs[i] - tol or ys[i] > 1.0 - rs[i] + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                dist_sq = dx * dx + dy * dy
                min_dist = rs[i] + rs[j]
                if dist_sq < min_dist * min_dist - tol * (min_dist * min_dist):
                    return False
        return True

    def objective_and_grad(params, beta):
        """
        Compute objective function and gradient.
        params: [x_0...x_n-1, y_0...y_n-1, r_0...r_n-1]
        beta: penalty weight for soft constraints.
        
        Objective: -sum(r) + beta * (sum of constraint violations)
        """
        xs = params[:N]
        ys = params[N:2*N]
        rs = params[2*N:]

        obj = -np.sum(rs)
        grad = np.zeros_like(params)
        grad[2*N:] = -1.0  # d(-sum(r))/dr = -1

        # Boundary constraints
        # x - r >= 0 => violation if x < r. Penalty: max(0, r - x)^2
        # 1 - x - r >= 0 => violation if x > 1 - r. Penalty: max(0, x - (1 - r))^2
        # y - r >= 0 => violation if y < r. Penalty: max(0, r - y)^2
        # 1 - y - r >= 0 => violation if y > 1 - r. Penalty: max(0, y - (1 - r))^2
        
        # x < r
        viol = rs - xs
        mask = viol > 0
        if np.any(mask):
            obj += beta * np.sum(viol[mask] ** 2)
            # d/dx: -2 * viol
            grad[:N][mask] += beta * (-2 * viol[mask])
            # d/dr: 2 * viol
            grad[2*N:][mask] += beta * (2 * viol[mask])

        # x > 1 - r
        viol = xs - (1.0 - rs)
        mask = viol > 0
        if np.any(mask):
            obj += beta * np.sum(viol[mask] ** 2)
            # d/dx: 2 * viol
            grad[:N][mask] += beta * (2 * viol[mask])
            # d/dr: -2 * viol
            grad[2*N:][mask] += beta * (-2 * viol[mask])

        # y < r
        viol = rs - ys
        mask = viol > 0
        if np.any(mask):
            obj += beta * np.sum(viol[mask] ** 2)
            # d/dy: -2 * viol
            grad[N:2*N][mask] += beta * (-2 * viol[mask])
            # d/dr: 2 * viol
            grad[2*N:][mask] += beta * (2 * viol[mask])

        # y > 1 - r
        viol = ys - (1.0 - rs)
        mask = viol > 0
        if np.any(mask):
            obj += beta * np.sum(viol[mask] ** 2)
            # d/dy: 2 * viol
            grad[N:2*N][mask] += beta * (2 * viol[mask])
            # d/dr: -2 * viol
            grad[2*N:][mask] += beta * (-2 * viol[mask])

        # Pairwise non-overlap constraints
        # dist(i, j) >= r_i + r_j
        # violation if dist < r_i + r_j
        # Penalty: max(0, r_i + r_j - dist)^2
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                
                r_sum = rs[i] + rs[j]
                viol = r_sum - dist
                if viol > 0:
                    obj += beta * viol ** 2
                    
                    # Gradients
                    # d(dist)/dx_j = dx/dist, d(dist)/dx_i = -dx/dist
                    # d(viol)/dx_j = -dx/dist, d(viol)/dx_i = dx/dist
                    # d(viol)/dr_i = 1, d(viol)/dr_j = 1
                    
                    # d(obj)/dx_j = 2 * beta * viol * (-dx/dist)
                    # d(obj)/dx_i = 2 * beta * viol * (dx/dist)
                    # d(obj)/dr_i = 2 * beta * viol * 1
                    # d(obj)/dr_j = 2 * beta * viol * 1
                    
                    factor = 2 * beta * viol / dist
                    grad[j] += -factor * dx
                    grad[i] += factor * dx
                    grad[N + j] += -factor * dy
                    grad[N + i] += factor * dy
                    grad[2*N + i] += 2 * beta * viol
                    grad[2*N + j] += 2 * beta * viol

        return obj, grad

    def local_refine(params, beta_start, time_limit):
        """
        Refine parameters using L-BFGS-B with increasing beta.
        """
        refine_start = time.time()
        best_params = params.copy()
        best_score = -params[2*N:].sum()  # sum of radii
        
        beta = beta_start
        
        # Bounds for L-BFGS-B
        # x in [0, 1], y in [0, 1], r in [1e-4, 0.5]
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(1e-4, 0.5)] * N
        
        while time.time() < deadline - 5 and time.time() - refine_start < time_limit:
            # Define objective for this beta
            def obj_func(p):
                return objective_and_grad(p, beta)[0]
            
            def grad_func(p):
                return objective_and_grad(p, beta)[1]

            res = minimize(
                obj_func,
                params,
                jac=grad_func,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            new_params = res.x
            new_score = -new_params[2*N:].sum()
            
            if new_score > best_score:
                best_score = new_score
                best_params = new_params.copy()
            
            params = new_params
            
            # Increase beta to enforce constraints more strictly
            beta *= 10.0
            
            if beta > 1e6:
                break

        return best_params, best_score

    # --- Main optimization loop ---

    # Phase 1: Initial grid setup
    xs, ys, rs = initial_grid(N)
    params = np.concatenate((xs, ys, rs))
    
    best_score = np.sum(rs)
    best_params = params.copy()

    # Phase 2: Gradient Descent Refinement
    # Start with a moderate beta to allow movement, then increase
    # We run this in stages to handle non-convexity better
    stages = [10, 100, 1000, 10000]
    
    current_params = params.copy()
    for beta in stages:
        if time.time() > deadline - 10:
            break
        current_params, score = local_refine(current_params, beta, time_limit=30.0)
        if score > best_score:
            best_score = score
            best_params = current_params.copy()

    # Phase 3: Final high-beta refinement to ensure validity
    if time.time() < deadline - 10:
        current_params, score = local_refine(best_params, 100000.0, time_limit=20.0)
        if score > best_score:
            best_score = score
            best_params = current_params.copy()

    # Extract final result
    final_params = best_params
    final_xs = final_params[:N]
    final_ys = final_params[N:2*N]
    final_rs = final_params[2*N:]

    # Final validation
    if not check_validity(final_xs, final_ys, final_rs):
        # Fallback: return a valid grid
        xs_f, ys_f, rs_f = initial_grid(N)
        # Try to relax the grid slightly using PBD-like projection just for safety
        # But the grid is valid by construction (with margin)
        final_xs, final_ys, final_rs = xs_f, ys_f, rs_f

    return np.column_stack((final_xs, final_ys, final_rs))
