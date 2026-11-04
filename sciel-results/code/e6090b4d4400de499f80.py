"""
Circle packing in unit square: maximize sum of radii for 26 disjoint circles.

Strategy:
1. Use a smooth log-sum-exp surrogate for the radius constraints (min of boundary and neighbor distances).
2. Optimize centers (x, y) with L-BFGS-B using the analytic gradient of the surrogate.
3. Radii are derived from the optimized centers.
4. Multiple random restarts to escape local minima.
5. A final polish pass with tighter tolerances.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    def compute_radii(x, y):
        """Compute radii from centers: min of half-distance to neighbors and boundary."""
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        min_dist = np.min(dist, axis=1)
        r = np.minimum(r_bound, 0.5 * min_dist)
        r = np.maximum(r, 1e-6)
        return r

    def smooth_objective_and_grad(params, beta=50.0):
        """
        Compute -sum(r) and its gradient with respect to params = [x, y].
        Uses analytic gradient of the log-sum-exp soft-min.
        
        r_i = softmin_j(c_ij) where c_ij = 0.5*dist(i,j) for j!=i, c_ii = r_bound_i
        r_i = -(1/beta) * log(sum_j exp(-beta * c_ij))
        """
        x = params[:N]
        y = params[N:2*N]

        # Boundary constraint
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-30)
        np.fill_diagonal(dist, 1e10)

        # Build constraint matrix: c_ij for all i,j
        # c_ij = 0.5 * dist(i,j) for i != j
        # c_ii = r_bound[i]
        c = 0.5 * dist
        np.fill_diagonal(c, r_bound)

        # Soft-min computation
        neg_beta_c = -beta * c
        max_neg = np.max(neg_beta_c, axis=1, keepdims=True)
        exp_terms = np.exp(neg_beta_c - max_neg)
        sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
        log_sum = np.log(sum_exp)
        r = -(max_neg + log_sum) / beta
        r = r[:, 0]

        # Weights: w_ij = exp(-beta*c_ij) / sum_j' exp(-beta*c_ij')
        w = exp_terms / sum_exp  # (N, N)

        # Gradient of c_ij w.r.t. parameters
        dist_safe = dist.copy()
        np.fill_diagonal(dist_safe, 1.0)

        # Off-diagonal gradients
        dc_dx_i = 0.5 * dx / dist_safe  # dc_ij/dx_i
        dc_dy_i = 0.5 * dy / dist_safe  # dc_ij/dy_i
        dc_dx_j = -0.5 * dx / dist_safe  # dc_ij/dx_j
        dc_dy_j = -0.5 * dy / dist_safe  # dc_ij/dy_j

        # Diagonal gradients for r_bound_i
        bounds_4 = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)  # (N, 4)
        beta_b = 100.0
        neg_bb = -beta_b * bounds_4
        max_bb = np.max(neg_bb, axis=1, keepdims=True)
        exp_bb = np.exp(neg_bb - max_bb)
        sum_bb = np.sum(exp_bb, axis=1, keepdims=True)
        w_b = exp_bb / sum_bb  # (N, 4)

        grad_rb_dx = w_b[:, 0] - w_b[:, 1]
        grad_rb_dy = w_b[:, 2] - w_b[:, 3]

        # Set diagonal entries
        np.fill_diagonal(dc_dx_i, grad_rb_dx)
        np.fill_diagonal(dc_dy_i, grad_rb_dy)
        np.fill_diagonal(dc_dx_j, grad_rb_dx)
        np.fill_diagonal(dc_dy_j, grad_rb_dy)

        # Total gradient w.r.t. x_k: sum_i dr_i/dx_k
        # dr_i/dx_k = sum_j w_ij * (dc_ij/dx_k)
        # For k = i: sum_j w_ij * dc_dx_i[i,j]
        # For k != i: only j=k contributes: w_ik * dc_dx_j[i,k]
        
        # dr_dx[k] = sum_i [w_ik * dc_dx_j[i,k]] + sum_{j!=k} w_kj * dc_dx_i[k,j]
        # Note: when i=k, the term w_kk * dc_dx_i[k,k] is included in the second sum (j=k)
        # But we need to be careful about double counting. Let's compute it directly.
        
        # For each k, dr_dx[k] = sum_i sum_j w_ij * dc_ij/dx_k
        # dc_ij/dx_k is nonzero only if k==i or k==j
        # If k==i: dc_ij/dx_i = dc_dx_i[i,j]
        # If k==j: dc_ij/dx_j = dc_dx_j[i,j]
        
        # So dr_dx[k] = sum_i [w_ki * dc_dx_i[i,k]]  [i=k case, but this is just w_kk * dc_dx_i[k,k]]
        #              + sum_{i!=k} [w_ik * dc_dx_j[i,k]]  [j=k case]
        #              + sum_{i!=k} sum_{j!=k} w_ij * 0  [no contribution]
        #              + sum_i [w_ki * dc_dx_i[i,k]]  [i=k case]
        
        # Actually, let's just compute it as:
        # dr_dx[k] = sum_i w_ik * dc_dx_j[i,k] + sum_{j} w_kj * dc_dx_i[k,j]
        # But the term i=k, j=k appears in both sums, so we need to subtract it once.
        # w_kk * dc_dx_j[k,k] + w_kk * dc_dx_i[k,k] - w_kk * dc_kk/dx_k
        # Since dc_dx_i[k,k] = dc_dx_j[k,k] = grad_rb_dx[k], and dc_kk/dx_k = grad_rb_dx[k]
        # So the correction is -w_kk * grad_rb_dx[k]
        
        # Let's compute the two sums:
        # Sum 1: sum_i w_ik * dc_dx_j[i,k] for all i
        # Sum 2: sum_j w_kj * dc_dx_i[k,j] for all j
        
        # We can compute these using matrix operations:
        # grad_x[k] = sum_i w[i,k] * dc_dx_j[i,k] + sum_j w[k,j] * dc_dx_i[k,j]
        
        # Let's compute this efficiently:
        # grad_x = (w.T * dc_dx_j).sum(axis=0) + (w * dc_dx_i).sum(axis=1)
        # But we need to handle the double-counting of the diagonal term.
        
        # Actually, the cleanest way is:
        # grad_x[k] = sum_i w[i,k] * dc_dx_j[i,k] + sum_j w[k,j] * dc_dx_i[k,j]
        # The diagonal term (i=k, j=k) is counted twice: once in each sum.
        # So we subtract w[k,k] * dc_dx_i[k,k] (or dc_dx_j[k,k], they're the same).
        
        grad_x = np.sum(w.T * dc_dx_j, axis=0) + np.sum(w * dc_dx_i, axis=1) - np.diag(w) * grad_rb_dx
        grad_y = np.sum(w.T * dc_dy_j, axis=0) + np.sum(w * dc_dy_i, axis=1) - np.diag(w) * grad_rb_dy

        # Objective is -sum(r)
        obj = -np.sum(r)
        
        return obj, np.concatenate([grad_x, grad_y])

    def evaluate(x, y):
        """Check validity and compute score."""
        r = compute_radii(x, y)
        if np.any(r < 1e-6):
            return -np.inf
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return -np.inf
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return -np.inf
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - 1e-9:
                    return -np.inf
        return np.sum(r)

    def make_hexagonal_init(rng):
        """Create a hexagonal packing initial guess."""
        x = np.zeros(N)
        y = np.zeros(N)
        
        spacing = 0.12
        row = 0
        col = 0
        idx = 0
        while idx < N:
            cx = spacing + col * 2 * spacing * np.cos(np.pi/6)
            cy = spacing + row * 2 * spacing * np.sin(np.pi/6)
            if col % 2 == 1:
                cy += spacing * np.sin(np.pi/6)
            
            if cx < 0.95 and cy < 0.95:
                x[idx] = cx + (rng.random() - 0.5) * 0.02
                y[idx] = cy + (rng.random() - 0.5) * 0.02
                idx += 1
            
            col += 1
            if col > 8:
                col = 0
                row += 1
        
        # Clip to valid range
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        
        return x, y

    def make_random_init(rng):
        """Create a random initial guess."""
        x = rng.uniform(0.1, 0.9, size=N)
        y = rng.uniform(0.1, 0.9, size=N)
        return x, y

    best_result = None
    best_score = -np.inf

    rng = np.random.default_rng(42)

    num_restarts = 30
    for trial in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Generate initial positions
        if trial % 3 == 0:
            x0, y0 = make_hexagonal_init(rng)
        else:
            x0, y0 = make_random_init(rng)
        
        params0 = np.concatenate([x0, y0])
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                smooth_objective_and_grad,
                params0,
                jac=True,
                method='L-BFGS-B',
                bounds=[(0.01, 0.99)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < 0:  # valid (since objective is -sum(r))
                xp = result.x[:N]
                yp = result.x[N:2*N]
                score = evaluate(xp, yp)
                if score > best_score:
                    best_score = score
                    r_sol = compute_radii(xp, yp)
                    best_result = np.column_stack([xp, yp, r_sol])
        except Exception:
            pass

    # Final polish: use tighter tolerances from best result
    if best_result is not None and time.time() < deadline - 10:
        x, y, r = best_result[:, 0], best_result[:, 1], best_result[:, 2]
        params0 = np.concatenate([x, y])
        
        try:
            result = minimize(
                smooth_objective_and_grad,
                params0,
                jac=True,
                method='L-BFGS-B',
                bounds=[(0.01, 0.99)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < 0:  # valid
                xp = result.x[:N]
                yp = result.x[N:2*N]
                score = evaluate(xp, yp)
                if score > best_score:
                    best_score = score
                    r_sol = compute_radii(xp, yp)
                    best_result = np.column_stack([xp, yp, r_sol])
        except Exception:
            pass

    # If no valid result found, return the trivial seed
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
