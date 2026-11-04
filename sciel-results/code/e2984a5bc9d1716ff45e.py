import numpy as np
import time
from scipy.optimize import minimize

N = 26
DEADLINE = 110.0  # Safety margin from 120s limit

def construct():
    start_time = time.time()
    
    def compute_radii_smooth(x, y, beta=100.0):
        """
        Compute radii using a smooth approximation of the minimum distance constraint.
        r_i = softmin( min(x_i, 1-x_i, y_i, 1-y_i), min_{j!=i} 0.5 * dist(i,j) )
        
        Returns radii and the gradient of the sum of radii w.r.t x, y.
        """
        # 1. Compute distances to boundaries
        # d_bound_i = min(x_i, 1-x_i, y_i, 1-y_i)
        # We use a soft-min over the 4 boundaries.
        bounds_4 = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)  # (N, 4)
        
        # Soft-min for boundaries
        beta_b = beta
        neg_bb = -beta_b * bounds_4
        max_bb = np.max(neg_bb, axis=1, keepdims=True)
        exp_bb = np.exp(neg_bb - max_bb)
        sum_bb = np.sum(exp_bb, axis=1, keepdims=True)
        w_b = exp_bb / sum_bb  # (N, 4)
        r_bound_soft = -(max_bb[:, 0] + np.log(sum_bb[:, 0])) / beta_b
        
        # Gradient of r_bound_soft w.r.t x_i, y_i
        # d(r_bound)/dx_i = w_b[i,0]*1 + w_b[i,1]*(-1) + 0 + 0
        grad_rb_dx = w_b[:, 0] - w_b[:, 1]
        grad_rb_dy = w_b[:, 2] - w_b[:, 3]
        
        # 2. Compute pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        np.fill_diagonal(dist, np.inf)
        
        half_dist = 0.5 * dist  # (N, N)
        
        # We need softmin over j != i of 0.5*dist(i,j) AND r_bound_soft[i]
        # Let's combine them into a single soft-min per row i.
        # Vector of constraints for row i: [r_bound_soft[i], 0.5*dist(i,0), ..., 0.5*dist(i,N-1)]
        # Note: dist(i,i) is inf, so 0.5*dist(i,i) is inf, which is fine for min.
        
        # Construct constraint matrix C (N, N+1)
        # Column 0: r_bound_soft
        # Columns 1..N: 0.5 * dist
        C = np.empty((N, N + 1))
        C[:, 0] = r_bound_soft
        C[:, 1:] = half_dist
        
        # Soft-min over columns for each row
        neg_beta_c = -beta * C
        max_neg_c = np.max(neg_beta_c, axis=1, keepdims=True)
        exp_c = np.exp(neg_beta_c - max_neg_c)
        sum_c = np.sum(exp_c, axis=1, keepdims=True)
        w_c = exp_c / sum_c  # (N, N+1)
        
        r_soft = -(max_neg_c[:, 0] + np.log(sum_c[:, 0])) / beta
        
        # Gradient calculation
        # r_i = softmin_j C_ij
        # dr_i/dp = sum_j w_ij * dC_ij/dp
        
        # dC_i0/dx_i = grad_rb_dx[i]
        # dC_i0/dy_i = grad_rb_dy[i]
        
        # dC_ij/dx_i (for j>0, i.e., j is neighbor index k=j-1)
        # C_i, (k+1) = 0.5 * sqrt((x_i-x_k)^2 + (y_i-y_k)^2)
        # d/dx_i = 0.5 * (x_i - x_k) / dist_ik
        
        # Let's compute gradients for the pairwise part
        # We need dC_ij/dx_m and dC_ij/dy_m for all i, j, m.
        # This is expensive if done naively. Let's exploit structure.
        
        # For a fixed m, dr_i/dx_m = w_i0 * dC_i0/dx_m + sum_{k!=i} w_i(k+1) * dC_i(k+1)/dx_m
        
        # dC_i0/dx_m is non-zero only if m==i.
        # dC_i(k+1)/dx_m is non-zero if m==i or m==k.
        
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        
        # Contribution from boundary term (column 0)
        # For each i, term is w_i0 * grad_rb_dx[i] added to grad_x[i]
        grad_x += w_c[:, 0] * grad_rb_dx
        grad_y += w_c[:, 0] * grad_rb_dy
        
        # Contribution from pairwise terms
        # For each pair (i, k) with i != k:
        # Term in r_i: w_i(k+1) * 0.5 * (x_i - x_k) / dist_ik
        # This contributes to grad_x[i] (as i is the first index) and grad_x[k] (as k is the second index).
        
        # Let's iterate or vectorize.
        # Vectorized approach:
        # dC_ik/dx_i = 0.5 * dx_ik / dist_ik
        # dC_ik/dx_k = -0.5 * dx_ik / dist_ik
        
        # We have weights w_i(k+1) for the constraint of i against k.
        # Let W_pair be matrix where W_pair[i, k] = w_c[i, k+1] for i!=k, 0 otherwise.
        
        W_pair = w_c[:, 1:]  # (N, N)
        np.fill_diagonal(W_pair, 0.0)
        
        # grad_x[i] += sum_k W_pair[i, k] * (0.5 * dx_ik / dist_ik)
        # grad_x[k] += sum_i W_pair[i, k] * (-0.5 * dx_ik / dist_ik)
        
        # Let A_ik = 0.5 * dx_ik / dist_ik
        # Let B_ik = 0.5 * dy_ik / dist_ik
        
        A = 0.5 * dx / dist
        B = 0.5 * dy / dist
        
        # grad_x[i] gets sum_k W_pair[i, k] * A[i, k]
        # This is (W_pair * A)[i]
        grad_x += np.sum(W_pair * A, axis=1)
        
        # grad_x[k] gets sum_i W_pair[i, k] * (-A[i, k])
        # Note: A[i, k] = -A[k, i]
        # So -A[i, k] = A[k, i]
        # We need sum_i W_pair[i, k] * A[k, i]
        # This is (W_pair.T * A.T)[k]? No.
        # Let's check indices.
        # Sum over i of W_pair[i, k] * (-A[i, k])
        # = - Sum_i W_pair[i, k] * A[i, k]
        # This is the k-th element of (W_pair.T * A.T)?
        # (W_pair.T)[k, i] = W_pair[i, k]
        # (A.T)[i, k] = A[k, i] = -A[i, k]
        # So (W_pair.T * A.T)[k] = Sum_i W_pair[i, k] * (-A[i, k]). Correct.
        
        grad_x -= np.sum(W_pair.T * A, axis=1) # Wait, A is not symmetric.
        # Let's re-verify.
        # We want contribution to grad_x[k] from all i.
        # Term is W_pair[i, k] * dC_ik/dx_k = W_pair[i, k] * (-0.5 * dx_ik / dist_ik) = - W_pair[i, k] * A[i, k].
        # So grad_x[k] += - Sum_i W_pair[i, k] * A[i, k].
        # This is exactly - (W_pair.T @ A)[k] if we treat A as a matrix.
        # Let's check matrix multiplication:
        # (W_pair.T @ A)[k] = Sum_i (W_pair.T)[k, i] * A[i, k] = Sum_i W_pair[i, k] * A[i, k].
        # Yes.
        
        # Similarly for y
        grad_y += np.sum(W_pair * B, axis=1)
        grad_y -= np.sum(W_pair.T * B, axis=1)
        
        return r_soft, grad_x, grad_y

    def objective_and_grad(params):
        x = params[:N]
        y = params[N:2*N]
        r, gx, gy = compute_radii_smooth(x, y)
        obj = -np.sum(r)
        grad = np.concatenate([gx, gy])
        return obj, grad

    def check_validity(x, y, r, tol=1e-8):
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        diff_x = x[:, None] - x[None, :]
        diff_y = y[:, None] - y[None, :]
        dist = np.sqrt(diff_x**2 + diff_y**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(min_dist, 0)
        if np.any(dist < min_dist - tol):
            return False
        return True

    def get_exact_radii(x, y):
        """Compute exact radii for final result."""
        r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist, axis=1)
        r = np.minimum(r_bound, 0.5 * min_dist)
        return r

    best_score = -np.inf
    best_x, best_y, best_r = None, None, None

    rng = np.random.RandomState(42)

    # Initial guesses: Grid, Random, and a few structured ones
    initial_guesses = []
    
    # 1. Grid
    side = int(np.ceil(np.sqrt(N)))
    x_grid = np.zeros(N)
    y_grid = np.zeros(N)
    for i in range(N):
        row = i // side
        col = i % side
        x_grid[i] = (col + 0.5) / side
        y_grid[i] = (row + 0.5) / side
    initial_guesses.append((x_grid, y_grid))
    
    # 2. Random uniform
    for _ in range(3):
        x_r = rng.uniform(0.1, 0.9, N)
        y_r = rng.uniform(0.1, 0.9, N)
        initial_guesses.append((x_r, y_r))
    
    # 3. Hexagonal packing approximation
    x_hex = np.zeros(N)
    y_hex = np.zeros(N)
    count = 0
    for row in range(6):
        for col in range(6):
            if count >= N: break
            x_hex[count] = 0.1 + col * 0.15
            y_hex[count] = 0.1 + row * 0.15 + (0.075 if row % 2 else 0)
            count += 1
    if count < N:
        # Fill remaining
        x_hex[count:] = rng.uniform(0.1, 0.9, N-count)
        y_hex[count:] = rng.uniform(0.1, 0.9, N-count)
    initial_guesses.append((x_hex, y_hex))

    # Optimization Loop
    num_restarts = 10
    max_iter_per_restart = 200
    
    for restart in range(num_restarts):
        if time.time() > DEADLINE - 10:
            break
            
        # Select initial guess
        if restart < len(initial_guesses):
            x_init, y_init = initial_guesses[restart]
        else:
            # Perturb the best solution found so far
            if best_x is not None:
                x_init = best_x + rng.normal(0, 0.05, N)
                y_init = best_y + rng.normal(0, 0.05, N)
                x_init = np.clip(x_init, 0.01, 0.99)
                y_init = np.clip(y_init, 0.01, 0.99)
            else:
                x_init = rng.uniform(0.1, 0.9, N)
                y_init = rng.uniform(0.1, 0.9, N)
        
        params_init = np.concatenate([x_init, y_init])
        
        # Run L-BFGS-B
        try:
            res = minimize(
                objective_and_grad,
                params_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': max_iter_per_restart, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            continue
            
        x_opt = res.x[:N]
        y_opt = res.x[N:2*N]
        
        # Clamp to valid region to ensure no NaNs or out-of-bounds in radius calc
        x_opt = np.clip(x_opt, 1e-6, 1 - 1e-6)
        y_opt = np.clip(y_opt, 1e-6, 1 - 1e-6)
        
        # Compute exact radii
        r_opt = get_exact_radii(x_opt, y_opt)
        
        # Check validity (should be valid by construction, but floating point might be tricky)
        if check_validity(x_opt, y_opt, r_opt):
            score = np.sum(r_opt)
            if score > best_score:
                best_score = score
                best_x, best_y, best_r = x_opt.copy(), y_opt.copy(), r_opt.copy()

    # If no valid result found (should not happen with grid), return safe grid
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i // side + 0.5) / side, (i % side + 0.5) / side, 0.5 / side - 1e-3) for i in range(N)]
        return np.array(cells)
        
    # Final repair: ensure strict validity
    for _ in range(20):
        diff_x = best_x[:, None] - best_x[None, :]
        diff_y = best_y[:, None] - best_y[None, :]
        dist = np.sqrt(diff_x**2 + diff_y**2)
        min_dist = best_r[:, None] + best_r[None, :]
        np.fill_diagonal(min_dist, 0)
        viol = np.maximum(0, min_dist - dist)
        np.fill_diagonal(viol, 0)
        if np.max(viol) < 1e-9:
            break
        idx_i, idx_j = np.unravel_index(np.argmax(viol), viol.shape)
        reduction = viol[idx_i, idx_j] / 2 + 1e-6
        best_r[idx_i] = max(best_r[idx_i] - reduction, 1e-6)
        best_r[idx_j] = max(best_r[idx_j] - reduction, 1e-6)
        
    # Final clamp
    best_r = np.maximum(best_r, 1e-6)
    best_x = np.clip(best_x, best_r, 1 - best_r)
    best_y = np.clip(best_y, best_r, 1 - best_r)
    
    return np.column_stack((best_x, best_y, best_r))
