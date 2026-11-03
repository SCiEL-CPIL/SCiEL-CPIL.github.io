import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _objective(x, w, h, penalty_factor):
    """
    Objective function for optimization.
    x: array of shape (n_circles * 3,) -> [x1, y1, r1, ..., xn, yn, rn]
    Returns: -sum(r) + penalty
    """
    n = N
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Base objective: maximize sum of radii -> minimize -sum(r)
    val = -np.sum(rs)
    
    # Penalty for boundary violations
    # x must be in [r, w-r], y in [r, h-r]
    # Violation is max(0, r - x) + max(0, x - (w-r)) + ...
    
    # Left/Right walls
    v_left = np.maximum(0, rs - xs)
    v_right = np.maximum(0, xs - (w - rs))
    
    # Bottom/Top walls
    v_bottom = np.maximum(0, rs - ys)
    v_top = np.maximum(0, ys - (h - rs))
    
    boundary_violation = np.sum(v_left**2) + np.sum(v_right**2) + np.sum(v_bottom**2) + np.sum(v_top**2)
    
    # Penalty for pairwise overlaps
    # We want dist(i,j) >= r_i + r_j
    # Violation = max(0, r_i + r_j - dist(i,j))
    
    # Use broadcasting to compute pairwise distances
    # dx[i,j] = xs[i] - xs[j]
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    sum_r = rs[:, None] + rs[None, :]
    
    # Only consider i < j to avoid double counting and self-overlap (dist=0, sum_r=2r>0 but we ignore diag)
    # Actually, penalty method usually sums all violations. 
    # Let's use a mask for i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    violations = np.maximum(0, sum_r - dist)
    overlap_violation = np.sum(violations[mask]**2)
    
    total_penalty = penalty_factor * (boundary_violation + overlap_violation)
    
    return val + total_penalty

def _objective_grad(x, w, h, penalty_factor):
    """
    Gradient of the objective function.
    """
    n = N
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    grad = np.zeros_like(x)
    
    # Gradient of -sum(r)
    grad[2::3] = -1.0
    
    # Gradient of boundary penalties
    # d/dx (max(0, r-x)^2) = -2(r-x) if r>x else 0
    # d/dx (max(0, x-(w-r))^2) = 2(x-(w-r)) if x>w-r else 0
    # d/dr (max(0, r-x)^2) = 2(r-x) if r>x else 0
    # d/dr (max(0, x-(w-r))^2) = 2(x-(w-r)) if x>w-r else 0 (note: d/dr of (w-r) is -1, so chain rule gives -1 * -1 = 1? No.)
    # Let f(r) = max(0, x - (w-r))^2. If x > w-r, f = (x - w + r)^2. df/dr = 2(x - w + r).
    # So both terms contribute positively to grad_r if violated.
    
    v_left = np.maximum(0, rs - xs)
    v_right = np.maximum(0, xs - (w - rs))
    v_bottom = np.maximum(0, rs - ys)
    v_top = np.maximum(0, ys - (h - rs))
    
    # x-gradient
    grad[0::3] += -2 * penalty_factor * v_left
    grad[0::3] += 2 * penalty_factor * v_right
    
    # y-gradient
    grad[1::3] += -2 * penalty_factor * v_bottom
    grad[1::3] += 2 * penalty_factor * v_top
    
    # r-gradient from boundary
    grad[2::3] += 2 * penalty_factor * (v_left + v_right + v_bottom + v_top)
    
    # Gradient of overlap penalties
    # Penalty term: sum_{i<j} (max(0, r_i+r_j - dist_ij))^2
    # Let V_ij = max(0, r_i+r_j - dist_ij)
    # dV_ij/dr_i = 1 if violated
    # dV_ij/dr_j = 1 if violated
    # dV_ij/dx_i = - (x_i - x_j) / dist_ij
    # dV_ij/dy_i = - (y_i - y_j) / dist_ij
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    dist_safe = np.where(dist < 1e-8, 1e-8, dist)
    
    sum_r = rs[:, None] + rs[None, :]
    violations = np.maximum(0, sum_r - dist)
    
    # Mask for i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    v_masked = violations * mask
    
    # Compute gradients for x and y
    # grad_x_i = sum_j 2 * penalty * v_ij * (-dx_ij / dist_ij)
    # We can use matrix multiplication
    # G_x[i] = -2 * penalty * sum_j (v_masked[i,j] * dx[i,j] / dist_safe[i,j])
    
    # Note: v_masked is symmetric in value but we only keep upper triangle. 
    # For the gradient of circle i, we sum over all j.
    # Since v_masked[i,j] is non-zero only for j>i in the upper triangle, 
    # we need to account for the pair (j,i) in the lower triangle for circle i's gradient?
    # Actually, the penalty is sum_{i<j} V_ij^2.
    # d/dx_i sum_{i<j} V_ij^2 + d/dx_i sum_{k<i} V_ki^2
    # The first part involves terms where i is the first index.
    # The second part involves terms where i is the second index.
    
    # Let's construct the full violation matrix for gradient calculation, 
    # but zero out the diagonal and ensure symmetry in contribution.
    # V_full[i,j] = max(0, r_i+r_j - dist_ij)
    # The penalty is 0.5 * sum_{i!=j} V_full[i,j]^2 (to make it symmetric and easy to sum)
    # Or just sum_{i<j} V_ij^2.
    # Let's stick to sum_{i<j}.
    
    # For circle i:
    # Contribution from pairs (i, j) where j > i:
    # d/dx_i V_ij^2 = 2 V_ij * dV_ij/dx_i = 2 V_ij * (-(x_i-x_j)/dist_ij)
    # Contribution from pairs (k, i) where k < i:
    # d/dx_i V_ki^2 = 2 V_ki * dV_ki/dx_i = 2 V_ki * (-(x_k-x_i)/dist_ki) = 2 V_ki * ((x_i-x_k)/dist_ki)
    
    # Using broadcasting:
    # term_x[i,j] = 2 * penalty * V_ij * (-dx[i,j] / dist_safe[i,j])
    # We want row sum for x_i? No.
    # For x_i, we need sum over j of contribution from pair (i,j) and sum over k of contribution from pair (k,i).
    
    # Let's compute the full matrix of partial derivatives of V w.r.t x_i
    # dV/dx_i = -dx[i,j]/dist
    # dV/dx_j = dx[i,j]/dist
    
    # Let A[i,j] = 2 * penalty * V_ij * (-dx[i,j] / dist_safe[i,j])
    # This A[i,j] is the derivative of the penalty term for pair (i,j) w.r.t x_i.
    # The total gradient for x_i is sum_{j>i} A[i,j] + sum_{k<i} A[k,i]
    # Note that A[k,i] = 2 * penalty * V_ki * (-dx[k,i] / dist_safe[k,i])
    # dx[k,i] = x_k - x_i = -dx[i,k]
    # dist[k,i] = dist[i,k]
    # V_ki = V_ik
    # So A[k,i] = 2 * penalty * V_ik * (dx[i,k] / dist_safe[i,k]) = - A[i,k]
    
    # So grad_x[i] = sum_{j>i} A[i,j] - sum_{k<i} A[i,k]
    # This is equivalent to sum_{j != i} A[i,j] ?
    # If we define A_full[i,j] = 2 * penalty * V_ij * (-dx[i,j] / dist_safe[i,j]) for all i,j
    # Then A_full[i,j] = -A_full[j,i].
    # grad_x[i] = sum_{j} A_full[i,j]
    
    # Let's verify:
    # grad_x[i] = sum_{j>i} A[i,j] + sum_{k<i} A[k,i]
    # Since A[k,i] = -A[i,k], sum_{k<i} A[k,i] = - sum_{k<i} A[i,k]
    # So grad_x[i] = sum_{j>i} A[i,j] - sum_{k<i} A[i,k]
    # This is NOT sum_{all j} A[i,j].
    # sum_{all j} A[i,j] = sum_{j>i} A[i,j] + sum_{k<i} A[i,k]
    # The signs are different!
    
    # Wait, let's re-evaluate the derivative.
    # Penalty P = sum_{i<j} V_ij^2
    # dP/dx_i = sum_{j>i} 2 V_ij dV_ij/dx_i + sum_{k<i} 2 V_ki dV_ki/dx_i
    # dV_ij/dx_i = -(x_i - x_j) / dist_ij
    # dV_ki/dx_i = -(x_k - x_i) / dist_ki = (x_i - x_k) / dist_ki
    
    # Let B[i,j] = 2 * penalty * V_ij * (-(x_i - x_j) / dist_ij)
    # Then dP/dx_i = sum_{j>i} B[i,j] + sum_{k<i} (-B[k,i])  (since B[k,i] uses x_k-x_i, so B[k,i] = 2p V_ki (-(x_k-x_i)/dist) = 2p V_ki ((x_i-x_k)/dist) = - 2p V_ki (-(x_i-x_k)/dist) ... wait)
    
    # Let's just compute the matrix M where M[i,j] = 2 * penalty * V_ij * (-(x_i - x_j) / dist_ij)
    # Then grad_x[i] = sum_{j>i} M[i,j] + sum_{k<i} (-M[k,i])
    # Note M[k,i] = 2 * penalty * V_ki * (-(x_k - x_i) / dist_ki) = 2 * penalty * V_ik * ((x_i - x_k) / dist_ik)
    # And M[i,k] = 2 * penalty * V_ik * (-(x_i - x_k) / dist_ik)
    # So M[k,i] = -M[i,k]
    # Therefore, sum_{k<i} (-M[k,i]) = sum_{k<i} M[i,k]
    # So grad_x[i] = sum_{j>i} M[i,j] + sum_{k<i} M[i,k] = sum_{j != i} M[i,j]
    
    # Great! So grad_x = row_sums of M.
    
    # Calculate M
    # V is violations matrix (symmetric)
    # dx is difference matrix (antisymmetric)
    # dist is distance matrix (symmetric)
    
    # M[i,j] = 2 * penalty_factor * violations[i,j] * (-dx[i,j] / dist_safe[i,j])
    # We must set diagonal to 0 to avoid division by zero issues, though dist_safe handles it.
    # Also, we should only count pairs i<j in the penalty sum, but the gradient logic above 
    # derived that using the full antisymmetric matrix M and summing rows works 
    # IF the penalty was sum_{i<j} V_ij^2.
    # Let's check:
    # P = sum_{i<j} V_ij^2
    # dP/dx_i = sum_{j>i} 2 V_ij dV_ij/dx_i + sum_{k<i} 2 V_ki dV_ki/dx_i
    # Let M_full[i,j] = 2 * penalty * V_ij * (-(x_i-x_j)/dist)
    # Then term 1 is sum_{j>i} M_full[i,j]
    # Term 2: dV_ki/dx_i = (x_i-x_k)/dist_ki.
    # M_full[k,i] = 2 * penalty * V_ki * (-(x_k-x_i)/dist_ki) = 2 * penalty * V_ki * ((x_i-x_k)/dist_ki)
    # So Term 2 is sum_{k<i} M_full[k,i]
    # So dP/dx_i = sum_{j>i} M_full[i,j] + sum_{k<i} M_full[k,i]
    # This is NOT row sum of M_full.
    # Row sum is sum_{j} M_full[i,j] = sum_{j>i} M_full[i,j] + sum_{k<i} M_full[i,k]
    # Since M_full[i,k] = -M_full[k,i], row sum = sum_{j>i} M_full[i,j] - sum_{k<i} M_full[k,i]
    # The sign is wrong for the second term!
    
    # So we cannot just use row sums of the antisymmetric matrix.
    # We need to construct the gradient carefully.
    
    # Let's use the explicit loop for clarity and correctness, 
    # but vectorize the inner part if possible. 
    # Given N=21, a simple loop over pairs is fast enough (21*20/2 = 210 pairs).
    
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    grad_r = np.zeros(n)
    
    # We already have dx, dy, dist, violations
    # Loop over pairs
    for i in range(n):
        for j in range(i+1, n):
            v = violations[i, j]
            if v > 0:
                dist_ij = dist[i, j]
                if dist_ij < 1e-8:
                    continue # or handle carefully
                nx = (xs[i] - xs[j]) / dist_ij
                ny = (ys[i] - ys[j]) / dist_ij
                
                # dP/dx_i = 2 * penalty * v * (-nx)
                # dP/dx_j = 2 * penalty * v * (nx)
                # dP/dy_i = 2 * penalty * v * (-ny)
                # dP/dy_j = 2 * penalty * v * (ny)
                # dP/dr_i = 2 * penalty * v * (1)
                # dP/dr_j = 2 * penalty * v * (1)
                
                factor = 2.0 * penalty_factor * v
                grad_x[i] += -factor * nx
                grad_x[j] += factor * nx
                grad_y[i] += -factor * ny
                grad_y[j] += factor * ny
                grad_r[i] += factor
                grad_r[j] += factor
                
    grad[0::3] += grad_x
    grad[1::3] += grad_y
    grad[2::3] += grad_r
    
    return val + total_penalty, grad

def _pack_single(w, h, init_circles, time_limit, penalty_factor=100.0):
    """
    Optimize circles for a fixed w, h starting from init_circles.
    """
    start_time = time.time()
    x0 = init_circles.flatten()
    
    # Bounds: 
    # x in [0, w], y in [0, h], r in [0, min(w,h)/2]
    max_r = min(w, h) / 2.0
    bounds = []
    for _ in range(N):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0, max_r))
        
    def obj_func(x):
        return _objective(x, w, h, penalty_factor)
        
    def obj_grad(x):
        return _objective_grad(x, w, h, penalty_factor)
        
    def callback(Xk):
        return time.time() - start_time > time_limit

    try:
        res = minimize(
            lambda x: obj_func(x),
            x0,
            jac=lambda x: obj_grad(x)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'maxfun': 500, 'callback': callback}
        )
        return res.x
    except Exception:
        return x0

def _generate_grid_init(w, h, n_circles, cols, rows):
    circles = np.zeros((n_circles, 3))
    count = 0
    for i in range(cols):
        for j in range(rows):
            if count >= n_circles:
                break
            x = (i + 0.5) * w / cols
            y = (j + 0.5) * h / rows
            cell_w = w / cols
            cell_h = h / rows
            r = 0.4 * min(cell_w, cell_h)
            circles[count, 0] = x
            circles[count, 1] = y
            circles[count, 2] = r
            count += 1
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_result = None
    best_score = 0.0
    
    # We will try a few aspect ratios and grid configurations.
    # w + h = 2
    aspect_ratios = [0.8, 1.0, 1.2, 0.7, 1.4]
    
    # Grid configs (cols, rows)
    grid_configs = [
        (4, 5), (5, 4),
        (3, 7), (7, 3),
        (4, 6), (6, 4),
        (5, 5)
    ]
    
    candidates = []
    
    for ar in aspect_ratios:
        if time.time() > deadline - 20:
            break
            
        w = 2.0 * ar / (1 + ar)
        h = 2.0 / (1 + ar)
        
        for cols, rows in grid_configs:
            if cols * rows < N:
                continue
                
            # Check aspect ratio match
            grid_ar = cols / rows
            box_ar = w / h
            if abs(grid_ar - box_ar) / box_ar > 0.4:
                continue
                
            init_circles = _generate_grid_init(w, h, N, cols, rows)
            
            # Time budget for this optimization
            time_left = deadline - 20 - time.time()
            if time_left < 5:
                break
                
            opt_time = min(10.0, time_left / 5.0)
            
            # Run optimization
            x_opt = _pack_single(w, h, init_circles, opt_time)
            pos_r = x_opt.reshape(-1, 3)
            
            # Validate
            if _validate_solution(pos_r, w, h, N):
                score = np.sum(pos_r[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = pos_r
                candidates.append((score, pos_r, w, h))
                
            if time.time() > deadline - 20:
                break
                
    # If we have candidates, try to refine the best one with a different method or more iterations
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_pos, best_w, best_h = candidates[0]
        
        # Try Nelder-Mead on the best candidate's aspect ratio? 
        # Or just run L-BFGS-B again with higher penalty?
        
        # Let's try a few more random restarts if time permits
        time_left = deadline - 5 - time.time()
        if time_left > 10:
            # Use best_w, best_h
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                # Random perturbation of best solution
                init_circles = best_pos.copy()
                init_circles[:, :2] += np.random.normal(0, 0.05, init_circles[:, :2].shape)
                init_circles[:, 2] *= 0.95 # Slightly shrink to allow movement
                
                opt_time = min(5.0, time_left / 3.0)
                x_opt = _pack_single(best_w, best_h, init_circles, opt_time)
                pos_r = x_opt.reshape(-1, 3)
                
                if _validate_solution(pos_r, best_w, best_h, N):
                    score = np.sum(pos_r[:, 2])
                    if score > best_score:
                        best_score = score
                        best_result = pos_r

    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        best_result = _generate_grid_init(w, h, N, 5, 5)
        # Ensure validity
        for i in range(N):
            x, y, r = best_result[i]
            min_d = min(x, w-x, y, h-y)
            best_result[i, 2] = min(r, min_d - 1e-4)
        # Simple relaxation
        best_result = _resolve_collisions(best_result, w, h, max_iters=100)

    return best_result

def _validate_solution(pos_r, w, h, n_circles, tol=1e-6):
    if pos_r.shape[0] != n_circles:
        return False
    
    xs = pos_r[:, 0]
    ys = pos_r[:, 1]
    rs = pos_r[:, 2]
    
    if np.any(rs < -tol):
        return False
    
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
        
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < (rs[i] + rs[j]) - tol:
                return False
    return True

def _resolve_collisions(circles, w, h, max_iters=1000, tol=1e-5):
    n = circles.shape[0]
    circles = circles.copy()
    
    for _ in range(max_iters):
        moved = False
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0:
                circles[i, 0] = r
                moved = True
            if x + r > w:
                circles[i, 0] = w - r
                moved = True
            if y - r < 0:
                circles[i, 1] = r
                moved = True
            if y + r > h:
                circles[i, 1] = h - r
                moved = True
        
        for i in range(n):
            for j in range(i+1, n):
                dx = circles[j, 0] - circles[i, 0]
                dy = circles[j, 1] - circles[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = circles[i, 2] + circles[j, 2]
                
                if dist < min_dist:
                    if dist < 1e-8:
                        circles[i, 0] += np.random.uniform(-0.01, 0.01)
                        circles[i, 1] += np.random.uniform(-0.01, 0.01)
                        circles[j, 0] -= np.random.uniform(-0.01, 0.01)
                        circles[j, 1] -= np.random.uniform(-0.01, 0.01)
                        moved = True
                    else:
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        circles[i, 0] -= nx * overlap * 0.5
                        circles[i, 1] -= ny * overlap * 0.5
                        circles[j, 0] += nx * overlap * 0.5
                        circles[j, 1] += ny * overlap * 0.5
                        moved = True
        
        if not moved:
            break
            
    return circles
