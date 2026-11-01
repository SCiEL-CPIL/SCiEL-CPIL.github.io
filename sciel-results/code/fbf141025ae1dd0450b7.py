"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s buffer for overhead
    
    # Initial grid setup
    side = 6
    x0 = np.zeros(N * 3)
    
    # Generate grid points
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            x0[idx * 3] = (i + 0.5) / side
            x0[idx * 3 + 1] = (j + 0.5) / side
            x0[idx * 3 + 2] = 0.5 / side - 1e-4
            idx += 1
    
    best_result = x0.copy()
    best_score = np.sum(best_result[2::3])
    
    def check_validity(x, y, r, tol=1e-7):
        """Vectorized validity check."""
        # Boundary checks
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        # Overlap checks using squared distances
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < (min_dist - tol)**2:
                    return False
        return True
    
    def objective(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        sum_r = np.sum(r)
        
        viol_x_low = np.maximum(0, r - x)
        viol_x_high = np.maximum(0, r - (1 - x))
        viol_y_low = np.maximum(0, r - y)
        viol_y_high = np.maximum(0, r - (1 - y))
        
        boundary_viol = np.sum(viol_x_low**2) + np.sum(viol_x_high**2) + \
                        np.sum(viol_y_low**2) + np.sum(viol_y_high**2)
        
        overlap_viol = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap_viol += (min_dist - dist)**2
        
        penalty = boundary_viol + overlap_viol
        lambda_val = 100.0
        return -sum_r + lambda_val * penalty
    
    def run_penalty_optimization(x_init, maxiter=1000, ftol=1e-10, gtol=1e-8):
        res = scipy.optimize.minimize(objective, x_init, method='L-BFGS-B', 
                                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        return res.x
    
    def solve_lp_tangent(x_init, max_iterations=10):
        """
        Solve LP with tangent plane approximation of non-convex distance constraints.
        For each pair (i,j), linearize: ||p_i - p_j|| >= r_i + r_j
        Using tangent plane at current point:
        (p_i - p_j)·(p_i - p_j)_current / ||p_i - p_j||_current >= r_i + r_j
        """
        x = x_init[0::3].copy()
        y = x_init[1::3].copy()
        r = x_init[2::3].copy()
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Build LP: maximize sum(r)
            # Variables: x_0..x_{N-1}, y_0..y_{N-1}, r_0..r_{N-1}
            n_vars = 3 * N
            
            # Objective: maximize sum(r) => minimize -sum(r)
            c = np.zeros(n_vars)
            c[2::3] = -1.0  # Minimize -r_i
            
            # Constraints: A_ub @ vars <= b_ub
            A_ub_rows = []
            b_ub_rows = []
            
            # Boundary constraints:
            # x_i >= r_i  =>  -x_i + r_i <= 0
            # x_i <= 1 - r_i  =>  x_i + r_i <= 1
            # y_i >= r_i  =>  -y_i + r_i <= 0
            # y_i <= 1 - r_i  =>  y_i + r_i <= 1
            
            for i in range(N):
                # -x_i + r_i <= 0
                row = np.zeros(n_vars)
                row[i] = -1.0
                row[2*N + i] = 1.0
                A_ub_rows.append(row)
                b_ub_rows.append(0.0)
                
                # x_i + r_i <= 1
                row = np.zeros(n_vars)
                row[i] = 1.0
                row[2*N + i] = 1.0
                A_ub_rows.append(row)
                b_ub_rows.append(1.0)
                
                # -y_i + r_i <= 0
                row = np.zeros(n_vars)
                row[N + i] = -1.0
                row[2*N + i] = 1.0
                A_ub_rows.append(row)
                b_ub_rows.append(0.0)
                
                # y_i + r_i <= 1
                row = np.zeros(n_vars)
                row[N + i] = 1.0
                row[2*N + i] = 1.0
                A_ub_rows.append(row)
                b_ub_rows.append(1.0)
            
            # Separation constraints via tangent plane:
            # For pair (i,j): (x_i - x_j, y_i - y_j) · (dx_ij, dy_ij) / dist_ij >= r_i + r_j
            # where (dx_ij, dy_ij) = (x_i - x_j, y_i - y_j) at current point
            # => (x_i - x_j)*dx_ij + (y_i - y_j)*dy_ij >= dist_ij * (r_i + r_j)
            # => -(x_i - x_j)*dx_ij/d - (y_i - y_j)*dy_ij/d + r_i + r_j <= 0
            # where d = dist_ij
            
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    
                    if d < 1e-10:
                        # Degenerate case: use unit vector
                        dx, dy = 1.0, 0.0
                        d = 1.0
                    
                    dx_norm = dx / d
                    dy_norm = dy / d
                    
                    # Constraint: -dx_norm*(x_i - x_j) - dy_norm*(y_i - y_j) + r_i + r_j <= 0
                    row = np.zeros(n_vars)
                    row[i] = -dx_norm
                    row[N + i] = -dy_norm
                    row[j] = dx_norm
                    row[N + j] = dy_norm
                    row[2*N + i] = 1.0
                    row[2*N + j] = 1.0
                    A_ub_rows.append(row)
                    b_ub_rows.append(0.0)
            
            A_ub = np.array(A_ub_rows)
            b_ub = np.array(b_ub_rows)
            
            # Bounds: 0 <= x_i, y_i <= 1, 0 <= r_i
            bounds = []
            for i in range(N):
                bounds.append((0, 1))  # x_i
            for i in range(N):
                bounds.append((0, 1))  # y_i
            for i in range(N):
                bounds.append((0, 1))  # r_i
            
            try:
                res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.success:
                    x_new = res.x[0:N].copy()
                    y_new = res.x[N:2*N].copy()
                    r_new = res.x[2*N:3*N].copy()
                    
                    # Check if new solution is actually feasible
                    if check_validity(x_new, y_new, r_new, tol=1e-6):
                        x, y, r = x_new, y_new, r_new
                    else:
                        # LP solution not feasible, keep old
                        pass
                else:
                    break
            except Exception:
                break
        
        # Reconstruct result
        result = np.zeros(3 * N)
        result[0::3] = x
        result[1::3] = y
        result[2::3] = r
        return result
    
    # Try penalty-based restarts first
    np.random.seed(42)
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        if restart == 0:
            x_current = x0.copy()
        else:
            x_current = x0.copy()
            noise_pos = np.random.normal(0, 0.01, N)
            noise_rad = np.random.normal(0, 0.001, N)
            x_current[0::3] += noise_pos
            x_current[1::3] += noise_pos
            x_current[2::3] += noise_rad
            
        try:
            x_opt = run_penalty_optimization(x_current)
            x = x_opt[0::3]
            y = x_opt[1::3]
            r = x_opt[2::3]
            
            if check_validity(x, y, r, tol=1e-6):
                actual_value = np.sum(r)
                if actual_value > best_score:
                    best_score = actual_value
                    best_result = x_opt.copy()
        except Exception:
            pass
    
    # Now try LP tangent plane approach from the best result
    if time.time() < deadline - 15:
        try:
            x_lp = solve_lp_tangent(best_result, max_iterations=8)
            x = x_lp[0::3]
            y = x_lp[1::3]
            r = x_lp[2::3]
            
            if check_validity(x, y, r, tol=1e-6):
                actual_value = np.sum(r)
                if actual_value > best_score:
                    best_score = actual_value
                    best_result = x_lp.copy()
        except Exception:
            pass
    
    # Final polish with high-precision penalty method
    if time.time() < deadline - 5:
        try:
            def objective_polish(params):
                x = params[0::3]
                y = params[1::3]
                r = params[2::3]
                sum_r = np.sum(r)
                
                viol_x_low = np.maximum(0, r - x)
                viol_x_high = np.maximum(0, r - (1 - x))
                viol_y_low = np.maximum(0, r - y)
                viol_y_high = np.maximum(0, r - (1 - y))
                
                boundary_viol = np.sum(viol_x_low**2) + np.sum(viol_x_high**2) + \
                                np.sum(viol_y_low**2) + np.sum(viol_y_high**2)
                
                overlap_viol = 0.0
                for i in range(N):
                    for j in range(i + 1, N):
                        dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                        min_dist = r[i] + r[j]
                        if dist < min_dist:
                            overlap_viol += (min_dist - dist)**2
                
                penalty = boundary_viol + overlap_viol
                lambda_val = 1000.0
                return -sum_r + lambda_val * penalty

            res = scipy.optimize.minimize(objective_polish, best_result, method='L-BFGS-B', 
                                          options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            x = res.x[0::3]
            y = res.x[1::3]
            r = res.x[2::3]
            
            if check_validity(x, y, r, tol=1e-8):
                actual_value = np.sum(r)
                if actual_value > best_score:
                    best_score = actual_value
                    best_result = res.x.copy()
                    
        except Exception:
            pass

    # Format output
    result = np.zeros((N, 3))
    result[:, 0] = best_result[0::3]
    result[:, 1] = best_result[1::3]
    result[:, 2] = best_result[2::3]
    
    return result

# EVOLVE-BLOCK-END
