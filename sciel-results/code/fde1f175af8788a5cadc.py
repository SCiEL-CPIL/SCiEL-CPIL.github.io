import numpy as np
import time
from scipy.optimize import minimize

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    best_score = 0.0
    best_solution = None
    
    # Penalty weight
    lam = 1000.0
    
    def unpack(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        return x, y, r
    
    def soft_max_smooth(violations, temperature=0.01):
        """Smooth approximation of max(0, v) using softplus-like function"""
        eps = temperature ** 2
        return 0.5 * (violations + np.sqrt(violations**2 + eps))
    
    def objective(v):
        x, y, r = unpack(v)
        
        # Sum of radii
        sum_r = np.sum(r)
        
        # Boundary penalties - use smooth max
        viol_x = r - x
        viol_1mx = r - (1 - x)
        viol_y = r - y
        viol_1my = r - (1 - y)
        
        pen_x = np.sum(soft_max_smooth(viol_x)**2)
        pen_1mx = np.sum(soft_max_smooth(viol_1mx)**2)
        pen_y = np.sum(soft_max_smooth(viol_y)**2)
        pen_1my = np.sum(soft_max_smooth(viol_1my)**2)
        pen_bound = pen_x + pen_1mx + pen_y + pen_1my
        
        # Overlap penalties - use smooth max
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        r_sum_sq = r_sum**2
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        violations = r_sum_sq[mask] - dist_sq[mask]
        smooth_viol = soft_max_smooth(violations)
        pen_overlap = np.sum(smooth_viol**2)
        
        total_penalty = pen_bound + pen_overlap
        
        return -sum_r + lam * total_penalty
    
    def objective_grad(v):
        x, y, r = unpack(v)
        
        grad = np.zeros(3*N)
        
        # Gradient of -sum(r)
        grad[2*N:3*N] -= 1.0
        
        # Boundary penalty gradients
        temp = 0.01
        eps = temp**2
        
        def smooth_grad(viol):
            return 0.5 * (1 + viol / np.sqrt(viol**2 + eps))
        
        # x boundaries
        viol_x = r - x
        viol_1mx = r - (1 - x)
        grad[0:N] += lam * 2 * soft_max_smooth(viol_x) * smooth_grad(viol_x) * (-1)  # d/dx of (r-x) is -1
        grad[0:N] += lam * 2 * soft_max_smooth(viol_1mx) * smooth_grad(viol_1mx) * (1)   # d/dx of (r-(1-x)) is 1
        # Wait, let me be more careful
        # pen_x = sum(soft_max(r_i - x_i)^2)
        # d/dx_i = 2 * soft_max(r_i - x_i) * smooth_grad(r_i - x_i) * d(r_i - x_i)/dx_i = 2 * soft_max * smooth_grad * (-1)
        # So grad_x_i += lam * (-2) * soft_max(r_i - x_i) * smooth_grad(r_i - x_i)
        # And for pen_1mx: d/dx_i of (r_i - (1-x_i))^2 = 2 * soft_max * smooth_grad * (1)
        
        # Let me redo this properly
        grad[0:N] = -1.0  # from -sum(r), but r is separate, so actually 0 for x
        grad[0:N] = 0.0
        grad[N:2*N] = 0.0
        grad[2*N:3*N] = -1.0  # from -sum(r)
        
        # Boundary: pen_bound = sum_i [soft_max(r_i-x_i)^2 + soft_max(r_i-(1-x_i))^2 + soft_max(r_i-y_i)^2 + soft_max(r_i-(1-y_i))^2]
        # d/dx_i: 2*soft_max(r_i-x_i)*sg(r_i-x_i)*(-1) + 2*soft_max(r_i-(1-x_i))*sg(r_i-(1-x_i))*(1)
        # d/dy_i: 2*soft_max(r_i-y_i)*sg(r_i-y_i)*(-1) + 2*soft_max(r_i-(1-y_i))*sg(r_i-(1-y_i))*(1)
        # d/dr_i: 2*soft_max(r_i-x_i)*sg(r_i-x_i)*(1) + 2*soft_max(r_i-(1-x_i))*sg(r_i-(1-x_i))*(1) + ... similar for y
        
        for i in range(N):
            # x contribution
            vx = r[i] - x[i]
            v1mx = r[i] - (1 - x[i])
            vy = r[i] - y[i]
            v1my = r[i] - (1 - y[i])
            
            sx = soft_max_smooth(np.array([vx]))[0]
            s1mx = soft_max_smooth(np.array([v1mx]))[0]
            sy = soft_max_smooth(np.array([vy]))[0]
            s1my = soft_max_smooth(np.array([v1my]))[0]
            
            gx = smooth_grad(np.array([vx]))[0]
            g1mx = smooth_grad(np.array([v1mx]))[0]
            gy = smooth_grad(np.array([vy]))[0]
            g1my = smooth_grad(np.array([v1my]))[0]
            
            grad[i] += lam * (-2*sx*gx + 2*s1mx*g1mx)
            grad[N+i] += lam * (-2*sy*gy + 2*s1my*g1my)
            grad[2*N+i] += lam * (2*sx*gx + 2*s1mx*g1mx + 2*sy*gy + 2*s1my*g1my)
        
        # Overlap penalty
        # pen_overlap = sum_{i<j} soft_max((r_i+r_j)^2 - dist_ij^2)^2
        # Let u_ij = (r_i+r_j)^2 - dist_ij^2
        # d/du_ij of soft_max(u)^2 = 2*soft_max(u)*sg(u)
        # d u_ij / dx_i = -2*(x_i - x_j) = 2*(x_j - x_i)
        # d u_ij / dx_j = 2*(x_i - x_j)
        # d u_ij / dr_i = 2*(r_i+r_j)
        # d u_ij / dr_j = 2*(r_i+r_j)
        
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        u = r_sum**2 - dist_sq
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        for i in range(N):
            for j in range(i+1, N):
                u_ij = u[i, j]
                s = soft_max_smooth(np.array([u_ij]))[0]
                sg = smooth_grad(np.array([u_ij]))[0]
                
                # Gradient factor
                factor = lam * 2 * s * sg
                
                # d/dx_i of u_ij = -2*(x_i - x_j)
                # d/dx_j of u_ij = -2*(x_j - x_i) = 2*(x_i - x_j)
                grad[i] += factor * (-2*(x[i] - x[j]))
                grad[j] += factor * (-2*(x[j] - x[i]))
                grad[N+i] += factor * (-2*(y[i] - y[j]))
                grad[N+j] += factor * (-2*(y[j] - y[i]))
                grad[2*N+i] += factor * 2*(r[i] + r[j])
                grad[2*N+j] += factor * 2*(r[i] + r[j])
        
        return grad
    
    def is_valid(v):
        x, y, r = unpack(v)
        # Check boundaries
        if np.any(x < r - 1e-9) or np.any(1 - x < r - 1e-9):
            return False
        if np.any(y < r - 1e-9) or np.any(1 - y < r - 1e-9):
            return False
        if np.any(r < -1e-9):
            return False
            
        # Check overlaps
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # dist >= r_sum for i != j
        dist[np.diag_indices(N)] = np.inf
        if np.any(dist < r_sum - 1e-9):
            return False
        return True
    
    def get_initial_guesses():
        guesses = []
        
        # Random
        for _ in range(3):
            x = np.random.rand(N)
            y = np.random.rand(N)
            r = np.random.rand(N) * 0.08 + 0.02
            v = np.concatenate([x, y, r])
            guesses.append(v)
            
        # Grid - 6x6 = 36, use first 32
        side = 6
        xs = (np.arange(side) + 0.5) / side
        ys = (np.arange(side) + 0.5) / side
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_x = grid_x.flatten()[:N]
        grid_y = grid_y.flatten()[:N]
        grid_r = np.full(N, 0.5 / side * 0.95)
        v_grid = np.concatenate([grid_x, grid_y, grid_r])
        guesses.append(v_grid)
        
        # Perturbed Grid
        v_pert = v_grid + np.random.randn(3*N) * 0.005
        guesses.append(v_pert)
        
        # Another perturbed grid with different seed
        np.random.seed(42)
        v_pert2 = v_grid + np.random.randn(3*N) * 0.01
        np.random.seed()
        guesses.append(v_pert2)
        
        # Hexagonal packing initial guess
        hex_positions = []
        r0 = 1.0 / 8
        dx = 2 * r0
        dy = np.sqrt(3) * r0
        for row in range(6):
            for col in range(6):
                x = dx * col + (dx/2 if row % 2 else 0) + r0
                y = dy * row + r0
                if x + r0 <= 1.0 and y + r0 <= 1.0:
                    hex_positions.append((x, y))
        
        # Take first 32
        hex_positions = hex_positions[:N]
        if len(hex_positions) == N:
            hx = np.array([p[0] for p in hex_positions])
            hy = np.array([p[1] for p in hex_positions])
            hr = np.full(N, r0 * 0.9)
            v_hex = np.concatenate([hx, hy, hr])
            guesses.append(v_hex)
            # Perturbed hex
            v_hex_pert = v_hex + np.random.randn(3*N) * 0.005
            guesses.append(v_hex_pert)
        
        return guesses
    
    guesses = get_initial_guesses()
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
    for i in range(N):
        bounds.append((0, 1)) # y
    for i in range(N):
        bounds.append((0, 0.5)) # r
    
    # First pass: moderate optimization with analytic gradients
    for v0 in guesses:
        if time.time() > deadline - 30:  # Reserve 30s for polish
            break
            
        try:
            res = minimize(objective, v0, jac=objective_grad, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 1500, 'ftol': 1e-9, 'gtol': 1e-7, 'maxfun': 5000})
            
            if time.time() > deadline - 30:
                break
                
            if res.success or res.fun < 1e-5:
                v_opt = res.x
                v_opt = np.clip(v_opt, [b[0] for b in bounds], [b[1] for b in bounds])
                
                if is_valid(v_opt):
                    score = np.sum(v_opt[2*N:3*N])
                    if score > best_score:
                        best_score = score
                        best_solution = v_opt
        except Exception:
            pass
    
    # If no valid solution found, return a trivial valid one
    if best_solution is None:
        side = 6
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Second pass: high-precision polish on the best solution
    if time.time() < deadline - 10:
        try:
            # Use tighter tolerances for polish
            res_polish = minimize(objective, best_solution, jac=objective_grad, method='L-BFGS-B', bounds=bounds,
                                 options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000})
            
            if time.time() < deadline - 5:
                v_polished = res_polish.x
                v_polished = np.clip(v_polished, [b[0] for b in bounds], [b[1] for b in bounds])
                
                if is_valid(v_polished):
                    score_polished = np.sum(v_polished[2*N:3*N])
                    if score_polished > best_score:
                        best_score = score_polished
                        best_solution = v_polished
        except Exception:
            pass
    
    # Third pass: try to improve with simulated annealing-like perturbation and re-optimization
    if time.time() < deadline - 20:
        for _ in range(5):
            if time.time() > deadline - 15:
                break
            
            # Perturb the best solution
            perturbation_scale = 0.02
            v_perturbed = best_solution + np.random.randn(3*N) * perturbation_scale
            v_perturbed = np.clip(v_perturbed, [b[0] for b in bounds], [b[1] for b in bounds])
            
            try:
                res_pert = minimize(objective, v_perturbed, jac=objective_grad, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 3000})
                
                if time.time() > deadline - 10:
                    break
                    
                if res.success or res_pert.fun < 1e-5:
                    v_opt = res_pert.x
                    v_opt = np.clip(v_opt, [b[0] for b in bounds], [b[1] for b in bounds])
                    
                    if is_valid(v_opt):
                        score = np.sum(v_opt[2*N:3*N])
                        if score > best_score:
                            best_score = score
                            best_solution = v_opt
            except Exception:
                pass
    
    x, y, r = unpack(best_solution)
    result = np.column_stack((x, y, r))
    return result
