import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    W = 1000.0 # Penalty weight, defined here for scope

    def generate_initial_grid(n_points):
        # Try to create a regular grid that fits n_points
        # 100 = 5 * 4 * 5
        # Grid in x: 5 points, y: 4 points, z: 5 points
        # Spacing: 1/(5-1) = 0.25, 1/(4-1) = 1/3, 1/(5-1) = 0.25
        # This gives min dist 0.25 -> r=0.125
        # Let's generate this grid
        nx, ny, nz = 5, 4, 5
        if nx * ny * nz != n_points:
            # Fallback to random if dimensions don't match, but 5*4*5 is 100
            pass
            
        xs = np.linspace(-0.5, 0.5, nx)
        ys = np.linspace(-0.5, 0.5, ny)
        zs = np.linspace(-0.5, 0.5, nz)
        
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Shuffle to break symmetry? Or keep order?
        # Keeping order is fine for optimization.
        return pts

    centers = generate_initial_grid(N)
    
    def calc_radius(c):
        # Efficient min distance calculation
        # For N=100, O(N^2) is fine
        d = np.linalg.norm(c[:, np.newaxis] - c[np.newaxis, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        min_b = np.min(0.5 - np.abs(c))
        return min(min_d / 2.0, min_b)
    
    r_init = calc_radius(centers)
    
    def objective(v):
        c = v[:3*N].reshape(N, 3)
        r = v[3*N]
        
        # Penalty for overlaps
        d = np.linalg.norm(c[:, np.newaxis] - c[np.newaxis, :], axis=2)
        np.fill_diagonal(d, np.inf)
        
        # Overlap amount: max(0, 2r - d)
        overlaps = np.maximum(0, 2*r - d)
        penalty_overlap = np.sum(overlaps**2)
        
        # Boundary violations
        b = 0.5 - np.abs(c)
        violations = np.maximum(0, r - b)
        penalty_boundary = np.sum(violations**2)
        
        return -r + W * (penalty_overlap + penalty_boundary)
        
    def gradient(v):
        c = v[:3*N].reshape(N, 3)
        r = v[3*N]
        
        grad = np.zeros_like(v)
        
        # Gradient of -r is -1 for the r component
        grad[3*N] = -1.0
        
        # Gradient of penalties
        d = np.linalg.norm(c[:, np.newaxis] - c[np.newaxis, :], axis=2)
        np.fill_diagonal(d, np.inf)
        
        overlaps = np.maximum(0, 2*r - d)
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        
        # Factor: 2 * W * overlaps / d
        # Handle d=0 case (diagonal) by setting factor to 0 there
        factor = np.zeros_like(d)
        mask = (d > 1e-12) & (overlaps > 0)
        factor[mask] = 2.0 * W * overlaps[mask] / d[mask]
        
        # grad_c_i = - sum_j factor[i, j] * diff[i, j]
        grad_c = -np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        grad[:3*N] = grad_c.reshape(-1)
        
        # Gradient of boundary penalty
        b = 0.5 - np.abs(c)
        violations = np.maximum(0, r - b)
        
        signs = np.sign(c)
        # If c is 0, sign is 0. Derivative of |x| at 0 is undefined, but subgradient 0 is safe if violation is 0.
        # If violation > 0, c cannot be 0 (since r > 0.5 - 0 = 0.5 is impossible for r<0.5).
        grad_b = -2.0 * W * violations * signs
        
        grad[:3*N] += grad_b.reshape(-1)
        
        # Gradient of r
        grad_r_overlap = 4.0 * W * np.sum(overlaps)
        grad_r_boundary = 2.0 * W * np.sum(violations)
        
        grad[3*N] += grad_r_overlap + grad_r_boundary
        
        return grad

    # Initial guess
    v0 = np.concatenate([centers.flatten(), [r_init]])
    
    bounds = [(-0.5, 0.5)] * (3*N) + [(0.0, 0.5)]
    
    best_v = v0.copy()
    best_score = calc_radius(centers)
    
    # Run optimization
    # Use L-BFGS-B
    # We will iterate a few times, restarting from best point with perturbation if stuck
    
    max_outer_loops = 20
    
    for outer_loop in range(max_outer_loops):
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                objective, 
                v0, 
                jac=gradient, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
        except Exception:
            break
            
        v_new = res.x
        c_new = v_new[:3*N].reshape(N, 3)
        
        # Check validity
        if np.all(np.isfinite(c_new)) and np.all(c_new >= -0.5) and np.all(c_new <= 0.5):
            score_new = calc_radius(c_new)
            
            if score_new > best_score:
                best_score = score_new
                best_v = v_new.copy()
                # Continue from this point, maybe with a tiny perturbation to escape flat areas?
                # Or just use it as new v0.
                v0 = v_new.copy()
            else:
                # If no improvement, perturb and try again
                # Increase perturbation slightly each time
                noise_scale = 0.01 * (outer_loop + 1)
                v0 = best_v + np.random.normal(0, noise_scale, size=best_v.shape)
                v0[:3*N] = np.clip(v0[:3*N], -0.49, 0.49)
                v0[3*N] = np.clip(v0[3*N], 0.0, 0.49)
                
        # Optional: increase W if constraints are heavily violated?
        # For now, keep W constant.
        
    # Final polish: run one more tight optimization from best_v
    if time.time() < deadline - 5:
        try:
            res_final = minimize(
                objective, 
                best_v, 
                jac=gradient, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            c_final = res_final.x[:3*N].reshape(N, 3)
            if np.all(np.isfinite(c_final)) and np.all(c_final >= -0.5) and np.all(c_final <= 0.5):
                score_final = calc_radius(c_final)
                if score_final > best_score:
                    best_v = res_final.x
        except Exception:
            pass

    c_best = best_v[:3*N].reshape(N, 3)
    return c_best
