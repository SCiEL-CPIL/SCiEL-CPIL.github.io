import numpy as np
import scipy.optimize as opt
import time

N = 26

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def compute_penalty(params, N):
        """
        params: array of shape (N, 3) with [x, y, r]
        Returns penalty value to minimize.
        We want to maximize sum(r), so we minimize -sum(r) + penalties.
        """
        x = params[:, 0]
        y = params[:, 1]
        r = params[:, 2]
        
        # Objective: -sum(r)
        obj = -np.sum(r)
        
        # Penalty for overlaps
        # Vectorized distance calculation
        # d_ij = sqrt((xi-xj)^2 + (yi-yj)^2)
        # We need sum over i < j of max(0, ri + rj - d_ij)^2
        
        # Efficient way:
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        
        r_sum = r[:, None] + r[None, :]
        
        # Overlap amount
        overlap = np.maximum(0, r_sum - d)
        
        # Only consider i < j to avoid double counting and self-interaction
        # We can use a mask or just sum the upper triangle
        iu = np.triu_indices(N, k=1)
        penalty_overlap = np.sum(overlap[iu] ** 2)
        
        # Penalty for boundaries
        # x >= r, x <= 1-r => r <= x and r <= 1-x
        # violation = max(0, r - x) + max(0, r - (1-x)) ...
        
        pen_bound = 0.0
        # Left
        pen_bound += np.sum(np.maximum(0, r - x) ** 2)
        # Right
        pen_bound += np.sum(np.maximum(0, r - (1 - x)) ** 2)
        # Bottom
        pen_bound += np.sum(np.maximum(0, r - y) ** 2)
        # Top
        pen_bound += np.sum(np.maximum(0, r - (1 - y)) ** 2)
        
        return obj + penalty_overlap + pen_bound

    def is_valid(params, N, tol=1e-4):
        x = params[:, 0]
        y = params[:, 1]
        r = params[:, 2]
        
        if np.any(r < 0):
            return False
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
            
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    def get_initial_guess():
        # Grid initialization
        side = int(np.ceil(np.sqrt(N)))
        # Use a grid of side x side, take first N
        # Center points
        xs = np.linspace(0.5/side, 1 - 0.5/side, side)
        ys = np.linspace(0.5/side, 1 - 0.5/side, side)
        
        pts = []
        for i in range(side):
            for j in range(side):
                if len(pts) >= N:
                    break
                pts.append((xs[i], ys[j]))
            if len(pts) >= N:
                break
        
        # If we didn't get enough (shouldn't happen for N=26, side=6 -> 36 pts), pad
        while len(pts) < N:
            pts.append((0.5, 0.5))
            
        pts = np.array(pts[:N])
        
        # Initial radii: small, based on grid spacing
        # Spacing is 1/side
        r_init = 0.45 / side
        # Add some noise to break symmetry
        params = np.zeros((N, 3))
        params[:, 0] = pts[:, 0]
        params[:, 1] = pts[:, 1]
        params[:, 2] = r_init
        
        # Add small random perturbation
        params += np.random.normal(0, 0.01, params.shape)
        
        # Clip to valid range roughly
        params[:, 0] = np.clip(params[:, 0], 0.1, 0.9)
        params[:, 1] = np.clip(params[:, 1], 0.1, 0.9)
        params[:, 2] = np.clip(params[:, 2], 0.01, 0.2)
        
        return params

    best_params = None
    best_val = -np.inf
    
    # We will try a few restarts if time permits
    max_restarts = 10 # Will break based on time
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        x0 = get_initial_guess()
        
        # Optimization
        # L-BFGS-B is good for bounded variables
        # Bounds: x in [0, 1], y in [0, 1], r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((0, 0.5)) # r
        
        # Flatten x0 for scipy
        x0_flat = x0.flatten()
        
        try:
            # Use L-BFGS-B
            # Maxiter needs to be tuned. 500-1000 is reasonable.
            res = opt.minimize(
                compute_penalty, 
                x0_flat, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            
            params_res = res.x.reshape(N, 3)
            
            # Check validity
            if is_valid(params_res, N):
                val = np.sum(params_res[:, 2])
                if val > best_val:
                    best_val = val
                    best_params = params_res.copy()
            else:
                # If invalid, try to repair by shrinking radii
                # This is a heuristic repair
                x = params_res[:, 0]
                y = params_res[:, 1]
                r = params_res[:, 2]
                
                # Shrink radii until valid?
                # Simple approach: multiply r by 0.95 and check
                # Or project: for each circle, reduce r to fit in box and not overlap neighbors?
                # For now, if invalid, discard or try to fix simply.
                # Let's try a simple shrink
                scale = 0.9
                for _ in range(10):
                    r_new = r * scale
                    params_test = np.column_stack((x, y, r_new))
                    if is_valid(params_test, N):
                        val = np.sum(r_new)
                        if val > best_val:
                            best_val = val
                            best_params = params_test.copy()
                        break
                    scale *= 0.95

        except Exception as e:
            pass

    if best_params is None:
        # Fallback: return the trivial grid if optimization failed completely
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_params

# EVOLVE-BLOCK-END
