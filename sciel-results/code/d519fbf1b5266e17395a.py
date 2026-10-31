import numpy as np
from scipy.optimize import minimize
import time

N = 26

def _compute_radius_sum(res):
    return np.sum(res[:, 2])

def _is_valid(res, tol=1e-5):
    """Check if the packing is valid."""
    if len(res) != N:
        return False
    # Check radii are positive
    if np.any(res[:, 2] <= 0):
        return False
    # Check boundaries
    if np.any(res[:, 0] - res[:, 2] < -tol):
        return False
    if np.any(res[:, 1] - res[:, 2] < -tol):
        return False
    if np.any(1 - res[:, 0] - res[:, 2] < -tol):
        return False
    if np.any(1 - res[:, 1] - res[:, 2] < -tol):
        return False
    
    # Check pairwise distances
    # Vectorized check
    dx = res[:, 0][:, None] - res[:, 0][None, :]
    dy = res[:, 1][:, None] - res[:, 1][None, :]
    dist_sq = dx**2 + dy**2
    r_sum = res[:, 2][:, None] + res[:, 2][None, :]
    # We need dist_sq >= r_sum^2 for i != j
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    if np.any(dist_sq[mask] < (r_sum**2)[mask] - tol):
        return False
    return True

def _fix_invalid(res, tol=1e-6):
    """Attempt to fix small violations by shrinking radii."""
    res = res.copy()
    # Shrink radii slightly
    res[:, 2] *= (1 - 1e-4)
    # If still invalid, shrink more aggressively
    if not _is_valid(res, tol=1e-4):
        res[:, 2] *= 0.9
    return res

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s buffer for safety
    
    best_res = None
    best_score = -1.0
    
    # Initial guess: Grid with jitter
    side = int(np.ceil(np.sqrt(N)))
    base_cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)][:N]
    
    # Define constraints for trust-constr
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    
    def bounds():
        b = []
        for _ in range(N):
            b.append((0, 1)) # x
            b.append((0, 1)) # y
            b.append((1e-6, 0.5)) # r (max 0.5 for one circle, but sum matters)
        return b

    def constraints():
        cons = []
        # Boundary constraints: x - r >= 0, etc.
        # trust-constr expects fun(x) >= 0
        
        # 1. x_i - r_i >= 0
        def c_x_left(x):
            res = x.reshape(N, 3)
            return res[:, 0] - res[:, 2]
        cons.append({'type': 'ineq', 'fun': c_x_left})
        
        # 2. y_i - r_i >= 0
        def c_y_left(x):
            res = x.reshape(N, 3)
            return res[:, 1] - res[:, 2]
        cons.append({'type': 'ineq', 'fun': c_y_left})
        
        # 3. 1 - x_i - r_i >= 0
        def c_x_right(x):
            res = x.reshape(N, 3)
            return 1 - res[:, 0] - res[:, 2]
        cons.append({'type': 'ineq', 'fun': c_x_right})
        
        # 4. 1 - y_i - r_i >= 0
        def c_y_right(x):
            res = x.reshape(N, 3)
            return 1 - res[:, 1] - res[:, 2]
        cons.append({'type': 'ineq', 'fun': c_y_right})
        
        # 5. Pairwise non-overlap: dist_sq - (r_i + r_j)^2 >= 0
        # This is O(N^2) constraints. N=26 -> 325 pairs.
        # This might be too many constraints for trust-constr to handle efficiently in one shot?
        # Alternatively, we can use a single constraint function that returns the minimum slack?
        # No, trust-constr handles arrays of inequality constraints.
        
        # However, 325 constraints is a lot.
        # Let's try to group them or use a different method if it's too slow.
        # But the directive specifically asked for trust-constr with hard constraints.
        
        # Let's implement the pairwise constraints.
        pairs = []
        for i in range(N):
            for j in range(i+1, N):
                pairs.append((i, j))
                
        def c_overlap(x):
            res = x.reshape(N, 3)
            slacks = np.zeros(len(pairs))
            for k, (i, j) in enumerate(pairs):
                dx = res[i, 0] - res[j, 0]
                dy = res[i, 1] - res[j, 1]
                dist_sq = dx**2 + dy**2
                r_sum = res[i, 2] + res[j, 2]
                slacks[k] = dist_sq - r_sum**2
            return slacks
            
        cons.append({'type': 'ineq', 'fun': c_overlap})
        
        return cons

    def objective(x):
        res = x.reshape(N, 3)
        return -np.sum(res[:, 2])

    def jac_objective(x):
        # Gradient of -sum(r_i)
        # x is [x1, y1, r1, ...]
        grad = np.zeros(3*N)
        for i in range(N):
            grad[i*3 + 2] = -1.0
        return grad

    # Run multiple restarts
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        if restart == 0:
            # Grid start
            x0 = np.zeros(3*N)
            for i, (cx, cy) in enumerate(base_cells):
                x0[i*3] = cx
                x0[i*3+1] = cy
                x0[i*3+2] = 0.05 # Small initial radius
        else:
            # Random start
            x0 = np.random.rand(3*N)
            # Ensure initial validity roughly
            # Random positions
            x0[0::3] = np.random.rand(N)
            x0[1::3] = np.random.rand(N)
            x0[2::3] = 0.02 # Very small radii
            
        try:
            res_opt = minimize(
                objective, 
                x0, 
                method='trust-constr', 
                jac=jac_objective,
                bounds=bounds(),
                constraints=constraints(),
                options={'maxiter': 500, 'gtol': 1e-8}
            )
            
            if res_opt.success:
                res_arr = res_opt.x.reshape(N, 3)
                if _is_valid(res_arr):
                    score = _compute_radius_sum(res_arr)
                    if score > best_score:
                        best_score = score
                        best_res = res_arr
                else:
                    # Try to fix
                    res_fixed = _fix_invalid(res_arr)
                    if _is_valid(res_fixed):
                        score = _compute_radius_sum(res_fixed)
                        if score > best_score:
                            best_score = score
                            best_res = res_fixed
                            
        except Exception as e:
            print(f"Optimization failed: {e}")
            continue

    # If no valid result found, return a valid trivial one
    if best_res is None:
        # Fallback: small grid
        res = np.zeros((N, 3))
        side = int(np.ceil(np.sqrt(N)))
        for i in range(N):
            cx, cy = base_cells[i]
            res[i, 0] = cx
            res[i, 1] = cy
            res[i, 2] = 0.01
        return res

    return best_res
