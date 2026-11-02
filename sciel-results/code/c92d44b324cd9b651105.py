"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint


def _valid(arr):
    """Check if the array represents a valid packing."""
    if arr.shape != (N, 3):
        return False
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r < 0) or np.any(x < 0) or np.any(y < 0):
        return False
    if np.any(x + r > 1 + 1e-9) or np.any(y + r > 1 + 1e-9):
        return False
    if np.any(x - r < -1e-9) or np.any(y - r < -1e-9):
        return False
    # Check pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True


def _score(arr):
    """Compute the sum of radii."""
    return float(np.sum(arr[:, 2]))


def _fix_positions(arr):
    """Fix positions to ensure they are strictly inside the square and non-overlapping."""
    x, y, r = arr[:, 0].copy(), arr[:, 1].copy(), arr[:, 2].copy()
    # Ensure inside
    for i in range(N):
        if x[i] < r[i]:
            x[i] = r[i]
        if x[i] > 1 - r[i]:
            x[i] = 1 - r[i]
        if y[i] < r[i]:
            y[i] = r[i]
        if y[i] > 1 - r[i]:
            y[i] = 1 - r[i]
    return np.column_stack([x, y, r])


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    best_arr = None
    best_score = -np.inf

    # 1. Initial Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4) for i in range(side) for j in range(side)]
    init_arr = np.array(cells[:N])
    
    # 2. Optimize Radii given Positions (Phase 1)
    # Variables: r_0, ..., r_{N-1}
    # Maximize sum(r)
    # Constraints:
    # r_i >= 0
    # r_i <= x_i
    # r_i <= 1 - x_i
    # r_i <= y_i
    # r_i <= 1 - y_i
    # r_i + r_j <= dist(i, j) for all i < j
    
    x0 = init_arr[:, 2]
    x_pos = init_arr[:, 0]
    y_pos = init_arr[:, 1]
    
    # Precompute distances
    dists = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dists[i, j] = np.sqrt((x_pos[i] - x_pos[j])**2 + (y_pos[i] - y_pos[j])**2)
    
    # Linear Constraints for SciPy
    # We want:
    # 1. r_i + r_j <= d_ij  =>  r_i + r_j - d_ij <= 0
    # 2. r_i <= x_i  =>  r_i - x_i <= 0
    # 3. r_i <= 1 - x_i  =>  r_i - (1 - x_i) <= 0
    # 4. r_i <= y_i  =>  r_i - y_i <= 0
    # 5. r_i <= 1 - y_i  =>  r_i - (1 - y_i) <= 0
    # 6. r_i >= 0
    
    A_list = []
    b_list = []
    
    # Pairwise constraints
    for i in range(N):
        for j in range(i + 1, N):
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_list.append(row)
            b_list.append(dists[i, j])
            
    # Wall constraints
    for i in range(N):
        for bound in [x_pos[i], 1 - x_pos[i], y_pos[i], 1 - y_pos[i]]:
            row = np.zeros(N)
            row[i] = 1.0
            A_list.append(row)
            b_list.append(bound)
            
    A = np.array(A_list)
    b = np.array(b_list)
    
    # LinearConstraint(A, -np.inf, b) means -inf <= A @ r <= b
    # But we also need r >= 0.
    # We can use bounds for r >= 0.
    bounds = [(0.0, 1.0)] * N
    
    cons = LinearConstraint(A, -np.inf, b)
    
    # Objective: minimize -sum(r)
    def objective(r):
        return -np.sum(r)
        
    def jac(r):
        return -np.ones(N)
        
    try:
        res = minimize(objective, x0, method='SLSQP', jac=jac, bounds=bounds, constraints=cons,
                       options={'maxiter': 100, 'ftol': 1e-9})
        if res.success:
            r_opt = res.x
            # Clip to ensure validity
            r_opt = np.clip(r_opt, 0, 1)
            arr1 = np.column_stack([x_pos, y_pos, r_opt])
            if _valid(arr1):
                s = _score(arr1)
                if s > best_score:
                    best_score = s
                    best_arr = arr1
    except Exception:
        pass

    if time.time() > deadline:
        return best_arr if best_arr is not None else init_arr

    # 3. Optimize Positions given Radii (Phase 2) - Iterate a few times
    current_arr = best_arr if best_arr is not None else init_arr
    current_score = best_score if best_score > -np.inf else _score(init_arr)
    
    # We will do a few iterations of:
    # 1. Fix positions, optimize radii.
    # 2. Fix radii, optimize positions.
    
    # For position optimization, we need to handle non-linear constraints:
    # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # This is non-convex. We can use SLSQP with these constraints.
    
    # Variables: x_0, y_0, ..., x_{N-1}, y_{N-1}
    # Objective: minimize 0 (we just want a feasible point that maximizes radius potential? 
    # Actually, if we fix radii, we just want to find a valid configuration.
    # But to improve the score, we need to allow radii to grow.
    # So a better approach is to optimize positions AND radii simultaneously?
    # That's 3N variables. 78 variables. SLSQP might struggle but let's try.
    
    # Alternative: Keep radii fixed, move positions to be "tighter" or allow radius increase.
    # Actually, if we move positions, we might be able to increase radii.
    # Let's try optimizing positions to maximize the minimum distance to walls and other circles.
    # Or simply, let's do a joint optimization of x, y, r.
    
    def joint_objective(vars):
        x = vars[:N]
        y = vars[N:2*N]
        r = vars[2*N:]
        return -np.sum(r)
        
    def joint_jac(vars):
        x = vars[:N]
        y = vars[N:2*N]
        r = vars[2*N:]
        return np.concatenate([np.zeros(N), np.zeros(N), -np.ones(N)])
        
    # Constraints for joint optimization are messy.
    # Let's stick to the alternating approach but improve the position step.
    # If we fix radii, we want to find x, y such that circles don't overlap.
    # If they already don't overlap, we can try to "push" them to increase the gap, 
    # which allows radius increase in the next step.
    # Maximizing the minimum gap is a good proxy.
    
    def min_gap(x, y, r):
        min_g = np.inf
        for i in range(N):
            # Gap to walls
            min_g = min(min_g, x[i] - r[i])
            min_g = min(min_g, 1 - x[i] - r[i])
            min_g = min(min_g, y[i] - r[i])
            min_g = min(min_g, 1 - y[i] - r[i])
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                min_g = min(min_g, d - r[i] - r[j])
        return min_g
        
    # We want to maximize min_gap.
    # This is a "max-min" problem.
    # Let's try to maximize the sum of gaps? No, max-min is better for uniformity.
    # But SLSQP minimizes. So minimize -min_gap.
    # -min_gap is non-smooth.
    # Let's use a barrier function or just penalize overlaps.
    # Actually, if we just want to improve the packing, we can try to maximize the sum of squared distances?
    
    # Let's try a simpler heuristic: 
    # Random restarts of the radius optimization with slightly perturbed positions.
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Perturb positions
        noise_scale = 0.01
        x_pert = x_pos + np.random.normal(0, noise_scale, N)
        y_pert = y_pos + np.random.normal(0, noise_scale, N)
        
        # Clip to [0,1]
        x_pert = np.clip(x_pert, 0, 1)
        y_pert = np.clip(y_pert, 0, 1)
        
        # Re-optimize radii with new positions
        dists_pert = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    dists_pert[i, j] = np.sqrt((x_pert[i] - x_pert[j])**2 + (y_pert[i] - y_pert[j])**2)
                    
        A_list = []
        b_list = []
        
        for i in range(N):
            for j in range(i + 1, N):
                row = np.zeros(N)
                row[i] = 1.0
                row[j] = 1.0
                A_list.append(row)
                b_list.append(dists_pert[i, j])
                
        for i in range(N):
            for bound in [x_pert[i], 1 - x_pert[i], y_pert[i], 1 - y_pert[i]]:
                row = np.zeros(N)
                row[i] = 1.0
                A_list.append(row)
                b_list.append(bound)
                
        A = np.array(A_list)
        b = np.array(b_list)
        cons = LinearConstraint(A, -np.inf, b)
        
        x0 = np.clip(r_opt, 0, 1) # Use previous radii as initial guess
        
        try:
            res = minimize(objective, x0, method='SLSQP', jac=jac, bounds=bounds, constraints=cons,
                           options={'maxiter': 50, 'ftol': 1e-8})
            if res.success:
                r_opt_new = res.x
                r_opt_new = np.clip(r_opt_new, 0, 1)
                arr_new = np.column_stack([x_pert, y_pert, r_opt_new])
                if _valid(arr_new):
                    s = _score(arr_new)
                    if s > current_score:
                        current_score = s
                        current_arr = arr_new
                        x_pos = x_pert
                        y_pos = y_pert
                        r_opt = r_opt_new
        except Exception:
            pass

    return current_arr

# EVOLVE-BLOCK-END
