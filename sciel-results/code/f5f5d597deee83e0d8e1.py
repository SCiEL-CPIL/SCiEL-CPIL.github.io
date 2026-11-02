import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial guess: 3x7 grid
    # Aspect ratio W/H approx 7/3
    # W + H = 2
    # W = 1.4, H = 0.6
    best_score = 0.0
    best_result = None
    
    # Try a few different aspect ratios
    # The optimal aspect ratio for 21 circles is likely close to the grid ratio, 
    # but might be adjusted to allow larger radii.
    # Let's sample a few W values.
    W_values = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
    
    for W in W_values:
        if time.time() > deadline:
            break
        H = 2.0 - W
        
        # Initialize positions in a 3x7 grid within W x H
        # 3 rows, 7 cols
        rows, cols = 3, 7
        # Center of each cell
        x_coords = np.linspace(W/cols, W - W/cols, cols)
        y_coords = np.linspace(H/rows, H - H/rows, rows)
        
        # Create grid of points
        xx, yy = np.meshgrid(x_coords, y_coords)
        x0 = xx.flatten()
        y0 = yy.flatten()
        
        # Only take first 21 (which is all of them 3*7=21)
        x0 = x0[:N]
        y0 = y0[:N]
        
        # Variables: x[0..20], y[0..20]
        # We fix W and H for this iteration.
        
        def objective(params):
            x = params[:N]
            y = params[N:2*N]
            
            # Calculate distances to boundaries
            d_left = x
            d_right = W - x
            d_bottom = y
            d_top = H - y
            
            # Combine boundary distances
            # We want the min distance to any boundary for each circle
            # d_bound_i = min(x_i, W-x_i, y_i, H-y_i)
            d_bound = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))
            
            # Calculate pairwise distances
            # d_ij = sqrt((xi-xj)^2 + (yi-yj)^2)
            # We want min over i<j of d_ij / 2
            # To avoid sqrt and loops, we can compute all pairwise distances
            # But N=21, so 21*20/2 = 210 pairs. This is fast.
            
            # Vectorized pairwise distance calculation
            # dx = x[:, None] - x[None, :]
            # dy = y[:, None] - y[None, :]
            # d = np.sqrt(dx**2 + dy**2)
            # Mask diagonal
            # min_pair_dist = np.min(d[np.triu_indices(N, k=1)])
            
            # For smooth min, we use log-sum-exp
            # Softmin of a set of values v is -1/beta * log(sum(exp(-beta*v)))
            # We want to maximize R, which is min(d_bound, d_pair/2).
            # So we want to minimize -R.
            # Let v = [d_bound_1, ..., d_bound_N, d_12/2, ..., d_20,21/2]
            # R_soft = -1/beta * log(sum(exp(-beta*v)))
            
            beta = 20.0 # Sharpness of min
            
            # Boundary terms
            # v_bound = d_bound
            # v_pair = d_pair / 2
            
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            
            # Get upper triangle indices
            iu = np.triu_indices(N, k=1)
            d_pairs = d[iu]
            v_pairs = d_pairs / 2.0
            
            # Combine all constraints into one vector v
            # v = concat(d_bound, v_pairs)
            v = np.concatenate((d_bound, v_pairs))
            
            # Handle negative values (overlaps or out of bounds)
            # If any v is negative, the soft min will be very negative.
            # We should penalize negative v heavily.
            # Actually, if v is negative, exp(-beta*v) is huge, log is huge, -1/beta*log is very negative.
            # This correctly drives the objective down (since we minimize -R).
            
            # Numerical stability for log-sum-exp
            # max_v = np.max(v)
            # log_sum_exp = max_v + np.log(np.sum(np.exp(v - max_v)))
            # R_soft = -log_sum_exp / beta
            
            # Wait, softmin is -1/beta * log(sum(exp(-beta*v)))
            # Let u = -beta * v
            # sum(exp(u))
            # max_u = np.max(u)
            # log_sum_exp = max_u + np.log(np.sum(np.exp(u - max_u)))
            # R_soft = -log_sum_exp / beta
            
            u = -beta * v
            max_u = np.max(u)
            # If max_u is very large positive, exp(u-max_u) is safe.
            # If max_u is very large negative, exp(u-max_u) is safe.
            log_sum_exp = max_u + np.log(np.sum(np.exp(u - max_u)))
            R_soft = -log_sum_exp / beta
            
            # We want to maximize R_soft, so minimize -R_soft
            return -R_soft

        def jac(params):
            # Gradient of -R_soft
            # R_soft = -1/beta * log(sum(exp(-beta*v)))
            # dR_soft/dv_k = -1/beta * (1/sum) * exp(-beta*v_k) * (-beta)
            # dR_soft/dv_k = exp(-beta*v_k) / sum(exp(-beta*v_j))
            # This is the softmax of -beta*v.
            # d(-R_soft)/dv_k = - dR_soft/dv_k
            
            # d v_k / d params
            # v_bound_i = min(x_i, W-x_i, y_i, H-y_i)
            # This is non-differentiable. The soft min smooths it, but the derivative of the soft min w.r.t v is smooth.
            # However, v itself depends on x,y with min operations.
            # The gradient of the soft min w.r.t x is sum_k (weight_k * dv_k/dx)
            # where weight_k = softmax(-beta*v)_k
            
            x = params[:N]
            y = params[N:2*N]
            
            d_bound = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))
            
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            iu = np.triu_indices(N, k=1)
            d_pairs = d[iu]
            v_pairs = d_pairs / 2.0
            
            v = np.concatenate((d_bound, v_pairs))
            
            u = -beta * v
            max_u = np.max(u)
            exp_u = np.exp(u - max_u)
            sum_exp = np.sum(exp_u)
            weights = exp_u / sum_exp
            
            # Gradient of -R_soft w.r.t v
            # d(-R_soft)/dv = -weights
            
            # Now chain rule: d(-R_soft)/dparams = sum_k (-weights[k] * dv_k/dparams)
            
            grad = np.zeros(2*N)
            
            # Boundary contributions
            # v_bound_i is min of 4 terms.
            # The soft min weight for each term in the global v vector is weights[i] for the i-th boundary term.
            # But v_bound_i is itself a min of 4 local terms.
            # The gradient of the soft min of the global set v w.r.t x_i is:
            # weights[i] * (gradient of v_bound_i w.r.t x_i)
            # gradient of v_bound_i w.r.t x_i is:
            # if x_i is the min, grad is 1.
            # if W-x_i is the min, grad is -1.
            # etc.
            # Since we are using a soft min for the GLOBAL set, we don't need to differentiate the local min explicitly if we treat v_bound_i as a variable?
            # No, v_bound_i is a function of x_i, y_i.
            # The chain rule is: dR/dx_i = sum_k (dR/dv_k * dv_k/dx_i)
            # For k corresponding to boundary i:
            # dv_bound_i/dx_i = 1 if x_i < W-x_i and x_i < y_i and x_i < H-y_i
            # This is discontinuous.
            # To make it differentiable, we should have used a soft min for the boundary terms too, or just accept the non-smoothness and let the optimizer handle it (L-BFGS-B uses finite differences if jac not provided, or we can provide an approximate jac).
            
            # Given the complexity of an exact jac, let's let scipy use finite differences.
            # With 42 variables, finite differences are 42 evaluations per step.
            # Each evaluation is fast (vectorized numpy).
            # This should be efficient enough.
            return None # Let scipy use finite differences
        
        # Optimize
        try:
            res = minimize(objective, np.concatenate((x0, y0)), method='L-BFGS-B', jac=jac, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success:
                x_opt = res.x[:N]
                y_opt = res.x[N:2*N]
                
                # Calculate the true max radius for this configuration
                d_bound = np.minimum(np.minimum(x_opt, W - x_opt), np.minimum(y_opt, H - y_opt))
                dx = x_opt[:, None] - x_opt[None, :]
                dy = y_opt[:, None] - y_opt[None, :]
                d = np.sqrt(dx**2 + dy**2)
                iu = np.triu_indices(N, k=1)
                d_pairs = d[iu]
                R = min(np.min(d_bound), np.min(d_pairs)/2)
                
                # Check validity: R must be positive
                if R > 0:
                    # Construct the result
                    # The result is an array of shape (N, 3): x, y, r
                    # The evaluator measures sum of radii.
                    # We return the circles.
                    result = np.column_stack((x_opt, y_opt, np.full(N, R)))
                    score = np.sum(result[:, 2])
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                        
        except Exception:
            pass

    if best_result is not None:
        return best_result
    
    # Fallback to seed
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

BET: {mechanism: "Physics-inspired smooth optimization of equal-radius grid packing with aspect ratio search", q: 0.9, delta: [0.01, 0.05]}
