"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Initialization: Hexagonal Packing
    # Start with a hexagonal grid to provide a good initial spread.
    r_est = 0.038
    spacing = 2 * r_est
    hex_spacing = spacing * np.sqrt(3) / 2

    points = []
    max_coord = 0.5 + spacing
    num_rows = int(2 * max_coord / hex_spacing) + 1
    num_cols = int(2 * max_coord / spacing) + 1

    for row in range(num_rows):
        y = -max_coord + row * hex_spacing
        offset = (row % 2) * (spacing / 2)
        for col in range(num_cols):
            x = -max_coord + col * spacing + offset
            points.append([x, y])

    points = np.array(points)
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]

    dist_to_center = np.sum(points**2, axis=1)
    idx = np.argsort(dist_to_center)
    positions = points[idx[:N]].copy()

    if len(positions) < N:
        while len(positions) < N:
            p = np.random.uniform(-0.49, 0.49, size=2)
            positions = np.vstack([positions, [p]])
    else:
        positions = positions[:N]

    # Add tiny jitter to break symmetry
    positions += np.random.normal(0, 0.001, size=positions.shape)
    positions = np.clip(positions, -0.49, 0.49)

    # 2. Optimization: Max-Min Fairness with Log-Barrier
    # Objective: Maximize the minimum radius r.
    # Equivalent to: Minimize -r + Penalty(overlap) + Penalty(boundary)
    # We use a smooth approximation for the min function and log-barriers for constraints.
    
    # Variables: flattened positions (N*2,)
    x0 = positions.flatten()
    
    # Constraints are handled via penalty in the objective to allow L-BFGS-B (which handles bounds but not general constraints easily without SLSQP).
    # We will use a penalty method inside the objective function.
    
    def objective_and_grad(x):
        pos = x.reshape(N, 2)
        
        # 1. Compute all pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        
        # 2. Compute boundary distances
        # d_i = min(x_i + 0.5, 0.5 - x_i, y_i + 0.5, 0.5 - y_i)
        d_x = np.minimum(pos[:, 0] + 0.5, 0.5 - pos[:, 0])
        d_y = np.minimum(pos[:, 1] + 0.5, 0.5 - pos[:, 1])
        d_bound = np.minimum(d_x, d_y)
        
        # 3. Compute the "radius" for each circle
        # r_i is limited by half the min distance to other circles and distance to boundary
        min_pair_dist = np.min(dist, axis=1)
        r_pair = min_pair_dist / 2.0
        r_i = np.minimum(r_pair, d_bound)
        
        # 4. Max-Min Fairness Objective
        # We want to maximize min(r_i).
        # Smooth approximation: -log-sum-exp(-r_i / t) * t
        # This approximates min(r_i) as t -> 0.
        t = 0.01
        # Clip r_i to avoid log(0) or negative issues if they go out of bounds (though we clamp pos)
        # Note: If a circle is outside, d_bound can be negative. 
        # We should add a hard penalty for being outside or overlapping too much.
        
        # Log-sum-exp trick for stability
        # We want to minimize -min(r_i). 
        # Let's use the soft-min: m = -t * log(sum(exp(-r_i/t)))
        # We want to MINIMIZE this m? No, we want to MAXIMIZE min(r_i).
        # So we MINIMIZE -min(r_i).
        # -min(r_i) approx -(-t * log(sum(exp(-r_i/t)))) = t * log(sum(exp(-r_i/t)))
        
        # However, if any r_i is very small (overlap), this term blows up?
        # exp(-large_positive) is small. 
        # If r_i is negative (overlap), -r_i/t is large positive, exp is large. Log is large.
        # So this acts as a penalty for small/negative r_i.
        
        # To be safe, clip r_i to a small positive number for the log-sum-exp calculation?
        # No, we want the penalty to be strong if r_i < 0.
        # Let's just compute it.
        
        # Gradient of t * log(sum(exp(-r_i/t))) wrt r_i is -1 / sum(exp(-r_j/t)) * exp(-r_i/t)
        # This is the softmax probability of i being the minimum.
        
        # To handle potential numerical issues if r_i is very negative:
        # We will add a separate quadratic penalty for violations of a hard constraint r_min_hard = 0.
        # But the log-sum-exp already penalizes small r_i.
        
        # Let's stick to the smooth min formulation.
        # Objective = t * log(sum(exp(-r_i / t)))
        
        # For stability, shift r_i by max(r_i)
        r_shifted = r_i - np.max(r_i)
        # exp(-r_i/t) = exp(-(r_shifted + max_r)/t) = exp(-max_r/t) * exp(-r_shifted/t)
        # log(sum(...)) = -max_r/t + log(sum(exp(-r_shifted/t)))
        # Obj = t * (-max_r/t + log(...)) = -max_r + t * log(sum(exp(-r_shifted/t)))
        
        # This is numerically stable.
        
        exp_terms = np.exp(-r_shifted / t)
        sum_exp = np.sum(exp_terms)
        log_sum = np.log(sum_exp)
        
        obj_val = -np.max(r_i) + t * log_sum
        
        # Add a small regularization to keep points inside and avoid singularities?
        # The log-sum-exp already penalizes small r_i.
        # If a point goes outside, d_bound < 0, so r_i < 0, which increases the objective (penalty).
        
        # Gradient calculation
        # d(obj)/d(r_i) = -1 (from -max_r term if i is max? No, max is not differentiable everywhere)
        # Actually, the derivative of the smooth min m(r) = -t * log(sum(exp(-r_j/t))) is:
        # dm/dr_i = -t * (1/sum) * exp(-r_i/t) * (-1/t) = exp(-r_i/t) / sum_exp = softmax_i
        # We are minimizing f = -m(r) = t * log(sum(exp(-r_i/t))) - max_r? 
        # Wait, my expansion: Obj = -max_r + t * log(sum(exp(-r_shifted/t)))
        # This is not exactly -min(r). 
        # Let's use the standard soft-min for minimization of max(-r_i) which is equivalent to maximizing min(r_i).
        # Minimize f(r) = t * log(sum(exp(-r_i/t)))
        # Gradient wrt r_i: -exp(-r_i/t) / sum_exp
        
        # But we also have the -max_r term in my stable formulation?
        # Let's just use the stable formulation for the value and derive the gradient correctly.
        # f(r) = t * log(sum(exp(-r_i/t)))
        # Let M = max(r_i). f(r) = t * log(exp(-M/t) * sum(exp(-(r_i-M)/t))) = -M + t * log(sum(exp(-(r_i-M)/t)))
        # df/dr_k = d(-M)/dr_k + t * d(log(S))/dr_k
        # If k is unique max, dM/dr_k = 1. Else 0. This is non-smooth.
        # The smooth approximation avoids the max by just using the log-sum-exp directly without the shift trick in the gradient derivation, or by acknowledging that the log-sum-exp itself is the smooth approximation of the max of -r_i.
        
        # Let's simply minimize g(r) = t * log(sum(exp(-r_i/t)))
        # This approximates max(-r_i) = -min(r_i).
        # So minimizing g maximizes min(r_i).
        
        # Gradient of g wrt r_i:
        # dg/dr_i = t * (1/S) * exp(-r_i/t) * (-1/t) = -exp(-r_i/t) / S
        
        # We need d(r_i)/dx_j.
        # r_i = min(min_{j!=i} dist(i,j)/2, d_bound_i)
        # This min is non-differentiable.
        # We need a smooth approximation for the min in r_i calculation too?
        # Or we can use sub-gradients? L-BFGS-B needs gradients.
        
        # Alternative: Use the penalty method on the constraints directly.
        # Minimize: -r_target + sum( max(0, 2*r_target - dist_ij)^2 ) + sum( max(0, r_target - d_bound_i)^2 )
        # This is differentiable (piecewise smooth).
        # We can iterate r_target upwards? Or optimize r_target as a variable?
        # Let's optimize r_target as a variable too!
        # Variables: x (positions), r (radius)
        # Objective: -r + lambda * (sum violations)
        # Violations:
        # V1 = sum_{i<j} max(0, 2r - dist_ij)^2
        # V2 = sum_i max(0, r - d_bound_i)^2
        
        # This is a smooth (C1) function.
        
        # Let's switch strategy to this Penalty Method with r as a variable.
        
        # Unpack x
        pos = x[:N*2].reshape(N, 2)
        r = x[N*2]
        
        # Compute distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        
        # Boundary distances
        d_x = np.minimum(pos[:, 0] + 0.5, 0.5 - pos[:, 0])
        d_y = np.minimum(pos[:, 1] + 0.5, 0.5 - pos[:, 1])
        d_bound = np.minimum(d_x, d_y)
        
        # Violations
        # Pairwise
        # We only care about pairs where 2r > dist
        # Let delta = 2r - dist
        # V1 = sum_{i<j} max(0, delta)^2
        # Gradient wrt r: 2 * delta * 1 (for i<j) -> sum over i<j of 2*delta
        # Gradient wrt pos: 
        # d(V1)/d(pos_i) = sum_{j != i} 2 * delta_ij * d(delta_ij)/d(pos_i)
        # delta_ij = 2r - dist_ij
        # d(delta_ij)/d(pos_i) = - d(dist_ij)/d(pos_i) = - (pos_i - pos_j) / dist_ij
        # So term is 2 * delta_ij * (pos_j - pos_i) / dist_ij
        
        # Boundary
        # V2 = sum_i max(0, r - d_bound_i)^2
        # Let delta_b_i = r - d_bound_i
        # V2 = sum max(0, delta_b_i)^2
        # Grad wrt r: sum 2 * delta_b_i
        # Grad wrt pos_i: 2 * delta_b_i * d(delta_b_i)/d(pos_i)
        # d(delta_b_i)/d(pos_i) = - d(d_bound_i)/d(pos_i)
        # d_bound_i is min of 4 terms. Gradient is the gradient of the active min.
        
        # To make it smooth, we can use a soft-min for d_bound?
        # Or just use the subgradient (pick the active constraint).
        # Since we are using L-BFGS-B, it can handle piecewise smooth if we provide a valid subgradient.
        
        # Let's compute the objective and gradient for the penalty formulation.
        # Objective = -r + mu * (V1 + V2)
        
        mu = 10.0 # Penalty weight
        
        # Pairwise part
        # We need to be careful with the double counting in sum_{i<j}.
        # Let's compute for all i != j and divide by 2.
        
        # delta_matrix = 2*r - dist
        # mask = delta_matrix > 0
        # V1 = 0.5 * sum_{i!=j} max(0, delta_ij)^2
        
        delta = 2*r - dist
        mask_pair = delta > 0
        # V1_val = 0.5 * np.sum(np.where(mask_pair, delta**2, 0))
        
        # Grad r: 0.5 * sum_{i!=j} 2 * delta_ij (where mask) = sum_{i!=j} delta_ij (where mask)
        # Note: sum_{i!=j} is 2 * sum_{i<j}. So 0.5 * 2 * sum_{i<j} = sum_{i<j}.
        # My formula: V1 = sum_{i<j} max(0, delta_ij)^2.
        # Let's stick to i<j to avoid confusion.
        
        # Create upper triangle indices
        i_idx, j_idx = np.triu_indices(N, k=1)
        delta_pair = 2*r - dist[i_idx, j_idx]
        mask_p = delta_pair > 0
        
        V1_val = np.sum(np.where(mask_p, delta_pair**2, 0))
        
        # Grad wrt r
        grad_r = np.sum(np.where(mask_p, 2*delta_pair, 0))
        
        # Grad wrt pos
        grad_pos = np.zeros_like(pos)
        if np.any(mask_p):
            # For each active pair (i, j):
            # Contribution to pos_i: 2 * delta_ij * (pos_j - pos_i) / dist_ij
            # Contribution to pos_j: 2 * delta_ij * (pos_i - pos_j) / dist_ij
            
            d_active = delta_pair[mask_p]
            i_act = i_idx[mask_p]
            j_act = j_idx[mask_p]
            dist_act = dist[i_act, j_act]
            
            # Vectorized update
            # We need to accumulate into grad_pos.
            # Use np.add.at
            vecs = (pos[j_act] - pos[i_act]) / dist_act[:, np.newaxis]
            forces = 2 * d_active[:, np.newaxis] * vecs
            
            np.add.at(grad_pos, i_act, -forces)
            np.add.at(grad_pos, j_act, forces)
            
        # Boundary part
        delta_b = r - d_bound
        mask_b = delta_b > 0
        V2_val = np.sum(np.where(mask_b, delta_b**2, 0))
        
        grad_r += np.sum(np.where(mask_b, 2*delta_b, 0))
        
        # Grad wrt pos for boundary
        # d_bound_i = min(x_i+0.5, 0.5-x_i, y_i+0.5, 0.5-y_i)
        # We need the gradient of the active minimum.
        # Let's determine which constraint is active for each i where mask_b is true.
        
        grad_pos_b = np.zeros_like(pos)
        if np.any(mask_b):
            i_b = np.where(mask_b)[0]
            db_act = delta_b[i_b]
            
            # For each i in i_b, find the active wall
            # d_x = min(x+0.5, 0.5-x)
            # d_y = min(y+0.5, 0.5-y)
            # d_bound = min(d_x, d_y)
            
            # Calculate distances to 4 walls
            d_left = pos[i_b, 0] + 0.5
            d_right = 0.5 - pos[i_b, 0]
            d_bottom = pos[i_b, 1] + 0.5
            d_top = 0.5 - pos[i_b, 1]
            
            # Find min
            d_vals = np.stack([d_left, d_right, d_bottom, d_top], axis=1)
            min_idx = np.argmin(d_vals, axis=1)
            
            # Gradient of d_bound wrt pos:
            # If left (0): grad = [1, 0] -> delta_b grad = -[1, 0]
            # If right (1): grad = [-1, 0] -> delta_b grad = -[-1, 0] = [1, 0]
            # If bottom (2): grad = [0, 1] -> delta_b grad = [0, -1]
            # If top (3): grad = [0, -1] -> delta_b grad = [0, 1]
            
            # Force = 2 * delta_b * (-grad_d_bound)
            # So:
            # Left: -2*db * [1, 0] = [-2*db, 0]
            # Right: -2*db * [-1, 0] = [2*db, 0]
            # Bottom: -2*db * [0, 1] = [0, -2*db]
            # Top: -2*db * [0, -1] = [0, 2*db]
            
            for k, idx in enumerate(min_idx):
                db_val = db_act[k]
                if idx == 0: # Left
                    grad_pos_b[k, 0] += -2 * db_val
                elif idx == 1: # Right
                    grad_pos_b[k, 0] += 2 * db_val
                elif idx == 2: # Bottom
                    grad_pos_b[k, 1] += -2 * db_val
                elif idx == 3: # Top
                    grad_pos_b[k, 1] += 2 * db_val
            
            grad_pos[i_b] += grad_pos_b
        
        # Total Objective
        obj = -r + mu * (V1_val + V2_val)
        
        # Total Gradient
        grad = np.zeros_like(x)
        grad[:N*2] = mu * grad_pos.flatten()
        grad[N*2] = -1 + mu * grad_r
        
        return obj, grad

    # Bounds for L-BFGS-B
    # Positions in [-0.5, 0.5]
    # Radius in [0, 0.5]
    bounds = [(-0.5, 0.5)] * (2*N) + [(0, 0.5)]
    
    # Initial guess for r
    r0 = 0.03
    x0 = np.concatenate([positions.flatten(), [r0]])
    
    # Run optimization
    result = None
    try:
        result = minimize(
            objective_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True, # Returns (f, g)
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
    except Exception:
        pass

    if result is not None and result.success is False:
        # Even if not fully successful, we might have a better point
        # Check if it improved
        pass

    # Extract best position found
    if result is not None:
        x_opt = result.x
        pos_opt = x_opt[:N*2].reshape(N, 2)
        r_opt = x_opt[N*2]
    else:
        pos_opt = positions
        r_opt = r0

    # Post-processing: Ensure validity
    # Clamp positions
    pos_opt = np.clip(pos_opt, -0.5, 0.5)
    
    # Calculate actual radius
    def _calc_radius(pos):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair = np.min(np.sqrt(dist_sq), axis=1) / 2.0
        
        d_x = np.minimum(pos[:, 0] + 0.5, 0.5 - pos[:, 0])
        d_y = np.minimum(pos[:, 1] + 0.5, 0.5 - pos[:, 1])
        d_bound = np.minimum(d_x, d_y)
        
        return np.min(np.minimum(min_pair, d_bound))
    
    final_r = _calc_radius(pos_opt)
    
    # If the optimizer found a radius larger than the actual valid radius (due to penalty relaxation),
    # we might need to project or just return the positions. The score is based on the positions.
    
    # One more refinement step: Local search on the boundary of the feasible set?
    # Or just return the positions.
    
    # Check time
    if time.time() > deadline:
        return pos_opt

    return pos_opt

# EVOLVE-BLOCK-END
