"""
Program to pack 200 equal circles in a unit square [-0.5, 0.5]^2.
Uses hexagonal initialization, smooth surrogate optimization (L-BFGS-B), and SLSQP polish.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 200

def construct():
    start_time = time.time()
    # Safety margin: 10 seconds before the 150s limit
    deadline = start_time + 140.0
    
    def time_exceeded():
        return time.time() > deadline

    # 1. Hexagonal Initialization
    # We want to place N points in a hexagonal pattern.
    # Optimal radius for 200 circles is ~0.0366.
    # Diameter ~ 0.0732.
    # Hex spacing s = 2 * r * sqrt(3) / 2 ? No.
    # In hex packing, distance between centers is 2r.
    # Let's just create a grid and crop.
    
    # Estimate spacing
    # Area = 1. N=200.
    # If we use a square grid, spacing is 1/sqrt(200) ~ 0.0707.
    # Hex is denser.
    
    # Generate a large hex grid and crop to center
    # Hex lattice vectors:
    # v1 = (s, 0)
    # v2 = (s/2, s*sqrt(3)/2)
    
    # Let's try to fit N points.
    # Rows: m. Points per row: ~ N/m.
    # Let m = 14. Points per row ~ 14.
    # Total ~ 196. Close enough.
    
    m_rows = 14
    m_cols = 15 # Alternate rows
    
    # Spacing
    # We want the grid to fit in [-0.5, 0.5].
    # Width of m_cols points with spacing s: (m_cols-1)*s + s/2 (for offset) ?
    # Let's just pick s = 0.072 (slightly larger than 1/sqrt(200) to be safe, then optimize)
    s = 0.072
    
    points = []
    for i in range(m_rows):
        for j in range(m_cols):
            x = (j - m_cols/2 + 0.5) * s
            y = (i - m_rows/2 + 0.5) * s * np.sqrt(3)
            if i % 2 == 1:
                x += s / 2.0
            points.append([x, y])
            
    points = np.array(points)
    
    # If we have too many, crop to center
    if len(points) > N:
        # Sort by distance from origin
        dists = np.sum(points**2, axis=1)
        idx = np.argsort(dists)
        points = points[idx[:N]]
    elif len(points) < N:
        # Add random points near boundary or just duplicate? 
        # Better: expand grid slightly
        # For now, just repeat last point (will be fixed by optimizer)
        # Or generate more
        # Let's just use what we have if close, but N must be exact.
        # If < N, add random points in the square
        extra = np.random.rand(N - len(points), 2) - 0.5
        points = np.vstack([points, extra])
        
    # Ensure shape is (N, 2)
    points = points[:N]
    
    # 2. Smooth Surrogate Optimization
    # Objective: Minimize -softmin(distances)
    # softmin(D) = -1/beta * log(sum(exp(-beta * d_k)))
    # We want to maximize softmin, so minimize -softmin.
    
    def objective_and_grad(x_flat, beta):
        # x_flat shape (2*N,)
        x = x_flat.reshape(N, 2)
        
        # Calculate distances
        # Pairwise
        # To save memory/time, we might not compute ALL pairwise if N is large, 
        # but N=200 is small enough (20k pairs).
        
        # Pairwise distances
        # Use broadcasting
        # diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        # dists = np.sqrt(np.sum(diff**2, axis=2))
        # This is O(N^2) memory. 200x200x2 is small.
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Mask diagonal
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        
        # Boundary distances
        # d_i_boundary = min(0.5 - x_i, 0.5 + x_i, 0.5 - y_i, 0.5 + y_i)
        # We treat these as separate terms in the sum
        # d_1 = 0.5 - x
        # d_2 = 0.5 + x
        # d_3 = 0.5 - y
        # d_4 = 0.5 + y
        
        d_boundaries = np.vstack([
            0.5 - x[:, 0],
            0.5 + x[:, 0],
            0.5 - x[:, 1],
            0.5 + x[:, 1]
        ])
        
        # Combine all distances into a flat array for log-sum-exp
        # We need to be careful with numerical stability.
        # log(sum(exp(-beta * d)))
        # Let d_min = min(d).
        # log(exp(-beta*d_min) * sum(exp(-beta*(d - d_min))))
        # = -beta*d_min + log(sum(exp(-beta*(d - d_min))))
        
        all_dists = np.concatenate([dists.flatten(), d_boundaries.flatten()])
        
        # Filter out any non-finite or negative (shouldn't happen if inside, but safety)
        # If a point is outside, distance is negative. 
        # We want to penalize this heavily.
        # The softmin will naturally favor smaller distances.
        # If distance is negative, exp(-beta * neg) = exp(pos) -> large.
        # This is good, it increases the objective (which we minimize), so it's a penalty.
        
        d_min = np.min(all_dists)
        # Stability shift
        shifted = all_dists - d_min
        # exp(-beta * shifted)
        # If beta is large, this can underflow. 
        # We only care about the log sum.
        
        # To avoid overflow/underflow in exp:
        # We compute log_sum_exp = -beta*d_min + log(sum(exp(-beta*shifted)))
        
        # If beta is very large, only terms where shifted is close to 0 matter.
        
        # Calculate sum of exps
        # Clip shifted to avoid huge positive exponents if d_min was very negative?
        # No, shifted is >= 0. So -beta*shifted <= 0. exp <= 1. Safe.
        
        exps = np.exp(-beta * shifted)
        sum_exps = np.sum(exps)
        
        # Objective: -softmin = -(-1/beta * log_sum_exp) = 1/beta * log_sum_exp
        # Wait.
        # softmin = -1/beta * log(sum(exp(-beta*d)))
        # We want to MAXIMIZE softmin.
        # So we MINIMIZE -softmin = 1/beta * log(sum(exp(-beta*d)))
        
        val = (1.0 / beta) * np.log(sum_exps)
        
        # Gradient
        # d/dx (1/beta * log(sum exp(-beta d_k)))
        # = 1/beta * 1/sum_exp * sum (d/dx exp(-beta d_k))
        # = 1/beta * 1/sum_exp * sum (-beta exp(-beta d_k) * d/dx d_k)
        # = -1/sum_exp * sum (exp(-beta d_k) * d/dx d_k)
        # = - sum (p_k * d/dx d_k)  where p_k = exp(-beta d_k) / sum_exp
        
        # We need d/dx d_k for each term.
        # This is expensive to compute analytically for all 20k+ terms.
        # We can use finite differences or approximate.
        # Given L-BFGS-B, we can provide the objective only and let it approximate gradient, 
        # or we can compute gradient numerically.
        # For N=200, numerical gradient (central diff) requires 2*N+1 evaluations.
        # 401 evaluations per gradient step.
        # Each evaluation is O(N^2) ~ 40k ops.
        # 401 * 40k ~ 16M ops per gradient.
        # L-BFGS-B might take 100-500 iterations.
        # 16M * 500 = 8B ops. Might be too slow in Python/NumPy if not vectorized well.
        
        # Let's try to compute the gradient analytically for the pairwise terms.
        # d/dx_i d_{ij} = (x_i - x_j) / d_{ij}
        
        # Let's stick to objective only for L-BFGS-B and let it use finite differences?
        # SciPy's L-BFGS-B supports `jac=None` which uses finite differences.
        # This is simpler to implement correctly.
        
        return val
    
    def objective_only(x_flat, beta):
        return objective_and_grad(x_flat, beta)

    # Optimization Loop
    best_x = points.flatten()
    best_score = -np.inf
    
    # Increase beta over time
    betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for beta in betas:
        if time_exceeded():
            break
            
        # Optimize
        res = minimize(
            objective_only, 
            best_x, 
            args=(beta,), 
            method='L-BFGS-B', 
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        
        if res.success or res.fun < best_score:
            best_x = res.x
            # Update best_score with actual min distance for tracking?
            # For now, just keep the x.
            
        if time_exceeded():
            break

    # 3. SLSQP Polish
    # Maximize r
    # Variables: x, y, r
    # Constraints:
    # 1. x_i >= -0.5 + r
    # 2. x_i <= 0.5 - r
    # 3. y_i >= -0.5 + r
    # 4. y_i <= 0.5 - r
    # 5. (x_i - x_j)^2 + (y_i - y_j)^2 >= (2r)^2
    
    # We will maximize r by minimizing -r.
    
    def slsqp_objective(z):
        # z shape (2*N + 1,)
        r = z[-1]
        # Return -r
        return -r

    def slsqp_constraints(z):
        r = z[-1]
        x = z[:2*N].reshape(N, 2)
        
        cons = []
        
        # Boundary constraints
        # x_i - r >= -0.5  =>  x_i - r + 0.5 >= 0
        # -x_i - r + 0.5 >= 0
        # y_i - r + 0.5 >= 0
        # -y_i - r + 0.5 >= 0
        
        # Vectorize boundary constraints
        # We can add them as a single array constraint if SLSQP supports it?
        # SLSQP 'ineq' constraint is a scalar function.
        # We need to loop or find a way to vectorize.
        # Looping 4*200 = 800 constraints is fine.
        
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[2*i] - z[-1] + 0.5})
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: -z[2*i] - z[-1] + 0.5})
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[2*i+1] - z[-1] + 0.5})
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: -z[2*i+1] - z[-1] + 0.5})
            
        # Pairwise constraints
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (2r)^2 >= 0
        
        # This is 20k constraints. This might be slow.
        # We can use a sparse approach or just accept the time.
        # Given 150s limit, and we've spent time on L-BFGS-B, we have ~100s left.
        
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq', 
                    'fun': lambda z, i=i, j=j: 
                        (z[2*i] - z[2*j])**2 + (z[2*i+1] - z[2*j+1])**2 - (2*z[-1])**2
                })
                
        return cons

    # Initial guess for SLSQP
    # Use the result from L-BFGS-B
    x_init = best_x.reshape(N, 2)
    # Estimate r from min distance
    # Compute min dist
    diff = x_init[:, np.newaxis, :] - x_init[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    min_bound = np.min([
        np.min(0.5 - x_init[:, 0]),
        np.min(0.5 + x_init[:, 0]),
        np.min(0.5 - x_init[:, 1]),
        np.min(0.5 + x_init[:, 1])
    ])
    
    r_init = 0.5 * min(min_pair, min_bound) * 0.99 # Slightly conservative
    
    z_init = np.concatenate([x_init.flatten(), [r_init]])
    
    # Run SLSQP
    # This might be very slow with 20k constraints.
    # If it's too slow, we skip it and return the L-BFGS-B result.
    
    if not time_exceeded():
        try:
            res_slsqp = minimize(
                slsqp_objective, 
                z_init, 
                method='SLSQP', 
                constraints=slsqp_constraints(z_init),
                options={'maxiter': 50, 'ftol': 1e-6}
            )
            
            if res_slsqp.success:
                best_x = res_slsqp.x[:2*N]
        except Exception:
            # If SLSQP fails or is too slow, keep previous best
            pass

    # Final Validation and Adjustment
    x_final = best_x.reshape(N, 2)
    
    # Ensure points are strictly inside or on boundary
    # If any point is outside, clamp it?
    # The score function handles it by returning invalid or low score.
    # We want to maximize score.
    
    # Compute score
    def compute_score(x):
        x = x.reshape(N, 2)
        # Pairwise
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Boundary
        min_bound = np.min([
            np.min(0.5 - x[:, 0]),
            np.min(0.5 + x[:, 0]),
            np.min(0.5 - x[:, 1]),
            np.min(0.5 + x[:, 1])
        ])
        
        return 0.5 * min(min_pair, min_bound)

    score = compute_score(x_final)
    
    return x_final

# EVOLVE-BLOCK-END
