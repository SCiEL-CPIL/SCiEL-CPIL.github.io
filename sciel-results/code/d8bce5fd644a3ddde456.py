import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    # 1. Initialization: 5x5x4 grid
    # x: 5 points, y: 5 points, z: 4 points
    # Spacing x: 0.2, offset 0.4
    # Spacing y: 0.2, offset 0.4
    # Spacing z: 0.25, offset 0.375
    xs = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    ys = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    zs = np.array([-0.375, -0.125, 0.125, 0.375])
    
    # Generate grid
    # Order: z outer, y middle, x inner (matches seed logic roughly, but seed was 5x5x5 truncated)
    # Let's just generate all 100
    centers = []
    for z in zs:
        for y in ys:
            for x in xs:
                centers.append([x, y, z])
    centers = np.array(centers)
    
    # Ensure shape is (N, 3)
    if len(centers) != N:
        # Fallback to seed if grid logic fails (it shouldn't)
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        centers = grid[:N]

    x0 = centers.flatten()
    
    # Precompute indices for pairs
    # Pairs i < j
    i_indices, j_indices = np.triu_indices(N, k=1)
    num_pairs = len(i_indices)
    
    # Wall constraints
    # For each point i and dim k, dist is 0.5 - abs(c_ik)
    # We can handle walls by adding them to the distance vector
    
    def soft_min_radius(x, beta):
        """
        Returns -soft_min_radius to be minimized.
        """
        c = x.reshape(N, 3)
        
        # Wall distances
        # c shape (N, 3)
        # abs_c shape (N, 3)
        wall_dists = 0.5 - np.abs(c) # (N, 3)
        
        # Pair distances
        # c_i - c_j for all pairs
        # c[i_indices] - c[j_indices]
        diffs = c[i_indices] - c[j_indices]
        pair_dists = 0.5 * np.linalg.norm(diffs, axis=1)
        
        # Combine
        # Total distances: N*3 wall dists + num_pairs pair dists
        # We want min of all these.
        # Soft min: -1/beta * log(sum(exp(-beta * d)))
        
        # To avoid overflow/underflow in exp, we can shift by min_d
        # But for L-BFGS, we need gradients.
        # Let's compute the weighted average gradient.
        
        all_dists = np.concatenate([wall_dists.flatten(), pair_dists])
        
        # Clip distances to be positive? No, if a sphere goes outside, dist is negative.
        # The soft min will penalize negative distances heavily if beta is large.
        # However, exp(-beta * negative) = exp(positive) -> overflow.
        # We should handle this carefully.
        # If any dist < 0, the "radius" is effectively 0 (or negative).
        # We can clip the input to exp to a reasonable range, or just let it be.
        # Better: use a stable log-sum-exp.
        
        max_d = np.max(all_dists)
        # If max_d is large, exp(-beta*d) is small.
        # We want sum(exp(-beta*d)).
        # Let m = min(all_dists).
        # sum(exp(-beta*d)) = exp(-beta*m) * sum(exp(-beta*(d-m)))
        # log(sum) = -beta*m + log(sum(exp(-beta*(d-m))))
        # -1/beta * log(sum) = m - 1/beta * log(sum(exp(-beta*(d-m))))
        
        m = np.min(all_dists)
        # If m is very negative, the radius is very negative.
        
        # Calculate weights w_i = exp(-beta * (d_i - m))
        # Note: d_i - m >= 0.
        # If beta is large, only terms where d_i is close to m have significant weight.
        
        # Gradient of r_soft wrt x:
        # r_soft = m - 1/beta * log(sum(exp(-beta*(d-m))))
        # Let S = sum(exp(-beta*(d-m)))
        # dr/dx = dm/dx - 1/beta * (1/S) * dS/dx
        # dS/dx = sum( exp(-beta*(d_i-m)) * (-beta) * dd_i/dx )
        # dr/dx = dm/dx + (1/S) * sum( exp(-beta*(d_i-m)) * dd_i/dx )
        # Note: dm/dx is the gradient of the min distance.
        # If we assume the min is unique, dm/dx = d_d_min/dx.
        # The formula above effectively computes a weighted average of the gradients of all distances,
        # where the weights are proportional to their "closeness" to the minimum.
        # Specifically, w_i = exp(-beta*(d_i-m)) / S.
        # Then dr/dx = sum(w_i * dd_i/dx).
        # This is a smooth approximation of the gradient of the min function.
        
        # Let's implement this.
        
        # Calculate d - m
        dm = all_dists - m
        
        # Calculate weights
        # To avoid underflow for large beta*(d-m), we can just use exp.
        # If beta*(d-m) > 50, exp is ~1e-22, which is fine in float64.
        weights = np.exp(-beta * dm)
        S = np.sum(weights)
        w = weights / S
        
        # Now compute gradient
        grad = np.zeros_like(x)
        
        # 1. Wall distances
        # w_wall is the first N*3 weights
        w_wall = w[:N*3].reshape(N, 3)
        
        # d(d_wall)/dx
        # d(0.5 - |c_ik|)/dc_ik = -sign(c_ik)
        # So grad_wall = -sign(c) * w_wall
        grad_wall = -np.sign(c) * w_wall
        grad += grad_wall.flatten()
        
        # 2. Pair distances
        w_pair = w[N*3:]
        
        # d(d_pair)/dx
        # d_pair = 0.5 * ||c_i - c_j||
        # dd_pair/dx_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
        # dd_pair/dx_j = 0.5 * (c_j - c_i) / ||c_i - c_j||
        # Note: diffs = c_i - c_j. norm = ||diffs||.
        # grad_i = 0.5 * diffs / norm
        # grad_j = -0.5 * diffs / norm
        
        # We need to accumulate these gradients into the full vector.
        # Use np.add.at for efficient accumulation
        
        # norm of diffs
        norms = np.linalg.norm(diffs, axis=1)
        # Avoid division by zero
        norms_safe = np.where(norms < 1e-10, 1e-10, norms)
        
        # Unit vectors
        unit_diffs = diffs / norms_safe[:, np.newaxis]
        
        # Gradients for each pair
        # grad_i = 0.5 * w_pair * unit_diffs
        # grad_j = -0.5 * w_pair * unit_diffs
        
        # Accumulate to i
        np.add.at(grad, i_indices * 3, 0.5 * w_pair * unit_diffs[:, 0])
        np.add.at(grad, i_indices * 3 + 1, 0.5 * w_pair * unit_diffs[:, 1])
        np.add.at(grad, i_indices * 3 + 2, 0.5 * w_pair * unit_diffs[:, 2])
        
        # Accumulate to j
        np.add.at(grad, j_indices * 3, -0.5 * w_pair * unit_diffs[:, 0])
        np.add.at(grad, j_indices * 3 + 1, -0.5 * w_pair * unit_diffs[:, 1])
        np.add.at(grad, j_indices * 3 + 2, -0.5 * w_pair * unit_diffs[:, 2])
        
        # We want to MINIMIZE -r_soft.
        # So return -r_soft and -grad
        r_soft = m - (1.0/beta) * np.log(S)
        return -r_soft, -grad

    def objective(x):
        return soft_min_radius(x, current_beta)[0]
        
    def jac(x):
        return soft_min_radius(x, current_beta)[1]

    # Optimization loop
    best_x = x0.copy()
    best_score = 0.0
    
    # Calculate initial score
    def calc_score(x):
        c = x.reshape(N, 3)
        # Check validity
        if not np.all(np.isfinite(c)):
            return 0.0
        if np.any(c < -0.5 - 1e-9) or np.any(c > 0.5 + 1e-9):
            return 0.0
            
        wall_dists = 0.5 - np.abs(c)
        min_wall = np.min(wall_dists)
        
        diffs = c[i_indices] - c[j_indices]
        pair_dists = 0.5 * np.linalg.norm(diffs, axis=1)
        min_pair = np.min(pair_dists)
        
        r = min(min_wall, min_pair)
        return r

    current_score = calc_score(best_x)
    best_score = current_score

    # Beta schedule
    # Start with low beta, increase
    betas = [10, 50, 200, 1000]
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
            
        current_beta = beta
        # Reset objective/jac to use current beta
        # scipy minimize passes x, so we just call the functions defined above which use global current_beta
        # Actually, it's better to define them inside the loop or use closures properly.
        
        def obj_curr(x):
            return soft_min_radius(x, beta)[0]
            
        def jac_curr(x):
            return soft_min_radius(x, beta)[1]

        try:
            res = minimize(obj_curr, best_x, jac=jac_curr, method='L-BFGS-B', 
                           bounds=[(-0.5, 0.5)] * (N*3),
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < -calc_score(best_x):
                # Check if new point is better
                new_score = calc_score(res.x)
                if new_score > best_score:
                    best_x = res.x
                    best_score = new_score
        except Exception:
            pass

    # Final polish with very high beta if time permits
    if time.time() < deadline - 10:
        beta = 5000
        def obj_final(x):
            return soft_min_radius(x, beta)[0]
        def jac_final(x):
            return soft_min_radius(x, beta)[1]
            
        try:
            res = minimize(obj_final, best_x, jac=jac_final, method='L-BFGS-B', 
                           bounds=[(-0.5, 0.5)] * (N*3),
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            new_score = calc_score(res.x)
            if new_score > best_score:
                best_x = res.x
                best_score = new_score
        except Exception:
            pass

    return best_x.reshape(N, 3)

# EVOLVE-BLOCK-END
