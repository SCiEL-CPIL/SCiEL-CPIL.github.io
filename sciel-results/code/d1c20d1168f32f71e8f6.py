import numpy as np
import scipy.optimize as opt
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer
    
    # 1. Generate Initial Guess: Hexagonal Packing
    def get_hex_initial():
        # Target radius ~0.07, diameter ~0.14
        d = 0.14
        dy = d * np.sqrt(3) / 2
        
        points = []
        # Generate a large hexagonal grid centered at 0,0
        for j in range(-10, 10):
            y = j * dy
            # Offset even rows
            offset = d / 2 if j % 2 != 0 else 0
            for i in range(-10, 10):
                x = i * d + offset
                # Check if inside square [-0.5, 0.5]
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
        
        points = np.array(points)
        
        if len(points) < N:
            # Fallback to grid if hexagonal is too sparse
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            return grid[:N]
        
        # Select N points closest to the center to form a compact cluster
        dists = np.sum(points**2, axis=1)
        idx = np.argsort(dists)[:N]
        return points[idx]

    c0 = get_hex_initial()
    
    # Ensure c0 is valid and inside bounds
    # Clip to bounds just in case
    c0 = np.clip(c0, -0.5, 0.5)
    
    best_score = 0.0
    best_pos = c0
    
    # Helper to calculate score
    def calc_score(c):
        # c is (N, 2)
        # Check validity
        if not np.all(np.isfinite(c)):
            return 0.0
        if not np.all(c >= -0.5) or not np.all(c <= 0.5):
            return 0.0
            
        # Pairwise distances
        # Use broadcasting for efficiency
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Zero out diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Boundary distances
        # min(0.5 - |x|, 0.5 - |y|)
        bx = 0.5 - np.abs(c[:, 0])
        by = 0.5 - np.abs(c[:, 1])
        min_bound = np.min(np.minimum(bx, by))
        
        r = min(min_pair / 2.0, min_bound)
        return r

    # Check initial score
    s0 = calc_score(c0)
    if s0 > best_score:
        best_score = s0
        best_pos = c0.copy()
        
    # Objective function for optimization
    # Maximize soft min of pairwise and boundary distances
    # Minimize -log_sum_exp(-dist/tau)
    
    def objective_and_grad(c_flat, tau=0.05):
        c = c_flat.reshape(N, 2)
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e10) # Avoid div by 0 in grad
        
        # Boundary distances
        bx = 0.5 - np.abs(c[:, 0])
        by = 0.5 - np.abs(c[:, 1])
        db = np.minimum(bx, by)
        
        # Combine distances
        # We want to maximize min(all_dist)
        # Soft min approx: -log(sum(exp(-dist/tau)))
        # We minimize L = log(sum(exp(-dist/tau)))
        
        # Pairwise terms
        # Only upper triangle to avoid double counting, but for log-sum-exp it doesn't matter much
        # Just sum all i<j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        d_pair = dists[mask]
        
        # Boundary terms
        d_bound = db
        
        # Combine
        # L = log( sum(exp(-d_pair/tau)) + sum(exp(-d_bound/tau)) )
        # To avoid overflow/underflow, use log-sum-exp trick
        # max_val = max(-d_pair/tau, -d_bound/tau)
        # But we are summing exps.
        
        # Simpler: just sum exps. Values are small.
        terms_pair = np.exp(-d_pair / tau)
        terms_bound = np.exp(-d_bound / tau)
        
        S = np.sum(terms_pair) + np.sum(terms_bound)
        
        if S == 0:
            return 0.0, np.zeros_like(c_flat)
            
        L = np.log(S)
        
        # Gradient
        # dL/dc = (1/S) * sum( d(terms)/dc )
        # d(exp(-d/tau))/dc = exp(-d/tau) * (-1/tau) * dd/dc
        # So dL/dc = -(1/tau) * (1/S) * sum( w_k * dd_k/dc )
        # where w_k = exp(-d_k/tau)
        
        grad = np.zeros_like(c)
        
        # Pairwise gradients
        # d_ij = norm(c_i - c_j)
        # d(d_ij)/d(c_i) = (c_i - c_j) / d_ij
        # d(d_ij)/d(c_j) = (c_j - c_i) / d_ij
        
        # We need to accumulate gradients for all pairs.
        # This is O(N^2).
        # For N=50, 2500 pairs.
        # Vectorized approach:
        # w_pair = terms_pair.reshape(N, N) ? No, mask is used.
        # Let's create a full matrix of weights for pairwise
        w_pair_full = np.zeros((N, N))
        w_pair_full[mask] = terms_pair
        
        # Gradient contribution from pairwise:
        # For each i, sum over j != i of w_ij * (c_i - c_j) / d_ij
        # Note: d_ij is symmetric.
        
        # We can compute this using matrix operations.
        # Let D be the distance matrix.
        # Let W be the weight matrix (w_ij).
        # Grad_i = sum_j W_ij * (c_i - c_j) / D_ij
        
        # Avoid division by zero (diagonal)
        D_safe = np.where(D_safe == 0, 1e-10, dists)
        
        # Vectorized sum:
        # (W * (1/D_safe)) @ c? No.
        # Sum_j W_ij (c_i - c_j) / D_ij = c_i * Sum_j W_ij/D_ij - Sum_j W_ij/D_ij * c_j
        
        # Factor_ij = W_ij / D_ij
        Factor = w_pair_full / D_safe
        # Set diagonal to 0
        np.fill_diagonal(Factor, 0)
        
        # Sum_j Factor_ij * c_j
        # This is Factor @ c
        Sum_F_c = Factor @ c
        # Sum_j Factor_ij
        Sum_F = Factor.sum(axis=1)
        
        grad_pair = c * Sum_F[:, np.newaxis] - Sum_F_c
        
        # Boundary gradients
        # db = min(bx, by)
        # If bx < by, db = bx = 0.5 - |x|
        # ddb/dx = -sign(x)
        # If by < bx, db = by = 0.5 - |y|
        # ddb/dy = -sign(y)
        
        # w_bound = terms_bound
        # Grad_bound_i = w_bound_i * ddb_i
        
        grad_bound = np.zeros_like(c)
        mask_x = bx < by
        mask_y = ~mask_x
        
        # For x
        # d(0.5 - |x|)/dx = -sign(x)
        grad_bound[mask_x, 0] = terms_bound[mask_x] * (-np.sign(c[mask_x, 0]))
        
        # For y
        grad_bound[mask_y, 1] = terms_bound[mask_y] * (-np.sign(c[mask_y, 1]))
        
        # Total Gradient
        # L = log(S)
        # dL/dc = -(1/tau) * (1/S) * (grad_pair + grad_bound)
        
        grad_total = -(1.0 / tau) * (1.0 / S) * (grad_pair + grad_bound)
        
        return L, grad_total.flatten()

    # Optimization
    # Use multiple tau values
    taus = [0.1, 0.05, 0.01]
    
    current_c = c0.flatten()
    
    for tau in taus:
        if time.time() > deadline:
            break
            
        try:
            res = opt.minimize(
                objective_and_grad,
                current_c,
                args=(tau,),
                method='L-BFGS-B',
                jac=True, # We provided gradient
                bounds=[(-0.5, 0.5)] * (N * 2),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            current_c = res.x
        except Exception as e:
            # If optimization fails, keep previous
            pass
        
        # Check score
        c_test = current_c.reshape(N, 2)
        s_test = calc_score(c_test)
        if s_test > best_score:
            best_score = s_test
            best_pos = c_test.copy()
            
    # Final check and return
    # Ensure best_pos is valid
    if calc_score(best_pos) > 0:
        return best_pos
    else:
        # Return initial if valid
        if calc_score(c0) > 0:
            return c0
        else:
            # Fallback to grid
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            return grid[:N]
