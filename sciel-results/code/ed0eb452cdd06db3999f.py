"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize
import math

N = 200

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Initialization: Jittered grid
    # A square grid of 14x14 = 196 is close to 200. 
    # 15x15 = 225. We can use 15x15 and remove 25, or just use a 14x14 grid + 4 extra.
    # Better: Use a 14x14 grid (196) and add 4 points in the gaps? 
    # Or just use a 15x15 grid and drop 25 random points?
    # Let's stick to a 14x14 grid (196) and add 4 points in the corners/gaps?
    # Actually, 200 points in 14x14 grid cells (size 1/14 ~ 0.071).
    # If we use 14x14, we have 4 spots left.
    # Let's try a 15x15 grid and keep the first 200? No, that's biased.
    # Let's use a 14x14 grid and add 4 points at (0,0), (0, 1/14), etc?
    # Simpler: Generate 200 points on a 14x14 grid with slight jitter, 
    # but 14*14=196. We need 4 more.
    # Let's use a 15x15 grid (225 points) and randomly select 200? 
    # Random selection might break symmetry.
    
    # Let's use a 14x14 grid (196) and add 4 points in the center of the 4 corner quadrants?
    # Or just use 200 points on a grid of 15x15, skipping 25 in a pattern?
    # Let's just use a 14x14 grid and add 4 points at (0, 0), (0, 0.5), (0.5, 0), (0.5, 0.5) scaled? No.
    
    # Best simple init: 14x14 grid covers 196. Add 4 points at the midpoints of the outer edges?
    # Let's just use 200 points from a 15x15 grid, but arranged to be symmetric.
    # 15x15 = 225. Remove 25.
    # Let's try 14x14 = 196 + 4 extra.
    
    m = 14
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Add 4 points. Where?
    # The grid spacing is 1/14.
    # The corners of the square are at +/- 0.5.
    # The grid points are at +/- (0.5 - 1/28) = +/- 13/28.
    # Distance to wall is 1/28 ~ 0.0357.
    # We can add points closer to the corners?
    # Let's add points at (0, 0), (0, 1/m), (1/m, 0), (1/m, 1/m)?
    # This is getting complicated.
    
    # Alternative Init: Random uniform, then relax.
    # But random is slow to converge.
    
    # Let's use the 14x14 grid and add 4 points at the exact corners of the sub-cells?
    # Actually, let's just use 200 points from a 15x15 grid, removing 25 in a checkerboard pattern?
    # 15x15 = 225.
    # Let's just use a 14x14 grid and add 4 points at (0, 0), (0, 0.5), (0.5, 0), (0.5, 0.5) is wrong.
    
    # Let's try this: 14x14 grid (196). Add 4 points at the midpoints of the sides of the square?
    # (0, 0.5), (0, -0.5), (0.5, 0), (-0.5, 0).
    # These are on the boundary.
    
    points = np.zeros((N, 2))
    points[:196] = grid
    points[196] = [0.0, 0.5]
    points[197] = [0.0, -0.5]
    points[198] = [0.5, 0.0]
    points[199] = [-0.5, 0.0]
    
    # Add small jitter to break symmetry
    points += np.random.uniform(-0.01, 0.01, (N, 2))
    
    def soft_min_energy(pos, beta):
        # pos is (N, 2)
        # Compute pairwise distances
        # dist^2 = (x1-x2)^2 + (y1-y2)^2
        # We want min dist.
        # Surrogate: -1/beta * log(sum(exp(-beta * dist)))
        # To avoid overflow, we can use log-sum-exp trick, but with N=200, N^2=40000, it's manageable.
        
        # Pairwise distances
        # Use broadcasting
        # diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # dist = np.linalg.norm(diff, axis=2)
        
        # This is O(N^2) memory. 200*200*2 = 80k floats. Fine.
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        
        # Mask diagonal
        np.fill_diagonal(dist, np.inf)
        
        # Wall distances
        # dist to wall = min(0.5 - |x|, 0.5 - |y|)
        wall_dist = np.min(0.5 - np.abs(pos), axis=1)
        
        # Combine all distances
        # We want min(dist_pair, wall_dist)
        # Create a matrix of all distances?
        # Or just sum the exp terms.
        
        # Sum of exp(-beta * dist) for all pairs
        # And sum of exp(-beta * wall_dist) for all points
        
        # To prevent overflow in exp(-beta * dist), we can subtract max?
        # No, we want log(sum(exp(-beta * d))).
        # If d is small, -beta * d is large negative? No, if d is small, -beta * d is close to 0?
        # Wait. We want to maximize min(d).
        # Let f(d) = -1/beta * log(sum(exp(-beta * d))).
        # As beta -> inf, this approaches min(d).
        # If d is small, exp(-beta * d) is large?
        # If d=0.01, beta=100, exp(-1) = 0.36.
        # If d=0.001, beta=100, exp(-0.1) = 0.9.
        # The term with smallest d dominates the sum.
        # So log(sum) ~ log(exp(-beta * d_min)) = -beta * d_min.
        # Then -1/beta * log(sum) ~ d_min.
        # Correct.
        
        # We need to compute sum(exp(-beta * dist)) safely.
        # Max of -beta * dist is -beta * min(dist).
        # Let M = -beta * min(dist).
        # sum(exp(-beta * dist)) = exp(M) * sum(exp(-beta * dist - M))
        # log(sum) = M + log(sum(exp(...)))
        
        # Let's just use np.logaddexp.reduce if available, or manual.
        # N=200, 19900 pairs.
        
        # Pairwise
        # dist is (N, N)
        # We only want upper triangle to avoid double counting? 
        # The objective is min over all i<j.
        # Summing over all i != j is fine, it just weights them by 2.
        
        # Wall distances are (N,)
        
        # Let's flatten dist (excluding diag)
        d_pairs = dist[dist < np.inf]
        d_walls = wall_dist
        
        all_d = np.concatenate([d_pairs, d_walls])
        
        # Calculate log-sum-exp of -beta * all_d
        # val = -beta * all_d
        # max_val = np.max(val)
        # log_sum = max_val + np.log(np.sum(np.exp(val - max_val)))
        # energy = -1/beta * log_sum
        
        val = -beta * all_d
        max_val = np.max(val)
        log_sum = max_val + np.log(np.sum(np.exp(val - max_val)))
        energy = -log_sum / beta
        
        # Gradient
        # d(energy)/d(pos)
        # This is complex to derive analytically in code quickly.
        # Use numerical gradient? Slow.
        # Use scipy.optimize.minimize with numerical gradient? 
        # N=200, 400 vars. Numerical gradient is 400 evaluations per step.
        # 100 steps = 40000 evaluations.
        # Each eval is O(N^2) = 40000 ops.
        # 40000 * 40000 = 1.6e9 ops. Too slow in Python/NumPy?
        # NumPy is fast, but 1.6e9 might take seconds.
        # 120s limit.
        
        # Better: Analytical gradient or use a library that handles it?
        # Let's try to derive the gradient.
        
        # E = -1/beta * log( sum_k exp(-beta d_k) )
        # dE/dx_i = -1/beta * ( 1/S * sum_k exp(-beta d_k) * (-beta * dd_k/dx_i) )
        # dE/dx_i = 1/S * sum_k exp(-beta d_k) * dd_k/dx_i
        # where S = sum_k exp(-beta d_k)
        # Let w_k = exp(-beta d_k) / S. Sum w_k = 1.
        # dE/dx_i = sum_k w_k * dd_k/dx_i
        
        # dd_pair/dx_i for pair (i, j):
        # d_ij = |x_i - x_j|
        # d(d_ij)/dx_i = (x_i - x_j) / d_ij
        # d(d_ij)/dx_j = (x_j - x_i) / d_ij
        
        # dd_wall/dx_i for wall distance of point i:
        # d_wi = min(0.5 - |x_ix|, 0.5 - |x_iy|)
        # Gradient is +/- 1 depending on which wall is closer and sign of coord.
        
        grad = np.zeros_like(pos)
        
        # Weights for pairs
        # w_pair = exp(-beta * d_pairs) / S
        # But we need to map back to (N, N) matrix?
        # Or just accumulate.
        
        # Let's compute weights for all terms in all_d.
        # all_d length is N*(N-1) + N
        # We need to know which term corresponds to which point.
        
        # This is getting complicated to implement efficiently without pre-computing indices.
        
        # Alternative: Use a simpler objective.
        # Maximize sum of exp(-beta * d)? No, that's repulsion.
        # The soft-min is the correct one for max-min.
        
        # Let's use scipy.optimize.minimize with method='BFGS' and provide the function.
        # If we don't provide gradient, it uses numerical.
        # Let's check speed of numerical gradient.
        # 400 dimensions.
        # One function eval: O(N^2) ~ 40k ops.
        # 400 evals for gradient: 16M ops.
        # BFGS step: 16M ops.
        # 100 steps: 1.6B ops.
        # In NumPy, 1.6B simple ops might take ~1-2 seconds?
        # Actually, Python overhead per call is high.
        # 400 calls to Python function per gradient step.
        # 100 steps = 40,000 Python calls.
        # Each call does NumPy vectorized ops.
        # This should be feasible within 110s.
        
        return energy

    # We will use scipy.optimize.minimize
    # We need to handle the beta increase.
    
    best_pos = points
    best_score = -np.inf
    
    def calc_score(pos):
        # Raw min distance
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)
        
        wall_dist = np.min(0.5 - np.abs(pos), axis=1)
        min_wall = np.min(wall_dist)
        
        return min(min_pair, min_wall)

    # Optimization Loop
    betas = [10, 50, 100, 200, 500, 1000]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        def objective(pos):
            # Flatten pos
            p = pos.reshape(-1)
            # Call soft_min_energy
            # Note: soft_min_energy returns scalar
            return soft_min_energy(p.reshape(N, 2), beta)
        
        # Initial point for this beta is the result of the previous beta
        # Use BFGS
        # We don't provide gradient to save implementation time, rely on numerical.
        # But numerical gradient is slow.
        # Let's try to provide a simple gradient if possible?
        # No, let's stick to numerical first. If it's too slow, we'll see.
        
        try:
            res = scipy.optimize.minimize(objective, best_pos.reshape(-1), method='BFGS', 
                                          options={'maxiter': 50, 'disp': False})
            new_pos = res.x.reshape(N, 2)
            
            # Check validity
            if not np.all(np.isfinite(new_pos)):
                continue
                
            # Check if it's better
            new_score = calc_score(new_pos)
            if new_score > best_score:
                best_score = new_score
                best_pos = new_pos
                
        except Exception as e:
            pass

    # Final polish with higher beta and more iterations if time permits
    if time.time() < deadline - 5:
        beta = 2000
        def objective_final(pos):
            p = pos.reshape(-1)
            return soft_min_energy(p.reshape(N, 2), beta)
            
        try:
            res = scipy.optimize.minimize(objective_final, best_pos.reshape(-1), method='BFGS', 
                                          options={'maxiter': 100, 'disp': False})
            new_pos = res.x.reshape(N, 2)
            if np.all(np.isfinite(new_pos)):
                new_score = calc_score(new_pos)
                if new_score > best_score:
                    best_pos = new_pos
                    best_score = new_score
        except:
            pass

    return best_pos

# EVOLVE-BLOCK-END
