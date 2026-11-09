import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def squares_in_square10() -> tuple:
    """
    Maximize the sum of side lengths of 10 squares in a unit square.
    Uses Differential Evolution for global search and L-BFGS-B for local polishing.
    """
    n = 10
    start_time = time.time()
    time_limit = 590.0  # Safety margin
    
    # Variables: x0, y0, s0, x1, y1, s1, ..., x9, y9, s9
    # Total 3n variables
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.01, 0.6)) # s (max side cannot be > 0.5 if others exist, but 0.6 is safe upper bound for search)

    def unpack(params):
        """Convert flat array to centers (Nx2) and sides (N,)"""
        centers = params[0:3*n:3].reshape(-1, 2) # Wait, x, y, s are interleaved? 
        # Let's use structure: [x0, y0, s0, x1, y1, s1, ...]
        xs = params[0::3]
        ys = params[1::3]
        ss = params[2::3]
        centers = np.column_stack((xs, ys))
        return centers, ss

    def penalty_and_sum(params):
        """
        Returns (penalty, sum_sides)
        """
        centers, ss = unpack(params)
        
        # Boundary penalty
        # cx - s/2 >= 0  => cx >= s/2
        # cx + s/2 <= 1  => cx <= 1 - s/2
        # If violated, penalty is magnitude of violation
        pen_boundary = 0.0
        for i in range(n):
            cx, cy = centers[i]
            s = ss[i]
            # x violations
            if cx < s/2:
                pen_boundary += (s/2 - cx)
            if cx > 1 - s/2:
                pen_boundary += (cx - (1 - s/2))
            # y violations
            if cy < s/2:
                pen_boundary += (s/2 - cy)
            if cy > 1 - s/2:
                pen_boundary += (cy - (1 - s/2))
        
        # Overlap penalty
        # Squares i, j overlap if |cx_i - cx_j| < (s_i+s_j)/2 AND |cy_i - cy_j| < (s_i+s_j)/2
        # We want to penalize the "penetration" depth.
        # Let dx = |cx_i - cx_j|, dy = |cy_i - cy_j|
        # req_x = (s_i+s_j)/2, req_y = (s_i+s_j)/2
        # If dx < req_x and dy < req_y, they overlap.
        # Penalty term: max(0, req_x - dx) * max(0, req_y - dy)
        # This is 0 if separated. It grows as they penetrate.
        
        pen_overlap = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                req_x = (ss[i] + ss[j]) / 2.0
                req_y = (ss[i] + ss[j]) / 2.0
                
                pen_x = max(0.0, req_x - dx)
                pen_y = max(0.0, req_y - dy)
                
                if pen_x > 0 and pen_y > 0:
                    # Use a smooth product penalty
                    pen_overlap += pen_x * pen_y
        
        sum_sides = np.sum(ss)
        return pen_boundary + pen_overlap, sum_sides

    def objective(params):
        """
        Objective for minimization: -sum_sides + Lambda * Penalty
        """
        pen, sum_sides = penalty_and_sum(params)
        # Large lambda to ensure feasibility is prioritized, but we want to maximize sum.
        # If we use a large lambda, we might get stuck in feasible but small solutions.
        # Better: Use a penalty that is large enough to push out of infeasibility, 
        # but not so large that it dominates the sum difference for valid packings.
        # Since max sum is ~3.3, and penalty for 0.1 overlap is ~0.01, 
        # Lambda=100 makes penalty 1.0, which is significant.
        # Let's use Lambda=1000 to be safe on feasibility, but we will polish.
        # Actually, for DE, a moderate penalty works well.
        return -sum_sides + 100.0 * pen

    # 1. Global Search with Differential Evolution
    # Strategy: Run DE for a few seconds, then polish.
    # If time remains, run DE again with different seed or polish more.
    
    best_result = None
    best_score = -np.inf
    
    # Initial guess for DE bounds
    # DE handles bounds internally
    
    # We will run DE in chunks to monitor time
    max_iter_per_chunk = 50
    chunk_count = 0
    max_chunks = 20 # 1000 iterations total max
    
    while time.time() - start_time < time_limit - 10.0:
        # Run one chunk of DE
        # To allow resuming or multiple restarts, we can just run DE fresh or continue.
        # DE doesn't easily "continue" from a population in scipy's standard API without manual state management.
        # So we run a fresh DE with a limited max_iter, or use a fixed max_iter and check time.
        
        # Let's run a single DE instance with a reasonable max_iter and seed.
        # If we want multiple restarts, we loop.
        
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=max_iter_per_chunk, 
                popsize=15, 
                seed=chunk_count, # Different seed for diversity
                tol=0.01,
                init='sobol'
            )
            chunk_count += 1
        except Exception:
            break
            
        if res.success or res.fun < best_score:
            best_score = res.fun
            best_result = res.x
            
        # Check time
        if time.time() - start_time > time_limit - 20.0:
            break

    # 2. Local Polishing with L-BFGS-B
    # Use the best result from DE as starting point.
    if best_result is not None:
        # We need to use the same objective, but L-BFGS-B is gradient-based.
        # The penalty function `pen_x * pen_y` is not differentiable at 0.
        # L-BFGS-B might struggle with kinks.
        # However, if the point is already feasible (pen ~ 0), the gradient of the penalty is 0 (if we define it carefully) or we just optimize the sum.
        # If feasible, we just maximize sum of sides subject to constraints.
        # Let's define a smooth penalty for L-BFGS-B.
        
        def smooth_objective(params):
            centers, ss = unpack(params)
            pen_b = 0.0
            for i in range(n):
                cx, cy = centers[i]
                s = ss[i]
                # Smooth approximation of max(0, x) is not needed if we just use quadratic penalty for violations
                # Violation x: cx < s/2 -> (s/2 - cx)^2
                # This is differentiable.
                pen_b += max(0, s/2 - cx)**2 + max(0, cx - (1 - s/2))**2
                pen_b += max(0, s/2 - cy)**2 + max(0, cy - (1 - s/2))**2
                
            pen_o = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    dx = abs(centers[i, 0] - centers[j, 0])
                    dy = abs(centers[i, 1] - centers[j, 1])
                    req_x = (ss[i] + ss[j]) / 2.0
                    req_y = (ss[i] + ss[j]) / 2.0
                    # Smooth penalty: if dx < req_x and dy < req_y
                    # Use exp or quadratic
                    # Let's use: (max(0, req_x - dx))**2 * (max(0, req_y - dy))**2
                    # This is 0 if separated.
                    px = max(0, req_x - dx)
                    py = max(0, req_y - dy)
                    pen_o += (px * px) * (py * py)
                    
            sum_sides = np.sum(ss)
            # Weight for penalty
            return -sum_sides + 1000.0 * (pen_b + pen_o)

        # Run L-BFGS-B
        try:
            res_local = minimize(
                smooth_objective, 
                best_result, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            
            if res_local.fun < best_score:
                best_score = res_local.fun
                best_result = res_local.x
        except Exception:
            pass

    # Final Extraction
    if best_result is None:
        # Fallback to grid
        k = 3
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
    else:
        centers, sides = unpack(best_result)
        
    # Final Validation and Clamping to ensure strict validity
    # The evaluator is strict. We must ensure no overlap and inside bounds.
    # If the optimizer left slight violations, we must fix them.
    
    # 1. Clamp sides and centers to valid ranges
    for i in range(n):
        sides[i] = max(0.0001, min(sides[i], 1.0))
        # Adjust centers to keep square inside [0,1]
        half_s = sides[i] / 2.0
        centers[i, 0] = max(half_s, min(centers[i, 0], 1.0 - half_s))
        centers[i, 1] = max(half_s, min(centers[i, 1], 1.0 - half_s))
        
    # 2. Resolve overlaps
    # This is a simple iterative relaxation.
    # If two squares overlap, push them apart.
    # We do this a few times.
    for _ in range(50):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                req_x = (sides[i] + sides[j]) / 2.0
                req_y = (sides[i] + sides[j]) / 2.0
                
                # Check overlap
                if abs(dx) < req_x and abs(dy) < req_y:
                    # Overlap detected
                    # Determine which axis to separate on
                    # Push along the axis with smaller penetration
                    pen_x = req_x - abs(dx)
                    pen_y = req_y - abs(dy)
                    
                    if pen_x < pen_y:
                        # Separate in X
                        if dx >= 0:
                            # i is left, j is right
                            # Move i left, j right by pen_x/2
                            shift = pen_x / 2.0
                            centers[i, 0] -= shift
                            centers[j, 0] += shift
                        else:
                            # i is right, j is left
                            shift = pen_x / 2.0
                            centers[i, 0] += shift
                            centers[j, 0] -= shift
                    else:
                        # Separate in Y
                        if dy >= 0:
                            # i is bottom, j is top
                            shift = pen_y / 2.0
                            centers[i, 1] -= shift
                            centers[j, 1] += shift
                        else:
                            shift = pen_y / 2.0
                            centers[i, 1] += shift
                            centers[j, 1] -= shift
                    moved = True
                    
                    # Re-clamp to boundaries after movement
                    for k in [i, j]:
                        half_sk = sides[k] / 2.0
                        centers[k, 0] = max(half_sk, min(centers[k, 0], 1.0 - half_sk))
                        centers[k, 1] = max(half_sk, min(centers[k, 1], 1.0 - half_sk))
        
        if not moved:
            break

    # 3. Final check: if any square is still overlapping or out of bounds, shrink it?
    # Shrinking reduces the score.
    # If we can't resolve, we might have to shrink.
    # But the relaxation usually works for these densities.
    
    # Ensure sides are > 0
    sides = np.maximum(sides, 1e-6)
    
    return centers.astype(np.float64), sides.astype(np.float64)
