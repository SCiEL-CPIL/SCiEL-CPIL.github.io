"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from scipy.optimize import minimize
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit
    
    best_score = -1
    best_result = None
    
    # Perimeter 4 => W + H = 2
    # We will optimize over aspect ratios.
    # Square (1x1) is a strong baseline.
    # We'll try a range of aspect ratios, but weight time towards the best performers.
    
    aspect_ratios = [1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15]
    
    # Simulated Annealing Parameters
    # We want to maximize sum of radii.
    # We can fix radii and move centers, or move both.
    # Moving both is more complex. 
    # Strategy: 
    # 1. Start with a random valid configuration.
    # 2. Define a "energy" function: -sum(r) + penalty(violations).
    # 3. Use SA to minimize energy.
    # 4. Periodically try to increase radii if no collisions.
    
    # A more robust approach for circle packing:
    # Fix the aspect ratio.
    # Use a "soft" constraint method where overlapping circles repel each other.
    # We want to maximize sum of radii. 
    # Let's use a specific heuristic:
    # 1. Initialize circles with small radii in a grid.
    # 2. Iteratively:
    #    a. Calculate forces: repulsion between overlapping circles, attraction to walls? No, just repulsion.
    #    b. Actually, a common method is to maximize the minimum distance between circles, then scale.
    #    But we want to maximize SUM of radii.
    
    # Let's use a direct optimization approach with a penalty function and Nelder-Mead or similar derivative-free method, 
    # but with many more restarts and a smarter penalty.
    # The previous program used trust-constr which is slow to set up and solve.
    # Nelder-Mead is fast per iteration but can get stuck.
    # Let's use a hybrid: 
    # 1. Coarse search with Nelder-Mead on (cx, cy, r) with a penalty function.
    # 2. Fine polish with a local gradient-based method (e.g., scipy.optimize.minimize with BFGS) on the penalty-free region if possible, or just continued Nelder-Mead.
    
    # Actually, a very effective strategy for this specific problem (max sum of radii) is:
    # 1. Assume all circles are equal size? No, the problem allows different sizes.
    #    However, for maximizing sum of radii with a fixed perimeter, equal radii is often near-optimal or optimal due to symmetry/isoperimetric inequalities, 
    #    unless the aspect ratio forces some circles to be smaller.
    #    Let's test the hypothesis: Are unequal radii beneficial?
    #    In a square, 21 circles. If all equal, r = ?
    #    If we allow different radii, we might fit a few large ones and many small ones.
    #    But usually, for "max sum of radii", making them as large as possible is key.
    #    Let's allow different radii in the optimization.
    
    def _penalty_function(x, W, H):
        """
        x: array of shape (3N,) -> [cx, cy, r]
        Returns penalty value (0 if valid, >0 if invalid)
        """
        cx = x[:N]
        cy = x[N:2*N]
        r = x[2*N:]
        
        penalty = 0.0
        
        # Boundary violations
        # cx - r < 0
        pen_left = np.maximum(0, r - cx)
        # cx + r > W
        pen_right = np.maximum(0, cx + r - W)
        # cy - r < 0
        pen_bottom = np.maximum(0, r - cy)
        # cy + r > H
        pen_top = np.maximum(0, cy + r - H)
        
        penalty += np.sum(pen_left) + np.sum(pen_right) + np.sum(pen_bottom) + np.sum(pen_top)
        
        # Overlap violations
        # dist(i, j) < r_i + r_j
        # We want dist >= r_i + r_j
        # Violation = (r_i + r_j) - dist
        # To make it smooth and differentiable-ish for NM, we can square it or just sum the excess.
        # Sum of excess is non-smooth at 0. Squaring is smooth.
        # Let's use squared excess to encourage larger separation.
        
        for i in range(N):
            for j in range(i+1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = math.hypot(dx, dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    excess = min_dist - dist
                    penalty += excess * excess # Quadratic penalty
        
        return penalty
    
    def _score(x, W, H):
        """
        Returns sum of radii if valid, else -inf or very negative.
        """
        if _penalty_function(x, W, H) > 1e-8:
            return -1e10
        r = x[2*N:]
        return np.sum(r)
    
    # We will use a multi-start Nelder-Mead approach.
    # To speed up, we will vectorize the penalty calculation where possible, 
    # but N=21 is small, so Python loops in penalty might be okay if NM iterations are limited.
    # However, NM can take 1000s of iterations. 210 pairs * 1000 iters = 210k ops. That's fine.
    
    # Optimization for penalty:
    # Precompute pairs?
    # Let's just write it cleanly.
    
    def _fast_penalty(x, W, H):
        cx = x[:N]
        cy = x[N:2*N]
        r = x[2*N:]
        
        # Boundary
        pen = 0.0
        pen += np.sum(np.maximum(0, r - cx))
        pen += np.sum(np.maximum(0, cx + r - W))
        pen += np.sum(np.maximum(0, r - cy))
        pen += np.sum(np.maximum(0, cy + r - H))
        
        # Overlap
        # Vectorized distance calculation is harder with penalty logic, but let's try
        # or just use loops since N is small.
        # Actually, let's use numpy broadcasting for pairs?
        # cx[:, None] - cx[None, :]
        # This creates a 21x21 matrix.
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        
        r_sum = r[:, None] + r[None, :]
        
        # dist < r_sum
        # We only care about i < j
        # excess = np.maximum(0, r_sum - dist)
        # penalty = np.sum(excess**2)
        # But this counts each pair twice. So divide by 2.
        
        excess = np.maximum(0, r_sum - dist)
        # Mask out diagonal
        np.fill_diagonal(excess, 0)
        
        pen += 0.5 * np.sum(excess**2)
        
        return pen
    
    def _objective(x, W, H):
        """
        Objective for minimization: 
        We want to maximize sum(r) subject to penalty == 0.
        A common trick: Minimize -sum(r) + lambda * penalty
        But lambda needs to be large enough to enforce constraints.
        Alternatively, minimize penalty primarily, then maximize sum(r).
        
        Let's use a lexicographic approach or a large penalty.
        f(x) = -sum(r) + M * penalty(x)
        If M is large, it will find a valid config first.
        Then, if we reduce M or use a different phase, we can grow radii.
        
        Better: 
        Phase 1: Minimize penalty. Find any valid config.
        Phase 2: Maximize sum(r) starting from Phase 1 result.
        
        But NM doesn't handle phase changes well.
        
        Let's try: Minimize ( -sum(r) + 1000 * penalty )
        This should push towards valid configs with large radii.
        """
        r = x[2*N:]
        pen = _fast_penalty(x, W, H)
        return -np.sum(r) + 1000.0 * pen

    # Let's iterate over aspect ratios
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
            
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 / (1.0 + ar)
        
        # Multiple restarts
        for restart in range(5):
            if time.time() > deadline:
                break
                
            rng = np.random.RandomState(restart * 100 + int(ar*100))
            
            # Initial guess
            # Grid initialization is good to start
            # 5x5 grid = 25 cells. We take 21.
            # Or random.
            # Let's mix: 50% grid, 50% random
            
            if restart % 2 == 0:
                # Grid
                side = 5
                cells = []
                for i in range(side):
                    for j in range(side):
                        cx = (i + 0.5) / side * W
                        cy = (j + 0.5) / side * H
                        r = 0.4 / side * min(W, H) # Slightly smaller to allow movement
                        cells.append((cx, cy, r))
                # Shuffle and take 21
                rng.shuffle(cells)
                cells = cells[:N]
                x0 = np.array([c[0] for c in cells] + [c[1] for c in cells] + [c[2] for c in cells])
            else:
                # Random
                cx = rng.uniform(0.1, W-0.1, N)
                cy = rng.uniform(0.1, H-0.1, N)
                r = rng.uniform(0.05, 0.15, N) # Small radii
                x0 = np.concatenate([cx, cy, r])
            
            # Bounds
            # cx in [0, W], cy in [0, H], r in [0, min(W,H)/2]
            r_max = min(W, H) / 2.0
            bounds = [(0.0, W)] * N + [(0.0, H)] * N + [(0.0, r_max)] * N
            
            # Use Nelder-Mead
            # We need to handle bounds. NM doesn't support bounds directly in older scipy, 
            # but recent versions do? 
            # Actually, scipy.optimize.minimize with method='Nelder-Mead' does NOT support bounds.
            # We must ensure x0 is within bounds and hope it stays there, 
            # or use a different method.
            # 'Powell' supports bounds? No, 'Powell' doesn't support bounds in standard scipy minimize either (only 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'trust-ncg', etc.).
            # 'COBYLA' supports constraints but not bounds directly (can add as constraints).
            # 'SLSQP' supports bounds and constraints.
            
            # Let's use SLSQP with the penalty method? 
            # SLSQP is a gradient-based method. It requires derivatives.
            # We can provide analytical gradients for the penalty function?
            # Or use numerical gradients.
            
            # Alternative: Use 'L-BFGS-B' which supports bounds.
            # But L-BFGS-B minimizes a smooth function. 
            # Our objective -sum(r) + 1000*penalty is not smooth at the boundary of validity (penalty is C1, but the switch might cause issues? No, penalty is defined everywhere).
            # penalty = sum(max(0, violation)^2). This is C1. 
            # So L-BFGS-B should work.
            
            # Let's try L-BFGS-B.
            
            def obj_func(x):
                return _objective(x, W, H)
                
            # We need a gradient for L-BFGS-B to be efficient.
            # Numerical gradient is slow.
            # Let's implement analytical gradient for _fast_penalty?
            # That's complex.
            
            # Let's try 'COBYLA' which is derivative-free and handles bounds via constraints?
            # No, COBYLA is for constraints.
            
            # Let's go back to 'Nelder-Mead' but manually enforce bounds by clipping x in the objective function?
            # If we clip x, the gradient info is lost/corrupted for NM? NM doesn't use gradients.
            # It uses simplex. If we clip, the simplex vertices might get stuck on the boundary.
            # This is a common workaround.
            
            def obj_func_clipped(x):
                # Clip to bounds
                cx = np.clip(x[:N], 0, W)
                cy = np.clip(x[N:2*N], 0, H)
                r = np.clip(x[2*N:], 0, r_max)
                x_clipped = np.concatenate([cx, cy, r])
                return _objective(x_clipped, W, H)
            
            # NM options
            # maxiter: 2000
            # xatol, fatol: 1e-6
            
            try:
                res = minimize(
                    obj_func_clipped,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                )
                
                x_opt = res.x
                # Unclip? No, x_opt might be outside bounds if NM pushed it there.
                # We must use the clipped version for evaluation.
                cx_opt = np.clip(x_opt[:N], 0, W)
                cy_opt = np.clip(x_opt[N:2*N], 0, H)
                r_opt = np.clip(x_opt[2*N:], 0, r_max)
                
                # Validate
                valid = True
                if np.any(cx_opt - r_opt < -1e-9) or np.any(cx_opt + r_opt - W > 1e-9):
                    valid = False
                if np.any(cy_opt - r_opt < -1e-9) or np.any(cy_opt + r_opt - H > 1e-9):
                    valid = False
                
                if valid:
                    # Check overlaps
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = math.hypot(cx_opt[i] - cx_opt[j], cy_opt[i] - cy_opt[j])
                            if dist < r_opt[i] + r_opt[j] - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = np.sum(r_opt)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack((cx_opt, cy_opt, r_opt))
                        
            except Exception:
                pass

    # Fallback
    if best_result is None:
        # 5x5 grid in 1x1 square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append((cx, cy, r))
        best_result = np.array(cells[:N])
        
    return best_result
# EVOLVE-BLOCK-END
