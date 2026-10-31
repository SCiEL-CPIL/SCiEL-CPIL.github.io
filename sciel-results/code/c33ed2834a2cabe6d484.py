"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit (and 150s kill)
    
    # Target: maximize sum of radii for 21 circles in rectangle with perimeter 4.
    # Let rectangle width be w, height h.
    # Constraint: 2*w + 2*h = 4  =>  w + h = 2.
    # So h = 2 - w.
    # We can parameterize by w in (0, 2).
    # For a given w, h is fixed.
    # We need to place 21 circles (x_i, y_i, r_i) such that:
    #   0 <= x_i - r_i
    #   x_i + r_i <= w
    #   0 <= y_i - r_i
    #   y_i + r_i <= h
    #   (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i != j
    # Maximize sum(r_i)
    
    # Since the problem is symmetric in w and h (just swap x and y), we can assume w >= 1 (since w+h=2, w>=1 implies w>=h).
    # Actually, the optimal aspect ratio is likely close to 1 (square) or slightly rectangular.
    # Let's try a few aspect ratios or optimize w as well?
    # Optimizing w adds a dimension. Let's first fix w=1 (square 1x1) and see.
    # Perimeter 4 => 2(w+h)=4 => w+h=2. If w=1, h=1. Area=1.
    # If w=1.5, h=0.5. Area=0.75.
    # If w=1.2, h=0.8. Area=0.96.
    # Generally, for packing circles, square is often good, but for specific N, rectangular might be better.
    # However, optimizing w is a 1D search. We can do a coarse search over w.
    
    best_val = -np.inf
    best_sol = None
    
    # Define the optimization problem for a fixed w
    def solve_for_width(w):
        h = 2.0 - w
        if h <= 0 or w <= 0:
            return -np.inf, None
        
        # Variables: x1...x21, y1...y21, r1...r21
        # Total 63 variables.
        
        # Initial guess: hexagonal packing in a 1x1 square scaled to w,h?
        # Or just grid.
        # Let's use a grid initialization.
        # 21 circles. 5x5 grid has 25. 4x5=20.
        # Let's use 5x5 grid, take first 21.
        side = 5
        xs = np.linspace(0.1, w-0.1, side)
        ys = np.linspace(0.1, h-0.1, side)
        X, Y = np.meshgrid(xs, ys)
        X = X.flatten()
        Y = Y.flatten()
        
        # Initial radii: small
        r0 = 0.05
        r = np.full(N, r0)
        
        x0 = np.concatenate([X[:N], Y[:N], r])
        
        def objective(v):
            # v: [x1..x21, y1..y21, r1..r21]
            x = v[0:N]
            y = v[N:2*N]
            r = v[2*N:3*N]
            
            # Constraints (penalty method)
            penalty = 0.0
            
            # Wall constraints
            # x_i - r_i >= 0  =>  r_i - x_i <= 0
            # x_i + r_i <= w  =>  x_i + r_i - w <= 0
            # y_i - r_i >= 0
            # y_i + r_i <= h
            
            # Let's use soft penalties for inequality constraints g(v) <= 0
            # g1 = r - x
            # g2 = x + r - w
            # g3 = r - y
            # g4 = y + r - h
            # g5 = (r_i + r_j)^2 - (x_i - x_j)^2 - (y_i - y_j)^2 <= 0  (non-overlap)
            
            # It's often better to minimize -sum(r) + lambda * sum(max(0, g)^2)
            
            # Wall penalties
            g1 = r - x
            g2 = x + r - w
            g3 = r - y
            g4 = y + r - h
            
            pen_walls = np.sum(np.maximum(g1, 0)**2) + np.sum(np.maximum(g2, 0)**2) + \
                        np.sum(np.maximum(g3, 0)**2) + np.sum(np.maximum(g4, 0)**2)
            
            # Non-overlap penalties
            # This is O(N^2). N=21, so 21*20/2 = 210 pairs.
            pen_overlap = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx*dx + dy*dy
                    min_dist = r[i] + r[j]
                    # Constraint: dist_sq >= min_dist^2
                    # g = min_dist^2 - dist_sq <= 0
                    g = min_dist*min_dist - dist_sq
                    if g > 0:
                        pen_overlap += g*g
            
            return -np.sum(r) + 10.0 * (pen_walls + pen_overlap)
        
        def objective_grad(v):
            # Numerical gradient is slow. Let's try to derive or use finite diff if needed.
            # Given time constraints, let's rely on L-BFGS-B with numerical gradient first?
            # Or use SLSQP?
            # Let's stick with penalty method and L-BFGS-B with numerical gradient for simplicity, 
            # but check speed.
            # Actually, let's just return the objective and let scipy compute gradient?
            # No, let's try to implement analytical gradient for walls, and numerical for overlap?
            # Or just use finite difference for all.
            # With 63 variables, finite difference is 64 evaluations per step.
            # L-BFGS-B might take 100-1000 steps.
            # 1000 * 64 * 210 ops ~ 13 million ops. Should be fast enough in numpy?
            # Let's try.
            return None # Let scipy do it? No, it's better to provide gradient if possible.
            # For now, let's use a simple finite difference approximation inside the optimizer?
            # scipy.optimize.minimize with method='L-BFGS-B' and jac=None uses finite differences.
        
        # Let's use L-BFGS-B
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-9})
            val = -res.fun + 10.0 * 0 # Wait, res.fun is the penalized objective.
            # We need to recover the actual sum of radii from the final x
            x = res.x[0:N]
            y = res.x[N:2*N]
            r = res.x[2*N:3*N]
            
            # Check validity
            valid = True
            for i in range(N):
                if r[i] < 0 or x[i] - r[i] < -1e-6 or x[i] + r[i] > w + 1e-6 or y[i] - r[i] < -1e-6 or y[i] + r[i] > h + 1e-6:
                    valid = False
                    break
            
            if valid:
                # Check overlaps
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < r[i] + r[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                return np.sum(r), res.x
            else:
                return -np.inf, None
        except Exception as e:
            return -np.inf, None

    # Try a range of widths
    # w from 0.8 to 1.2 (symmetric around 1)
    widths = np.linspace(0.8, 1.2, 9)
    
    for w in widths:
        if time.time() > deadline - 5:
            break
        val, sol = solve_for_width(w)
        if val > best_val:
            best_val = val
            best_sol = sol
    
    # If we found a solution, format it
    if best_sol is not None:
        x = best_sol[0:N]
        y = best_sol[N:2*N]
        r = best_sol[2*N:3*N]
        
        # Determine w and h from the best solution? 
        # We iterated over w. We need to know which w gave the best result.
        # Let's modify the loop to track best_w.
        
        # Re-run logic to track best_w
        best_val = -np.inf
        best_sol = None
        best_w = 1.0
        
        for w in widths:
            if time.time() > deadline - 5:
                break
            val, sol = solve_for_width(w)
            if val > best_val:
                best_val = val
                best_sol = sol
                best_w = w
        
        if best_sol is not None:
            x = best_sol[0:N]
            y = best_sol[N:2*N]
            r = best_sol[2*N:3*N]
            h = 2.0 - best_w
            
            # Construct the output array (n, 3)
            # The problem asks for circles in a variable rectangle of perimeter 4.
            # The evaluator likely checks if the circles fit in *some* rectangle of perimeter 4.
            # Or does it check the bounding box of the circles?
            # "bounding rectangle has perimeter at most 4"
            # So the evaluator will take the min/max x and y of the circles, compute the bounding box,
            # and check its perimeter.
            # If so, we should return the coordinates as is.
            # The evaluator will compute:
            # x_min = min(x - r), x_max = max(x + r)
            # y_min = min(y - r), y_max = max(y + r)
            # w_bbox = x_max - x_min
            # h_bbox = y_max - y_min
            # perim = 2 * (w_bbox + h_bbox)
            # If perim <= 4, it's valid.
            
            # So we just return x, y, r.
            
            result = np.column_stack((x, y, r))
            return result
    else:
        # Fallback: simple grid in 1x1 square
        side = 5
        xs = np.linspace(0.1, 0.9, side)
        ys = np.linspace(0.1, 0.9, side)
        X, Y = np.meshgrid(xs, ys)
        X = X.flatten()
        Y = Y.flatten()
        r = np.full(N, 0.09)
        result = np.column_stack((X[:N], Y[:N], r[:N]))
        return result

# EVOLVE-BLOCK-END
