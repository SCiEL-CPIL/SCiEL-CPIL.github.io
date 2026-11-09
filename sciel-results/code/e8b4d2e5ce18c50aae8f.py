import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Maximizes the total side length of 10 non-overlapping axis-aligned squares
    inside the unit square [0, 1]^2 using Multi-start Local Optimization 
    with structural initializations.
    
    Strategy:
    1. Define a smooth objective function: minimize ( -sum(sides) + penalty )
    2. Penalty terms for boundary violations and overlaps are smooth (quadratic).
    3. Use L-BFGS-B to refine configurations.
    4. Initializations are based on structural templates:
       - Uniform grid (fallback)
       - "1 Big + 9 Small" (One large square in a corner, 9 smaller in the rest)
       - "2 Big + 8 Small" (Two large squares, 8 smaller)
       - Random perturbations of these templates.
    """
    n = 10
    start_time = time.time()
    hard_deadline = start_time + 600.0 - 10.0  # 10s safety margin

    # Precompute indices for pair-wise checks
    i_idx, j_idx = np.triu_indices(n, k=1)

    def compute_penalty_and_objective(x):
        """
        x: array of shape (20,) -> [cx1, cy1, s1, cx2, cy2, s2, ...]
        Returns: scalar objective to minimize.
        Objective = -sum(sides) + lambda_boundary * boundary_viol + lambda_overlap * overlap_viol
        """
        centers = x[0::2]  # cx, cy
        sides = x[1::2]    # s
        
        # Ensure sides are positive (clamp for penalty calculation, but let optimizer handle)
        # For smoothness, we penalize s < 0 heavily.
        
        total_sides = np.sum(sides)
        
        # Boundary violations
        # x_min = cx - s/2, x_max = cx + s/2
        # Violation if x_min < 0 or x_max > 1
        # Smooth penalty: max(0, 0 - x_min)^2 + max(0, x_max - 1)^2
        x_min = centers[0::2] - sides / 2.0
        x_max = centers[0::2] + sides / 2.0
        y_min = centers[1::2] - sides / 2.0
        y_max = centers[1::2] + sides / 2.0
        
        # Vectorized smooth max using np.where is not differentiable at 0, but L-BFGS-B handles kinks okay.
        # Better: use np.maximum(0, val) ** 2
        bound_pen_x = np.sum(np.maximum(0, -x_min)**2) + np.sum(np.maximum(0, x_max - 1)**2)
        bound_pen_y = np.sum(np.maximum(0, -y_min)**2) + np.sum(np.maximum(0, y_max - 1)**2)
        boundary_viol = bound_pen_x + bound_pen_y
        
        # Overlap violations
        # Squares i, j overlap if |cx_i - cx_j| < (s_i + s_j)/2 AND |cy_i - cy_j| < (s_i + s_j)/2
        # Smooth penalty for overlap:
        # dx = |cx_i - cx_j|
        # dy = |cy_i - cy_j|
        # sep_x = (s_i + s_j)/2 - dx
        # sep_y = (s_i + s_j)/2 - dy
        # If sep_x > 0 and sep_y > 0, they overlap.
        # Penalty = max(0, sep_x) * max(0, sep_y)
        # This is smooth if we use max(0, val) directly? No, product of max is C0 continuous but not C1.
        # L-BFGS-B can handle this, but let's try a smoother approximation or just use it.
        # Actually, max(0, a)*max(0, b) is differentiable almost everywhere.
        
        cx_i = centers[i_idx, 0] if centers.ndim > 1 else centers[i_idx]
        # Wait, centers is 1D array [cx1, cy1, cx2, cy2...]? No, I extracted centers = x[0::2] which is [cx1, cy1, cx2...]?
        # No, x is [cx1, cy1, s1, cx2, cy2, s2...].
        # x[0::2] -> [cx1, cx2, cx3...]
        # x[1::2] -> [cy1, cy2, cy3...]
        # My previous extraction:
        # centers = x[0::2] -> This is WRONG. 
        # Let's re-extract properly.
        
        # x layout: [cx1, cy1, s1, cx2, cy2, s2, ...]
        # cx = x[0::3]
        # cy = x[1::3]
        # s = x[2::3]
        
        cx = x[0::3]
        cy = x[1::3]
        s = x[2::3]
        
        total_sides = np.sum(s)
        
        x_min = cx - s / 2.0
        x_max = cx + s / 2.0
        y_min = cy - s / 2.0
        y_max = cy + s / 2.0
        
        bound_pen_x = np.sum(np.maximum(0, -x_min)**2) + np.sum(np.maximum(0, x_max - 1)**2)
        bound_pen_y = np.sum(np.maximum(0, -y_min)**2) + np.sum(np.maximum(0, y_max - 1)**2)
        boundary_viol = bound_pen_x + bound_pen_y
        
        # Overlaps
        dx = np.abs(cx[i_idx] - cx[j_idx])
        dy = np.abs(cy[i_idx] - cy[j_idx])
        
        sep_x = (s[i_idx] + s[j_idx]) / 2.0 - dx
        sep_y = (s[i_idx] + s[j_idx]) / 2.0 - dy
        
        # Penalty is area of overlap if positive
        # max(0, sep_x) * max(0, sep_y)
        overlap_pen = np.sum(np.maximum(0, sep_x) * np.maximum(0, sep_y))
        
        # Weights
        # We want to maximize total_sides. 
        # Minimize: -total_sides + W_bound * boundary_viol + W_overlap * overlap_pen
        # If penalty is 0, objective is -total_sides.
        # If penalty is high, it dominates.
        
        W_bound = 1000.0
        W_overlap = 1000.0
        
        objective = -total_sides + W_bound * boundary_viol + W_overlap * overlap_pen
        
        return objective

    def is_feasible(centers, sides, tol=1e-7):
        """
        Strict feasibility check.
        centers: (10, 2)
        sides: (10,)
        """
        x_min = centers[:, 0] - sides / 2
        x_max = centers[:, 0] + sides / 2
        y_min = centers[:, 1] - sides / 2
        y_max = centers[:, 1] + sides / 2
        
        if np.any(x_min < -tol) or np.any(x_max > 1 + tol) or np.any(y_min < -tol) or np.any(y_max > 1 + tol):
            return False
        
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        
        dx = np.abs(cx[i_idx] - cx[j_idx])
        dy = np.abs(cy[i_idx] - cy[j_idx])
        
        # t = (s_i + s_j)/2 - tol
        # Overlap if dx < t AND dy < t
        # Valid if NOT (dx < t AND dy < t) => dx >= t OR dy >= t
        t_x = (s[i_idx] + s[j_idx]) / 2.0 - tol
        t_y = (s[i_idx] + s[j_idx]) / 2.0 - tol
        
        violations = (dx < t_x) & (dy < t_y)
        
        return not np.any(violations)

    def get_solution_from_x(x):
        """Convert flat array x to centers (10,2) and sides (10,)"""
        cx = x[0::3]
        cy = x[1::3]
        s = x[2::3]
        centers = np.column_stack((cx, cy))
        return centers, s

    def generate_initializations():
        """
        Generate a list of initial x vectors for multi-start.
        """
        inits = []
        
        # 1. Uniform Grid (3x3 + 1 extra, or 4x4 subset)
        # 3x3 grid of 1/3 squares
        # 9 squares of size 1/3. 10th square?
        # If we put 9 squares of size 1/3, they fill the square. No room for 10th.
        # So uniform grid of 10 squares must be smaller.
        # 4x4 grid: 10 squares of size 1/4 = 0.25. Sum = 2.5.
        # 3x3 grid of size 1/3 is impossible for 10 squares.
        # Let's try a 3x3 grid where 9 squares are size 1/3, and 1 square is tiny?
        # No, they must not overlap.
        
        # Let's use the "1 Big + 9 Small" template.
        # Big square: side 0.5, at (0.25, 0.25) -> [0, 0.5]x[0, 0.5]
        # Remaining space: L-shape.
        # Top strip: [0, 1]x[0.5, 1] (height 0.5, width 1)
        # Right strip: [0.5, 1]x[0, 0.5] (width 0.5, height 0.5)
        # We can fit 9 small squares?
        # In the top strip (1x0.5), we can fit 2 rows of 2.5? No.
        # Let's say small squares have side 0.25.
        # Top strip (1x0.5): 4 squares of 0.25 fit in 1x0.5? 
        # 0.25*4 = 1.0 width. 0.25*2 = 0.5 height. Yes, 8 squares fit in 1x0.5?
        # 4 cols x 2 rows = 8 squares.
        # Right strip (0.5x0.5): 1 square of 0.5? No, we have 1 big already.
        # We have 10 squares total. 1 big + 9 small.
        # If small = 0.25.
        # 8 smalls in top strip (1x0.5).
        # 1 small in right strip (0.5x0.5).
        # Total 1 + 9 = 10 squares.
        # Sum = 0.5 + 9*0.25 = 0.5 + 2.25 = 2.75.
        
        # Let's construct this x.
        x_template = np.zeros(30)
        # Square 0: Big
        x_template[0] = 0.25  # cx
        x_template[1] = 0.25  # cy
        x_template[2] = 0.5   # s
        
        # Squares 1-8: Top strip (y from 0.5 to 1.0)
        # y_center = 0.75 (mid of 0.5 to 1.0)
        # x_centers: 0.125, 0.375, 0.625, 0.875 (for 4 cols)
        # Rows: 2 rows? Height is 0.5. 0.25*2 = 0.5.
        # y_centers: 0.625, 0.875
        top_xs = [0.125, 0.375, 0.625, 0.875]
        top_ys = [0.625, 0.875]
        idx = 1
        for y in top_ys:
            for x in top_xs:
                x_template[3*idx] = x
                x_template[3*idx+1] = y
                x_template[3*idx+2] = 0.25
                idx += 1
                # idx goes 1 to 8
        
        # Square 9: Right strip (x from 0.5 to 1.0, y from 0 to 0.5)
        # Center (0.75, 0.25), side 0.5? No, side 0.25 to match others?
        # If side 0.25, it fits easily.
        x_template[3*9] = 0.75
        x_template[3*9+1] = 0.25
        x_template[3*9+2] = 0.25
        
        inits.append(x_template)
        
        # 2. Perturbed versions of the template
        rng = np.random.default_rng(42)
        for _ in range(20):
            x_pert = x_template + rng.normal(0, 0.05, 30)
            # Clip sides to be positive
            x_pert[2::3] = np.clip(x_pert[2::3], 0.01, 0.6)
            inits.append(x_pert)
            
        # 3. Random initialization
        for _ in range(10):
            x_rand = np.zeros(30)
            s_rand = np.random.uniform(0.1, 0.4, 10)
            x_rand[2::3] = s_rand
            x_rand[0::3] = np.random.uniform(s_rand/2, 1 - s_rand/2, 10)
            x_rand[1::3] = np.random.uniform(s_rand/2, 1 - s_rand/2, 10)
            inits.append(x_rand)
            
        return inits

    best_centers = None
    best_sides = None
    best_sum = -np.inf
    
    inits = generate_initializations()
    
    # Bounds for L-BFGS-B
    # cx, cy in [0, 1] (loose, penalty handles it)
    # s in [0.01, 0.6]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # cx
        bounds.append((0.0, 1.0)) # cy
        bounds.append((0.01, 0.6)) # s

    for x0 in inits:
        if time.time() > hard_deadline:
            break
            
        try:
            res = minimize(
                compute_penalty_and_objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            x_opt = res.x
            centers, sides = get_solution_from_x(x_opt)
            
            # Check feasibility
            if is_feasible(centers, sides):
                current_sum = np.sum(sides)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = centers.copy()
                    best_sides = sides.copy()
                    
        except Exception:
            continue

    # Fallback if no feasible solution found (should not happen with templates)
    if best_centers is None:
        # 4x4 grid
        k = 4
        side = 1.0 / k
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.column_stack((cx, cy))
        best_sides = np.full(10, side)
        best_sum = np.sum(best_sides)

    return best_centers, best_sides

# EVOLVE-BLOCK-END
