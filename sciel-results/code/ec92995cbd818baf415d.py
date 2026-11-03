"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    best_score = -1
    best_result = None

    # Perimeter 4 => W + H = 2
    # We will try a few aspect ratios. 
    # The optimal aspect ratio for 21 circles might not be exactly 1.0.
    # We will optimize positions and radii for a fixed W, H.
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.95, 1.05]
    
    for w_ratio in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        w = 2.0 * w_ratio / (1.0 + w_ratio)
        h = 2.0 - w
        
        if w <= 0 or h <= 0:
            continue
        
        # Initialize centers and radii
        # A good initialization is crucial for non-convex problems.
        # We can use a simple grid or hexagonal pattern scaled to fit.
        # Let's try a few random restarts and a structured start.
        
        best_local_score = -1
        best_local_result = None
        
        num_restarts = 5
        if w_ratio != 1.0:
            num_restarts = 2 # Fewer restarts for less promising aspects
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize positions
            if restart == 0:
                # Structured: Hexagonal-like grid
                # Try to fit a grid of circles
                # For 21 circles, maybe 6x4 grid minus some, or 5x5 minus 4?
                # Let's just scatter them roughly uniformly.
                # A simple way: divide rectangle into cells and place centers there.
                # 21 circles. 5 cols x 5 rows = 25 cells. Remove 4.
                # Or just random uniform.
                
                # Random uniform initialization
                centers = np.random.uniform(0.1, w-0.1, (N, 2))
                radii = np.full(N, 0.05) # Small initial radius
            else:
                # Random initialization with some noise
                centers = np.random.uniform(0.1, w-0.1, (N, 2))
                radii = np.full(N, 0.05)
            
            # Successive Convex Relaxation (SCR)
            # We want to maximize sum(r_i)
            # Constraints:
            # 1. x_i >= r_i, x_i <= w - r_i
            # 2. y_i >= r_i, y_i <= h - r_i
            # 3. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
            
            # Variable vector z = [x_1, y_1, r_1, x_2, y_2, r_2, ...]
            # Length 3N
            
            def pack_solution(z):
                xs = z[0::3]
                ys = z[1::3]
                rs = z[2::3]
                return xs, ys, rs
            
            def is_valid(z, tol=1e-6):
                xs, ys, rs = pack_solution(z)
                # Check bounds
                if np.any(xs < -tol) or np.any(xs > w + tol):
                    return False
                if np.any(ys < -tol) or np.any(ys > h + tol):
                    return False
                if np.any(rs < -tol):
                    return False
                # Check circle-circle
                for i in range(N):
                    if xs[i] < rs[i] - tol or xs[i] > w - rs[i] + tol:
                        return False
                    if ys[i] < rs[i] - tol or ys[i] > h - rs[i] + tol:
                        return False
                for i in range(N):
                    for j in range(i+1, N):
                        dist2 = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                        sum_r = rs[i] + rs[j]
                        if dist2 < sum_r**2 - tol:
                            return False
                return True
            
            def objective(z):
                # Maximize sum(r) => Minimize -sum(r)
                rs = z[2::3]
                return -np.sum(rs)
            
            # Constraints for SCR subproblem
            # We linearize the non-convex constraint:
            # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
            # At current point z_k, let d_ij = x_i - x_j, e_ij = y_i - y_j, s_ij = r_i + r_j
            # The tangent plane to the feasible region (exterior of circle) is:
            # 2*d_ij*(x_i - x_j) + 2*e_ij*(y_i - y_j) - 2*s_ij*(r_i + r_j) >= (d_ij^2 + e_ij^2 - s_ij^2)
            # Wait, the constraint is g(z) = (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0
            # Gradient wrt z:
            # d/dx_i = 2(x_i-x_j), d/dx_j = -2(x_i-x_j)
            # d/dy_i = 2(y_i-y_j), d/dy_j = -2(y_i-y_j)
            # d/dr_i = -2(r_i+r_j), d/dr_j = -2(r_i+r_j)
            # Linear approx: g(z_k) + grad_g(z_k) . (z - z_k) >= 0
            # g(z_k) + grad . z >= grad . z_k
            # Note: if circles are already overlapping or touching, g(z_k) <= 0.
            # To ensure we move into feasible region, we might need to handle the case where g(z_k) < 0 carefully.
            # Standard SCR handles this by just using the linear cut.
            
            # However, solving 21 circles with ~210 quadratic constraints linearized is a QP.
            # scipy.optimize.minimize with 'SLSQP' or 'trust-constr' can handle linear constraints.
            # But building the constraint list every iteration is expensive.
            
            # Alternative: Penalty Method with SLSQP
            # This is often more robust for coding simplicity in short timeframes.
            # Let's try a penalty method approach where we minimize:
            # -sum(r) + M * sum( max(0, (r_i+r_j)^2 - dist^2 )^2 ) + M * sum( max(0, r_i - x_i)^2 + ... )
            
            def penalized_objective(z, M=100.0):
                xs, ys, rs = pack_solution(z)
                obj = -np.sum(rs)
                
                # Penalty for bounds
                # x_i >= r_i => max(0, r_i - x_i)
                # x_i <= w - r_i => max(0, x_i - (w - r_i))
                # Same for y
                
                p_x_low = np.maximum(0, rs - xs)
                p_x_high = np.maximum(0, xs - (w - rs))
                p_y_low = np.maximum(0, rs - ys)
                p_y_high = np.maximum(0, ys - (h - rs))
                
                penalty_bounds = np.sum(p_x_low**2) + np.sum(p_x_high**2) + np.sum(p_y_low**2) + np.sum(p_y_high**2)
                
                # Penalty for overlap
                # max(0, (r_i+r_j)^2 - dist^2)
                overlap_pen = 0.0
                for i in range(N):
                    for j in range(i+1, N):
                        dist2 = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                        sum_r = rs[i] + rs[j]
                        viol = sum_r**2 - dist2
                        if viol > 0:
                            overlap_pen += viol**2
                
                return obj + M * (penalty_bounds + overlap_pen)
            
            # We will use SLSQP with the penalty objective.
            # We need to be careful with M. Too small -> infeasible. Too large -> ill-conditioned.
            # We can ramp up M.
            
            current_z = np.concatenate([centers.ravel(), radii.ravel()])
            
            # Simple SLSQP loop with increasing penalty
            M = 10.0
            for k in range(20):
                if time.time() > deadline - 2:
                    break
                    
                res = minimize(
                    penalized_objective,
                    current_z,
                    args=(M,),
                    method='SLSQP',
                    options={'maxiter': 50, 'ftol': 1e-9}
                )
                
                if res.success:
                    current_z = res.x
                    # Check if feasible with current M
                    if is_valid(current_z, tol=1e-5):
                        score = -objective(current_z)
                        if score > best_local_score:
                            best_local_score = score
                            best_local_result = current_z.copy()
                        
                        # If feasible, we can stop increasing M or just keep refining
                        # Actually, if feasible, we want to maximize sum(r) subject to constraints.
                        # The penalty method converges to a solution of the constrained problem as M->inf.
                        # If we are already feasible, we should just try to improve the objective 
                        # using standard constrained optimization?
                        # SLSQP with penalty is a heuristic.
                        # Let's just increase M slightly to tighten constraints if any violation remains.
                        if np.any(np.array([penalized_objective(current_z, 0.0) - objective(current_z)] > 1e-6)):
                             # There is penalty contribution, so not fully feasible or M too low
                             M *= 2.0
                        else:
                             # Feasible and penalty is 0. 
                             # We can try to do a few more iterations with high M to polish.
                             M *= 1.5
                    else:
                        # Not feasible, increase penalty
                        M *= 2.0
                else:
                    break
            
            # Final check on best local result
            if best_local_result is not None:
                if best_local_score > best_score:
                    best_score = best_local_score
                    best_result = best_local_result.copy()
        
        # End restarts for this aspect ratio

    # Fallback: simple grid if optimization failed
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
        
    # Construct final output array (N, 3)
    # best_result is z = [x1, y1, r1, x2, y2, r2, ...]
    if best_result is not None:
        xs = best_result[0::3]
        ys = best_result[1::3]
        rs = best_result[2::3]
        final_array = np.column_stack((xs, ys, rs))
        return final_array
    else:
        # Should not happen due to fallback
        return np.zeros((N, 3))

# EVOLVE-BLOCK-END
