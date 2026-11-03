"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize, LinearConstraint

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Perimeter 4 => w + h = 2.
    # We parameterize the rectangle by aspect ratio ar = w/h.
    # h = 2 / (1 + ar), w = 2 * ar / (1 + ar).
    
    def get_wh(ar):
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        return w, h

    def is_valid(circles, w, h, tol=1e-5):
        """Check if circles are disjoint and inside [0, w] x [0, h]."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if x < r - tol or x > w - r + tol:
                return False
            if y < r - tol or y > h - r + tol:
                return False
            if r <= 0:
                return False
            for j in range(i+1, n):
                x2, y2, r2 = circles[j]
                dx = x - x2
                dy = y - y2
                dist_sq = dx*dx + dy*dy
                min_dist = r + r2
                if dist_sq < min_dist*min_dist - tol:
                    return False
        return True

    def solve_qp_subproblem(x0, y0, r0, w, h, max_iter=50):
        """
        Solve the SCR subproblem for fixed rectangle w,h.
        Variables: x_i, y_i, r_i for i=0..N-1
        Objective: Minimize -sum(r_i)
        Constraints:
          1. Boundary: r_i <= x_i <= w-r_i, r_i <= y_i <= h-r_i
          2. Non-overlap (linearized at current point):
             (x_i - x_j) * (x_i^k - x_j^k) + (y_i - y_j) * (y_i^k - y_j^k) >= (r_i + r_j) * ||d^k||
             where d^k is the vector from center j to center i at iteration k.
             If d^k is zero, use a random direction or skip? 
             Standard SCR uses the direction from the previous iterate.
             If they are coincident, we need a valid direction.
        """
        n = N
        dim = 3 * n
        
        # Current positions
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        
        best_val = -np.sum(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        for k in range(max_iter):
            if time.time() > deadline - 2:
                break
                
            # Build constraints
            # 1. Linear constraints for non-overlap
            A_rows = []
            b_rows = []
            
            for i in range(n):
                for j in range(i+1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx*dx + dy*dy
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-10 else 1.0
                    if dist < 1e-10:
                        # Coincident centers: pick arbitrary direction
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    
                    # Constraint: (x_i - x_j) * dx + (y_i - y_j) * dy >= (r_i + r_j) * dist
                    # Rewrite: dx*x_i - dx*x_j + dy*y_i - dy*y_j - dist*r_i - dist*r_j >= 0
                    # Form: A @ vars >= b
                    # We will use LinearConstraint(A, lb, ub)
                    # A row: [0...0, dx, 0...0, -dx, 0...0, -dist, 0...0, -dist, 0...]
                    row = np.zeros(dim)
                    row[i] = dx
                    row[j] = -dx
                    row[n + i] = dy
                    row[n + j] = -dy
                    row[2*n + i] = -dist
                    row[2*n + j] = -dist
                    
                    A_rows.append(row)
                    b_rows.append(0.0)
            
            # 2. Boundary constraints
            # r_i <= x_i  => x_i - r_i >= 0
            # x_i <= w - r_i => x_i + r_i <= w
            # r_i <= y_i => y_i - r_i >= 0
            # y_i <= h - r_i => y_i + r_i <= h
            # r_i >= 0 (handled by bounds)
            
            # Let's combine all linear constraints into one matrix
            # Inequalities: A_ineq @ z >= b_ineq
            # scipy LinearConstraint takes (lb, ub) for A @ z.
            # So we define A such that lb <= A @ z <= ub.
            
            # For non-overlap: A_nolap @ z >= 0  =>  0 <= A_nolap @ z <= inf
            # For boundaries:
            # x_i - r_i >= 0 => 0 <= (x_i - r_i) <= inf
            # x_i + r_i <= w => -inf <= (x_i + r_i) <= w
            # y_i - r_i >= 0 => 0 <= (y_i - r_i) <= inf
            # y_i + r_i <= h => -inf <= (y_i + r_i) <= h
            
            A_all = []
            lb_all = []
            ub_all = []
            
            # Non-overlap
            if A_rows:
                A_nolap = np.array(A_rows)
                A_all.append(A_nolap)
                lb_all.append(np.zeros(len(A_rows)))
                ub_all.append(np.full(len(A_rows), np.inf))
            
            # Boundary x
            for i in range(n):
                # x_i - r_i >= 0
                row1 = np.zeros(dim)
                row1[i] = 1.0
                row1[2*n + i] = -1.0
                A_all.append(row1)
                lb_all.append(0.0)
                ub_all.append(np.inf)
                
                # x_i + r_i <= w
                row2 = np.zeros(dim)
                row2[i] = 1.0
                row2[2*n + i] = 1.0
                A_all.append(row2)
                lb_all.append(-np.inf)
                ub_all.append(w)
            
            # Boundary y
            for i in range(n):
                # y_i - r_i >= 0
                row3 = np.zeros(dim)
                row3[n + i] = 1.0
                row3[2*n + i] = -1.0
                A_all.append(row3)
                lb_all.append(0.0)
                ub_all.append(np.inf)
                
                # y_i + r_i <= h
                row4 = np.zeros(dim)
                row4[n + i] = 1.0
                row4[2*n + i] = 1.0
                A_all.append(row4)
                lb_all.append(-np.inf)
                ub_all.append(h)
            
            A_matrix = np.vstack(A_all)
            lb_vector = np.concatenate(lb_all)
            ub_vector = np.concatenate(ub_all)
            
            constraints = [LinearConstraint(A_matrix, lb_vector, ub_vector)]
            
            # Bounds for variables
            # x_i in [0, w], y_i in [0, h], r_i in [0, min(w,h)/2]
            bounds = []
            for i in range(n):
                bounds.append((0.0, w))
            for i in range(n):
                bounds.append((0.0, h))
            for i in range(n):
                bounds.append((1e-6, min(w, h)/2))
            
            # Objective: -sum(r_i)
            c_vec = np.zeros(dim)
            for i in range(n):
                c_vec[2*n + i] = -1.0
            
            # Initial point for this subproblem iteration is the current x, y, r
            z0 = np.concatenate([x, y, r])
            
            # Solve QP
            try:
                res = minimize(
                    lambda z: c_vec @ z, # Linear objective
                    z0,
                    method='trust-constr',
                    jac=c_vec,
                    hess=np.zeros((dim, dim)), # Linear objective has 0 Hessian
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 100, 'gtol': 1e-8, 'xtol': 1e-8}
                )
                
                if res.success or res.status == 0:
                    z_new = res.x
                    x_new = z_new[:n]
                    y_new = z_new[n:2*n]
                    r_new = z_new[2*n:]
                    
                    # Check validity of new solution
                    if is_valid(np.column_stack([x_new, y_new, r_new]), w, h, tol=1e-4):
                        val = -np.sum(r_new)
                        if val < best_val:
                            best_val = val
                            best_x, best_y, best_r = x_new, y_new, r_new
                        x, y, r = x_new, y_new, r_new
                    else:
                        # If invalid, we might need to step back or just stop
                        # SCR should converge to valid if started valid and steps are small enough
                        # But trust-constr might jump. Let's check if we can use it.
                        # If it violates constraints significantly, we stop this subproblem loop
                        # and try to repair or accept previous.
                        pass
                else:
                    # Optimization failed, break loop
                    break
                    
            except Exception:
                break

        return best_x, best_y, best_r, -best_val

    best_circles = None
    best_ar = 1.0
    best_value = -np.inf
    
    # Multi-start strategy
    # Try different aspect ratios
    aspect_ratios = [0.8, 1.0, 1.2, 1.5, 2.0]
    
    for ar_init in aspect_ratios:
        if time.time() > deadline - 5:
            break
            
        w, h = get_wh(ar_init)
        
        # Initialize with a grid
        # Try to fit 3x7 or 4x5 or 5x5 depending on aspect ratio
        # For now, use a simple grid init
        init_circles = np.zeros((N, 3))
        
        # Determine grid dimensions
        if w > h:
            rows, cols = 3, 7
        else:
            rows, cols = 7, 3
            
        # Adjust for N=21
        # 3x7 = 21 exactly.
        # 4x5 = 20 (need 1 more? no, 21 circles. 3x7 is perfect.)
        # 5x5 = 25 (too many).
        # So 3x7 is the natural grid for 21.
        
        # If aspect ratio is very different, the grid cells will be rectangular.
        # That's fine, SCR will fix it.
        
        margin = 0.0
        # Center of cells
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx >= N:
                    break
                cx = (j + 0.5) * (w / cols)
                cy = (i + 0.5) * (h / rows)
                # Initial radius: small enough to not overlap in grid
                # Cell width w/cols, height h/rows
                # Max radius without overlap is 0.5 * min(w/cols, h/rows)
                # Start with 0.4 * that
                r_init = 0.4 * min(w/cols, h/rows)
                init_circles[idx] = [cx, cy, r_init]
        
        # Add small noise
        init_circles[:, :2] += np.random.normal(0, 0.001, size=(N, 2))
        init_circles[:, 2] = np.maximum(init_circles[:, 2], 1e-4)
        
        x0 = init_circles[:, 0]
        y0 = init_circles[:, 1]
        r0 = init_circles[:, 2]
        
        # Check if initial is valid. If not, shrink radii until valid.
        # Actually, SCR requires a feasible starting point for the QP?
        # No, QP solvers can handle infeasible starts, but SCR theory assumes feasibility.
        # Let's ensure start is valid by shrinking radii.
        scale = 1.0
        while not is_valid(init_circles, w, h, tol=1e-6):
            scale *= 0.9
            init_circles[:, 2] = r0 * scale
            if scale < 1e-4:
                break
        
        # Run SCR
        x_opt, y_opt, r_opt, val = solve_qp_subproblem(x0, y0, init_circles[:, 2], w, h)
        
        if val > best_value:
            best_value = val
            best_ar = ar_init
            best_circles = np.column_stack([x_opt, y_opt, r_opt])
            
    if best_circles is not None:
        return best_circles
    else:
        # Fallback: Return a valid small grid
        # 3x7 grid in 1.4 x 0.6
        w, h = 1.4, 0.6
        fallback = np.zeros((N, 3))
        idx = 0
        for i in range(3):
            for j in range(7):
                cx = (j + 0.5) * (w / 7)
                cy = (i + 0.5) * (h / 3)
                r = 0.4 * min(w/7, h/3)
                fallback[idx] = [cx, cy, r]
                idx += 1
        return fallback

# EVOLVE-BLOCK-END
