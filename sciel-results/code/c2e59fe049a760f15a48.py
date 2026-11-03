"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    N = 21
    best_value = -np.inf
    best_result = None

    # Helper to check validity strictly
    def is_valid(circles, w, h):
        if circles.shape != (N, 3):
            return False
        # Check bounds
        if np.any(circles[:, 0] < -1e-8) or np.any(circles[:, 0] > w + 1e-8):
            return False
        if np.any(circles[:, 1] < -1e-8) or np.any(circles[:, 1] > h + 1e-8):
            return False
        if np.any(circles[:, 2] <= 0):
            return False
            
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < circles[i, 2] + circles[j, 2] - 1e-8:
                    return False
        return True

    # Successive Convex Relaxation Solver
    # Variables: x = [x_1..x_N, y_1..y_N, r_1..r_N]
    # Objective: Minimize -sum(r)
    # Constraints:
    # 1. x_i >= r_i, x_i <= w - r_i
    # 2. y_i >= r_i, y_i <= h - r_i
    # 3. r_i > 0
    # 4. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    #    Linearized at current iterate (xk, yk, rk):
    #    (x_i - x_j) * (xk_i - xk_j) + (y_i - y_j) * (yk_i - yk_j) >= rk_i + rk_j
    #    (This is a tangent plane approximation to the convex set of non-overlapping circles)
    
    def solve_scr(w, h, x0, max_iters=50, tol=1e-6):
        x = x0.copy()
        
        # Bounds for SLSQP
        # x: [0, w], y: [0, h], r: [0, 1] (max possible radius is small)
        bounds = []
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((0.01, 0.5))
            
        cons = []
        
        # We will re-create constraints in each iteration because they depend on current x
        # But SLSQP takes a fixed list. We can use a mutable container or just rebuild.
        # Rebuilding is safer for clarity, though slightly slower. Given N=21, it's fine.
        
        for k in range(max_iters):
            if time.time() > deadline:
                break
                
            # Define constraints based on current x
            xk, yk, rk = x[:N], x[N:2*N], x[2*N:3*N]
            
            # 1. Boundary constraints (linear, constant)
            # x_i - r_i >= 0
            # w - x_i - r_i >= 0
            # y_i - r_i >= 0
            # h - y_i - r_i >= 0
            
            # 2. Overlap constraints (linearized)
            # For each pair (i, j), i < j:
            # (x_i - x_j) * (xk_i - xk_j) + (y_i - y_j) * (yk_i - yk_j) - (r_i + r_j) >= 0
            
            # We will construct the constraint functions dynamically.
            # Since SLSQP requires a fixed structure, we define a list of dicts.
            # However, the coefficients change.
            # A common trick is to define the constraint function to look at the *current* state 
            # if we were doing sequential quadratic programming manually. 
            # But with SLSQP, we must provide the constraint definition.
            # Actually, SCR usually solves a QP or LP. SLSQP can solve the linearized problem.
            
            # Let's define the constraints for this iteration.
            local_cons = []
            
            # Boundary constraints
            # These are always the same
            for i in range(N):
                # x_i - r_i >= 0
                local_cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: x[i] - x[2*N + i],
                    'jac': lambda x, i=i: np.concatenate([np.eye(N)[i], -np.eye(N)[i], -np.eye(N)[i]])
                })
                # w - x_i - r_i >= 0
                local_cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: w - x[i] - x[2*N + i],
                    'jac': lambda x, i=i: np.concatenate([-np.eye(N)[i], -np.eye(N)[i], -np.eye(N)[i]])
                })
                # y_i - r_i >= 0
                local_cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: x[N + i] - x[2*N + i],
                    'jac': lambda x, i=i: np.concatenate([np.zeros(N), np.eye(N)[i], -np.eye(N)[i]])
                })
                # h - y_i - r_i >= 0
                local_cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: h - x[N + i] - x[2*N + i],
                    'jac': lambda x, i=i: np.concatenate([np.zeros(N), -np.eye(N)[i], -np.eye(N)[i]])
                })

            # Overlap constraints
            for i in range(N):
                for j in range(i + 1, N):
                    # Linearization point
                    dxk = xk[i] - xk[j]
                    dyk = yk[i] - yk[j]
                    distk = np.sqrt(dxk*dxk + dyk*dyk)
                    
                    # If distk is 0, we have a singularity. Perturb slightly.
                    if distk < 1e-8:
                        dxk = 1e-4
                        dyk = 0.0
                        distk = 1e-4
                    
                    # Unit vector from j to i at linearization point
                    ux = dxk / distk
                    uy = dyk / distk
                    
                    # Constraint: ux * (x_i - x_j) + uy * (y_i - y_j) >= r_i + r_j
                    # => ux * x_i - ux * x_j + uy * y_i - uy * y_j - r_i - r_j >= 0
                    
                    def make_overlap_fun(i, j, ux, uy):
                        def fun(x):
                            return ux * (x[i] - x[j]) + uy * (x[N + i] - x[N + j]) - x[2*N + i] - x[2*N + j]
                        def jac(x):
                            grad = np.zeros(3 * N)
                            grad[i] = ux
                            grad[j] = -ux
                            grad[N + i] = uy
                            grad[N + j] = -uy
                            grad[2*N + i] = -1
                            grad[2*N + j] = -1
                            return grad
                        return fun, jac

                    fun, jac = make_overlap_fun(i, j, ux, uy)
                    local_cons.append({
                        'type': 'ineq',
                        'fun': fun,
                        'jac': jac
                    })

            # Objective: Minimize -sum(r)
            # x[2*N:3*N] are radii
            def objective(x):
                return -np.sum(x[2*N:3*N])

            def objective_jac(x):
                grad = np.zeros(3 * N)
                grad[2*N:3*N] = -1
                return grad

            try:
                res = minimize(
                    objective, 
                    x, 
                    jac=objective_jac,
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=local_cons,
                    options={'maxiter': 100, 'ftol': 1e-8, 'disp': False}
                )
                
                if res.success:
                    new_x = res.x
                    # Check improvement
                    old_val = -objective(x)
                    new_val = -objective(new_x)
                    
                    if new_val > old_val + 1e-8:
                        x = new_x
                    else:
                        # Convergence or no improvement
                        break
                else:
                    # If SLSQP fails, try a smaller step or break
                    break
                    
            except Exception as e:
                # If optimization fails, break
                break
        
        return x

    # Try multiple aspect ratios
    # Perimeter 4 => w + h = 2
    # Common good aspect ratios for 3x7 grid: w > h
    # 1.4 / 0.6 = 2.33
    # 1.6 / 0.4 = 4.0
    # 1.8 / 0.2 = 9.0
    # 1.2 / 0.8 = 1.5
    
    aspect_ratios = [1.4/0.6, 1.6/0.4, 1.8/0.2, 1.2/0.8, 1.0, 0.6/1.4]

    for ar in aspect_ratios:
        if time.time() > deadline:
            break
            
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Initialize with staggered grid
        rows = 3
        cols = 7
        row_h = h / rows
        col_w = w / cols
        
        x0 = np.zeros(N * 3)
        idx = 0
        for i in range(rows):
            for j in range(cols):
                # Center x
                cx = (j + 0.5) * col_w
                # Stagger: shift odd rows by half cell width
                if i % 2 == 1:
                    cx += col_w / 2.0
                # Ensure within bounds
                cx = np.clip(cx, 0.1, w - 0.1)
                
                cy = (i + 0.5) * row_h
                
                # Initial radius: slightly less than half the min cell dimension
                r = min(col_w, row_h) * 0.45
                
                x0[idx*3] = cx
                x0[idx*3+1] = cy
                x0[idx*3+2] = r
                idx += 1
        
        # Run SCR
        try:
            x_res = solve_scr(w, h, x0)
            circles = x_res.reshape(N, 3)
            
            # Clip to bounds to handle numerical drift
            circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], w - circles[:, 2])
            circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], h - circles[:, 2])
            
            if is_valid(circles, w, h):
                val = np.sum(circles[:, 2])
                if val > best_value:
                    best_value = val
                    best_result = circles.copy()
        except Exception:
            pass

    # Fallback: Simple grid if SCR failed completely
    if best_result is None or best_value < 2.0:
        w, h = 1.4, 0.6
        cells = []
        for i in range(3):
            for j in range(7):
                cx = (j + 0.5) * w / 7
                cy = (i + 0.5) * h / 3
                cells.append([cx, cy, 0.0999])
        best_result = np.array(cells)
        best_value = np.sum(best_result[:, 2])

    return best_result
# EVOLVE-BLOCK-END
