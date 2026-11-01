import numpy as np
from scipy.optimize import linprog
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s out of 150s
    
    best_result = None
    best_score = -np.inf

    def is_valid(xy, r, tol=1e-6):
        n = len(r)
        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False
        
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False
        
        return True

    def get_score(xy, r):
        return np.sum(r)

    def repair_solution(xy, r, max_iters=1000):
        """
        Simple iterative repair: if circles overlap, push them apart.
        If they hit walls, push other circles away.
        This is a heuristic to make the LP solution strictly feasible.
        """
        xy = xy.copy()
        r = np.clip(r, 1e-9, 0.5).copy()
        
        for it in range(max_iters):
            moved = False
            # Check boundaries first
            for i in range(N):
                if xy[i, 0] < r[i]:
                    xy[i, 0] = r[i]
                    moved = True
                elif xy[i, 0] > 1 - r[i]:
                    xy[i, 0] = 1 - r[i]
                    moved = True
                if xy[i, 1] < r[i]:
                    xy[i, 1] = r[i]
                    moved = True
                elif xy[i, 1] > 1 - r[i]:
                    xy[i, 1] = 1 - r[i]
                    moved = True
            
            # Check overlaps
            for i in range(N):
                for j in range(i+1, N):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist - 1e-9:
                        if dist < 1e-9:
                            dx, dy = 1.0, 0.0
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist
                        
                        overlap = min_dist - dist
                        # Move both apart
                        # Simple 50-50 split
                        xy[i, 0] -= 0.5 * overlap * dx
                        xy[i, 1] -= 0.5 * overlap * dy
                        xy[j, 0] += 0.5 * overlap * dx
                        xy[j, 1] += 0.5 * overlap * dy
                        moved = True
            
            # Re-clamp after movement
            for i in range(N):
                xy[i, 0] = np.clip(xy[i, 0], r[i], 1 - r[i])
                xy[i, 1] = np.clip(xy[i, 1], r[i], 1 - r[i])
            
            if not moved:
                break
        
        return xy, r

    def solve_lp(xy_init, r_init, max_lp_iters=20):
        """
        Iteratively solve the LP relaxation.
        Variables: x_1..x_N, y_1..y_N, r_1..r_N (Total 3N)
        Maximize sum(r_i)
        Subject to:
        1. r_i >= 0
        2. x_i - r_i >= 0
        3. 1 - x_i - r_i >= 0
        4. y_i - r_i >= 0
        5. 1 - y_i - r_i >= 0
        6. Linearized separation: 
           (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
           Approximated by tangent plane at current (x, y, r):
           2(x_i^k - x_j^k)(x_i - x_j) + 2(y_i^k - y_j^k)(y_i - y_j) >= (r_i^k + r_j^k)^2 - ||(x_i^k - x_j^k, y_i^k - y_j^k)||^2 + 2(r_i^k + r_j^k)(r_i + r_j - r_i^k - r_j^k)
           
           Wait, the standard linearization for dist >= sum_r is:
           (x_i - x_j) * (x_i^k - x_j^k) + (y_i - y_j) * (y_i^k - y_j^k) >= (r_i + r_j) * ||(x_i^k - x_j^k, y_i^k - y_j^k)|| - 0.5 * ||(x_i^k - x_j^k, y_i^k - y_j^k)||^2 + 0.5 * (r_i^k + r_j^k)^2
           
           Let's use a simpler valid inequality for the convex set { (x,y,r) : dist >= r_i+r_j }? No, the set is non-convex.
           The feasible region for fixed centers is convex in r? No.
           
           Actually, let's look at the constraint:
           (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
           
           If we fix centers x, y, this is linear in r:
           r_i + r_j <= sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
           
           This is a linear constraint!
           So, if we alternate between:
           1. Fix x, y. Solve LP for r. (Max sum r s.t. r_i+r_j <= dist_ij, boundary constraints on r given x,y)
           2. Fix r. Solve LP for x, y? No, fixing r makes the constraint (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2, which is non-convex.
           
           However, we can linearize the non-convex constraint using the tangent plane method (Cutting Plane).
           At current point (x^k, y^k, r^k), the constraint is:
           2(x_i^k - x_j^k)(x_i - x_j) + 2(y_i^k - y_j^k)(y_i - y_j) - 2(r_i^k + r_j^k)(r_i + r_j) >= 0
           This is not quite right because the RHS is 0 but the LHS is not a linearization of the function f(x,y,r) = dist^2 - (r_i+r_j)^2.
           
           Let f(x,y,r) = (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0.
           Gradient w.r.t x_i: 2(x_i^k - x_j^k)
           Gradient w.r.t x_j: -2(x_i^k - x_j^k)
           Gradient w.r.t y_i: 2(y_i^k - y_j^k)
           Gradient w.r.t y_j: -2(y_i^k - y_j^k)
           Gradient w.r.t r_i: -2(r_i^k + r_j^k)
           Gradient w.r.t r_j: -2(r_i^k + r_j^k)
           
           Linear approximation:
           f(x^k, y^k, r^k) + grad_x_i * (x_i - x_i^k) + grad_x_j * (x_j - x_j^k) + ... >= 0
           
           Since f(x^k, y^k, r^k) >= 0 is satisfied (or close to it), we can use this as a cutting plane.
           However, if the current point is infeasible, f < 0.
           
           Let's try a simpler approach: 
           Alternate between:
           A. Fix x, y. Solve for r.
              Max sum r
              s.t. r_i + r_j <= dist_ij (for all i,j)
                   r_i <= x_i
                   r_i <= 1 - x_i
                   r_i <= y_i
                   r_i <= 1 - y_i
                   r_i >= 0
              
           B. Fix r. Solve for x, y? 
              This is hard because of non-convexity. 
              But we can use a local search or gradient ascent on x,y to maximize the minimum slack.
              Or, we can use the tangent plane LP on the full 3N variables.
              
           Let's implement the Full LP with cutting planes.
           Variables z = [x_1..x_N, y_1..y_N, r_1..r_N]
           
           Initial guess: Grid.
           """
        x = xy_init.copy()
        y = xy_init.copy()
        r = r_init.copy()
        
        # Accumulate constraints
        A_ub = []
        b_ub = []
        
        # Boundary constraints (Always active)
        # -x_i + r_i <= 0  =>  x_i - r_i >= 0
        # In linprog form: A_ub @ z <= b_ub
        # x_i - r_i <= 0 is not right. We want x_i - r_i >= 0 => -x_i + r_i <= 0.
        
        for i in range(N):
            # Left: x_i >= r_i  => -x_i + r_i <= 0
            row = np.zeros(3*N)
            row[i] = -1
            row[2*N + i] = 1
            A_ub.append(row)
            b_ub.append(0)
            
            # Right: 1 - x_i >= r_i => x_i + r_i <= 1
            row = np.zeros(3*N)
            row[i] = 1
            row[2*N + i] = 1
            A_ub.append(row)
            b_ub.append(1)
            
            # Bottom: y_i >= r_i => -y_i + r_i <= 0
            row = np.zeros(3*N)
            row[N + i] = -1
            row[2*N + i] = 1
            A_ub.append(row)
            b_ub.append(0)
            
            # Top: 1 - y_i >= r_i => y_i + r_i <= 1
            row = np.zeros(3*N)
            row[N + i] = 1
            row[2*N + i] = 1
            A_ub.append(row)
            b_ub.append(1)
            
            # r_i >= 0 => -r_i <= 0
            row = np.zeros(3*N)
            row[2*N + i] = -1
            A_ub.append(row)
            b_ub.append(0)
            
        # Objective: Maximize sum(r) => Minimize -sum(r)
        c = np.zeros(3*N)
        c[2*N:] = -1
        
        current_x = x
        current_y = y
        current_r = r
        
        for it in range(max_lp_iters):
            if time.time() > deadline:
                break
                
            # Add cutting planes for separation
            # For each pair (i, j), add the linearized constraint at current (x, y, r)
            # Only if current point is "close" to violation or always?
            # To keep the LP small, we might limit the number of active cuts.
            # However, N=32 is small. 32*31/2 = 496 pairs.
            # 496 * 32*32 matrix operations are fine.
            
            # We will rebuild A_ub with current cuts.
            # Actually, adding cuts to the previous LP is better, but scipy.linprog doesn't support warm starts easily with new constraints in a simple API without re-solving.
            # Since N is small, re-solving from scratch with all accumulated cuts is feasible.
            
            A_full = [row for row in A_ub] # Base constraints
            b_full = [b for b in b_ub]
            
            # Add separation cuts
            for i in range(N):
                for j in range(i+1, N):
                    dx_k = current_x[i] - current_x[j]
                    dy_k = current_y[i] - current_y[j]
                    dr_k = current_r[i] + current_r[j]
                    
                    # Gradient of f = dist^2 - (r_i+r_j)^2
                    # df/dx_i = 2*dx_k
                    # df/dx_j = -2*dx_k
                    # df/dy_i = 2*dy_k
                    # df/dy_j = -2*dy_k
                    # df/dr_i = -2*dr_k
                    # df/dr_j = -2*dr_k
                    
                    # f(x^k) + grad . (z - z^k) >= 0
                    # grad . z >= grad . z^k - f(x^k)
                    
                    # Note: If f(x^k) < 0, the point is infeasible. The cut pushes it towards feasibility.
                    # If f(x^k) > 0, the cut is a valid inequality for the convex hull approximation.
                    
                    f_k = dx_k**2 + dy_k**2 - dr_k**2
                    
                    # Coefficients for z
                    row = np.zeros(3*N)
                    row[i] = 2 * dx_k
                    row[j] = -2 * dx_k
                    row[N + i] = 2 * dy_k
                    row[N + j] = -2 * dy_k
                    row[2*N + i] = -2 * dr_k
                    row[2*N + j] = -2 * dr_k
                    
                    # RHS
                    # grad . z^k - f_k
                    # (2*dx_k)*x_i^k + (-2*dx_k)*x_j^k + ... - f_k
                    # 2*dx_k*(x_i^k - x_j^k) + 2*dy_k*(y_i^k - y_j^k) - 2*dr_k*(r_i^k + r_j^k) - f_k
                    # 2*dx_k^2 + 2*dy_k^2 - 2*dr_k^2 - (dx_k^2 + dy_k^2 - dr_k^2)
                    # = dx_k^2 + dy_k^2 - dr_k^2 = f_k
                    # Wait.
                    # LHS = grad . z
                    # Inequality: grad . z >= grad . z^k - f_k
                    # grad . z^k = 2*dx_k^2 + 2*dy_k^2 - 2*dr_k^2
                    # RHS = 2*dx_k^2 + 2*dy_k^2 - 2*dr_k^2 - (dx_k^2 + dy_k^2 - dr_k^2) = dx_k^2 + dy_k^2 - dr_k^2 = f_k
                    # So: grad . z >= f_k
                    
                    # In linprog form: -grad . z <= -f_k
                    row = -row
                    A_full.append(row)
                    b_full.append(-f_k)
                    
            A_full = np.array(A_full)
            b_full = np.array(b_full)
            
            try:
                res = linprog(c, A_ub=A_full, b_ub=b_full, method='highs')
                if res.success:
                    z_new = res.x
                    x_new = z_new[:N]
                    y_new = z_new[N:2*N]
                    r_new = z_new[2*N:]
                    
                    # Check validity of new point
                    # If valid, we have a better or equal score.
                    # Update current point for next iteration's cuts.
                    current_x = x_new
                    current_y = y_new
                    current_r = r_new
                    
                    # Optional: If the LP solution is strictly feasible, we might be done or close.
                    # The LP maximizes sum(r). If it's feasible, it's a good candidate.
                else:
                    # If LP fails, maybe too many constraints or numerical issues.
                    break
            except Exception:
                break
                
        return current_x, current_y, current_r

    def get_initial_config(restart):
        rng = np.random.RandomState(42 + restart)
        if restart == 0:
            side = 6
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
        else:
            # Perturbed grid
            side = 6
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
            xy += rng.uniform(-0.02, 0.02, size=(N, 2))
            # Ensure inside
            for i in range(N):
                xy[i, 0] = np.clip(xy[i, 0], r[i], 1-r[i])
                xy[i, 1] = np.clip(xy[i, 1], r[i], 1-r[i])
        return xy, r

    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        xy, r = get_initial_config(restart)
        
        # 1. Solve LP relaxation
        x_lp, y_lp, r_lp = solve_lp(xy, r, max_lp_iters=10)
        
        # 2. Repair the LP solution to be strictly feasible
        x_rep, r_rep = repair_solution(x_lp, r_lp, max_iters=200)
        
        if is_valid(x_rep, r_rep):
            score = get_score(x_rep, r_rep)
            if score > best_score:
                best_score = score
                best_result = (x_rep.copy(), r_rep.copy())
                
        # 3. Also try repairing the initial config just in case LP diverged
        x_rep_init, r_rep_init = repair_solution(xy, r, max_iters=200)
        if is_valid(x_rep_init, r_rep_init):
            score = get_score(x_rep_init, r_rep_init)
            if score > best_score:
                best_score = score
                best_result = (x_rep_init.copy(), r_rep_init.copy())

    if best_result is not None:
        xy, r = best_result
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1-r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1-r)
        if is_valid(xy, r):
            return np.column_stack([xy, r])
    
    # Fallback: grid
    side = 6
    cells = []
    for i in range(N):
        row = i // side
        col = i % side
        cells.append([(col + 0.5) / side, (row + 0.5) / side, 0.5 / side - 1e-3])
    return np.array(cells)
