"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

def _solve_circle_placement(w, h, n, max_iter=200, tol=1e-6):
    """
    Given fixed rectangle w x h (perimeter 2(w+h) = 4 => w+h=2),
    find n circle centers (x_i, y_i) and radii r_i to maximize sum r_i
    subject to:
      r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i
      r_i + r_j <= ||(x_i,y_i) - (x_j,y_j)||  for i != j
      r_i >= 0
    
    We use a convex relaxation: instead of the non-convex constraint
      r_i + r_j <= sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
    we use the linearized (first-order) approximation at the current iterate,
    or we use the fact that for fixed centers, the optimal radii can be found
    via LP. We alternate:
      1. Fix centers, solve LP for radii
      2. Fix radii, adjust centers to increase minimal pair distances
    Actually, a simpler approach: use a penalty/gradient method.
    
    Better approach: Use the fact that this is a convex optimization problem if we
    parameterize differently. Actually, the problem is NOT convex in general because
    the constraint r_i + r_j <= ||p_i - p_j|| is convex in (r_i, r_j, p_i, p_j) 
    when written as r_i + r_j <= ||p_i - p_j||. Wait, is it?
    
    The set {(r_i, r_j, p_i, p_j) : r_i + r_j <= ||p_i - p_j||} is convex?
    No, because r_i + r_j is linear and ||p_i - p_j|| is convex, so the inequality
    linear <= convex defines a non-convex set.
    
    However, if we fix the centers, the problem in radii is an LP.
    If we fix the radii, the problem in centers is: maximize sum r_i (constant) 
    subject to ||p_i - p_j|| >= r_i + r_j, which is a non-convex feasibility problem.
    
    Let's use a different approach: iterative relinearization.
    At each iteration, given current (p, r), linearize the distance constraint:
    ||p_i - p_j|| >= r_i + r_j
    At current point, the gradient of ||p_i - p_j|| w.r.t. p_i is (p_i - p_j)/||p_i - p_j||.
    So the linearized constraint is:
    (p_i - p_j)/||p_i - p_j|| · (q_i - q_j) >= r_i + r_j  [where q is new center]
    Wait, that's not quite right. The linearization of f(p) = ||p_i - p_j|| at p0 is:
    f(p) >= f(p0) + ∇f(p0)·(p - p0)
    So ||p_i - p_j|| >= ||p_i^0 - p_j^0|| + (p_i^0 - p_j^0)/||p_i^0 - p_j^0|| · ((p_i - p_i^0) - (p_j - p_j^0))
    
    This gives a linear lower bound on the distance. Using this as a constraint:
    r_i + r_j <= ||p_i^0 - p_j^0|| + (p_i^0 - p_j^0)/||p_i^0 - p_j^0|| · ((p_i - p_i^0) - (p_j - p_j^0))
    
    This is a linear constraint in (p_i, p_j, r_i, r_j). So at each iteration, we solve an LP
    with these linearized constraints plus the boundary constraints. This is a standard
    cutting-plane / linearization approach.
    """
    # Initialize with equal circles in a grid
    side = int(np.ceil(np.sqrt(n)))
    xs = np.array([(i + 0.5) * w / side for i in range(side)][:n])
    ys = np.array([(j + 0.5) * h / side for j in range(side)][:n])
    # Reshape to n points
    if len(xs) < n:
        xs = np.append(xs, np.array([w * 0.5] * (n - len(xs))))
    if len(ys) < n:
        ys = np.append(ys, np.array([h * 0.5] * (n - len(ys))))
    
    centers = np.column_stack([xs[:n], ys[:n]])
    # Initial radii: min distance to boundary
    r = np.minimum(np.minimum(centers[:, 0], w - centers[:, 0]),
                   np.minimum(centers[:, 1], h - centers[:, 1]))
    r = np.maximum(r, 1e-6)
    
    for it in range(max_iter):
        # Build LP: variables are [x_1, y_1, r_1, x_2, y_2, r_2, ..., x_n, y_n, r_n]
        # Objective: maximize sum r_i => minimize -sum r_i
        nvars = 3 * n
        c = np.zeros(nvars)
        for i in range(n):
            c[3*i + 2] = -1.0  # coefficient of r_i
        
        # Constraints:
        # 1. r_i <= x_i => x_i - r_i >= 0 => -x_i + r_i <= 0
        # 2. r_i <= w - x_i => x_i + r_i <= w
        # 3. r_i <= y_i => -y_i + r_i <= 0
        # 4. r_i <= h - y_i => y_i + r_i <= h
        # 5. r_i >= 0
        # 6. Linearized distance constraints
        
        A_rows = []
        b_rows = []
        
        for i in range(n):
            # -x_i + r_i <= 0
            row = np.zeros(nvars)
            row[3*i] = -1.0
            row[3*i+2] = 1.0
            A_rows.append(row)
            b_rows.append(0.0)
            
            # x_i + r_i <= w
            row = np.zeros(nvars)
            row[3*i] = 1.0
            row[3*i+2] = 1.0
            A_rows.append(row)
            b_rows.append(w)
            
            # -y_i + r_i <= 0
            row = np.zeros(nvars)
            row[3*i+1] = -1.0
            row[3*i+2] = 1.0
            A_rows.append(row)
            b_rows.append(0.0)
            
            # y_i + r_i <= h
            row = np.zeros(nvars)
            row[3*i+1] = 1.0
            row[3*i+2] = 1.0
            A_rows.append(row)
            b_rows.append(h)
            
            # r_i >= 0 => -r_i <= 0
            row = np.zeros(nvars)
            row[3*i+2] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)
        
        # Linearized distance constraints
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-10:
                    # Perturb slightly
                    dx = 1e-6
                    dy = 0
                    dist = 1e-6
                
                # Normal vector
                nx = dx / dist
                ny = dy / dist
                
                # Linearized constraint:
                # r_i + r_j <= dist + nx*(x_i - x_j) + ny*(y_i - y_j) - nx*(centers[i,0]-centers[j,0]) - ny*(centers[i,1]-centers[j,1])
                # Wait, let me re-derive.
                # f(p) = ||p_i - p_j||, linearized at p0:
                # f(p) >= f(p0) + ∇_{p_i}f(p0)·(p_i - p_i^0) + ∇_{p_j}f(p0)·(p_j - p_j^0)
                # ∇_{p_i}f = (p_i^0 - p_j^0)/||p_i^0 - p_j^0|| = (nx, ny)
                # ∇_{p_j}f = -(nx, ny)
                # So f(p) >= dist + nx*(x_i - x_i^0) + ny*(y_i - y_i^0) - nx*(x_j - x_j^0) - ny*(y_j - y_j^0)
                # Constraint: r_i + r_j <= f(p)
                # So: r_i + r_j - nx*(x_i - x_i^0) - ny*(y_i - y_i^0) + nx*(x_j - x_j^0) + ny*(y_j - y_j^0) <= dist
                # Rearranging:
                # -nx*x_i - ny*y_i + nx*x_j + ny*y_j + r_i + r_j <= dist - nx*(x_i^0 - x_j^0) - ny*(y_i^0 - y_j^0)
                # But nx*(x_i^0 - x_j^0) + ny*(y_i^0 - y_j^0) = (dx/dx_dist)*dx + (dy/dx_dist)*dy = dist
                # So RHS = dist - dist = 0
                # Wait, that can't be right. Let me recompute.
                # nx = dx/dist, ny = dy/dist where dx = x_i^0 - x_j^0, dy = y_i^0 - y_j^0
                # nx*dx + ny*dy = (dx^2 + dy^2)/dist = dist
                # So RHS = dist - dist = 0
                # So constraint: -nx*x_i - ny*y_i + nx*x_j + ny*y_j + r_i + r_j <= 0
                # Hmm, that seems off. Let me just use the standard form.
                
                # The constraint is:
                # r_i + r_j <= dist + nx*(x_i - x_i^0) + ny*(y_i - y_i^0) - nx*(x_j - x_j^0) - ny*(y_j - y_j^0)
                # => r_i + r_j - nx*x_i - ny*y_i + nx*x_j + ny*y_j <= dist - nx*x_i^0 - ny*y_i^0 + nx*x_j^0 + ny*y_j^0
                # => r_i + r_j - nx*x_i - ny*y_i + nx*x_j + ny*y_j <= dist - (nx*dx + ny*dy)
                # => r_i + r_j - nx*x_i - ny*y_i + nx*x_j + ny*y_j <= dist - dist = 0
                
                row = np.zeros(nvars)
                row[3*i] = -nx
                row[3*i+1] = -ny
                row[3*i+2] = 1.0
                row[3*j] = nx
                row[3*j+1] = ny
                row[3*j+2] = 1.0
                A_rows.append(row)
                b_rows.append(0.0)
        
        A = np.array(A_rows)
        b = np.array(b_rows)
        
        # Solve LP using scipy
        try:
            res = minimize(
                lambda x: c @ x,
                x0=np.column_stack([centers, r]).flatten(),
                method='HiGHS',
                bounds=[(0, w)]*n + [(0, h)]*n + [(0, min(w,h))]*n,
                constraints={'A_ub': A, 'b_ub': b}
            )
            if not res.success:
                break
            x_opt = res.x
            new_centers = np.column_stack([x_opt[0:n], x_opt[n:2*n]])
            new_r = x_opt[2*n:3*n]
            
            # Check convergence
            diff = np.max(np.abs(new_centers - centers)) + np.max(np.abs(new_r - r))
            centers = new_centers
            r = new_r
            if diff < tol:
                break
        except Exception:
            break
    
    return centers, r


def construct():
    start_time = time.time()
    deadline = 110.0  # safety margin
    
    best_sum = 0.0
    best_result = None
    
    # Try different aspect ratios
    # w + h = 2, so h = 2 - w
    # w in (0, 2)
    aspect_ratios = np.linspace(0.4, 1.6, 25)  # w from 0.4 to 1.6
    
    for w in aspect_ratios:
        if time.time() > deadline - 5:
            break
        h = 2.0 - w
        if h <= 0:
            continue
        
        centers, r = _solve_circle_placement(w, h, N, max_iter=50, tol=1e-5)
        s = np.sum(r)
        if s > best_sum:
            best_sum = s
            best_result = np.column_stack([centers, r])
    
    if best_result is None:
        # Fallback: equal circles in square
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
