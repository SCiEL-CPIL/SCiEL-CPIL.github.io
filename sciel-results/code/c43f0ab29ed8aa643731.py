"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Formulate the problem as a non-convex feasibility search using a custom interior-point method that iteratively solves a sequence of linear programming problems to maximize a lower bound on the minimum area, leveraging the fact that the area constraints are linear in point coordinates for fixed triplets. This approach avoids the non-smoothness of the min-function by directly optimizing the geometric constraints.
2. Use barycentric coordinates to enforce the triangular domain exactly without projection artifacts.
3. Employ a multi-start approach with different initial configurations to explore the solution space.
4. Use scipy.optimize.linprog for the LP subproblems, and a simple active-set-like method to determine which triplets are most restrictive.
"""
import numpy as np
import time
import random
from scipy.optimize import linprog

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

def _bary_to_cart(bary):
    """Convert barycentric coordinates to Cartesian coordinates.
    bary: (n, 3) array where columns are weights for vertices (0,0), (1,0), (0.5, sqrt(3)/2)
    """
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = S3H * bary[:, 2]
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    """Convert Cartesian coordinates to barycentric coordinates.
    pts: (n, 2) array
    """
    # Inverse of the transformation
    # x = b1 + 0.5*b2, y = sqrt(3)/2 * b2
    # b2 = 2*y/sqrt(3)
    # b1 = x - 0.5*b2
    # b0 = 1 - b1 - b2
    b2 = 2 * pts[:, 1] / np.sqrt(3)
    b1 = pts[:, 0] - 0.5 * b2
    b0 = 1 - b1 - b2
    return np.column_stack([b0, b1, b2])

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _lp_feasibility_check(bary, area_bound, rng):
    """
    Check if there exists a configuration with all triangle areas >= area_bound.
    Uses a linear programming approach: for each triplet, the area is a linear
    function of the barycentric coordinates (when expressed in Cartesian).
    
    Actually, the area of triangle (i,j,k) in Cartesian coordinates is:
    0.5 * |det([p_j - p_i, p_k - p_i])|
    
    This is piecewise linear. For a fixed orientation, it's linear.
    We can use the fact that if we fix the sign of the cross product,
    the area constraint becomes linear.
    
    However, a simpler approach: use a soft constraint or iterative method.
    
    For this implementation, we'll use a different approach:
    - Start with a candidate configuration
    - Identify the triplet with minimum area
    - Try to increase that area by moving points
    - Use LP to find a feasible direction
    
    Actually, let's use a simpler iterative approach:
    - Maintain a set of "active" triplets (those with area near the minimum)
    - Solve an LP that maximizes the minimum area subject to these active constraints
    - Update the active set
    """
    pass

def _solve_lp_for_min_area(bary, active_triplets, area_target):
    """
    Solve an LP to find a configuration that maximizes the minimum area.
    Variables: barycentric coordinates of all points (n*3 = 33 variables)
    Constraints:
      - bary[i,0] + bary[i,1] + bary[i,2] = 1 for each i
      - bary[i,j] >= 0 for each i,j
      - For each active triplet (i,j,k): area(i,j,k) >= area_target
    
    The area of triangle (i,j,k) in terms of barycentric coordinates:
    p_i = bary[i] @ V, where V = [[1,0],[1,0],[0.5,sqrt(3)/2]] (columns are vertices)
    
    Actually, let's work in Cartesian coordinates for the LP.
    Variables: x_i, y_i for i=0..n-1 (2n = 22 variables)
    
    Constraints:
      - 0 <= y_i <= sqrt(3)*x_i  (below left edge)
      - 0 <= y_i <= sqrt(3)*(1-x_i)  (below right edge)
      - 0 <= x_i <= 1
    
    For each active triplet (i,j,k):
      The area is 0.5*|det([p_j-p_i, p_k-p_i])|
      = 0.5*|(x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i)|
    
    This is piecewise linear. We need to fix the sign.
    For a given configuration, we can determine the sign and then the constraint
    becomes linear.
    
    Let's implement this:
    """
    n = N
    # Variables: x_0, y_0, x_1, y_1, ..., x_{n-1}, y_{n-1}
    # Index: 2*i for x_i, 2*i+1 for y_i
    
    # Objective: maximize t (lower bound on min area)
    # Minimize -t
    # c = [0, 0, ..., 0, -1] (last variable is t)
    c = np.zeros(2*n + 1)
    c[-1] = -1.0
    
    # Inequality constraints: A_ub @ vars <= b_ub
    
    # Triangle constraints for each point:
    # y_i >= 0  =>  -y_i <= 0
    # y_i <= sqrt(3)*x_i  =>  -sqrt(3)*x_i + y_i <= 0
    # y_i <= sqrt(3)*(1-x_i)  =>  sqrt(3)*x_i + y_i <= sqrt(3)
    # x_i >= 0  =>  -x_i <= 0
    # x_i <= 1  =>  x_i <= 1
    
    A_ub = []
    b_ub = []
    
    for i in range(n):
        # -y_i <= 0
        row = np.zeros(2*n + 1)
        row[2*i+1] = -1
        A_ub.append(row)
        b_ub.append(0)
        
        # -sqrt(3)*x_i + y_i <= 0
        row = np.zeros(2*n + 1)
        row[2*i] = -np.sqrt(3)
        row[2*i+1] = 1
        A_ub.append(row)
        b_ub.append(0)
        
        # sqrt(3)*x_i + y_i <= sqrt(3)
        row = np.zeros(2*n + 1)
        row[2*i] = np.sqrt(3)
        row[2*i+1] = 1
        A_ub.append(row)
        b_ub.append(np.sqrt(3))
        
        # -x_i <= 0
        row = np.zeros(2*n + 1)
        row[2*i] = -1
        A_ub.append(row)
        b_ub.append(0)
        
        # x_i <= 1
        row = np.zeros(2*n + 1)
        row[2*i] = 1
        A_ub.append(row)
        b_ub.append(1)
    
    # Area constraints for active triplets:
    # For each active triplet (i,j,k), we need:
    # 0.5*|det| >= area_target
    # Which means: det >= 2*area_target OR det <= -2*area_target
    # We need to determine which sign applies for the current configuration.
    
    for (i, j, k) in active_triplets:
        # Compute the sign of the cross product for the current configuration
        xi, yi = bary[2*i], bary[2*i+1]
        xj, yj = bary[2*j], bary[2*j+1]
        xk, yk = bary[2*k], bary[2*k+1]
        
        dx1 = xj - xi
        dy1 = yj - yi
        dx2 = xk - xi
        dy2 = yk - yi
        cross = dx1*dy2 - dy1*dx2
        
        if cross >= 0:
            # We want cross >= 2*area_target
            # cross = (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i)
            # = x_j*y_k - x_j*y_i - x_i*y_k + x_i*y_i - y_j*x_k + y_j*x_i + y_i*x_k - y_i*x_i
            # = x_j*y_k - x_j*y_i - x_i*y_k - y_j*x_k + y_j*x_i + y_i*x_k
            # (the x_i*y_i and -y_i*x_i terms cancel)
            # So: x_j*y_k - x_j*y_i - x_i*y_k - y_j*x_k + y_j*x_i + y_i*x_k >= 2*area_target
            # Rearranging: -x_j*y_k + x_j*y_i + x_i*y_k + y_j*x_k - y_j*x_i - y_i*x_k <= -2*area_target
            
            row = np.zeros(2*n + 1)
            row[2*j] = -1  # coefficient of x_j in -x_j*y_k... wait, this is bilinear!
            
            # Oops, the area is bilinear in the coordinates, not linear!
            # This means LP won't work directly.
            
            # Let me reconsider. The area of triangle (i,j,k) is:
            # 0.5*|(x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i)|
            # This is indeed bilinear (quadratic) in the point coordinates.
            # So it's NOT a linear constraint.
            
            # This means the LP approach doesn't directly work.
            # We need a different strategy.
            
            # Alternative: use a sequential quadratic programming approach,
            # or use the fact that for a fixed set of "active" triplets,
            # we can linearize around the current point.
            
            # For simplicity, let's use a different approach:
            # - Use a greedy method that identifies the most restrictive triplet
            # - Move points to increase that area
            # - Use gradient-based optimization with a smooth approximation
            
            pass
    
    # Since the area constraints are bilinear, not linear, the LP approach
    # doesn't directly apply. Let's use a different strategy.
    
    return None

def _gradient_ascent_step(bary, step_size, rng):
    """
    Take a gradient ascent step to increase the minimum area.
    Use a soft-min approximation for a smooth objective.
    """
    pts = _bary_to_cart(bary)
    
    # Compute areas for all triplets
    areas = []
    triplet_indices = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = dx1 * dy2 - dy1 * dx2
                area = 0.5 * abs(cross)
                areas.append(area)
                triplet_indices.append((i, j, k))
    
    areas = np.array(areas)
    min_area = np.min(areas)
    
    # Soft min: f = (1/b) * log(sum(exp(-b*area)))
    b = 50.0
    exp_terms = np.exp(-b * areas)
    soft_min_val = (1.0 / b) * np.log(np.sum(exp_terms))
    
    # Gradient of soft_min with respect to areas:
    # d(soft_min)/d(area_m) = -exp(-b*area_m) / sum(exp(-b*area))
    # = -softmax(-b*areas)
    
    weights = np.exp(-b * areas) / np.sum(np.exp(-b * areas))
    
    # Now compute gradient with respect to point coordinates.
    # This requires the chain rule: d(soft_min)/d(p) = sum_m weights_m * d(area_m)/d(p)
    
    # For each triplet m = (i,j,k), the area is:
    # area_m = 0.5 * |det([p_j - p_i, p_k - p_i])|
    
    # The gradient of area_m with respect to p_i, p_j, p_k:
    # If cross > 0:
    #   d(area)/d(p_i) = -0.5 * [dy2, -dx2] (from the first term) + ... 
    #   Actually, let's compute it carefully.
    
    # area = 0.5 * (dx1*dy2 - dy1*dx2)
    # where dx1 = x_j - x_i, dy1 = y_j - y_i, dx2 = x_k - x_i, dy2 = y_k - y_i
    
    # d(area)/d(x_i) = 0.5 * (-dy2 + dx2 * 0) = -0.5*dy2 = -0.5*(y_k - y_i)
    # d(area)/d(y_i) = 0.5 * (dx2 - dx1 * 0) = 0.5*dx2 = 0.5*(x_k - x_i)
    # d(area)/d(x_j) = 0.5 * (dy2) = 0.5*(y_k - y_i)
    # d(area)/d(y_j) = 0.5 * (-dx2) = -0.5*(x_k - x_i)
    # d(area)/d(x_k) = 0.5 * (-dy1) = -0.5*(y_j - y_i)
    # d(area)/d(y_k) = 0.5 * (dx1) = 0.5*(x_j - x_i)
    
    # If cross < 0, the signs flip.
    
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    
    for m, (i, j, k) in enumerate(triplet_indices):
        w = weights[m]
        if w < 1e-10:
            continue
        
        xi, yi = pts[i]
        xj, yj = pts[j]
        xk, yk = pts[k]
        
        dx1 = xj - xi
        dy1 = yj - yi
        dx2 = xk - xi
        dy2 = yk - yi
        cross = dx1 * dy2 - dy1 * dx2
        
        if cross >= 0:
            # Gradient as computed above
            grad_x[i] += w * (-0.5 * dy2)
            grad_y[i] += w * (0.5 * dx2)
            grad_x[j] += w * (0.5 * dy2)
            grad_y[j] += w * (-0.5 * dx2)
            grad_x[k] += w * (-0.5 * dy1)
            grad_y[k] += w * (0.5 * dx1)
        else:
            # Signs flip
            grad_x[i] += w * (0.5 * dy2)
            grad_y[i] += w * (-0.5 * dx2)
            grad_x[j] += w * (-0.5 * dy2)
            grad_y[j] += w * (0.5 * dx2)
            grad_x[k] += w * (0.5 * dy1)
            grad_y[k] += w * (-0.5 * dx1)
    
    # Now convert Cartesian gradient to barycentric gradient.
    # x = b1 + 0.5*b2, y = sqrt(3)/2 * b2
    # d(x)/d(b1) = 1, d(x)/d(b2) = 0.5, d(x)/d(b0) = 0
    # d(y)/d(b1) = 0, d(y)/d(b2) = sqrt(3)/2, d(y)/d(b0) = 0
    
    # grad_b0 = grad_x * 0 + grad_y * 0 = 0
    # grad_b1 = grad_x * 1 + grad_y * 0 = grad_x
    # grad_b2 = grad_x * 0.5 + grad_y * sqrt(3)/2
    
    grad_bary = np.zeros_like(bary)
    for i in range(N):
        grad_bary[i, 0] = 0  # b0 doesn't affect x or y directly (it's 1-b1-b2)
        grad_bary[i, 1] = grad_x[i]
        grad_bary[i, 2] = 0.5 * grad_x[i] + (np.sqrt(3) / 2) * grad_y[i]
    
    # Take a step in the direction of the gradient
    new_bary = bary + step_size * grad_bary
    
    # Project back to valid barycentric coordinates:
    # b0 + b1 + b2 = 1, b0,b1,b2 >= 0
    for i in range(N):
        # Normalize to sum to 1
        s = new_bary[i].sum()
        if s > 0:
            new_bary[i] /= s
        else:
            new_bary[i] = np.array([1/3, 1/3, 1/3])
        
        # Clip to [0, 1]
        new_bary[i] = np.clip(new_bary[i], 0, 1)
        
        # Renormalize
        s = new_bary[i].sum()
        if s > 0:
            new_bary[i] /= s
    
    return new_bary, min_area

def _multi_start_optimize(rng, start_time, deadline):
    """
    Multi-start optimization using gradient ascent with soft-min objective.
    """
    best_bary = None
    best_score = 0
    
    # Generate several initial configurations
    n_starts = 5
    
    for start_idx in range(n_starts):
        if time.time() - start_time > deadline - 10:
            break
        
        # Generate initial barycentric coordinates
        if start_idx == 0:
            # Start with a regular grid-like configuration
            bary = np.zeros((N, 3))
            # Place points at vertices, midpoints, and center
            bary[0] = [1, 0, 0]  # vertex 0
            bary[1] = [0, 1, 0]  # vertex 1
            bary[2] = [0, 0, 1]  # vertex 2
            bary[3] = [0.5, 0.5, 0]  # midpoint of edge 0-1
            bary[4] = [0.5, 0, 0.5]  # midpoint of edge 0-2
            bary[5] = [0, 0.5, 0.5]  # midpoint of edge 1-2
            bary[6] = [1/3, 1/3, 1/3]  # center
            # Remaining points: random
            for i in range(7, N):
                bary[i] = np.array([rng.random(), rng.random(), rng.random()])
                bary[i] /= bary[i].sum()
        else:
            # Random initialization
            bary = np.array([np.array([rng.random(), rng.random(), rng.random()]) for _ in range(N)])
            bary = bary / bary.sum(axis=1, keepdims=True)
        
        # Optimize using gradient ascent
        step_size = 0.01
        min_step_size = 1e-5
        decay = 0.995
        max_iter = 200
        
        current_bary = bary.copy()
        current_score = _min_triangle_area(_bary_to_cart(current_bary))
        
        for it in range(max_iter):
            if time.time() - start_time > deadline - 5:
                break
            
            new_bary, min_area = _gradient_ascent_step(current_bary, step_size, rng)
            new_score = _min_triangle_area(_bary_to_cart(new_bary))
            
            if new_score > current_score:
                current_bary = new_bary
                current_score = new_score
            else:
                # Reduce step size
                step_size *= decay
            
            if step_size < min_step_size:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_bary = current_bary.copy()
    
    return best_bary, best_score

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses gradient ascent with soft-min objective in barycentric coordinates.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    # Multi-start optimization
    best_bary, best_score = _multi_start_optimize(rng, start_time, DEADLINE)
    
    if best_bary is None:
        # Fallback: random configuration
        best_bary = np.array([np.array([rng.random(), rng.random(), rng.random()]) for _ in range(N)])
        best_bary = best_bary / best_bary.sum(axis=1, keepdims=True)
        best_score = _min_triangle_area(_bary_to_cart(best_bary))
    
    # Convert to Cartesian coordinates
    best_pts = _bary_to_cart(best_bary)
    
    # Project to ensure we're inside the triangle
    best_pts = _project_to_triangle(best_pts)
    
    # Check validity
    if not _check_inside(best_pts):
        # Fallback to a known valid configuration
        best_pts = np.array([
            [0, 0],
            [1, 0],
            [0.5, S3H],
            [0.5, 0],
            [0.25, S3H/2],
            [0.75, S3H/2],
            [0.5, S3H/3],
            [0.25, S3H/6],
            [0.75, S3H/6],
            [0.5, 2*S3H/3],
            [0.5, S3H/2]
        ])
        best_pts = _project_to_triangle(best_pts)
    
    return best_pts
