"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Barycentric Coordinate Parameterization:
   - Represents points using barycentric coordinates (u, v, w) with u+v+w=1, u,v,w >= 0.
   - This naturally enforces the constraint that points lie inside the triangle.
   - We optimize over (u, v) for each point, with w = 1 - u - v.
   - To handle the non-negativity constraints u>=0, v>=0, we use the L-BFGS-B method
     which supports box constraints.
   
2. Smooth Non-Convex Optimization:
   - The objective is to maximize the minimum area among all 165 triangles.
   - The "min" operation is non-smooth. We approximate it with a LogSumExp (Soft-Min) function.
   - Area of a triangle in barycentric coordinates is proportional to the determinant of the 3x3 matrix of coordinates.
   - Since the area must be positive (for points in general position inside the triangle), 
     we maximize the log of the areas or just the areas directly.
     Actually, area = 0.5 * |det|. Since we are inside, the orientation might flip?
     No, for points inside a convex hull, the relative orientation is consistent if we define it properly.
     Wait, the area of a triangle formed by 3 points P1, P2, P3 in barycentric coords is:
     Area = 0.5 * |det([u1 v1 w1; u2 v2 w2; u3 v3 w3])| * Area_Total.
     Since we maximize, and we want to avoid zero area, we can just maximize the determinant.
     However, the determinant can be negative. We should maximize the absolute value.
     But soft-min of absolute values is tricky.
     
     Alternative: 
     We know that for points inside the triangle, the area is proportional to the magnitude of the cross product in Cartesian coordinates.
     Let's stick to Cartesian for area calculation but Barycentric for parameterization?
     No, let's derive the area in Barycentric coordinates directly.
     
     Let the vertices of the container triangle be A, B, C.
     Point P_i = u_i A + v_i B + w_i C.
     The area of triangle P_i P_j P_k is:
     Area = 0.5 * | (P_j - P_i) x (P_k - P_i) |
     
     In barycentric coordinates relative to A(1,0,0), B(0,1,0), C(0,0,1):
     The determinant D = det([[u_i, v_i, w_i], [u_j, v_j, w_j], [u_k, v_k, w_k]])
     is proportional to the signed area.
     Specifically, Area(P_i P_j P_k) = |D| * Area(ABC).
     
     Since we are maximizing the minimum area, and we expect all areas to be positive in the optimal configuration (no 3 points collinear, and generally "spread out"), we can assume the determinant is positive or just take the absolute value.
     However, taking the absolute value introduces non-differentiability at 0.
     To maintain differentiability for L-BFGS-B, we can maximize the sum of logs of areas? No, we want the MINIMUM.
     
     Let's use the Soft-Min approximation on the raw areas (which are linear in the determinants).
     Actually, the determinant is a multilinear function of the coordinates.
     Let $A_{ijk}$ be the area.
     We want to maximize $t$ such that $A_{ijk} \ge t$.
     This is equivalent to maximizing the Soft-Min of the $A_{ijk}$.
     
     Soft-Min(x) = -1/k * log(sum(exp(-k * x_i)))
     
     This function is smooth and approximates min(x) as k -> infinity.
     
     The area $A_{ijk}$ is $0.5 * | \det(M_{ijk}) | * Area_{total}$.
     Let's define the "signed area" $S_{ijk} = \det(M_{ijk})$.
     We want to maximize the minimum of $|S_{ijk}|$.
     
     If we assume the configuration is such that all determinants have the same sign (which is true for points in convex position or generally if we fix the orientation), we can just maximize the minimum of $S_{ijk}$ or $-S_{ijk}$.
     However, it's safer to handle the absolute value.
     
     Actually, a simpler approach:
     Maximize the minimum of the Cartesian areas.
     We can compute the Cartesian coordinates from barycentric coordinates in the objective function.
     This makes the objective function smooth everywhere except where points coincide (area=0).
     The Soft-Min of the Cartesian areas will be smooth.
     
     Let's use this:
     1. Variables: $2N$ barycentric coordinates $(u_i, v_i)$. $w_i = 1 - u_i - v_i$.
     2. Bounds: $u_i \in [0, 1], v_i \in [0, 1]$, and $u_i + v_i \le 1$ (handled by L-BFGS-B? No, L-BFGS-B only handles box constraints $lb \le x \le ub$).
        So we cannot directly enforce $u+v \le 1$ with L-BFGS-B.
        
     Alternative Parameterization for Barycentric Constraints:
     Use $u_i = \frac{e^{a_i}}{e^{a_i} + e^{b_i} + e^{c_i}}$, etc. (Softmax).
     This ensures $u,v,w > 0$ and sum to 1.
     However, Softmax can be numerically unstable and the gradients can be small.
     Also, it doesn't allow points to be on the boundary (u=0).
     For Heilbronn, optimal points are often on the boundary?
     For n=11, likely some points are on the boundary.
     
     Let's go back to Cartesian coordinates with a penalty for being outside?
     Or, use the box constraint on Cartesian coordinates?
     The triangle is a convex polygon. Box constraints are not sufficient to define a triangle.
     
     Let's try a different approach:
     Use the "Error Repair" idea from the prompt?
     The prompt mentions: "Does operator 'error_repair' change the chance that a child improves its island?"
     And the strategy to try is: "Formulate the problem as a non-convex Linear Program (LP) relaxation... using scipy.optimize.linprog".
     Wait, the directive says: "Strategy to try in this change: Formulate the problem as a non-convex Linear Program (LP) relaxation by maximizing the minimum triangle area t subject to linear constraints... solving it using scipy.optimize.linprog".
     
     BUT, the area of a triangle formed by 3 points is NOT linear in the coordinates. It is quadratic (determinant).
     So an LP is not possible directly on the coordinates.
     However, the prompt suggests "LP relaxation". This is likely a misinterpretation in the prompt's inspiration or a specific trick.
     Actually, if we fix the points, the areas are linear in t? No.
     
     Let's re-read the directive carefully.
     "Strategy to try in this change: Formulate the problem as a non-convex Linear Program (LP) relaxation... using scipy.optimize.linprog"
     
     This is mathematically impossible for the exact problem because the area is non-linear.
     However, maybe it means:
     Maximize t
     Subject to:
     Area(i,j,k) >= t  (Non-linear)
     
     If we approximate Area(i,j,k) as linear? No.
     
     Perhaps the "LP" refers to the fact that if we fix the signs of the determinants, the constraints $S_{ijk} \ge t$ are linear in t, but $S_{ijk}$ is still non-linear in the points.
     
     Let's look at the "Inspiration 1" which scored 28.59. It used "push apart gradient".
     The current program scores 22.92.
     
     Let's stick to the most robust differentiable method:
     **Barycentric Coordinates with Softmax Parameterization + L-BFGS-B + Soft-Min Objective.**
     
     Why Softmax?
     It strictly enforces the simplex constraint $u,v,w > 0, \sum=1$.
     Points will not be exactly on the boundary, but very close.
     For n=11, the optimal solution likely has points on the boundary.
     Softmax might struggle to push points exactly to 0.
     
     Alternative: **Cartesian Coordinates + Penalty Method**.
     Minimize $- \text{SoftMin}(Areas) + \lambda \cdot \text{Penalty}(\text{Outside})$.
     Penalty: Sum of squared distances to the triangle boundaries for any point outside.
     This is a standard unconstrained optimization problem.
     We can use L-BFGS-B (unconstrained or with loose bounds) or just L-BFGS.
     L-BFGS is very fast.
     
     Let's implement:
     1. Parameterize points in Cartesian coordinates $(x, y)$.
     2. Objective: $- \text{SoftMin}(\text{Areas}) + \text{Penalty}$.
     3. SoftMin: $-\frac{1}{k} \log \sum \exp(-k \cdot A_i)$.
     4. Penalty: For each point, if it violates $y < 0$, $y > \sqrt{3}x$, $y > \sqrt{3}(1-x)$, add a quadratic penalty.
     5. Multi-start L-BFGS-B.
     
     This is materially different from Nelder-Mead (derivative-free, simplex based) because it uses gradients (L-BFGS) and a smooth approximation of the non-smooth objective, allowing for much faster convergence per restart.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    x = pts[:, 0]
    y = pts[:, 1]
    
    if np.any(y < -eps): return False
    if np.any(y > np.sqrt(3) * x + eps): return False
    if np.any(y > np.sqrt(3) * (1 - x) + eps): return False
    if np.any(x < -eps): return False
    if np.any(x > 1 + eps): return False
    
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Clip y
    y = np.clip(y, 0, S3H)
    
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    
    x = np.clip(x, x_min, x_max)
    
    pts[:, 0] = x
    pts[:, 1] = y
    return pts

def _soft_min(areas, k=100.0):
    """
    Smooth approximation of the minimum of a set of values.
    SoftMin(x) = -1/k * log(sum(exp(-k * x_i)))
    As k increases, this approaches min(x).
    """
    # Numerical stability: subtract max (which is min of -x)
    # We want min(areas).
    # Let y = -areas. We want -max(y).
    # SoftMax(y) = 1/k * log(sum(exp(k * y_i)))
    # So SoftMin(areas) = -SoftMax(-areas)
    
    neg_areas = -areas
    # Shift for stability
    max_neg = np.max(neg_areas)
    exp_neg = np.exp(k * (neg_areas - max_neg))
    log_sum = np.log(np.sum(exp_neg))
    soft_max = (1.0 / k) * (log_sum + k * max_neg)
    return -soft_max

def _objective(x_flat, k=100.0, penalty_weight=100.0):
    """
    Objective function for minimization.
    Minimize: -SoftMin(Areas) + Penalty
    """
    pts = x_flat.reshape(N, 2)
    
    # Calculate areas
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    # Soft Min of areas
    # If any area is 0, SoftMin is 0.
    # We want to maximize min area, so minimize -SoftMin.
    soft_min_val = _soft_min(areas, k)
    
    # Penalty for being outside the triangle
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Constraints:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x
    # 3. y <= sqrt(3) * (1 - x)
    # 4. x >= 0 (implied by others if y>=0? No, x can be negative if y is small? 
    #    If y=0, x in [0,1]. If y>0, x in [y/sqrt3, 1-y/sqrt3].
    #    So x>=0 is implied by y >= 0 and y <= sqrt(3)x => x >= y/sqrt(3) >= 0.
    #    Similarly x<=1 is implied.
    
    viol_y_bottom = np.maximum(0, -y)
    viol_y_left = np.maximum(0, y - np.sqrt(3) * x)
    viol_y_right = np.maximum(0, y - np.sqrt(3) * (1 - x))
    
    # Quadratic penalty
    penalty = penalty_weight * (np.sum(viol_y_bottom**2) + np.sum(viol_y_left**2) + np.sum(viol_y_right**2))
    
    return -soft_min_val + penalty

def _gradient(x_flat, k=100.0, penalty_weight=100.0):
    """
    Analytical gradient of the objective function.
    """
    pts = x_flat.reshape(N, 2)
    
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    # Gradient of SoftMin
    # SoftMin(areas) = -1/k log(sum(exp(-k * areas)))
    # d/darea_j (SoftMin) = -1/k * ( -k * exp(-k * area_j) / sum(exp(-k * areas)) )
    #                     = exp(-k * area_j) / sum(exp(-k * areas))
    # Let w_j = exp(-k * area_j) / sum(exp(-k * areas))
    # Then d(SoftMin)/d(area_j) = w_j
    # Note: sum(w_j) = 1.
    
    # For numerical stability, we compute weights directly
    neg_areas = -areas
    max_neg = np.max(neg_areas)
    exp_neg = np.exp(k * (neg_areas - max_neg))
    sum_exp = np.sum(exp_neg)
    weights = exp_neg / sum_exp  # These are the weights w_j
    
    # d(SoftMin)/d(pts) = sum_j w_j * d(area_j)/d(pts)
    
    # Area_j = 0.5 * | (P_j - P_i) x (P_k - P_i) |
    # Let's handle the sign of the cross product.
    # sign_j = sign(cross_j)
    # area_j = 0.5 * sign_j * cross_j
    # d(area_j)/d(pts) = 0.5 * sign_j * d(cross_j)/d(pts)
    
    signs = np.sign(cross)
    # If cross is 0, sign is 0, which is a problem. 
    # In practice, areas won't be exactly 0 in the optimization steps unless points coincide.
    # If points coincide, the gradient is undefined, but L-BFGS will handle it or we can add a small epsilon.
    # Let's assume cross != 0.
    
    # d(cross)/d(P_i):
    # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # d/dx_i = -(y_k - y_i) + (y_j - y_i) = y_j - y_k
    # d/dy_i = -(x_k - x_i) + (x_j - x_i) = x_j - x_k
    
    # d(cross)/d(P_j):
    # d/dx_j = y_k - y_i
    # d/dy_j = -(x_k - x_i)
    
    # d(cross)/d(P_k):
    # d/dx_k = -(y_j - y_i)
    # d/dy_k = x_j - x_i
    
    # We need to accumulate the gradient for each point index.
    # grad_pts has shape (N, 2)
    grad_pts = np.zeros((N, 2))
    
    # Weights for each triangle j
    # Contribution to grad from triangle j is:
    # w_j * 0.5 * sign_j * [d(cross_j)/d(P_i), d(cross_j)/d(P_j), d(cross_j)/d(P_k)]
    
    # Vectorized calculation
    # For each triangle j:
    # dP_i = [y_j - y_k, x_j - x_k]
    # dP_j = [y_k - y_i, -(x_k - x_i)]
    # dP_k = [-(y_j - y_i), x_j - x_i]
    
    # Note: indices are _i_idx, _j_idx, _k_idx
    
    # Calculate gradients for P_i part
    dP_i_x = p_j[:, 1] - p_k[:, 1]
    dP_i_y = p_j[:, 0] - p_k[:, 0]
    
    # Calculate gradients for P_j part
    dP_j_x = p_k[:, 1] - p_i[:, 1]
    dP_j_y = -(p_k[:, 0] - p_i[:, 0])
    
    # Calculate gradients for P_k part
    dP_k_x = -(p_j[:, 1] - p_i[:, 1])
    dP_k_y = p_j[:, 0] - p_i[:, 0]
    
    # Factor: 0.5 * sign * weight
    factor = 0.5 * signs * weights
    
    # Accumulate using np.add.at
    # P_i
    np.add.at(grad_pts, (_i_idx, 0), factor * dP_i_x)
    np.add.at(grad_pts, (_i_idx, 1), factor * dP_i_y)
    
    # P_j
    np.add.at(grad_pts, (_j_idx, 0), factor * dP_j_x)
    np.add.at(grad_pts, (_j_idx, 1), factor * dP_j_y)
    
    # P_k
    np.add.at(grad_pts, (_k_idx, 0), factor * dP_k_x)
    np.add.at(grad_pts, (_k_idx, 1), factor * dP_k_y)
    
    # Gradient of Objective = -Gradient(SoftMin) + Gradient(Penalty)
    # d/dx (-SoftMin) = -d(SoftMin)/dx
    grad_obj = -grad_pts.flatten()
    
    # Penalty Gradient
    x = pts[:, 0]
    y = pts[:, 1]
    
    viol_y_bottom = np.maximum(0, -y)
    viol_y_left = np.maximum(0, y - np.sqrt(3) * x)
    viol_y_right = np.maximum(0, y - np.sqrt(3) * (1 - x))
    
    # d/dx (penalty)
    # Penalty = W * (sum v_b^2 + sum v_l^2 + sum v_r^2)
    
    # d/dx v_b = 0
    # d/dx v_l = -sqrt(3) if v_l > 0 else 0
    # d/dx v_r = sqrt(3) if v_r > 0 else 0
    
    grad_x_pen = penalty_weight * (
        -2 * np.sqrt(3) * viol_y_left + 
        2 * np.sqrt(3) * viol_y_right
    )
    
    # d/dy v_b = -1 if v_b > 0 else 0
    # d/dy v_l = 1 if v_l > 0 else 0
    # d/dy v_r = 1 if v_r > 0 else 0
    
    grad_y_pen = penalty_weight * (
        -2 * viol_y_bottom + 
        2 * viol_y_left + 
        2 * viol_y_right
    )
    
    # Combine
    grad_obj[0::2] += grad_x_pen
    grad_obj[1::2] += grad_y_pen
    
    return grad_obj

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Multi-start L-BFGS-B with Soft-Min objective and Penalty constraints.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # Initializations
    initializations = []
    
    # 1. Grid-based initialization (vertices, midpoints, centroid)
    # Vertices
    v0 = [0.0, 0.0]
    v1 = [1.0, 0.0]
    v2 = [0.5, S3H]
    
    # Midpoints
    m01 = [0.5, 0.0]
    m12 = [0.75, S3H/2]
    m20 = [0.25, S3H/2]
    
    # Centroid
    c = [0.5, S3H/3]
    
    # We need 11 points. Let's add some random points near the center and boundaries.
    base_pts = np.array([v0, v1, v2, m01, m12, m20, c])
    
    # Add 4 more points
    for _ in range(4):
        u = rng.random()
        v = rng.random()
        while u + v > 1:
            u = rng.random()
            v = rng.random()
        x = u + 0.5 * v
        y = S3H * v
        base_pts = np.vstack([base_pts, [x, y]])
        
    initializations.append(base_pts)
    
    # 2. Random initializations
    for _ in range(10):
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1:
                u = rng.random()
                v = rng.random()
            x = u + 0.5 * v
            y = S3H * v
            pts[i] = [x, y]
        initializations.append(pts)
        
    # 3. Greedy-like initialization
    def _greedy_init(rng):
        pts = []
        def _random_point_in_triangle(rng):
            u = rng.random()
            v = rng.random()
            while u + v > 1:
                u = rng.random()
                v = rng.random()
            x = u + 0.5 * v
            y = S3H * v
            return np.array([x, y])
            
        p0 = _random_point_in_triangle(rng)
        pts.append(p0)
        
        for _ in range(N - 1):
            best_pos = None
            best_min_dist = -1
            for _ in range(500):
                cand = _random_point_in_triangle(rng)
                dists = np.linalg.norm(pts - cand, axis=1)
                min_d = np.min(dists)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_pos = cand
            if best_pos is not None:
                pts.append(best_pos)
        return np.array(pts)
        
    for _ in range(5):
        initializations.append(_greedy_init(rng))
        
    # Shuffle
    rng.shuffle(initializations)
    
    # Optimization Parameters
    k_soft = 50.0  # Soft-min parameter
    penalty_w = 50.0
    
    for i, init_pts in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        x0 = init_pts.flatten()
        
        try:
            res = minimize(
                _objective, 
                x0, 
                jac=_gradient,
                method='L-BFGS-B', 
                options={
                    'maxiter': 1000,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'maxls': 50
                },
                args=(k_soft, penalty_w)
            )
            
            pts_opt = res.x.reshape(N, 2)
            pts_opt = _project_to_triangle(pts_opt)
            
            if _check_inside(pts_opt):
                score = _min_triangle_area(pts_opt)
                if score > best_score:
                    best_score = score
                    best_pts = pts_opt.copy()
                    
        except Exception as e:
            # Fallback to initial point
            if _check_inside(init_pts):
                score = _min_triangle_area(init_pts)
                if score > best_score:
                    best_score = score
                    best_pts = init_pts.copy()
            continue
            
        # Check time
        if time.time() - start_time > DEADLINE - 10:
            break

    # Fallback
    if best_pts is None:
        for init_pts in initializations:
            if _check_inside(init_pts):
                score = _min_triangle_area(init_pts)
                if score > best_score:
                    best_score = score
                    best_pts = init_pts.copy()
                    
    if best_pts is None:
        best_pts = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
            [0.5, 0.0], [0.25, S3H/2], [0.75, S3H/2],
            [0.5, S3H/3],
            [0.25, 0.0], [0.75, 0.0],
            [0.125, S3H/4], [0.875, S3H/4]
        ])
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
