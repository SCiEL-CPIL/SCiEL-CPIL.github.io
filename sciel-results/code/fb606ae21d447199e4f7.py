"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Barycentric coordinate parameterization to enforce the triangle constraint exactly.
2. Formulate as a Constrained Nonlinear Program (NLP):
   Maximize t
   Subject to:
     Area(i, j, k) >= t for all 165 triplets
     Barycentric coordinates >= 0 (points inside triangle)
     Barycentric coordinates sum to 1 (implicitly handled by construction or constraint)
3. Use SciPy's SLSQP optimizer which handles non-linear constraints effectively.
4. Multi-start with diverse initializations (grid-based, random, greedy-like).
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
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

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

def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates to Cartesian coordinates.
    bary: shape (N, 3) where each row sums to 1 and all elements >= 0.
    Vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    Note: Standard barycentric (u, v, w) corresponds to:
    P = u*A + v*B + w*C
    A=(0,0), B=(1,0), C=(0.5, S3H)
    x = v*1 + w*0.5
    y = w*S3H
    Here we use variables b1, b2, b3 corresponding to weights of A, B, C.
    So x = b2 + 0.5*b3, y = S3H*b3.
    """
    # bary shape (N, 3)
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = S3H * bary[:, 2]
    return np.column_stack((x, y))

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    # Barycentric coords: (1-u-v, u, v)
    # Wait, standard mapping:
    # P = (1-u-v)A + uB + vC
    # x = u*1 + v*0.5
    # y = v*S3H
    # Let's just return Cartesian for init, then convert to bary for optimization.
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _cartesian_to_bary(pts):
    """Convert Cartesian points to barycentric coordinates relative to A(0,0), B(1,0), C(0.5, S3H)."""
    # x = b2 + 0.5 b3
    # y = S3H b3
    # => b3 = y / S3H
    # => b2 = x - 0.5 b3
    # => b1 = 1 - b2 - b3
    b3 = pts[:, 1] / S3H
    b2 = pts[:, 0] - 0.5 * b3
    b1 = 1.0 - b2 - b3
    return np.column_stack((b1, b2, b3))

def _objective_and_constraints(x_flat):
    """
    x_flat: shape (N*3 + 1,)
    First N*3 elements are barycentric coordinates (b1, b2, b3) for each point.
    Last element is t.
    
    Objective: Minimize -t (to maximize t)
    Constraints:
    1. For each point i, b1_i + b2_i + b3_i == 1 (Equality)
    2. For each point i, b1_i >= 0, b2_i >= 0, b3_i >= 0 (Inequality)
    3. For each triplet (i,j,k), Area(i,j,k) - t >= 0 (Inequality)
    """
    N_bary = N * 3
    t = x_flat[N_bary]
    bary = x_flat[:N_bary].reshape(N, 3)
    
    # Convert to Cartesian for area calculation
    pts = _barycentric_to_cartesian(bary)
    
    # Calculate areas
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Objective: -t
    obj = -t
    
    # Constraints
    # SLSQP expects a dictionary or a list of constraint dicts.
    # It's more efficient to define them as separate functions or use a single function returning array.
    # Here we use a single function returning an array of inequalities and equalities.
    
    # Equality constraints: sum of barycentric coords for each point is 1
    eq_cons = np.sum(bary, axis=1) - 1.0
    
    # Inequality constraints:
    # 1. Barycentric coords >= 0
    ineq_bary = bary.flatten()
    
    # 2. Area >= t  =>  Area - t >= 0
    ineq_areas = areas - t
    
    # Combine all inequalities into one array
    ineq_all = np.concatenate([ineq_bary, ineq_areas])
    
    return obj, eq_cons, ineq_all

def _obj_func(x_flat):
    obj, _, _ = _objective_and_constraints(x_flat)
    return obj

def _jac_func(x_flat):
    """
    Jacobian of the objective and constraints.
    This is complex to compute analytically for SLSQP if we pass a single function.
    SLSQP can handle finite differences, but it's slow.
    However, implementing the full Jacobian for 165 area constraints + 33 bary constraints is tedious.
    Let's try to provide gradients for the objective and constraints separately if possible,
    or rely on SLSQP's internal FD if performance allows.
    
    Given the time limit, let's first try with finite differences.
    If that's too slow, we'll need to implement gradients.
    N=11, vars = 34. FD cost is ~34 function evals per gradient step.
    Each function eval involves 165 area calcs.
    This should be fast enough in NumPy.
    """
    return None # Let SLSQP use FD

def _construct_constraints():
    """Construct the constraint dictionaries for SLSQP."""
    # We define the constraints as a list of dictionaries.
    # type 'eq' or 'ineq'
    # fun: function returning array
    # jac: function returning array (optional)
    
    # To use the combined array approach, we can define one 'eq' and one 'ineq' constraint.
    
    def eq_func(x):
        _, eq, _ = _objective_and_constraints(x)
        return eq
        
    def ineq_func(x):
        _, _, ineq = _objective_and_constraints(x)
        return ineq
        
    # SLSQP constraints format:
    # {'type': 'eq', 'fun': ..., 'jac': ...}
    # {'type': 'ineq', 'fun': ..., 'jac': ...}
    
    cons = [
        {'type': 'eq', 'fun': eq_func},
        {'type': 'ineq', 'fun': ineq_func}
    ]
    return cons

def _greedy_init(rng):
    """
    Greedy initialization: place points one by one to maximize the minimum distance to existing points.
    This provides a decent starting point for the NLP.
    """
    pts = []
    # Start with a vertex or center? Center is safer.
    # Let's start with a random point, then add others greedily.
    p0 = _random_point_in_triangle(rng)
    pts.append(np.array(p0))
    
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        # Sample many candidates
        for _ in range(500):
            cand = np.array(_random_point_in_triangle(rng))
            # Calculate min distance to existing points
            dists = np.linalg.norm(pts - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        pts.append(best_pos)
    
    return np.array(pts)

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses SLSQP with barycentric coordinates and auxiliary variable t.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # Multi-start optimization
    # We will try a few different initializations:
    # 1. Greedy
    # 2. Random
    # 3. Grid-like (vertices + midpoints + center)
    
    initializations = []
    
    # 1. Greedy
    try:
        p_greedy = _greedy_init(rng)
        initializations.append(p_greedy)
    except Exception:
        pass
    
    # 2. Random
    p_rand = np.array([_random_point_in_triangle(rng) for _ in range(N)])
    p_rand = _project_to_triangle(p_rand)
    initializations.append(p_rand)
    
    # 3. Structured: Vertices, Midpoints, Center
    # Vertices: (0,0), (1,0), (0.5, S3H)
    # Midpoints: (0.5,0), (0.75, S3H/2), (0.25, S3H/2)
    # Center: (0.5, S3H/3)
    # Need 11 points.
    # Let's add some more structured points.
    p_struct = [
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.5, 0.0], [0.75, S3H/2], [0.25, S3H/2],
        [0.5, S3H/3],
        [0.25, 0.0], [0.75, 0.0],
        [0.125, S3H/4], [0.875, S3H/4]
    ]
    p_struct = np.array(p_struct)
    p_struct = _project_to_triangle(p_struct)
    initializations.append(p_struct)
    
    cons = _construct_constraints()
    
    for s, init_pts in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Convert to barycentric
        bary = _cartesian_to_bary(init_pts)
        
        # Ensure barycentric coords are valid (sum to 1, >= 0)
        # The conversion should handle sum=1 if points are in triangle.
        # But due to floating point, might be slightly off.
        # SLSQP will handle the equality constraint.
        # However, initial guess should be close to feasible.
        # Let's normalize barycentric coords to sum to 1 exactly.
        bary = bary / np.sum(bary, axis=1, keepdims=True)
        
        # Initial t: estimate min area
        t_init = _min_triangle_area(init_pts) * 0.9 # Slightly lower to be safe
        
        x0 = np.concatenate([bary.flatten(), [t_init]])
        
        # Bounds for SLSQP
        # Barycentric coords: [0, 1]
        # t: [0, 1] (area is at most 1/2 * base * height = 1/2 * 1 * S3H = S3H/2 approx 0.433, but t is area, so max is ~0.433)
        # Actually, max area of a triangle inside unit area triangle is 1/4? No.
        # The area of the container is S3H/2.
        # Max area of a sub-triangle is at most S3H/2.
        # Let's set upper bound for t to 1.0 for safety.
        
        bounds = []
        for _ in range(N):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 1))
        bounds.append((0, 1)) # t
        
        try:
            result = minimize(
                _obj_func,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            
            if result.success:
                x_opt = result.x
                bary_opt = x_opt[:N*3].reshape(N, 3)
                # t_opt = x_opt[N*3]
                
                pts_opt = _barycentric_to_cartesian(bary_opt)
                pts_opt = _project_to_triangle(pts_opt)
                
                if _check_inside(pts_opt):
                    score = _min_triangle_area(pts_opt)
                    if score > best_score:
                        best_score = score
                        best_pts = pts_opt.copy()
                        
        except Exception as e:
            # print(f"Optimization failed for start {s}: {e}")
            pass
            
    # If no optimization succeeded, fall back to the best initial guess
    if best_pts is None:
        # Pick the best from initializations
        for init_pts in initializations:
            score = _min_triangle_area(init_pts)
            if score > best_score:
                best_score = score
                best_pts = init_pts.copy()
                
    if best_pts is None:
        best_pts = _project_to_triangle(np.array([_random_point_in_triangle(rng) for _ in range(N)]))
        
    # Final check
    if not _check_inside(best_pts):
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
