import numpy as np
from scipy.optimize import linprog
import itertools
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Equilateral triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Area = sqrt(3)/4
    # We work in barycentric-like coordinates or just Cartesian.
    
    # The triangle T is defined by:
    # y >= 0
    # y <= sqrt(3) * x
    # y <= -sqrt(3) * (x - 1)  => y <= sqrt(3) - sqrt(3)*x
    
    # Let's use a change of variables to a square or simplex for easier LP?
    # No, let's stick to Cartesian (x, y) for points.
    # Variables: x_i, y_i for i=0..10, and t (minimum area * 2, i.e., min determinant magnitude)
    # Actually, area = 0.5 * |det|. So 2*area = |det|.
    # We want to maximize min |det|.
    # Let t = min |det|.
    # Constraints: |det(i,j,k)| >= t for all i,j,k.
    # This is non-convex due to absolute value.
    # To make it convex, we fix the sign of the determinant for each triplet.
    # det(i,j,k) >= t  OR  det(i,j,k) <= -t.
    
    # Strategy:
    # 1. Generate a random initial configuration of 11 points.
    # 2. Determine the sign of det(i,j,k) for all triplets.
    # 3. Solve LP: max t subject to:
    #    - sign(i,j,k) * det(i,j,k) >= t
    #    - Points inside triangle.
    # 4. Check if the solution is valid (i.e., the signs didn't change).
    #    If signs changed, we might need to re-fix signs or restart.
    #    Actually, if we fix signs based on current config, and the LP moves points,
    #    the signs might flip. If they flip, the constraint `sign * det >= t` might
    #    be violated or force t to be negative (if t is constrained >= 0).
    #    Wait, if sign=1, constraint is det >= t. If det becomes negative, det >= t (t>0) is violated.
    #    So the LP will find a local optimum within that orientation class.
    #    We can try multiple random initial configurations to find different orientation classes.
    
    # Let's refine:
    # Variables: 2*N + 1 (x_i, y_i, t)
    # Constraints:
    # 1. Inside triangle:
    #    y_i >= 0
    #    y_i - sqrt(3)*x_i <= 0
    #    y_i + sqrt(3)*x_i - sqrt(3) <= 0
    # 2. Area constraints:
    #    For each triplet (i,j,k):
    #    s_ijk * ( (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) ) >= t
    #    where s_ijk is +1 or -1.
    
    # Objective: maximize t => minimize -t.
    
    def get_signs(points):
        signs = {}
        for i, j, k in itertools.combinations(range(N), 3):
            p1, p2, p3 = points[i], points[j], points[k]
            det = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
            signs[(i,j,k)] = 1 if det > 0 else -1
        return signs

    def solve_lp(signs):
        # n_vars = 2*N + 1
        # indices: 0..N-1 are x, N..2N-1 are y, 2N is t
        n_vars = 2 * N + 1
        t_idx = 2 * N
        
        # Objective: minimize -t
        c = np.zeros(n_vars)
        c[t_idx] = -1
        
        # Inequality constraints: A_ub @ x <= b_ub
        A_ub = []
        b_ub = []
        
        # Triangle constraints
        # y_i >= 0  => -y_i <= 0
        for i in range(N):
            row = np.zeros(n_vars)
            row[N + i] = -1
            A_ub.append(row)
            b_ub.append(0)
            
        # y_i - sqrt(3)*x_i <= 0
        for i in range(N):
            row = np.zeros(n_vars)
            row[i] = -np.sqrt(3)
            row[N + i] = 1
            A_ub.append(row)
            b_ub.append(0)
            
        # y_i + sqrt(3)*x_i - sqrt(3) <= 0
        for i in range(N):
            row = np.zeros(n_vars)
            row[i] = np.sqrt(3)
            row[N + i] = 1
            A_ub.append(row)
            b_ub.append(np.sqrt(3))
            
        # Area constraints
        # s * det >= t  =>  -s * det <= -t
        # det = (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i)
        #     = x_j*y_k - x_j*y_i - x_i*y_k + x_i*y_i - y_j*x_k + y_j*x_i + y_i*x_k - y_i*x_i
        # Wait, this is bilinear in x,y!
        # LP cannot handle bilinear terms.
        # Ah, the determinant is linear in the coordinates of each point if the other two are fixed,
        # but here all are variables.
        # (x_j - x_i)*(y_k - y_i) is x_j*y_k - x_j*y_i - x_i*y_k + x_i*y_i.
        # This is not linear in the vector of all variables.
        # So this is NOT a Linear Program. It's a Quadratic Program (QP) or even non-convex.
        
        # My strategy description in the prompt said "transforming ... into a sequence of convex Linear Programs".
        # This is incorrect. The area of a triangle with variable vertices is a quadratic function of the coordinates.
        # Maximizing the minimum of quadratic functions is non-convex.
        
        # So the LP approach is invalid for variable points.
        
        # Let's rethink.
        # If we fix the points, the signs are fixed.
        # But we are optimizing the points.
        
        # Alternative:
        # Use scipy.optimize.minimize with a smooth surrogate or penalty.
        # Or use a local search / simulated annealing as hinted by "Simulated annealing with topological swaps".
        
        # Let's implement a robust local search / simulated annealing.
        # Objective: minimize -min_area.
        # Or maximize min_area.
        
        return None # LP approach failed

    # Let's implement Simulated Annealing / Local Search.
    # 1. Start with a good initial guess (e.g., grid or random).
    # 2. Define a neighborhood: move one point slightly, or swap two points.
    # 3. Evaluate min_area.
    # 4. Accept if better, or with probability exp(-delta/T).
    
    def min_area(points):
        min_a = float('inf')
        for i, j, k in itertools.combinations(range(N), 3):
            p1, p2, p3 = points[i], points[j], points[k]
            cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
            area = 0.5 * abs(cross)
            if area < min_a:
                min_a = area
        return min_a

    def is_inside_triangle(p):
        x, y = p
        if y < -1e-9: return False
        if y > np.sqrt(3) * x + 1e-9: return False
        if y > -np.sqrt(3) * (x - 1) + 1e-9: return False
        return True

    def project_to_triangle(p):
        # Simple projection: if outside, move towards centroid or clamp.
        # Centroid: (0.5, sqrt(3)/6)
        # This is a rough heuristic.
        x, y = p
        # Clamp to bounds
        if y < 0: y = 0
        if y > np.sqrt(3) * x: 
            # Move along normal? Or just reduce y and adjust x?
            # Let's just pull it in.
            # For simplicity, if invalid, return a random valid point near it?
            # Better: use a proper projection or just reject the move.
            pass
        # Rejecting moves is safer.
        return None

    # Let's use a different approach: 
    # Start with a symmetric configuration (e.g., 1 center, 10 around).
    # Then use Nelder-Mead or BFGS on a smooth approximation?
    # Or just random restarts + local refinement.
    
    best_points = None
    best_val = -1
    
    # Initial guess: 1 center, 10 points on a circle or grid.
    # Let's try a grid-like distribution first.
    # Or random.
    
    def random_points():
        pts = []
        for _ in range(N):
            while True:
                u, v = np.random.random(2)
                if u + v < 1:
                    x = u + 0.5 * v
                    y = np.sqrt(3) / 2 * v
                    if is_inside_triangle([x, y]):
                        pts.append([x, y])
                        break
        return np.array(pts)

    def local_refine(points, max_iter=1000):
        # Use Nelder-Mead on -min_area
        from scipy.optimize import minimize
        
        def objective(pts_flat):
            pts = pts_flat.reshape(N, 2)
            # Check validity
            for p in pts:
                if not is_inside_triangle(p):
                    return 1e10 # Penalty
            return -min_area(pts)
        
        try:
            res = minimize(objective, points.flatten(), method='Nelder-Mead', 
                           options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.success and res.fun < 1e9:
                return res.x.reshape(N, 2)
        except:
            pass
        return points

    # Run multiple restarts
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        pts = random_points()
        # Initial local refine
        pts = local_refine(pts, max_iter=200)
        
        val = min_area(pts)
        if val > best_val:
            best_val = val
            best_points = pts.copy()
            
    # Final polish on the best point found
    if best_points is not None:
        best_points = local_refine(best_points, max_iter=1000)
        best_val = min_area(best_points)

    return best_points

# EVOLVE-BLOCK-START
# (The above code is the new implementation)
# EVOLVE-BLOCK-END
