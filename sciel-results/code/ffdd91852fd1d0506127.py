import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

N = 11
h = np.sqrt(3) / 2
# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, h])

# Precompute all triplets for vectorized area computation
_TRIPLETS = np.array(list(itertools.combinations(range(N), 3)), dtype=np.int32)

def barycentric_to_cartesian(bary):
    """
    bary: array of shape (N, 3) where each row sums to 1 and is non-negative.
    Returns Cartesian coordinates.
    """
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = h * bary[:, 2]
    return np.column_stack((x, y))

def min_triangle_area(points):
    """
    points: (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    """
    p1 = points[_TRIPLETS[:, 0]]
    p2 = points[_TRIPLETS[:, 1]]
    p3 = points[_TRIPLETS[:, 2]]
    
    v12 = p2 - p1
    v13 = p3 - p1
    
    cross = v12[:, 0] * v13[:, 1] - v12[:, 1] * v13[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def soft_min_surrogate(points, beta=50.0):
    """
    Computes a smooth approximation of the min triangle area using log-sum-exp.
    Minimizing -soft_min is equivalent to maximizing the soft_min.
    """
    p1 = points[_TRIPLETS[:, 0]]
    p2 = points[_TRIPLETS[:, 1]]
    p3 = points[_TRIPLETS[:, 2]]
    
    v12 = p2 - p1
    v13 = p3 - p1
    
    cross = v12[:, 0] * v13[:, 1] - v12[:, 1] * v13[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp for stability: log(sum(exp(x))) = max_x + log(sum(exp(x - max_x)))
    # We want min(areas). 
    # min(a) = -log(sum(exp(-beta * a))) / beta
    # Let's maximize this quantity.
    # objective = - min(areas)
    # soft_obj = log(sum(exp(-beta * areas))) / beta
    
    max_neg = np.max(-beta * areas)
    if max_neg < -700: # Avoid underflow
        return 0.0
    sum_exp = np.sum(np.exp(-beta * areas - max_neg))
    soft_min_val = -(max_neg + np.log(sum_exp)) / beta
    return soft_min_val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer for 150s limit
    
    # We work in Cartesian coordinates directly for simplicity in DE bounds,
    # but we must ensure points stay inside the triangle.
    # DE with bounds is tricky for polygon constraints.
    # Instead, we use Barycentric coordinates for the state vector.
    # State: B (N, 3) flattened.
    # Constraints: B >= 0, sum(B_row) = 1.
    # DE doesn't handle equality constraints well directly.
    # We can fix one barycentric coordinate per point (e.g., b0 = 1 - b1 - b2)
    # and let DE optimize b1, b2 for each point.
    # Bounds for b1, b2: 
    # b1 >= 0, b2 >= 0, b1 + b2 <= 1.
    # This is a simplex constraint.
    # We can map DE's hypercube [0,1] to the simplex using the "stick-breaking" method or similar,
    # or just use DE on the 2N dimensions and project/penalize.
    # Projecting is better.
    
    # Let's use a transformation:
    # u, v in [0, 1]
    # r = sqrt(u)
    # theta = 2 * pi * v
    # This maps to a circle, not a triangle.
    
    # Simpler: Use DE on 2N variables representing (b1, b2) for each point.
    # If b1 < 0 or b2 < 0 or b1+b2 > 1, we project to the nearest point in the simplex.
    # Projection onto simplex {x >= 0, sum x = 1} is easy, but here the constraint is on (b1, b2)
    # such that b1>=0, b2>=0, b1+b2<=1.
    # This is the 2D simplex.
    # Projection:
    # If b1 < 0 or b2 < 0 or b1+b2 > 1:
    #   Find closest point in triangle (0,0), (1,0), (0,1) in b1-b2 space.
    
    def project_to_simplex_2d(b1, b2):
        # b1, b2 are arrays of shape (N,)
        b1 = b1.copy()
        b2 = b2.copy()
        
        # Check constraints
        # 1. b1 >= 0
        mask1 = b1 < 0
        b1[mask1] = 0
        # 2. b2 >= 0
        mask2 = b2 < 0
        b2[mask2] = 0
        # 3. b1 + b2 <= 1
        mask3 = b1 + b2 > 1
        if np.any(mask3):
            # For points violating b1+b2 <= 1, we need to project onto b1+b2=1
            # The projection of (b1, b2) onto line b1+b2=1 is not simply scaling
            # because the line has slope -1.
            # Distance squared to line x+y=1: (x+y-1)^2 / 2
            # Gradient is (1,1).
            # Closest point on line x+y=1 to (x,y) is (x + (1-x-y)/2, y + (1-x-y)/2)
            diff = 1 - (b1[mask3] + b2[mask3])
            b1[mask3] += 0.5 * diff
            b2[mask3] += 0.5 * diff
            
            # After this, b1 or b2 might become negative again?
            # If we project (0.8, 0.8) to sum=1 -> (0.5, 0.5). OK.
            # If we project (-0.1, 0.5) -> b1 was clipped to 0 first?
            # Let's do iterative projection or just clip again.
            # Clip again to be safe
            b1[b1 < 0] = 0
            b2[b2 < 0] = 0
            # Re-check sum
            mask4 = b1 + b2 > 1
            if np.any(mask4):
                diff = 1 - (b1[mask4] + b2[mask4])
                b1[mask4] += 0.5 * diff
                b2[mask4] += 0.5 * diff
                b1[b1 < 0] = 0
                b2[b2 < 0] = 0
        
        return b1, b2

    def objective_flat(x):
        """
        x: array of shape (2*N,) containing [b1_0, b2_0, b1_1, b2_1, ...]
        """
        b1 = x[0::2]
        b2 = x[1::2]
        
        # Project to valid simplex
        b1, b2 = project_to_simplex_2d(b1, b2)
        
        # Construct full barycentric coords
        b0 = 1.0 - b1 - b2
        bary = np.column_stack((b0, b1, b2))
        
        # Convert to Cartesian
        pts = barycentric_to_cartesian(bary)
        
        # Use soft min for smoothness during DE
        return -soft_min_surrogate(pts, beta=50.0)

    # Define bounds for DE
    # b1, b2 can technically be anything, but we bound them to [-0.5, 1.5] to cover the simplex
    # with some margin for the projection logic to work effectively.
    bounds = [(-0.5, 1.5)] * (2 * N)
    
    # Run Differential Evolution
    # maxiter: limit by time.
    # popsize: 20-50 is typical.
    # tol: 1e-6
    
    best_x = None
    best_val = float('inf')
    
    try:
        # DE is stochastic, so we run it for a set number of iterations or until time limit
        # We can't easily pass a time limit to DE directly, so we estimate.
        # Let's run for 200 iterations. Each iteration evaluates popsize * (2N) points.
        # Popsize 30 -> 30 * 22 = 660 evals per iter.
        # 200 iters -> 132,000 evals.
        # Each eval is vectorized area calc for 165 triangles. Very fast.
        # Should fit in 10-20s.
        
        res = differential_evolution(
            objective_flat, 
            bounds, 
            maxiter=300, 
            popsize=30, 
            tol=1e-8, 
            seed=42,
            polish=False # We will do our own polish
        )
        
        if res.success or res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to random if DE fails
        best_x = np.random.uniform(-0.5, 1.5, 2*N)
        best_val = objective_flat(best_x)

    # Convert best_x to barycentric and Cartesian
    b1 = best_x[0::2]
    b2 = best_x[1::2]
    b1, b2 = project_to_simplex_2d(b1, b2)
    b0 = 1.0 - b1 - b2
    best_bary = np.column_stack((b0, b1, b2))
    best_pts = barycentric_to_cartesian(best_bary)
    
    # Check time
    if time.time() > deadline - 5.0:
        return best_pts

    # Local Polish with Nelder-Mead
    # Objective: Minimize -min_triangle_area (exact)
    def objective_exact_flat(x):
        b1 = x[0::2]
        b2 = x[1::2]
        b1, b2 = project_to_simplex_2d(b1, b2)
        b0 = 1.0 - b1 - b2
        bary = np.column_stack((b0, b1, b2))
        pts = barycentric_to_cartesian(bary)
        return -min_triangle_area(pts)

    try:
        # Nelder-Mead is robust for small N
        res_nm = minimize(
            objective_exact_flat, 
            best_x, 
            method='Nelder-Mead', 
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if res_nm.fun < best_val:
            best_x = res_nm.x
            best_val = res_nm.fun
            
    except Exception as e:
        print(f"NM failed: {e}")

    # Final conversion
    b1 = best_x[0::2]
    b2 = best_x[1::2]
    b1, b2 = project_to_simplex_2d(b1, b2)
    b0 = 1.0 - b1 - b2
    final_bary = np.column_stack((b0, b1, b2))
    final_pts = barycentric_to_cartesian(final_bary)
    
    # Sanity check: ensure points are inside
    # The projection ensures b1>=0, b2>=0, b1+b2<=1, which implies inside the triangle.
    
    return final_pts

# EVOLVE-BLOCK-END
