"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def _triangle_area(p1, p2, p3):
    """Calculate area of triangle formed by 3 points."""
    # 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Calculate the minimum area among all triangles formed by the points."""
    n = len(points)
    min_area = float('inf')
    # Iterate through all combinations of 3 points
    # Using itertools.combinations is cleaner but might be slower in tight loops if not careful.
    # For N=11, there are 165 triangles. A triple loop is fine.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _inside_triangle(p):
    """Check if a point is inside the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    # y >= 0
    if y < -1e-9:
        return False
    # y <= sqrt(3) * x
    if y > np.sqrt(3) * x + 1e-9:
        return False
    # y <= sqrt(3) * (1 - x)
    if y > np.sqrt(3) * (1 - x) + 1e-9:
        return False
    return True

def _objective(x_flat):
    """
    Objective function for minimization.
    We want to maximize the minimum area, so we minimize the negative minimum area.
    We also add a large penalty if any point is outside the triangle to guide the optimizer,
    although constraints will handle this strictly.
    """
    points = x_flat.reshape(-1, 2)
    
    # Check if all points are inside
    # If not, return a large penalty to push them back in.
    # This helps with DE which might sample outside bounds if bounds are loose, 
    # but we will use strict bounds in DE.
    # For SLSQP, we use constraints.
    
    min_area = _min_triangle_area(points)
    return -min_area

def _constraints(x_flat):
    """
    Define constraints for SLSQP.
    Returns a list of constraint dicts.
    """
    points = x_flat.reshape(-1, 2)
    cons = []
    
    # Inequality constraints: g(x) >= 0
    # 1. y_i >= 0
    for i in range(N):
        def c_y_nonneg(x, i=i):
            return x[2*i + 1]
        cons.append({'type': 'ineq', 'fun': c_y_nonneg})
        
    # 2. y_i - sqrt(3)*x_i <= 0  =>  sqrt(3)*x_i - y_i >= 0
    for i in range(N):
        def c_left_edge(x, i=i):
            return np.sqrt(3) * x[2*i] - x[2*i + 1]
        cons.append({'type': 'ineq', 'fun': c_left_edge})
        
    # 3. y_i - sqrt(3)*(1-x_i) <= 0 => sqrt(3)*(1-x_i) - y_i >= 0
    for i in range(N):
        def c_right_edge(x, i=i):
            return np.sqrt(3) * (1 - x[2*i]) - x[2*i + 1]
        cons.append({'type': 'ineq', 'fun': c_right_edge})
            
    return cons

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Bounds for DE: x in [0, 1], y in [0, sqrt(3)/2]
    # The triangle is within this box.
    bounds = [(0, 1), (0, np.sqrt(3)/2)] * N
    
    # Initial population strategy: 
    # Use a few random starts and maybe a grid start.
    # DE will handle the rest.
    
    best_result = None
    best_score = -np.inf
    
    # 1. Differential Evolution
    # maxiter: number of iterations. 
    # popsize: size of population.
    # We need to be careful with time.
    # Let's try a moderate number of iterations first.
    
    try:
        # Use a fixed seed for reproducibility in the construct function? 
        # The task says "construct() -> array". It doesn't explicitly forbid randomness, 
        # but deterministic is safer for evaluation consistency if seeds aren't managed.
        # However, DE is stochastic. We can set seed in the function.
        
        # We'll run DE with a time limit check inside the callback if possible, 
        # but DE doesn't support easy early stopping via time in scipy directly.
        # We can estimate time per iteration.
        # Alternatively, just run a fixed number of iterations that fits in time.
        
        # Let's estimate: 165 area calcs per eval. 
        # DE with popsize=15, maxiter=50. 15*50 = 750 evals. 
        # Each eval is ~165 * (few ops). Very fast.
        # So we can afford more.
        
        res = differential_evolution(
            _objective, 
            bounds, 
            seed=42,
            maxiter=100, 
            popsize=20, 
            tol=1e-6,
            init='sobol' # More uniform initialization
        )
        
        if time.time() > deadline:
            return res.x.reshape(-1, 2)
            
        best_result = res.x
        best_score = -res.fun
        
    except Exception as e:
        # Fallback to random if DE fails
        rng = np.random.default_rng(42)
        points = []
        while len(points) < N:
            u = rng.random()
            v = rng.random()
            if u + v < 1:
                x = u + 0.5 * v
                y = (np.sqrt(3) / 2) * v
                points.append([x, y])
        best_result = np.array(points).flatten()
        best_score = -_min_triangle_area(best_result.reshape(-1, 2))
        
    # 2. SLSQP Polish
    # Use the best result from DE as the initial guess for SLSQP.
    # SLSQP is a local optimizer, so it will refine the DE solution.
    
    try:
        # We might want to run SLSQP multiple times from slightly perturbed versions of the best DE result
        # to avoid getting stuck in a local minimum if DE didn't find the global one perfectly.
        
        local_candidates = [best_result]
        
        # Add some random perturbations around the best DE result
        rng = np.random.default_rng(42)
        for _ in range(5):
            perturbed = best_result + rng.normal(0, 0.01, size=best_result.shape)
            # Clip to bounds roughly to help SLSQP
            perturbed[0::2] = np.clip(perturbed[0::2], 0, 1)
            perturbed[1::2] = np.clip(perturbed[1::2], 0, np.sqrt(3)/2)
            local_candidates.append(perturbed)
            
        # Also add a few random starts for SLSQP
        for _ in range(5):
            pts = []
            while len(pts) < N:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    x = u + 0.5 * v
                    y = (np.sqrt(3) / 2) * v
                    pts.append([x, y])
            local_candidates.append(np.array(pts).flatten())
            
        for x0 in local_candidates:
            if time.time() > deadline - 10: # Leave 10s for last polish
                break
                
            res_local = minimize(
                _objective, 
                x0, 
                method='SLSQP', 
                constraints=_constraints,
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            
            if res_local.success and res_local.fun < best_score:
                best_score = res_local.fun
                best_result = res_local.x
                
    except Exception:
        pass
        
    # Final check: ensure points are strictly inside. 
    # SLSQP might land on the boundary. The problem says "on or inside". 
    # However, if a point is exactly on the boundary, area might be 0 if 3 points are collinear on the edge.
    # SLSQP should avoid collinearities if the objective penalizes small areas.
    
    return best_result.reshape(-1, 2)

# EVOLVE-BLOCK-END
