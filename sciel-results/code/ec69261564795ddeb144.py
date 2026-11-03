import numpy as np
import time
from scipy.optimize import basinhopping, minimize

N = 11
H = np.sqrt(3) / 2

def min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    points: numpy array of shape (N, 2)
    """
    min_a = float('inf')
    n = len(points)
    # Vectorized calculation for speed
    # We need all combinations of 3 points
    # i, j, k
    # Area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) |
    
    # To be fast, we can loop over i and vectorize j, k
    for i in range(n):
        # Vectors from i to all other points
        # But we need pairs (j, k) where j, k > i
        # This is tricky to fully vectorize without generating indices.
        # Given N=11, N choose 3 = 165 triangles.
        # Simple loops are fine if optimized, or we can precompute.
        pass
        
    # Let's stick to the efficient loop but optimized
    # Actually, let's just use the triple loop but with numpy arrays for the points
    # It's only 165 evaluations per objective call.
    min_a = 1e9
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                v2 = pk - pi
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_a:
                    min_a = area
    return min_a

def objective(x):
    """
    Objective function for scipy.optimize.
    x: 22D array (11 points)
    Returns: -min_area (we minimize negative area to maximize area)
             + penalty if points are outside
    """
    points = x.reshape(N, 2)
    
    # Check if all points are inside the triangle
    # Triangle vertices: A(0,0), B(1,0), C(0.5, H)
    # Barycentric coordinates or half-plane tests.
    # Half-plane tests are faster.
    # Line AB: y >= 0
    # Line BC: y <= -sqrt(3)*(x - 1) => sqrt(3)*x + y <= sqrt(3)
    # Line CA: y <= sqrt(3)*x
    
    # Points must satisfy:
    # 1. y >= -eps
    # 2. sqrt(3)*x + y <= sqrt(3) + eps
    # 3. y <= sqrt(3)*x + eps
    
    eps = 1e-9
    ys = points[:, 1]
    xs = points[:, 0]
    
    # Condition 1
    if np.any(ys < -eps):
        return 1e6 - min_triangle_area(points) # Large penalty
    
    # Condition 2
    # sqrt(3)*x + y - sqrt(3) <= 0
    cond2 = np.sqrt(3) * xs + ys - np.sqrt(3)
    if np.any(cond2 > eps):
        return 1e6 - min_triangle_area(points)
        
    # Condition 3
    # y - sqrt(3)*x <= 0
    cond3 = ys - np.sqrt(3) * xs
    if np.any(cond3 > eps):
        return 1e6 - min_triangle_area(points)
        
    area = min_triangle_area(points)
    return -area

def generate_initial_points():
    """
    Generate diverse initial configurations.
    """
    inits = []
    
    # 1. Grid-based initialization (K=10 lattice, pick 11)
    # We'll just use a few random subsets of a fine grid
    k = 12
    lattice = []
    for j in range(k):
        for i in range(k - j):
            x = (i + 0.5 * j) / (k - 1)
            y = j * H / (k - 1)
            lattice.append([x, y])
    lattice = np.array(lattice)
    
    # Random subsets
    for _ in range(5):
        idx = np.random.choice(len(lattice), N, replace=False)
        inits.append(lattice[idx].flatten())
        
    # 2. Center + Rings
    # Center
    center = np.array([0.5, H/3])
    # Ring 1: 3 points
    # Ring 2: 7 points?
    # Let's just use a symmetric pattern
    # 1 center, 3 inner, 7 outer?
    # Simpler: Random points
    for _ in range(10):
        # Random points in triangle
        r1 = np.random.rand(N)
        r2 = np.random.rand(N)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        xs = r1 + 0.5 * r2
        ys = r2 * H
        inits.append(np.column_stack((xs, ys)).flatten())
        
    return inits

def construct():
    start_time = time.time()
    time_limit = 115
    
    best_points_global = None
    best_score_global = -float('inf') # We maximize area, so we store max area
    
    inits = generate_initial_points()
    
    # We will run Basin-Hopping on each initial point
    # Basin-Hopping parameters
    # stepsize: global move size
    # niter: number of iterations per basin (local opt)
    # miniter: minimum iterations
    
    # Since time is limited, we distribute time among inits.
    # Each init gets a chance to run basin hopping.
    # We can run them in parallel? No, single thread.
    # We'll run a few for a fixed number of iterations or until time runs out.
    
    # Let's define a callback to check time
    def time_check():
        return time.time() - start_time > time_limit
    
    # We'll iterate through inits.
    # For each init, run basin hopping.
    # To be safe, we limit the number of basin hopping iterations per start.
    
    for i_init, x0 in enumerate(inits):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Define local optimizer
        # Nelder-Mead is good for this small dimension
        local_min = lambda x: minimize(objective, x, method='Nelder-Mead', 
                                       options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000, 'adaptive': True})
        
        # Basin Hopping
        try:
            res = basinhopping(objective, x0, minimizer_kwargs=dict(method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500, 'adaptive': True}),
                               niter=50, stepsize=0.5, T=1.0, accept_test=None, 
                               callback=lambda x, f, accepted, stepaccept, info: time_check())
            
            if not time_check():
                # res.x is the best found
                x_best = res.x
                points_best = x_best.reshape(N, 2)
                
                # Verify it's inside (basinhopping might drift slightly due to floating point)
                # And calculate the score
                # Note: objective returns -area. So area = -objective(x_best)
                # But if penalty was applied, objective > 0.
                score = -res.fun
                if res.fun > 0: # Penalty was applied, invalid
                    score = -1e6
                    
                if score > best_score_global:
                    best_score_global = score
                    best_points_global = points_best.copy()
                    
        except Exception as e:
            # If basin hopping fails, try a simple local minimization
            try:
                res_local = minimize(objective, x0, method='Nelder-Mead', 
                                     options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 2000, 'adaptive': True})
                if not time_check():
                    x_best = res_local.x
                    points_best = x_best.reshape(N, 2)
                    score = -res_local.fun
                    if res_local.fun > 0:
                        score = -1e6
                    if score > best_score_global:
                        best_score_global = score
                        best_points_global = points_best.copy()
            except:
                pass
                
    # If no valid points found, return a default valid configuration (e.g., grid)
    if best_points_global is None:
        # Fallback: simple grid
        k = 12
        lattice = []
        for j in range(k):
            for i in range(k - j):
                x = (i + 0.5 * j) / (k - 1)
                y = j * H / (k - 1)
                lattice.append([x, y])
        lattice = np.array(lattice)
        idx = np.random.choice(len(lattice), N, replace=False)
        best_points_global = lattice[idx]
        
    return best_points_global
