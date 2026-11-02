import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import itertools

N = 26

# EVOLVE-BLOCK-START

def _hex_initial(n, width=1.0, height=1.0, margin=0.05):
    """
    Generate initial positions for n circles in a hexagonal pattern.
    """
    # Estimate grid size
    # For hex packing, density is ~0.9069
    # Area per circle ~ pi * r^2. 
    # We want to fit n circles in area 1.
    # Approximate r ~ sqrt(1 / (n * 1.1))
    r_est = np.sqrt(1.0 / (n * 1.1))
    
    # Hex lattice parameters
    a = 2.0 * r_est
    dx = a
    dy = a * np.sqrt(3) / 2
    
    points = []
    row = 0
    while len(points) < n * 2: # Generate extra to fill
        for col in range(int(width / dx) + 2):
            x = col * dx + (row % 2) * (dx / 2) + margin
            y = row * dy + margin
            if 0 <= x <= width and 0 <= y <= height:
                points.append((x, y))
        row += 1
        
    # If we don't have enough, add random points or expand
    # For N=26, this should be plenty.
    if len(points) < n:
        # Fallback to grid if hex fails (unlikely)
        side = int(np.ceil(np.sqrt(n)))
        for i in range(side):
            for j in range(side):
                points.append(((i + 0.5) / side, (j + 0.5) / side))
                
    # Select best n points (closest to center or just first n)
    # Shuffling helps avoid symmetry traps
    np.random.seed(42)
    np.random.shuffle(points)
    points = np.array(points[:n])
    
    # Initial radii
    radii = np.full(n, r_est * 0.9)
    
    return np.hstack([points, radii.reshape(-1, 1)])

def _get_delaunay_neighbors(tri):
    """
    Get unique neighbor pairs from Delaunay triangulation.
    """
    neighbors = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                idx1, idx2 = simplex[i], simplex[j]
                pair = (min(idx1, idx2), max(idx1, idx2))
                neighbors.add(pair)
    return list(neighbors)

def _validate(circles):
    """
    Check if circles are inside [0,1]^2 and disjoint.
    Returns True if valid.
    """
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Boundary check
    if np.any(x - r < -1e-6) or np.any(x + r > 1 + 1e-6):
        return False
    if np.any(y - r < -1e-6) or np.any(y + r > 1 + 1e-6):
        return False
    if np.any(r < 0):
        return False
        
    # Overlap check
    n = len(circles)
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dist < r[i] + r[j] - 1e-6:
                return False
    return True

def _score(circles):
    """
    Calculate the score: sum of radii if valid, 0 otherwise.
    """
    if _validate(circles):
        return np.sum(circles[:, 2])
    return 0.0

def _slsqp_optimize(circles, neighbors, max_iter=200):
    """
    Optimize radii and positions using SLSQP given a fixed set of neighbors.
    Variables: x, y, r for each circle.
    """
    n = len(circles)
    x0 = circles.flatten()
    
    def objective(vars):
        # Maximize sum of radii -> Minimize negative sum
        r = vars[2::3]
        return -np.sum(r)
        
    def gradient(vars):
        # Gradient of -sum(r)
        grad = np.zeros_like(vars)
        grad[2::3] = -1.0
        return grad

    # Constraints:
    # 1. Boundary constraints for each circle i:
    #    x_i - r_i >= 0
    #    1 - x_i - r_i >= 0
    #    y_i - r_i >= 0
    #    1 - y_i - r_i >= 0
    # 2. Overlap constraints for neighbors (i, j):
    #    (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= epsilon
    #    Note: SLSQP requires g(vars) >= 0
    
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        # x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda vars, i=i: vars[i] - vars[2 + 3*i],
            'jac': lambda vars, i=i: np.concatenate([np.zeros(3*i), [1], np.zeros(2*i), [-1], np.zeros(3*(n-1-i))])
        })
        # 1 - x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda vars, i=i: 1 - vars[i] - vars[2 + 3*i],
            'jac': lambda vars, i=i: np.concatenate([np.zeros(3*i), [-1], np.zeros(2*i), [-1], np.zeros(3*(n-1-i))])
        })
        # y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda vars, i=i: vars[i+1] - vars[2 + 3*i],
            'jac': lambda vars, i=i: np.concatenate([np.zeros(3*i), [0, 1], np.zeros(2*i), [-1], np.zeros(3*(n-1-i))])
        })
        # 1 - y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda vars, i=i: 1 - vars[i+1] - vars[2 + 3*i],
            'jac': lambda vars, i=i: np.concatenate([np.zeros(3*i), [0, -1], np.zeros(2*i), [-1], np.zeros(3*(n-1-i))])
        })
        
    # Neighbor overlap constraints
    eps = 1e-8
    for i, j in neighbors:
        def dist_sq(vars, i=i, j=j):
            xi, yi = vars[i], vars[i+1]
            xj, yj = vars[j], vars[j+1]
            return (xi - xj)**2 + (yi - yj)**2
            
        def rad_sum_sq(vars, i=i, j=j):
            ri, rj = vars[2+3*i], vars[2+3*j]
            return (ri + rj)**2
            
        def constraint_fun(vars, i=i, j=j):
            return dist_sq(vars) - rad_sum_sq(vars) - eps
            
        def constraint_jac(vars, i=i, j=j):
            xi, yi = vars[i], vars[i+1]
            xj, yj = vars[j], vars[j+1]
            ri, rj = vars[2+3*i], vars[2+3*j]
            
            jac = np.zeros_like(vars)
            # d/dxi (xi-xj)^2 = 2(xi-xj)
            jac[i] = 2 * (xi - xj)
            # d/dyi (yi-yj)^2 = 2(yi-yj)
            jac[i+1] = 2 * (yi - yj)
            # d/dxj (xi-xj)^2 = -2(xi-xj)
            jac[j] = -2 * (xi - xj)
            # d/dyj (yi-yj)^2 = -2(yi-yj)
            jac[j+1] = -2 * (yi - yj)
            # d/dri (ri+rj)^2 = 2(ri+rj)
            jac[2+3*i] = -2 * (ri + rj)
            # d/drj (ri+rj)^2 = 2(ri+rj)
            jac[2+3*j] = -2 * (ri + rj)
            return jac
            
        constraints.append({
            'type': 'ineq',
            'fun': constraint_fun,
            'jac': constraint_jac
        })
        
    # Bounds: x, y in [0,1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((1e-6, 0.5)) # r

    try:
        res = minimize(
            objective, 
            x0, 
            jac=gradient, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-9}
        )
        if res.success:
            return res.x.reshape(n, 3)
        else:
            # If SLSQP fails, return original or slightly modified
            return circles
    except Exception:
        return circles

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # 1. Initialization
    circles = _hex_initial(N)
    
    # Ensure initial validity (shrink radii if necessary)
    for _ in range(10):
        if _validate(circles):
            break
        circles[:, 2] *= 0.95
        
    best_circles = circles.copy()
    best_score = _score(circles)
    
    # 2. Iterative Refinement with Delaunay
    # We iterate: Compute Delaunay -> SLSQP Optimize -> Check Time
    # SLSQP is expensive, so we limit iterations.
    
    for iteration in range(20):
        if time.time() > deadline:
            break
            
        # Compute Delaunay Triangulation of centers
        centers = circles[:, :2]
        try:
            tri = Delaunay(centers)
        except Exception:
            # Points might be collinear or identical, perturb slightly
            centers = centers + np.random.normal(0, 1e-5, centers.shape)
            tri = Delaunay(centers)
            
        neighbors = _get_delaunay_neighbors(tri)
        
        # Optimize using SLSQP
        # Note: SLSQP can be slow. We use a moderate maxiter.
        new_circles = _slsqp_optimize(circles, neighbors, max_iter=100)
        
        # Check validity and score
        new_score = _score(new_circles)
        
        if new_score > best_score:
            best_score = new_score
            best_circles = new_circles.copy()
            circles = new_circles.copy()
        else:
            # If it didn't improve, we might be stuck. 
            # Try a small random perturbation to escape local minima
            # This helps SLSQP find a better local optimum in the next iteration
            circles = best_circles.copy()
            # Perturb positions slightly
            circles[:, :2] += np.random.normal(0, 0.01, circles[:, :2].shape)
            # Keep radii
            # Ensure validity after perturbation
            for _ in range(10):
                if _validate(circles):
                    break
                circles[:, :2] -= np.random.normal(0, 0.005, circles[:, :2].shape) # Backtrack
            
    # 3. Final Polish
    # One last attempt with tighter tolerance or different initial guess
    if time.time() < deadline - 5:
        # Try to refine the best found
        centers = best_circles[:, :2]
        try:
            tri = Delaunay(centers)
            neighbors = _get_delaunay_neighbors(tri)
            polished = _slsqp_optimize(best_circles, neighbors, max_iter=300)
            if _score(polished) > best_score:
                best_circles = polished
        except Exception:
            pass

    return best_circles

# EVOLVE-BLOCK-END
