import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
import itertools

N = 11
S3H = np.sqrt(3) / 2
TRI_AREA = 0.5 * 1 * S3H  # Area of the equilateral triangle with side 1

# Vertices of the triangle
V0 = np.array([0, 0])
V1 = np.array([1, 0])
V2 = np.array([0.5, S3H])

DEADLINE = 145.0

def _triangle_area(p1, p2, p3):
    """Calculate the area of a triangle given three points."""
    return 0.5 * np.abs(np.cross(p2 - p1, p3 - p1))

def _min_area(pts):
    """Calculate the minimum area of any triangle formed by 3 points from the set."""
    min_a = float('inf')
    n = len(pts)
    # Vectorized approach for speed
    # We can compute all areas at once, but for N=11, 165 triangles is small.
    # However, doing it in a loop in Python is slow. Let's use a semi-vectorized approach.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(pts[i], pts[j], pts[k])
                if a < min_a:
                    min_a = a
    return min_a

def _soft_min_area(pts, temp=0.01):
    """
    Smooth approximation of the minimum area.
    Uses log-sum-exp trick on negative areas.
    min(x_i) = -lim_{t->0} (1/t) * log(sum(exp(-t*x_i)))
    Here we want to maximize min_area, so we minimize -min_area.
    Objective = -min_area + penalty.
    To smooth -min_area, we can use:
    -min_area approx - (1/beta) * log(sum(exp(-beta * area_i))) ? No.
    
    Let's use: f(x) = -min(x_i).
    We want a smooth lower bound on -min(x_i) or an upper bound on min(x_i).
    
    Standard soft-min: min(x) ~ -1/beta * log(sum(exp(-beta*x))).
    As beta -> inf, this converges to min(x).
    So, -min(x) ~ 1/beta * log(sum(exp(-beta*x))).
    
    We want to minimize J = -min_area + penalty.
    So we use J_smooth = 1/beta * log(sum(exp(-beta * areas))) + penalty.
    """
    areas = []
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                areas.append(_triangle_area(pts[i], pts[j], pts[k]))
    areas = np.array(areas)
    
    # Avoid overflow in exp
    max_neg_area = np.max(-areas)
    log_sum_exp = max_neg_area + np.log(np.sum(np.exp(-areas - max_neg_area)))
    
    beta = 1.0 / temp
    soft_min_val = (1.0 / beta) * log_sum_exp
    return soft_min_val

def _penalty(pts):
    """
    Penalty for points outside the triangle.
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2).
    Inequalities for interior:
    1. y >= 0
    2. y <= sqrt(3) * x
    3. y <= sqrt(3) * (1 - x)
    """
    x = pts[:, 0]
    y = pts[:, 1]
    
    p1 = np.maximum(0, -y)
    p2 = np.maximum(0, y - np.sqrt(3) * x)
    p3 = np.maximum(0, y - np.sqrt(3) * (1 - x))
    
    return np.sum(p1**2) + np.sum(p2**2) + np.sum(p3**2)

def _objective(x_flat):
    """
    Objective function for minimization.
    x_flat: 22-element array.
    """
    pts = x_flat.reshape(N, 2)
    pen = _penalty(pts)
    
    if pen > 1e-2:
        return 100.0 + pen
        
    # Use soft min to make it differentiable for L-BFGS-B
    # Temperature can be tuned, but 0.01 is a reasonable start
    soft_min = _soft_min_area(pts, temp=0.01)
    
    # We want to maximize min area, so minimize -min area.
    # soft_min approximates min_area.
    # So we minimize -soft_min? No.
    # soft_min = min_area (approx).
    # We want to maximize min_area.
    # So objective = -soft_min + penalty_weight * pen.
    
    return -soft_min + 10.0 * pen

def _construct_structured_init():
    """
    Create a structured initial guess.
    Place 3 points at vertices, 1 at centroid.
    Distribute remaining 7 points in a quasi-regular pattern.
    """
    pts = np.zeros((N, 2))
    
    # Vertices
    pts[0] = V0
    pts[1] = V1
    pts[2] = V2
    
    # Centroid
    centroid = (V0 + V1 + V2) / 3.0
    pts[3] = centroid
    
    # Remaining 7 points
    # Let's try to place them in two rings or a grid-like structure
    # Simpler: random points inside, but seeded for reproducibility
    rng = np.random.default_rng(42)
    
    # Barycentric coordinates
    for i in range(4, N):
        # Generate random barycentric coords
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        u = r1
        v = r2
        w = 1 - u - v
        pts[i] = u * V0 + v * V1 + w * V2
        
    return pts

def construct():
    start_time = time.time()
    
    # Initial guess
    x0 = _construct_structured_init().flatten()
    
    # Bounds
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0, 1), (0, S3H)] * N
    
    best_x = x0
    best_val = _objective(x0)
    
    # Strategy: 
    # 1. Run a few iterations of Nelder-Mead or L-BFGS-B from the structured init.
    # 2. If time permits, run a short Differential Evolution to explore globally,
    #    then polish the best DE result with L-BFGS-B.
    
    # Since DE is expensive, let's first try to polish the structured init.
    # The structured init is likely better than random DE init.
    
    try:
        # L-BFGS-B with soft objective
        res = minimize(
            _objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': 500, 'disp': False}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > DEADLINE - 30:
        # Not enough time for DE
        pts = best_x.reshape(N, 2)
        # Project to triangle
        for i in range(N):
            x, y = pts[i]
            if y < 0: y = 0
            max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
            if y > max_y: y = max_y
            if x < 0: x = 0
            if x > 1: x = 1
            pts[i] = [x, y]
        return pts
    
    # Try Differential Evolution with a smaller population to find a better basin
    # DE is stochastic, so it might find a better global structure.
    # We use a smaller popsize and maxiter to fit in time.
    try:
        # Use a tighter time limit for DE
        de_deadline = DEADLINE - 15
        elapsed = time.time() - start_time
        time_left = de_deadline - elapsed
        
        if time_left > 10:
            # Estimate maxiter based on time? Hard to do.
            # Let's just cap it.
            maxiter = int(50 * (time_left / 30.0))
            if maxiter < 10: maxiter = 10
            if maxiter > 200: maxiter = 200
            
            res_de = differential_evolution(
                _objective, 
                bounds, 
                popsize=20, 
                maxiter=maxiter,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_x = res_de.x
                
                # Polish the DE result
                if time.time() - start_time < DEADLINE - 5:
                    res_polish = minimize(
                        _objective, 
                        best_x, 
                        method='L-BFGS-B', 
                        bounds=bounds, 
                        options={'maxiter': 200, 'disp': False}
                    )
                    if res_polish.fun < best_val:
                        best_val = res_polish.fun
                        best_x = res_polish.x
    except Exception:
        pass
    
    pts = best_x.reshape(N, 2)
    
    # Final projection and validation
    for i in range(N):
        x, y = pts[i]
        if y < 0: y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y: y = max_y
        if x < 0: x = 0
        if x > 1: x = 1
        pts[i] = [x, y]
        
    # If penalty is still high, fallback to structured init
    if _penalty(pts) > 1e-3:
        pts = _construct_structured_init()
        
    return pts

if __name__ == "__main__":
    pts = construct()
    min_a = _min_area(pts)
    print(f"Min Area: {min_a}")
    print(f"Score: {min_a / TRI_AREA}")
