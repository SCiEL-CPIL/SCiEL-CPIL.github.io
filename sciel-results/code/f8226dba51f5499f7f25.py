"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)

def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * np.min(cross)

def _penalty(points):
    """Calculate penalty for points outside the triangle.
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    Constraints:
    1. y >= 0
    2. y <= sqrt(3)*x  => sqrt(3)*x - y >= 0
    3. y <= -sqrt(3)*(x-1) => sqrt(3)*(1-x) - y >= 0
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Distances to edges (signed)
    d1 = y
    d2 = np.sqrt(3) * x - y
    d3 = np.sqrt(3) * (1 - x) - y
    
    # If any distance is negative, the point is outside.
    # We want to minimize the squared magnitude of the violation.
    violations = np.maximum(0, -d1)**2 + np.maximum(0, -d2)**2 + np.maximum(0, -d3)**2
    return np.sum(violations)

def _objective(params):
    """Objective function: -min_area + lambda * penalty.
    We want to minimize this.
    """
    points = params.reshape(N, 2)
    # Penalty weight: high enough to force points inside, but not so high it dominates the area term
    # The area is ~0.003, penalty should be comparable when slightly outside.
    pen = _penalty(points)
    area = _min_area(points)
    
    # If penalty is large, the point is far outside.
    # We can return a large value if penalty is huge to discourage DE from staying there.
    if pen > 1.0:
        return 1e6 + pen
    
    # We maximize area, so minimize -area.
    # Add penalty to objective.
    return -area + 10.0 * pen

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Bounds for x and y
    # x in [0, 1], y in [0, H]
    bounds = [(0, 1), (0, H)] * N
    
    best_params = None
    best_score = -np.inf
    
    # Initial guess: Grid points
    # Create a triangular grid
    def get_initial_guess(seed_offset=0):
        pts = []
        # 11 points:
        # 1 at center?
        # 3 at vertices?
        # 7 others?
        # Let's try a simple grid
        # Row 0: 1 pt (top)
        # Row 1: 2 pts
        # Row 2: 3 pts
        # Row 3: 5 pts?
        # Total 11.
        
        # Simpler: Just use random points inside for DE initial population, 
        # but provide a good starting point for local search.
        
        # Symmetric guess:
        # Vertices: (0,0), (1,0), (0.5, H)
        # Midpoints of edges: (0.5, 0), (0.25, H/2), (0.75, H/2)
        # Centroid: (0.5, H/3)
        # That's 7 points.
        # 4 more points...
        
        # Let's just use a random point inside for the "best_params" initialization if we don't have a good one.
        # But DE generates its own population.
        
        # For local polish, we need a good starting point.
        # We will collect candidates from DE.
        return None

    # Run Differential Evolution
    # Use 'latinhypercube' for better initial spread
    # popsize 30, maxiter 100 should be fast enough?
    # 22 variables. 30*100 = 3000 evaluations.
    # Each evaluation is O(165) ops. Very fast.
    # We can afford more iterations.
    
    # Let's run DE for 60 seconds
    de_deadline = start_time + 60
    
    try:
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=200, # Let it run, we check time
            popsize=40,
            tol=1e-8,
            seed=42,
            polish=False, # We will polish manually to save time inside DE
            init='latinhypercube',
            update_interval=1,
            callback=lambda x, convergence: (time.time() > de_deadline and False) or None
        )
        # Note: DE callback in scipy returns True to stop.
        # The above lambda is tricky. Let's just run it and hope it finishes, 
        # or use a wrapper.
        
        # Actually, scipy DE doesn't have a clean "stop after time" callback that works perfectly with all versions.
        # We can just run it with a reasonable maxiter and popsize that fits in time.
        # 22 vars, pop 40. 1000 iters = 40k evals.
        # 40k evals * 165 ops = 6.6M ops. Should take < 1s in NumPy.
        # So we can run many more iterations.
        
        # Let's run for 80 seconds worth of computation.
        
    except Exception as e:
        pass

    # The above try block is messy with time limits. 
    # Better approach: Run DE with a fixed, moderate budget, then polish.
    
    # Reset time logic
    start_time = time.time()
    deadline = start_time + 145
    
    # Phase 1: Global Optimization with Differential Evolution
    # We'll run it for a fixed number of iterations that we know is fast.
    # Let's try popsize=30, maxiter=100.
    
    best_params = None
    best_obj = np.inf
    
    # Try a few different DE runs with different seeds/strategies if time permits
    # But DE is stochastic. One good run is usually enough if params are right.
    
    # Let's use a wrapper to enforce time limit inside DE if possible, 
    # or just run a safe amount.
    
    # Safe amount: 200 iterations, popsize 30.
    try:
        res_de = differential_evolution(
            _objective,
            bounds,
            maxiter=100,
            popsize=30,
            tol=1e-6,
            seed=42,
            polish=False,
            init='latinhypercube'
        )
        if res_de.fun < best_obj:
            best_obj = res_de.fun
            best_params = res_de.x.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 5:
        points = best_params.reshape(N, 2) if best_params is not None else np.random.rand(N, 2)
        return points

    # Phase 2: Local Polish
    # Use the best result from DE as starting point for Nelder-Mead or L-BFGS-B.
    # Nelder-Mead is derivative-free and robust for non-smooth min-area functions.
    
    if best_params is not None:
        try:
            res_nm = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res_nm.fun < best_obj:
                best_obj = res_nm.fun
                best_params = res_nm.x.copy()
        except Exception:
            pass
        
        # Try L-BFGS-B as well, it can be faster for smooth parts
        if time.time() < deadline - 5:
            try:
                res_lbfgs = minimize(
                    _objective,
                    best_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000}
                )
                if res_lbfgs.fun < best_obj:
                    best_obj = res_lbfgs.fun
                    best_params = res_lbfgs.x.copy()
            except Exception:
                pass

    # Phase 3: Multi-start Local Polish
    # If we have time, run local polish on other good candidates from DE?
    # DE only returns the best. We didn't save the population.
    # We can do a quick random restart local search.
    
    if time.time() < deadline - 10:
        rng = np.random.default_rng(123)
        for i in range(5): # 5 random restarts
            if time.time() > deadline - 5:
                break
            
            # Random points inside triangle
            pts = np.zeros((N, 2))
            for j in range(N):
                while True:
                    u = rng.random()
                    v = rng.random()
                    if u + v < 1:
                        # Map to triangle
                        # x = v + 0.5*(1-u-v), y = H*(1-u-v)
                        pts[j, 0] = v + 0.5 * (1 - u - v)
                        pts[j, 1] = H * (1 - u - v)
                        break
            
            init_params = pts.flatten()
            
            try:
                res_r = minimize(
                    _objective,
                    init_params,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res_r.fun < best_obj:
                    best_obj = res_r.fun
                    best_params = res_r.x.copy()
            except Exception:
                pass

    # Final check
    if best_params is None:
        # Fallback
        pts = np.zeros((N, 2))
        rng = np.random.default_rng(0)
        for j in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    pts[j, 0] = v + 0.5 * (1 - u - v)
                    pts[j, 1] = H * (1 - u - v)
                    break
        best_params = pts.flatten()

    points = best_params.reshape(N, 2)
    
    # Clamp points to triangle strictly to avoid numerical errors in evaluation
    for i in range(N):
        x, y = points[i]
        # Check constraints
        # 1. y >= 0
        if y < 0:
            y = 0
        # 2. y <= sqrt(3)*x
        max_y_1 = np.sqrt(3) * x
        if y > max_y_1:
            y = max_y_1
        # 3. y <= sqrt(3)*(1-x)
        max_y_2 = np.sqrt(3) * (1 - x)
        if y > max_y_2:
            y = max_y_2
            
        # Also ensure x is in [0,1]
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, H)
        
        # Re-check if clamping pushed it out (e.g. x clamped to 0, but y > 0)
        # If x=0, y must be 0.
        if x < 1e-9:
            y = 0
        if x > 1 - 1e-9:
            y = 0
            
        points[i] = [x, y]

    return points
