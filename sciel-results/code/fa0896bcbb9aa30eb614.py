import numpy as np
import time
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Triangle vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    # Function to check if a point is inside the triangle
    # Using barycentric coordinates or simple inequalities
    def is_inside(p):
        # p is (x, y)
        # y >= 0
        if p[1] < 0: return False
        # y <= sqrt(3) * x  (Line AC: y = sqrt(3)x)
        if p[1] > np.sqrt(3) * p[0] + 1e-9: return False
        # y <= sqrt(3) * (1-x) (Line BC: y = -sqrt(3)(x-1))
        if p[1] > np.sqrt(3) * (1 - p[0]) + 1e-9: return False
        return True

    def penalty(pts):
        """Calculate penalty for points outside the triangle."""
        pen = 0.0
        for p in pts:
            if not is_inside(p):
                # Distance to boundary approximation
                # Simple penalty: square of violation
                # Check each constraint
                v1 = p[1] # y
                v2 = np.sqrt(3) * p[0] - p[1]
                v3 = np.sqrt(3) * (1 - p[0]) - p[1]
                # We want v1 >= 0, v2 >= 0, v3 >= 0
                if v1 < 0: pen += v1**2
                if v2 < 0: pen += v2**2
                if v3 < 0: pen += v3**2
        return pen

    def objective(params):
        """
        params: flat array of length 2*N (x1, y1, x2, y2, ...)
        Returns: -min_area + penalty
        """
        pts = params.reshape((N, 2))
        
        # Calculate min triangle area
        min_area = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    v1 = pts[j] - pts[i]
                    v2 = pts[k] - pts[i]
                    area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if area < min_area:
                        min_area = area
        
        # Calculate penalty
        pen = penalty(pts)
        
        # We want to maximize min_area, so minimize -min_area
        # Add penalty to keep points inside
        return -min_area + 100.0 * pen

    def generate_initial_guesses():
        guesses = []
        
        # Guess 1: Grid in reference triangle mapped to physical
        # Reference triangle (0,0), (1,0), (0,1)
        # Map (u,v) to x,y:
        # x = u + 0.5*v
        # y = sqrt(3)/2 * v
        # We need 11 points.
        # Rows in u-v space:
        # Row 0 (v=0): u in [0,1]. 4 points: 0, 1/3, 2/3, 1
        # Row 1 (v=1/3): u in [0, 2/3]. 3 points: 0, 1/3, 2/3
        # Row 2 (v=2/3): u in [0, 1/3]. 2 points: 0, 1/3
        # Row 3 (v=1): u=0. 1 point.
        # Total 10. Add one more in the middle?
        # Let's just use a denser grid or random valid points for variety.
        
        # Simple random valid points
        rng = np.random.default_rng(42)
        for _ in range(5):
            pts = []
            while len(pts) < N:
                u = rng.random()
                v = rng.random()
                if u + v <= 1.0:
                    x = u + 0.5 * v
                    y = (np.sqrt(3) / 2) * v
                    pts.append([x, y])
            guesses.append(np.array(pts).flatten())
            
        # Structured guess 1: 4-3-2-1-1?
        # Let's try a hexagonal-like packing
        # This is a known good configuration for N=11?
        # Let's just rely on the optimizer from random starts.
        # But a good start helps.
        
        # Structured guess: Rows of 4, 4, 3?
        # Row 0 (y=0.05): 4 points
        # Row 1 (y=0.15): 4 points
        # Row 2 (y=0.25): 3 points
        # Total 11.
        # Let's create a symmetric grid.
        # Centroid is (0.5, sqrt(3)/6).
        
        # Let's add a few more random starts to cover the space.
        for _ in range(10):
            pts = []
            while len(pts) < N:
                u = rng.random()
                v = rng.random()
                if u + v <= 1.0:
                    x = u + 0.5 * v
                    y = (np.sqrt(3) / 2) * v
                    # Add small jitter
                    x += rng.normal(0, 0.01)
                    y += rng.normal(0, 0.01)
                    # Project back if outside? The penalty handles it, but start inside is better.
                    if is_inside([x,y]):
                        pts.append([x, y])
            guesses.append(np.array(pts).flatten())
            
        return guesses

    best_params = None
    best_score = -float('inf') # We are maximizing min_area, so score is min_area
    # Actually, objective returns -min_area + pen. So lower objective is better.
    # Let's track best objective value.
    best_obj = float('inf')

    guesses = generate_initial_guesses()
    
    # Sort guesses by initial objective? Might be expensive. Just iterate.
    for i, x0 in enumerate(guesses):
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback: return first guess
        best_params = guesses[0]

    # Extract points
    pts = best_params.reshape((N, 2))
    
    # Final cleanup: ensure points are strictly inside
    # If any point is outside, project it
    for i in range(N):
        p = pts[i]
        if not is_inside(p):
            # Simple projection: move towards centroid
            centroid = np.array([0.5, np.sqrt(3)/6])
            # Binary search for t in [0,1] such that (1-t)p + t*centroid is inside
            lo, hi = 0.0, 1.0
            for _ in range(20):
                mid = (lo + hi) / 2
                test_p = (1 - mid) * p + mid * centroid
                if is_inside(test_p):
                    hi = mid
                else:
                    lo = mid
            pts[i] = (1 - hi) * p + hi * centroid
            # If still outside (numerical error), snap to centroid
            if not is_inside(pts[i]):
                pts[i] = centroid

    return pts
