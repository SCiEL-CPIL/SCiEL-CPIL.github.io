"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 21

# EVOLVE-BLOCK-START
def _pack_rect(w, h, circles):
    """
    Pack circles in a rectangle of width w and height h.
    circles: list of (cx, cy, r)
    Returns True if all circles are within the rectangle and non-overlapping.
    """
    for cx, cy, r in circles:
        if cx - r < 0 or cx + r > w or cy - r < 0 or cy + r > h:
            return False
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < circles[i][2] + circles[j][2] - 1e-9:
                return False
    return True

def _total_radius(circles):
    return sum(c[2] for c in circles)

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    # Start with a grid-based initial solution
    # Perimeter 4 => 2(w+h) = 4 => w+h = 2
    # Try various aspect ratios
    best_circles = None
    best_score = 0
    
    # Initial grid: 5x5 grid, use 21 circles
    # Rectangle: w=1, h=1 (square, perimeter 4)
    w0, h0 = 1.0, 1.0
    
    # Create initial grid positions
    side = 5
    initial_circles = []
    for i in range(side):
        for j in range(side):
            if len(initial_circles) >= N:
                break
            cx = (i + 0.5) / side * w0
            cy = (j + 0.5) / side * h0
            r = 0.5 / side * min(w0, h0) - 1e-3
            initial_circles.append((cx, cy, r))
    
    # Try to improve using optimization
    # Variables: for each circle, (cx, cy, r)
    # But we need to handle the rectangle constraint
    # Let's fix the rectangle to be 1x1 first and optimize circle positions and radii
    
    def objective(x):
        if time.time() > deadline:
            return 0
        circles = []
        for i in range(N):
            cx = x[i*3]
            cy = x[i*3+1]
            r = x[i*3+2]
            circles.append((cx, cy, r))
        
        # Check constraints
        for cx, cy, r in circles:
            if r <= 0 or cx - r < -1e-9 or cx + r > w0 + 1e-9 or cy - r < -1e-9 or cy + r > h0 + 1e-9:
                return 0
        for i in range(N):
            for j in range(i+1, N):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < circles[i][2] + circles[j][2] - 1e-9:
                    return 0
        
        return _total_radius(circles)
    
    # Bounds for optimization
    bounds = [(0, w0), (0, h0), (0, 0.5)] * N
    
    # Initial guess
    x0 = []
    for cx, cy, r in initial_circles:
        x0.extend([cx, cy, r])
    
    # Try a few random restarts
    best_result = None
    for attempt in range(5):
        if time.time() > deadline:
            break
            
        if attempt == 0:
            x_init = x0
        else:
            x_init = np.random.uniform(
                [b[0] for b in bounds],
                [b[1] for b in bounds]
            )
        
        # Use Nelder-Mead for local optimization
        try:
            res = minimize(
                lambda x: -objective(x),
                x_init,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            if objective(res.x) > 0:
                circles = []
                for i in range(N):
                    circles.append((res.x[i*3], res.x[i*3+1], res.x[i*3+2]))
                
                if _pack_rect(w0, h0, circles):
                    score = _total_radius(circles)
                    if score > best_score:
                        best_score = score
                        best_result = circles
        except Exception:
            pass
    
    # If we found a good result, return it
    if best_result is not None:
        return np.array(best_result)
    
    # Fallback to initial grid
    return np.array(initial_circles[:N])
# EVOLVE-BLOCK-END
