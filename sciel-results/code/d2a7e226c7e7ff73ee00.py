"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _max_radius_at_position(x, y, circles, skip_idx):
    """Compute max radius for a circle at (x,y) given other circles."""
    r_max = min(x, 1.0 - x, y, 1.0 - y)
    if r_max <= 0:
        return 0.0
    for j, (xj, yj, rj) in enumerate(circles):
        if j == skip_idx:
            continue
        d = np.sqrt((x - xj)**2 + (y - yj)**2)
        r_limit = d - rj
        if r_limit < r_max:
            r_max = r_limit
    return max(0.0, r_max)

def _best_fit_search(circles, iterations=100, grid_res=50):
    """For each circle, find best position on a grid to maximize its radius."""
    circles = [list(c) for c in circles]
    n = len(circles)
    
    for it in range(iterations):
        improved = False
        for i in range(n):
            xi, yi, ri = circles[i]
            best_r = ri
            best_x, best_y = xi, yi
            
            # Search on a grid around current position
            search_radius = max(ri, 0.05)
            for gx in range(grid_res + 1):
                for gy in range(grid_res + 1):
                    nx = xi + (gx / grid_res - 0.5) * 2 * search_radius
                    ny = yi + (gy / grid_res - 0.5) * 2 * search_radius
                    if nx < 0 or nx > 1 or ny < 0 or ny > 1:
                        continue
                    r_new = _max_radius_at_position(nx, ny, circles, i)
                    if r_new > best_r:
                        best_r = r_new
                        best_x, best_y = nx, ny
            
            if best_r > ri + 1e-10:
                circles[i] = [best_x, best_y, best_r]
                improved = True
        
        if not improved:
            break
    
    return circles

def _is_valid(circles, eps=1e-9):
    """Check validity of circle packing."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if x - r < -eps or x + r > 1 + eps or y - r < -eps or y + r > 1 + eps:
            return False
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            d2 = (xi-xj)**2 + (yi-yj)**2
            if d2 < (ri+rj - eps)**2:
                return False
    return True

def _initial_grid():
    """Create initial grid of circles."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
    return cells

def _initial_random(n_trials=20):
    """Try multiple random initializations and return the best."""
    best_circles = None
    best_sum = -1
    
    for _ in range(n_trials):
        random.seed()
        circles = []
        for _ in range(N):
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            r = 0.02
            circles.append([x, y, r])
        
        # Quick best-fit to get a reasonable starting point
        circles = _best_fit_search(circles, iterations=10, grid_res=20)
        s = sum(c[2] for c in circles)
        if s > best_sum:
            best_sum = s
            best_circles = [list(c) for c in circles]
    
    return best_circles

def _high_precision_polish(circles, iterations=50, grid_res=100):
    """Fine-grained best-fit search for final polish."""
    return _best_fit_search(circles, iterations=iterations, grid_res=grid_res)

def _simulated_annealing(circles, start_temp=0.01, end_temp=0.001, max_iter=2000):
    """Simulated annealing to escape local optima."""
    circles = [list(c) for c in circles]
    n = len(circles)
    current_sum = sum(c[2] for c in circles)
    best_circles = [list(c) for c in circles]
    best_sum = current_sum
    
    for it in range(max_iter):
        temp = start_temp * (end_temp / start_temp) ** (it / max_iter)
        
        # Pick a random circle to move
        i = random.randint(0, n-1)
        xi, yi, ri = circles[i]
        
        # Random perturbation
        dx = random.gauss(0, 0.05)
        dy = random.gauss(0, 0.05)
        new_x = np.clip(xi + dx, 0.01, 0.99)
        new_y = np.clip(yi + dy, 0.01, 0.99)
        
        new_r = _max_radius_at_position(new_x, new_y, circles, i)
        new_sum = current_sum - ri + new_r
        
        delta = new_sum - current_sum
        if delta > 0 or random.random() < np.exp(delta / temp):
            circles[i] = [new_x, new_y, new_r]
            current_sum = new_sum
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = [list(c) for c in circles]
    
    return best_circles

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with grid initialization
    circles = _initial_grid()
    best_circles = [list(c) for c in circles]
    best_sum = sum(c[2] for c in circles)
    
    # Best-fit search to optimize positions
    if time.time() < deadline - 60:
        circles = _best_fit_search(circles, iterations=50, grid_res=40)
        s = sum(c[2] for c in circles)
        if s > best_sum:
            best_sum = s
            best_circles = [list(c) for c in circles]
    
    # Try random initializations
    if time.time() < deadline - 40:
        rand_circles = _initial_random(n_trials=5)
        if rand_circles is not None:
            rand_circles = _best_fit_search(rand_circles, iterations=30, grid_res=40)
            s = sum(c[2] for c in rand_circles)
            if s > best_sum:
                best_sum = s
                best_circles = [list(c) for c in rand_circles]
    
    # Simulated annealing to escape local optima
    if time.time() < deadline - 30:
        sa_circles = _simulated_annealing(best_circles, start_temp=0.01, end_temp=0.001, max_iter=1500)
        s = sum(c[2] for c in sa_circles)
        if s > best_sum:
            best_sum = s
            best_circles = [list(c) for c in sa_circles]
    
    # High precision polish
    if time.time() < deadline - 10:
        polished = _high_precision_polish(best_circles, iterations=30, grid_res=80)
        s = sum(c[2] for c in polished)
        if s > best_sum:
            best_sum = s
            best_circles = [list(c) for c in polished]
    
    # Final validation
    if not _is_valid(best_circles):
        # Fallback to grid if invalid
        best_circles = _initial_grid()
    
    return np.array(best_circles)
# EVOLVE-BLOCK-END
