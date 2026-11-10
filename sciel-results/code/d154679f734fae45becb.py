import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    best_score = -np.inf
    best_centers = None
    best_sides = None
    
    def calculate_penalty_and_objective(params, lambda_overlap=1000.0, lambda_bound=1000.0):
        # params: [x0, y0, s0, x1, y1, s1, ...]
        centers = params[0:2*n].reshape(n, 2)
        sides = params[2*n:3*n]
        
        # Objective: -sum(sides)
        obj = -np.sum(sides)
        
        # Boundary Penalty
        bound_pen = 0.0
        for i in range(n):
            x, y, s = centers[i, 0], centers[i, 1], sides[i]
            # x bounds: s/2 <= x <= 1 - s/2
            pen_x = 0.0
            if x < s/2:
                pen_x += (s/2 - x)**2
            if x > 1 - s/2:
                pen_x += (x - (1 - s/2))**2
            
            pen_y = 0.0
            if y < s/2:
                pen_y += (s/2 - y)**2
            if y > 1 - s/2:
                pen_y += (y - (1 - s/2))**2
            
            bound_pen += pen_x + pen_y
        
        # Overlap Penalty
        overlap_pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                sx = (sides[i] + sides[j])/2
                sy = (sides[i] + sides[j])/2
                
                pen_x = 0.0
                if dx < sx:
                    pen_x = (sx - dx)**2
                pen_y = 0.0
                if dy < sy:
                    pen_y = (sy - dy)**2
                
                overlap_pen += pen_x * pen_y
        
        return obj + lambda_overlap * overlap_pen + lambda_bound * bound_pen

    def get_centers_sides(params):
        centers = params[0:2*n].reshape(n, 2)
        sides = params[2*n:3*n]
        return centers, sides

    def is_valid(centers, sides):
        tol = 1e-7
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        for i in range(n):
            x, y, s = centers[i, 0], centers[i, 1], sides[i]
            if s <= 0:
                return False
            if not (0 - tol <= x - s/2 and x + s/2 <= 1 + tol):
                return False
            if not (0 - tol <= y - s/2 and y + s/2 <= 1 + tol):
                return False
        
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j])/2 - tol
                if dx < t and dy < t:
                    return False
        return True

    def update_best(params):
        nonlocal best_score, best_centers, best_sides
        centers, sides = get_centers_sides(params)
        if is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()

    def generate_greedy_seed():
        """Generates a valid initial configuration using greedy random placement."""
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        np.random.shuffle(np.arange(n)) # Randomize order to create diversity
        # Try to place squares from largest to smallest potential
        # Simple random placement
        for i in range(n):
            placed = False
            attempts = 0
            while not placed and attempts < 100:
                attempts += 1
                s = np.random.uniform(0.1, 0.4)
                x = np.random.uniform(s/2, 1 - s/2)
                y = np.random.uniform(s/2, 1 - s/2)
                
                # Check overlap with already placed
                overlap = False
                for j in range(i):
                    dx = abs(x - centers[j, 0])
                    dy = abs(y - centers[j, 1])
                    t = (s + sides[j])/2
                    if dx < t and dy < t:
                        overlap = True
                        break
                if not overlap:
                    centers[i] = [x, y]
                    sides[i] = s
                    placed = True
            if not placed:
                # Fallback: tiny square in corner
                centers[i] = [0.05, 0.05]
                sides[i] = 0.05
        return np.concatenate([centers.flatten(), sides])

    # Seeds
    seeds = []
    
    # Seed 1: Greedy random
    np.random.seed(42)
    seeds.append(generate_greedy_seed())
    
    # Seed 2: Grid 2x5
    c2 = np.zeros((10, 2))
    s2 = np.full(10, 0.2)
    for i in range(10):
        col = i % 5
        row = i // 5
        c2[i, 0] = (col + 0.5) * 0.2
        c2[i, 1] = (row + 0.5) * 0.2
    seeds.append(np.concatenate([c2.flatten(), s2]))
    
    # Seed 3: Corner + Center
    c3 = np.array([
        [0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8], # Corners s=0.4
        [0.5, 0.5], # Center s=0.2
        [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5], # Edges s=0.2?
        [0.5, 0.3] # Fill
    ])
    s3 = np.array([0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    # This seed might have overlaps, optimizer will fix
    seeds.append(np.concatenate([c3.flatten(), s3]))

    # Optimization
    for seed in seeds:
        if time.time() > deadline:
            break
            
        try:
            # Use Powell method, it's robust and derivative-free
            res = minimize(
                calculate_penalty_and_objective,
                seed,
                method='Powell',
                bounds=[(1e-4, 1.0)] * (3 * n),
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            update_best(res.x)
        except Exception:
            pass

    # If we have time, do a polish on the best solution
    if best_centers is not None and time.time() < deadline - 30:
        try:
            best_params = np.concatenate([best_centers.flatten(), best_sides])
            res_polish = minimize(
                calculate_penalty_and_objective,
                best_params,
                method='SLSQP',
                bounds=[(1e-4, 1.0)] * (3 * n),
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            update_best(res_polish.x)
        except Exception:
            pass

    # Fallback
    if best_centers is None:
        best_centers = np.zeros((10, 2))
        best_sides = np.full(10, 0.1)
        # Simple valid fallback: 10 tiny squares in a row?
        for i in range(10):
            best_centers[i, 0] = 0.05 + i * 0.09
            best_centers[i, 1] = 0.5
            best_sides[i] = 0.05

    return best_centers, best_sides
