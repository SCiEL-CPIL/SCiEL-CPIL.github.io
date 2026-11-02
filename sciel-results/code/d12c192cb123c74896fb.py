"""Rewritten program for circle packing in unit square (26 circles)."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random


def construct():
    """
    Pack 26 circles in the unit square [0,1]^2 maximizing sum of radii.
    Uses a heuristic: start from a good grid-like arrangement, then use
    local optimization to push circles apart and grow radii.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Number of circles
    n = N
    
    # Initial positions: use a grid-like layout with some jitter
    # For 26 circles, a 5x6 grid (with one extra) or similar
    # Let's try a 5x6 grid = 30, use 26
    # Or 4x7 = 28, use 26
    # Let's use a denser arrangement
    
    # Strategy: Start with circles arranged in rows
    # Try different initial configurations and pick the best
    
    def evaluate(circles):
        """
        circles: array of shape (n, 3) with (x, y, r)
        Returns: sum of radii if valid, else -1
        """
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Check bounds: circles must be inside [0,1]^2
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return -1
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return -1
        
        # Check non-overlap
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - 1e-9:
                    return -1
        
        return np.sum(r)
    
    def make_initial_config(cols, rows, jitter=0.0):
        """Create initial circle configuration on a grid."""
        circles = []
        # Place circles in a grid pattern
        for row in range(rows):
            for col in range(cols):
                if len(circles) >= n:
                    break
                x = (col + 0.5) / cols
                y = (row + 0.5) / rows
                # Add jitter
                if jitter > 0:
                    x += (random.random() - 0.5) * jitter
                    y += (random.random() - 0.5) * jitter
                # Initial radius estimate
                r = min(0.5 / cols, 0.5 / rows) * 0.95
                circles.append([x, y, r])
        return np.array(circles[:n])
    
    def optimize_circles(circles):
        """
        Optimize circle positions and radii using gradient-based method.
        We maximize sum of radii subject to non-overlap and boundary constraints.
        """
        n = len(circles)
        
        # Parameters: for each circle, (x, y, r)
        # We'll use a penalty-based approach with scipy.optimize
        
        def objective(params):
            """Negative of sum of radii (we minimize negative)."""
            x = params[0::3]
            y = params[1::3]
            r = params[2::3]
            
            # Penalty for going out of bounds
            penalty = 0.0
            for i in range(n):
                if x[i] - r[i] < 0:
                    penalty += (x[i] - r[i]) ** 2
                if x[i] + r[i] > 1:
                    penalty += (x[i] + r[i] - 1) ** 2
                if y[i] - r[i] < 0:
                    penalty += (y[i] - r[i]) ** 2
                if y[i] + r[i] > 1:
                    penalty += (y[i] + r[i] - 1) ** 2
            
            # Penalty for overlapping
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2
            
            # We want to maximize sum(r), so minimize -sum(r) + penalty
            return -np.sum(r) + 100.0 * penalty
        
        # Initial parameters
        x0 = circles[:, 0]
        y0 = circles[:, 1]
        r0 = circles[:, 2]
        params0 = np.zeros(3 * n)
        params0[0::3] = x0
        params0[1::3] = y0
        params0[2::3] = r0
        
        # Bounds: x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))  # x
            bounds.append((0.0, 1.0))  # y
            bounds.append((0.0, 0.5))  # r
        
        # Try optimization
        try:
            result = minimize(objective, params0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
            
            # Extract results
            x_opt = result.x[0::3]
            y_opt = result.x[1::3]
            r_opt = result.x[2::3]
            
            # Clip to valid range
            x_opt = np.clip(x_opt, 0, 1)
            y_opt = np.clip(y_opt, 0, 1)
            r_opt = np.clip(r_opt, 0, 0.5)
            
            # Ensure circles are within bounds
            for i in range(n):
                r_opt[i] = min(r_opt[i], x_opt[i], 1 - x_opt[i], y_opt[i], 1 - y_opt[i])
            
            return np.column_stack([x_opt, y_opt, r_opt])
        except Exception:
            return circles
    
    def refine_with_push(circles, iterations=100):
        """
        Simple iterative refinement: push overlapping circles apart and grow radii.
        """
        n = len(circles)
        circles = circles.copy()
        
        for _ in range(iterations):
            # Check for overlaps and push apart
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = circles[i, 2] + circles[j, 2]
                    
                    if dist < min_dist and dist > 1e-10:
                        # Push apart
                        overlap = min_dist - dist
                        push_x = (dx / dist) * overlap * 0.5
                        push_y = (dy / dist) * overlap * 0.5
                        circles[i, 0] += push_x
                        circles[i, 1] += push_y
                        circles[j, 0] -= push_x
                        circles[j, 1] -= push_y
                        moved = True
            
            # Try to grow radii slightly
            for i in range(n):
                # Max possible radius given current position
                max_r = min(circles[i, 0], 1 - circles[i, 0],
                           circles[i, 1], 1 - circles[i, 1])
                # Also check against other circles
                for j in range(n):
                    if i == j:
                        continue
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    available = (dist - circles[j, 2])
                    max_r = min(max_r, available)
                
                if max_r > circles[i, 2]:
                    # Grow towards max_r gradually
                    circles[i, 2] = min(max_r, circles[i, 2] * 1.01 + 0.001)
                    moved = True
            
            if not moved:
                break
        
        # Final clip
        for i in range(n):
            circles[i, 2] = min(circles[i, 2],
                               circles[i, 0], 1 - circles[i, 0],
                               circles[i, 1], 1 - circles[i, 1])
        
        return circles
    
    # Try multiple initial configurations
    best_score = -1
    best_circles = None
    
    # Configuration 1: 5x6 grid (use 26)
    configs = [
        (5, 6),  # 30 cells, use 26
        (4, 7),  # 28 cells, use 26
        (6, 5),  # 30 cells, use 26
        (7, 4),  # 28 cells, use 26
    ]
    
    for cols, rows in configs:
        for jitter in [0.0, 0.05, 0.1]:
            circles = make_initial_config(cols, rows, jitter)
            if len(circles) < n:
                continue
            
            # Optimize
            optimized = optimize_circles(circles)
            
            # Refine with push method
            refined = refine_with_push(optimized, iterations=200)
            
            # Evaluate
            score = evaluate(refined)
            if score > best_score:
                best_score = score
                best_circles = refined
    
    # Also try a random initial configuration
    for _ in range(5):
        x = np.random.uniform(0.1, 0.9, n)
        y = np.random.uniform(0.1, 0.9, n)
        r = np.full(n, 0.05)
        circles = np.column_stack([x, y, r])
        
        optimized = optimize_circles(circles)
        refined = refine_with_push(optimized, iterations=200)
        
        score = evaluate(refined)
        if score > best_score:
            best_score = score
            best_circles = refined
    
    # If no valid solution found, fall back to a simple grid
    if best_circles is None or best_score <= 0:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    # Final validation and adjustment
    x = best_circles[:, 0]
    y = best_circles[:, 1]
    r = best_circles[:, 2]
    
    # Ensure all circles are valid
    for i in range(n):
        # Clip radius to fit in bounds
        r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
    
    best_circles[:, 2] = r
    
    return best_circles

# EVOLVE-BLOCK-END
