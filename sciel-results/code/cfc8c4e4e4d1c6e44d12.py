"""Circle packing in perimeter-4 rectangle using sequential greedy placement and local refinement."""
import numpy as np
import time
import random
from scipy.spatial import Voronoi, QhullError

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def is_valid(circles, W, H, tol=1e-7):
        """Check if packing is valid: no overlaps, all inside rectangle."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if r < -tol:
                return False
            if x < r - tol or x > W - r + tol:
                return False
            if y < r - tol or y > H - r + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist_sq = dx * dx + dy * dy
                min_dist = circles[i][2] + circles[j][2]
                if dist_sq < min_dist * min_dist - tol:
                    return False
        return True
    
    def max_radius_at_point(x, y, circles, W, H):
        """
        Estimate the maximum radius a circle can have at (x, y) given existing circles.
        Uses distance to walls and to all existing circles.
        """
        # Distance to walls
        r_wall = min(x, W - x, y, H - y)
        
        # Distance to existing circles
        r_min = r_wall
        for cx, cy, cr in circles:
            dx = x - cx
            dy = y - cy
            dist = np.sqrt(dx * dx + dy * dy)
            r_candidate = dist - cr
            if r_candidate < r_min:
                r_min = r_candidate
        
        return max(0.0, r_min)
    
    def greedy_pack(W, H, order=None, time_limit=None):
        """
        Sequentially place circles in the given order (or random order if None).
        Each circle is placed at the position that maximizes its radius,
        using a grid-based search for the optimal position.
        """
        if order is None:
            order = list(range(N))
        
        circles = []
        
        # Grid resolution for searching best position
        # Adaptive: start coarse, refine for later circles
        base_grid = 30
        
        for idx, circle_id in enumerate(order):
            if time_limit is not None and time.time() > time_limit:
                break
                
            # If no circles placed yet, place at center
            if len(circles) == 0:
                circles.append((W / 2, H / 2, min(W, H) / 4))
                continue
            
            # Search for best position using grid
            # Grid density increases as we add more circles (less free space)
            grid_size = base_grid + idx
            
            best_r = 0
            best_x = W / 2
            best_y = H / 2
            
            # Coarse grid search
            for i in range(grid_size + 1):
                x = (i / grid_size) * W
                for j in range(grid_size + 1):
                    y = (j / grid_size) * H
                    r = max_radius_at_point(x, y, circles, W, H)
                    if r > best_r:
                        best_r = r
                        best_x = x
                        best_y = y
            
            # Local refinement around best position
            if best_r > 0:
                step = W / (grid_size * 5)
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        x = best_x + di * step
                        y = best_y + dj * step
                        if 0 <= x <= W and 0 <= y <= H:
                            r = max_radius_at_point(x, y, circles, W, H)
                            if r > best_r:
                                best_r = r
                                best_x = x
                                best_y = y
            
            if best_r > 1e-6:
                circles.append((best_x, best_y, best_r))
        
        return circles
    
    def refine_packing(circles, W, H, time_limit=None):
        """
        Local hill-climbing refinement: try to grow each circle slightly
        and adjust positions to maintain validity.
        """
        if not circles:
            return circles
        
        circles = list(circles)
        n = len(circles)
        t0 = time.time()
        
        # Multiple passes of local improvement
        for _ in range(50):
            if time_limit is not None and time.time() > time_limit:
                break
            
            improved = False
            for i in range(n):
                if time_limit is not None and time.time() > time_limit:
                    break
                
                x, y, r = circles[i]
                old_r = r
                
                # Try to grow this circle
                r_new = r * 1.005
                
                # Check if we can grow without moving
                valid = True
                for j in range(n):
                    if i == j:
                        continue
                    dx = x - circles[j][0]
                    dy = y - circles[j][1]
                    dist_sq = dx * dx + dy * dy
                    min_dist = r_new + circles[j][2]
                    if dist_sq < min_dist * min_dist - 1e-8:
                        valid = False
                        break
                if x < r_new - 1e-8 or x > W - r_new + 1e-8:
                    valid = False
                if y < r_new - 1e-8 or y > H - r_new + 1e-8:
                    valid = False
                
                if valid:
                    circles[i] = (x, y, r_new)
                    improved = True
                    continue
                
                # Try small position adjustments to enable growth
                for _ in range(10):
                    angle = random.uniform(0, 2 * np.pi)
                    step = 0.01 * r
                    x_try = x + step * np.cos(angle)
                    y_try = y + step * np.sin(angle)
                    
                    # Clamp to bounds
                    x_try = np.clip(x_try, r_new, W - r_new)
                    y_try = np.clip(y_try, r_new, H - r_new)
                    
                    valid = True
                    for j in range(n):
                        if i == j:
                            continue
                        dx = x_try - circles[j][0]
                        dy = y_try - circles[j][1]
                        dist_sq = dx * dx + dy * dy
                        min_dist = r_new + circles[j][2]
                        if dist_sq < min_dist * min_dist - 1e-8:
                            valid = False
                            break
                    if x_try < r_new - 1e-8 or x_try > W - r_new + 1e-8:
                        valid = False
                    if y_try < r_new - 1e-8 or y_try > H - r_new + 1e-8:
                        valid = False
                    
                    if valid:
                        circles[i] = (x_try, y_try, r_new)
                        improved = True
                        break
            
            if not improved:
                break
        
        return circles
    
    # Main optimization loop
    random.seed(42)
    np.random.seed(42)
    
    best_sum_r = 0.0
    best_config = None
    best_W = 1.0
    best_H = 1.0
    
    # Try different aspect ratios
    aspect_ratios = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4]
    
    for W_ratio in aspect_ratios:
        if time.time() > deadline - 20:
            break
        
        W = W_ratio
        H = 2.0 - W
        if W <= 0.2 or H <= 0.2:
            continue
        
        # Try multiple random orders for greedy placement
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            
            # Generate random insertion order
            order = list(range(N))
            random.shuffle(order)
            
            # Greedy packing
            circles = greedy_pack(W, H, order, time_limit=deadline - 5)
            
            if len(circles) < N:
                continue
            
            # Refine
            circles = refine_packing(circles, W, H, time_limit=deadline - 5)
            
            # Validate
            if not is_valid(circles, W, H, tol=1e-6):
                continue
            
            total_r = sum(c[2] for c in circles)
            if total_r > best_sum_r:
                best_sum_r = total_r
                best_config = np.array(circles)
                best_W = W
                best_H = H
    
    # If we haven't found a good solution, try a simple grid fallback
    if best_config is None or len(best_config) < N:
        # Simple 5x5 grid in square
        W = 1.0
        H = 1.0
        cells = []
        for i in range(5):
            for j in range(5):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / 5
                y = (j + 0.5) / 5
                r = 0.5 / 5 * 0.8
                cells.append((x, y, r))
        best_config = np.array(cells[:N])
        best_W = W
        best_H = H
    
    return best_config

# EVOLVE-BLOCK-END
