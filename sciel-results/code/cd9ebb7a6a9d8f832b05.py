# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def squares_in_square10() -> tuple:
    """
    Optimizes the placement of 10 squares in a unit square to maximize the sum of their side lengths.
    Uses shelf-based constraint satisfaction combined with local optimization.
    """
    n = 10
    start_time = time.time()
    max_time = 580.0  # Safety margin before 600s limit

    def check_validity(cx, cy, s, tol=1e-7):
        """Check if configuration is valid."""
        if np.any(cx < -tol) or np.any(cx > 1 + tol):
            return False
        if np.any(cy < -tol) or np.any(cy > 1 + tol):
            return False
        if np.any(s < -tol):
            return False
        if np.any(cx - s/2 < -tol) or np.any(cx + s/2 > 1 + tol):
            return False
        if np.any(cy - s/2 < -tol) or np.any(cy + s/2 > 1 + tol):
            return False
            
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                min_dx = (s[i] + s[j]) / 2.0 - tol
                min_dy = (s[i] + s[j]) / 2.0 - tol
                if dx < min_dx and dy < min_dy:
                    return False
        return True

    def shelf_placement(sides, horizontal_shelves, vertical_shelves):
        """
        Place squares using shelf method.
        horizontal_shelves: list of shelf indices for each square (which horizontal shelf)
        vertical_shelves: list of shelf indices for each square (which vertical shelf)
        
        A square is on horizontal shelf h if it's separated from others by y-coordinate.
        A square is on vertical shelf v if it's separated from others by x-coordinate.
        
        For non-overlap: for each pair (i,j), either they're on different horizontal shelves 
        (separated in y) OR different vertical shelves (separated in x).
        """
        cx = np.zeros(n)
        cy = np.zeros(n)
        
        # Group squares by horizontal shelf
        h_groups = {}
        for i in range(n):
            h = horizontal_shelves[i]
            if h not in h_groups:
                h_groups[h] = []
            h_groups[h].append(i)
        
        # For each horizontal shelf, place squares in order by vertical shelf
        # The y-coordinate of shelf h determines the vertical position
        # We need to determine shelf heights and positions
        
        # First, determine the y-ranges for each horizontal shelf
        # Shelf h contains squares that must be separated in y from squares in other shelves
        # Within a shelf, squares can overlap in y but must be separated in x
        
        # Let's compute the required height for each horizontal shelf
        shelf_heights = {}
        for h, indices in h_groups.items():
            # The height needed is the max side length of squares in this shelf
            # But we also need to consider that squares in the same shelf must fit side by side
            shelf_heights[h] = max(sides[i] for i in indices)
        
        # Sort shelves by some criterion and assign y-positions
        # Total height must be <= 1
        total_height = sum(shelf_heights.values())
        
        if total_height > 1 + 1e-9:
            return None  # Cannot fit vertically
        
        # Assign y-centers to shelves
        shelf_y = {}
        current_y = 0.0
        for h in sorted(h_groups.keys()):
            shelf_y[h] = current_y + shelf_heights[h] / 2
            current_y += shelf_heights[h]
        
        # Now for each shelf, place squares horizontally
        # Within a shelf, squares must be separated in x
        # Sort squares in each shelf by their vertical shelf index (which determines x-order)
        for h, indices in h_groups.items():
            # Sort by vertical shelf index
            indices_sorted = sorted(indices, key=lambda i: vertical_shelves[i])
            
            # Place squares in order
            current_x = 0.0
            for i in indices_sorted:
                cx[i] = current_x + sides[i] / 2
                cy[i] = shelf_y[h]
                current_x += sides[i]
            
            # Check if total width fits
            if current_x > 1 + 1e-9:
                return None  # Cannot fit horizontally in this shelf
        
        return cx, cy

    def try_shelf_configuration(sides, h_shelves, v_shelves):
        """Try a specific shelf configuration and return centers if valid."""
        result = shelf_placement(sides, h_shelves, v_shelves)
        if result is None:
            return None
        
        cx, cy = result
        if check_validity(cx, cy, sides):
            return cx, cy
        return None

    def local_optimize(cx, cy, s, max_iter=5000):
        """Local optimization to improve the configuration."""
        best_val = np.sum(s)
        best_cx, best_cy, best_s = cx.copy(), cy.copy(), s.copy()
        
        for _ in range(max_iter):
            if time.time() - start_time > max_time:
                break
            
            # Try small perturbations
            idx = np.random.randint(0, 3*n)
            if idx >= 2*n:
                delta = np.random.normal(0.002, 0.005)
            else:
                delta = np.random.normal(0, 0.002)
                
            cx_new = cx.copy()
            cy_new = cy.copy()
            s_new = s.copy()
            
            if idx < n:
                cx_new[idx] += delta
            elif idx < 2*n:
                cy_new[idx - n] += delta
            else:
                s_new[idx - 2*n] += delta
            
            # Clip to valid ranges
            cx_new = np.clip(cx_new, 0, 1)
            cy_new = np.clip(cy_new, 0, 1)
            s_new = np.clip(s_new, 0.01, 1)
            
            if check_validity(cx_new, cy_new, s_new):
                new_val = np.sum(s_new)
                if new_val > best_val:
                    best_val = new_val
                    best_cx, best_cy, best_s = cx_new.copy(), cy_new.copy(), s_new.copy()
        
        return best_cx, best_cy, best_s, best_val

    best_score = -np.inf
    best_result = None
    
    # Strategy 1: Shelf-based search
    # Try different numbers of horizontal and vertical shelves
    # For 10 squares, try configurations like:
    # - 1 horizontal shelf, all separated vertically (10 shelves)
    # - 2 horizontal shelves, 5 each
    # - 5 horizontal shelves, 2 each
    # - 10 horizontal shelves, 1 each
    # And similar for vertical
    
    shelf_configs = []
    
    # Horizontal shelf configurations: list of (num_h_shelves, squares_per_shelf)
    # We'll try various distributions
    for num_h in range(1, 11):
        # Distribute 10 squares into num_h shelves
        # Try equal distribution first
        base = 10 // num_h
        remainder = 10 % num_h
        dist = [base + (1 if i < remainder else 0) for i in range(num_h)]
        shelf_configs.append(('h', num_h, dist))
    
    # Vertical shelf configurations
    for num_v in range(1, 11):
        base = 10 // num_v
        remainder = 10 % num_v
        dist = [base + (1 if i < remainder else 0) for i in range(num_v)]
        shelf_configs.append(('v', num_v, dist))
    
    # Also try mixed: some horizontal, some vertical shelves
    # For now, focus on pure horizontal or pure vertical
    
    random.seed(42)
    
    for config_type, num_shelves, dist in shelf_configs:
        if time.time() - start_time > max_time - 60:
            break
        
        # Generate random side lengths that might fit
        # Total area constraint: sum(s_i^2) <= 1
        # Try various side length distributions
        
        # Strategy: Start with equal sides, then vary
        # If num_shelves horizontal shelves, each shelf has height = max side in shelf
        # Total height = sum of shelf heights <= 1
        
        # For a given shelf configuration, find max side lengths
        # This is complex, so let's try random samples
        
        for trial in range(50):
            if time.time() - start_time > max_time - 30:
                break
            
            # Generate random side lengths
            # Constraint: for horizontal shelves, sum of max sides per shelf <= 1
            # For vertical shelves, sum of max sides per shelf <= 1
            
            if config_type == 'h':
                # Horizontal shelves: separate in y
                # Each shelf has some squares, shelf height = max side in shelf
                # Total height <= 1
                
                # Assign squares to shelves
                shelf_assignment = []
                for shelf_idx, count in enumerate(dist):
                    shelf_assignment.extend([shelf_idx] * count)
                random.shuffle(shelf_assignment)
                
                # Try to find side lengths
                # Start with equal, then optimize
                sides = np.random.rand(n) * 0.15 + 0.05  # Random sides between 0.05 and 0.2
                
                # Try to scale up
                # Check if feasible
                shelf_max = {}
                for i in range(n):
                    h = shelf_assignment[i]
                    shelf_max[h] = max(shelf_max.get(h, 0), sides[i])
                
                total_h = sum(shelf_max.values())
                
                if total_h <= 1:
                    # Scale up
                    scale = 1.0 / total_h
                    sides_scaled = sides * scale
                    # But this might make some shelves too wide
                    # Check width constraint per shelf
                    valid = True
                    for h in range(num_shelves):
                        indices_in_shelf = [i for i in range(n) if shelf_assignment[i] == h]
                        if indices_in_shelf:
                            width = sum(sides_scaled[i] for i in indices_in_shelf)
                            if width > 1 + 1e-9:
                                valid = False
                                break
                    
                    if valid:
                        v_shelves = list(range(n))  # All different vertical shelves
                        result = try_shelf_configuration(sides_scaled, shelf_assignment, v_shelves)
                        if result is not None:
                            cx, cy = result
                            val = np.sum(sides_scaled)
                            if val > best_score:
                                best_score = val
                                best_result = (cx.copy(), cy.copy(), sides_scaled.copy())
                
                # Also try local optimization from a good starting point
                if best_result is not None and trial % 10 == 0:
                    cx0, cy0, s0 = best_result
                    cx_opt, cy_opt, s_opt, val_opt = local_optimize(cx0, cy0, s0, max_iter=2000)
                    if val_opt > best_score:
                        best_score = val_opt
                        best_result = (cx_opt.copy(), cy_opt.copy(), s_opt.copy())
            
            elif config_type == 'v':
                # Vertical shelves: separate in x
                # Each shelf has some squares, shelf width = max side in shelf
                # Total width <= 1
                
                shelf_assignment = []
                for shelf_idx, count in enumerate(dist):
                    shelf_assignment.extend([shelf_idx] * count)
                random.shuffle(shelf_assignment)
                
                sides = np.random.rand(n) * 0.15 + 0.05
                
                shelf_max = {}
                for i in range(n):
                    v = shelf_assignment[i]
                    shelf_max[v] = max(shelf_max.get(v, 0), sides[i])
                
                total_w = sum(shelf_max.values())
                
                if total_w <= 1:
                    scale = 1.0 / total_w
                    sides_scaled = sides * scale
                    valid = True
                    for v in range(num_shelves):
                        indices_in_shelf = [i for i in range(n) if shelf_assignment[i] == v]
                        if indices_in_shelf:
                            height = sum(sides_scaled[i] for i in indices_in_shelf)
                            if height > 1 + 1e-9:
                                valid = False
                                break
                    
                    if valid:
                        h_shelves = list(range(n))  # All different horizontal shelves
                        result = try_shelf_configuration(sides_scaled, h_shelves, shelf_assignment)
                        if result is not None:
                            cx, cy = result
                            val = np.sum(sides_scaled)
                            if val > best_score:
                                best_score = val
                                best_result = (cx.copy(), cy.copy(), sides_scaled.copy())
                
                if best_result is not None and trial % 10 == 0:
                    cx0, cy0, s0 = best_result
                    cx_opt, cy_opt, s_opt, val_opt = local_optimize(cx0, cy0, s0, max_iter=2000)
                    if val_opt > best_score:
                        best_score = val_opt
                        best_result = (cx_opt.copy(), cy_opt.copy(), s_opt.copy())
    
    # Strategy 2: Known good packings as starting points
    # 2x5 grid
    cx = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.1, 0.3, 0.5, 0.7, 0.9])
    cy = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8])
    s = np.full(10, 0.2)
    if check_validity(cx, cy, s):
        val = np.sum(s)
        if val > best_score:
            best_score = val
            best_result = (cx.copy(), cy.copy(), s.copy())
    
    # 4x4 grid with 6 removed and resized
    # Try 3x3 + 1
    cx = np.array([0.15, 0.5, 0.85, 0.15, 0.5, 0.85, 0.15, 0.5, 0.85, 0.5])
    cy = np.array([0.15, 0.15, 0.15, 0.5, 0.5, 0.5, 0.85, 0.85, 0.85, 0.5])
    s = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
    if check_validity(cx, cy, s):
        val = np.sum(s)
        if val > best_score:
            best_score = val
            best_result = (cx.copy(), cy.copy(), s.copy())
    
    # Final local optimization
    if best_result is not None:
        cx0, cy0, s0 = best_result
        if time.time() - start_time < max_time - 30:
            cx_opt, cy_opt, s_opt, val_opt = local_optimize(cx0, cy0, s0, max_iter=10000)
            if val_opt > best_score:
                best_score = val_opt
                best_result = (cx_opt.copy(), cy_opt.copy(), s_opt.copy())
    
    if best_result is None:
        # Fallback
        k = 4
        side = 0.25
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(10, side, dtype=np.float64)
        return centers, sides
    
    cx, cy, s = best_result
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = s.astype(np.float64)
    
    if not check_validity(cx, cy, sides):
        k = 4
        side = 0.25
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(10, side, dtype=np.float64)
        
    return centers, sides

# EVOLVE-BLOCK-END
