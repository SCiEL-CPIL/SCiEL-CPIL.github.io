import numpy as np
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    time_limit = 580.0  # Safety margin for 600s limit
    
    # Candidate side lengths for the 10 squares.
    # We try to maximize sum(s).
    # A common high-performing structure for 10 squares is a mix of sizes.
    # Let's generate a few candidate "size profiles" and layouts.
    
    # Profile 1: 4 large, 6 small (2x2 grid of large, surrounding small?)
    # Profile 2: 2 large, 8 small
    # Profile 3: 10 equal (baseline)
    # Profile 4: 3 large, 7 small
    
    # We will define specific layouts for these profiles and then refine.
    
    def is_valid(cx, cy, s, tol=1e-7):
        # Check bounds
        if np.any(cx - s/2 < -tol) or np.any(cx + s/2 > 1 + tol):
            return False
        if np.any(cy - s/2 < -tol) or np.any(cy + s/2 > 1 + tol):
            return False
        
        # Check overlaps
        # Vectorized check for efficiency
        # dist_x[i,j] = |cx_i - cx_j|
        # dist_y[i,j] = |cy_i - cy_j|
        # overlap condition: dist_x < (s_i + s_j)/2 - tol AND dist_y < (s_i + s_j)/2 - tol
        
        cx = cx[:, np.newaxis]
        cy = cy[:, np.newaxis]
        s = s[:, np.newaxis]
        
        dx = np.abs(cx - cx.T)
        dy = np.abs(cy - cy.T)
        
        # t[i,j] = (s_i + s_j)/2 - tol
        t = (s + s.T) / 2.0 - tol
        
        # Overlap if dx < t AND dy < t
        # We need to check if ANY pair (i != j) satisfies this
        # Mask out diagonal
        mask = ~np.eye(n, dtype=bool)
        
        overlaps = (dx < t) & (dy < t) & mask
        
        if np.any(overlaps):
            return False
            
        return True

    def refine(cx, cy, s):
        """
        Local refinement using coordinate descent.
        Only accepts moves that maintain feasibility.
        Tries to expand squares and move them to fill gaps.
        """
        cx = cx.copy()
        cy = cy.copy()
        s = s.copy()
        
        improved = True
        max_iter = 10000
        iter_count = 0
        
        # Step size for expansion
        eps_s = 1e-4
        # Step size for movement
        eps_c = 1e-4
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Time check
            if time.time() - start_time > time_limit - 5:
                break
            
            # 1. Try to expand each square
            for i in range(n):
                # Try expanding s[i] slightly
                new_s = s.copy()
                new_s[i] += eps_s
                if new_s[i] > 1.0:
                    continue
                    
                if is_valid(cx, cy, new_s):
                    s = new_s
                    improved = True
            
            # 2. Try to move each square to center of its "free space" or towards center
            # A simple heuristic: try moving in 4 directions
            for i in range(n):
                for dx_dir, dy_dir in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    new_cx = cx.copy()
                    new_cy = cy.copy()
                    new_cx[i] += dx_dir * eps_c
                    new_cy[i] += dy_dir * eps_c
                    
                    if is_valid(new_cx, new_cy, s):
                        cx = new_cx
                        cy = new_cy
                        improved = True
            
            # 3. Try to move squares to reduce "wasted" space?
            # For now, simple expansion and nudging is enough to polish.
            # To get better results, we might need to try "shuffling" which is hard in local search.
            
        return cx, cy, s

    best_score = -np.inf
    best_result = None

    # Define a few structural layouts to start from.
    # Layout A: 2x5 grid of equal squares
    s_A = np.full(10, 0.2)
    cx_A = np.array([0.1, 0.3, 0.5, 0.7, 0.9] * 2)
    cy_A = np.array([0.1]*5 + [0.3]*5)
    # This is a 2x5 grid.
    # Wait, 0.2 side means width 0.2. 5 columns = 1.0 width. 2 rows = 0.4 height.
    # This leaves lots of space. We can expand.
    
    # Layout B: 1 large square (0.5x0.5) in corner, rest fill remaining?
    # Let's try a "guillotine" cut style.
    # Cut 1: Vertical split at x=0.5. Left half 0.5x1.0, Right half 0.5x1.0.
    # Left half: 2 squares of 0.5x0.5.
    # Right half: 8 squares? 4x2 grid of 0.25x0.25?
    # 4 cols * 0.25 = 1.0 width? No, right half width is 0.5.
    # So 2 cols of 0.25. 4 rows of 0.25.
    # Total 8 squares of 0.25.
    # Sum = 2*0.5 + 8*0.25 = 1.0 + 2.0 = 3.0.
    
    s_B = np.array([0.5, 0.5] + [0.25]*8)
    cx_B = np.array([0.25, 0.25, 
                      0.625, 0.875, 0.625, 0.875, 
                      0.625, 0.875, 0.625, 0.875])
    cy_B = np.array([0.25, 0.75, 
                      0.125, 0.125, 0.375, 0.375, 
                      0.625, 0.625, 0.875, 0.875])
    
    # Layout C: 3 large (0.4), 7 small?
    # Let's try 3 squares of 0.4 in a row? 3*0.4 = 1.2 > 1.0. No.
    # 2 squares of 0.4 in a row (0.8 width).
    # Remaining 0.2 width strip?
    
    # Let's just refine Layout A and B and a few random valid ones.
    
    layouts = []
    
    # Add Layout A
    layouts.append((cx_A, cy_A, s_A))
    
    # Add Layout B
    layouts.append((cx_B, cy_B, s_B))
    
    # Add a random valid layout generator
    # Start with 10 tiny squares, place them randomly, ensure no overlap.
    def generate_random_valid():
        cx = np.zeros(10)
        cy = np.zeros(10)
        s = np.zeros(10)
        
        # Place one by one
        for i in range(10):
            placed = False
            attempts = 0
            while not placed and attempts < 1000:
                attempts += 1
                # Try a random size between 0.1 and 0.3
                s_i = 0.1 + 0.2 * np.random.rand()
                
                # Random position
                cx_i = s_i/2 + (1 - s_i) * np.random.rand()
                cy_i = s_i/2 + (1 - s_i) * np.random.rand()
                
                # Check against existing
                valid = True
                for j in range(i):
                    t = (s[i] + s[j])/2 - 1e-7
                    if abs(cx_i - cx[j]) < t and abs(cy_i - cy[j]) < t:
                        valid = False
                        break
                
                if valid:
                    cx[i] = cx_i
                    cy[i] = cy_i
                    s[i] = s_i
                    placed = True
            
            if not placed:
                # Fallback: very small square in corner
                s_i = 0.01
                cx[i] = s_i/2
                cy[i] = s_i/2
                s[i] = s_i
        
        return cx, cy, s

    # Generate 5 random layouts
    for _ in range(5):
        layouts.append(generate_random_valid())
    
    # Refine each layout
    for cx, cy, s in layouts:
        if time.time() - start_time > time_limit - 10:
            break
            
        ref_cx, ref_cy, ref_s = refine(cx, cy, s)
        
        score = np.sum(ref_s)
        if score > best_score:
            best_score = score
            best_result = (ref_cx, ref_cy, ref_s)
            
        # Verify validity of best result
        if not is_valid(ref_cx, ref_cy, ref_s):
            # Should not happen if refine is correct, but check just in case
            print("Refinement produced invalid result!")
            # Fall back to initial layout if valid
            if is_valid(cx, cy, s):
                if np.sum(s) > best_score:
                    best_score = np.sum(s)
                    best_result = (cx, cy, s)

    # If no valid result found, use the grid fallback
    if best_result is None:
        k = 4
        side = 1.0/k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1)
        sides = np.full(n, side)
        return centers, sides

    cx, cy, s = best_result
    
    # Final safety check
    if not is_valid(cx, cy, s):
        # Fallback to grid
        k = 4
        side = 1.0/k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        s = np.full(n, side)
        
    centers = np.stack([cx, cy], axis=1)
    return centers, s
