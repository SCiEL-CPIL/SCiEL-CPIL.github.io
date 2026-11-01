import numpy as np

N = 12

def construct():
    # We search for the minimal side length L
    # L is the side length of the outer hexagon.
    # The safe region for centers is a hexagon of side L-1.
    
    def is_inside_safe_hex(x, y, K):
        # K is the side length of the safe hexagon
        if K <= 0:
            return False
        # Flat-top hexagon inequalities
        if abs(x) > K:
            return False
        if abs(y) > (np.sqrt(3) / 2) * K:
            return False
        if abs(x / np.sqrt(3) + y / 2) > K / 2:
            return False
        if abs(-x / np.sqrt(3) + y / 2) > K / 2:
            return False
        return True

    def find_packing(L):
        K = L - 1
        if K <= 0:
            return None
        
        # Generate candidate points
        # Grid step 0.1 for precision
        step = 0.1
        max_x = K
        max_y = (np.sqrt(3) / 2) * K
        
        x_vals = np.arange(-max_x, max_x + step, step)
        y_vals = np.arange(-max_y, max_y + step, step)
        
        candidates = []
        for x in x_vals:
            for y in y_vals:
                if is_inside_safe_hex(x, y, K):
                    candidates.append((x, y))
        
        if len(candidates) < N:
            return None
            
        # Backtracking search
        placed = []
        
        def backtrack(idx, placed):
            if len(placed) == N:
                return True
            
            # Optimization: Sort candidates by distance from origin to prefer central packing?
            # Or just iterate.
            for i, (x, y) in enumerate(candidates):
                # Check overlap with already placed
                valid = True
                for (px, py) in placed:
                    dist_sq = (x - px)**2 + (y - py)**2
                    if dist_sq < 3.0 - 1e-9: # min dist sqrt(3)
                        valid = False
                        break
                
                if valid:
                    placed.append((x, y))
                    if backtrack(idx + 1, placed):
                        return True
                    placed.pop()
            
            return False

        # To make it faster, we can shuffle or sort candidates
        # Sorting by distance to center might help find a compact cluster faster
        candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
        
        if backtrack(0, placed):
            return placed
        return None

    # Search L from 4.0 downwards
    # Start with a known feasible upper bound
    best_packing = None
    best_L = None
    
    # Try L values
    for L in np.arange(4.0, 3.0, -0.1):
        packing = find_packing(L)
        if packing:
            best_packing = packing
            best_L = L
            # We found a solution. Can we go lower?
            # Continue to find the minimal L
            # But for safety, we can break if we are confident
            # Let's continue to find the true minimum in the range
    
    if best_packing is None:
        # Fallback to a very large L if search fails
        # This shouldn't happen for L=4
        best_L = 10.0
        # Generate a fallback packing (e.g. in a line)
        # But L=10 is huge. Let's just return the seed if all else fails?
        # The seed was 40.0.
        # Let's try L=5.0
        packing = find_packing(5.0)
        if packing:
            best_packing = packing
            best_L = 5.0
        else:
            # Hardcode a simple grid for 12 hexagons
            # 3 rows of 4?
            # Spacing sqrt(3)
            s = np.sqrt(3)
            coords = []
            for r in range(3):
                for c in range(4):
                    x = c * s
                    y = r * 1.5 # Vertical spacing for hex grid is 1.5 * side? No.
                    # Vertical distance between rows in hex grid is sqrt(3)/2 * 2 = sqrt(3)? No.
                    # Distance between centers is sqrt(3).
                    # Row offset x: sqrt(3)/2.
                    # Row offset y: 1.5 * 1 = 1.5?
                    # Apothem is sqrt(3)/2.
                    # Vertical distance = 2 * apothem = sqrt(3).
                    # So y = r * sqrt(3).
                    # x offset for odd rows: sqrt(3)/2.
                    if r % 2 == 1:
                        x += s / 2
                    coords.append([x, y, 0.0])
            # Center the coords
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            cx = (max(xs) + min(xs)) / 2
            cy = (max(ys) + min(ys)) / 2
            for c in coords:
                c[0] -= cx
                c[1] -= cy
            
            best_packing = coords
            best_L = 10.0 # Safe large value

    # Return in the required format
    # inner: list of [x, y, angle_degrees]
    # center: (x, y) -> (0.0, 0.0)
    # side: best_L
    # angle: 0.0
    
    inner = [[p[0], p[1], 0.0] for p in best_packing]
    return inner, (0.0, 0.0), best_L, 0.0

# EVOLVE-BLOCK-END
