<<<<<<< SEARCH
def _initial_model():
    """
    Construct a geometrically inspired initial layout.
    Strategy: 
    1. Place 4 circles in corners.
    2. Place remaining 22 in a hexagonal-like grid in the center.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 1. Corners
    # Let's start them slightly away from the absolute corner to avoid singularity issues in gradients if they are exactly at 0,0 with r>0
    corner_r = 0.1
    corners = [
        (0.1, 0.1),
        (0.9, 0.1),
        (0.1, 0.9),
        (0.9, 0.9)
    ]
    for i in range(4):
        pos[i] = corners[i]
        radii[i] = corner_r
        
    # 2. Center Grid
    # We have 22 circles left.
    # A 5x4 grid = 20 circles. Plus 2 extra.
    # Let's make a 5x5 grid for indices 4..24 (21 circles) and 1 extra at 25.
    # Actually, 4 + 22 = 26.
    # Let's use a hexagonal pattern for the center.
    
    # Simpler: 5x5 grid of small circles, then optimizer expands.
    # But to help the optimizer, let's make the center ones smaller so they don't overlap corners initially.
    
    # Let's place 22 circles in a 5x5 grid (minus 1 center? no, just 5x5 = 25 spots, we have 22).
    # Let's use a 5x5 grid for indices 4..24 (21 circles) and put the last one (25) in the center of the remaining gap or just another spot.
    
    # Better: 
    # Indices 4..8: Row y=0.2
    # Indices 9..13: Row y=0.4
    # Indices 14..18: Row y=0.6
    # Indices 19..23: Row y=0.8
    # This is 4 rows of 5 = 20 circles.
    # Plus 2 extra circles (24, 25).
    
    r_center = 0.08
    y_coords = [0.2, 0.4, 0.6, 0.8]
    x_coords = [0.2, 0.4, 0.6, 0.8, 1.0-0.2] # 0.2, 0.4, 0.6, 0.8, 0.8
    
    idx = 4
    for y in y_coords:
        for x in x_coords:
            if idx < N-2: # Leave last 2 for extras
                pos[idx] = [x, y]
                radii[idx] = r_center
                idx += 1
                
    # Now idx should be 4 + 20 = 24.
    # We have indices 24 and 25 left.
    # Place them in the "holes" of the grid or just center.
    # Let's put them at (0.5, 0.3) and (0.5, 0.7) roughly, or just (0.5, 0.5) split.
    # Actually, (0.5, 0.5) is the center of the 5x5 grid.
    # The grid points are 0.2, 0.4, 0.6, 0.8.
    # Midpoints are 0.3, 0.5, 0.7.
    # Let's place 24 at (0.5, 0.3) and 25 at (0.5, 0.7).
    # Or (0.3, 0.5) and (0.7, 0.5).
    # Let's try (0.5, 0.5) is crowded.
    # Let's place 24 at (0.3, 0.5) and 25 at (0.7, 0.5).
    
    pos[24] = [0.3, 0.5]
    radii[24] = 0.07
    pos[25] = [0.7, 0.5]
    radii[25] = 0.07
    
    # Random jitter to break symmetry
    rng = np.random.RandomState(42)
    pos += rng.uniform(-0.01, 0.01, pos.shape)
    pos = np.clip(pos, 0.05, 0.95)
    
    return pos, radii
=======
def _initial_grid():
    """
    Construct a 5x5 grid + 1 extra initial layout.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 5x5 grid
    idx = 0
    for i in range(5):
        for j in range(5):
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = 0.09
            idx += 1
            
    # 26th circle in the center gap? No, grid is full.
    # Let's just put it in a corner or shrink the grid.
    # Actually, 5x5 = 25 circles. We need 26.
    # Let's use a 5x5 grid with slightly smaller radii and put the 26th in the middle of the square, 
    # but the center of the square is occupied by the grid point (2,2).
    # Let's shift the grid or use a different layout.
    
    # Alternative: 4x4 grid = 16 circles. 10 left.
    # Let's stick to the previous logic but clean it up.
    # 4 corners + 22 center.
    
    corner_r = 0.1
    corners = [
        (0.1, 0.1),
        (0.9, 0.1),
        (0.1, 0.9),
        (0.9, 0.9)
    ]
    for i in range(4):
        pos[i] = corners[i]
        radii[i] = corner_r
        
    # 22 circles in a 5x5 grid pattern in the center region [0.2, 0.8]x[0.2, 0.8]
    # 5x5 = 25 spots. We have 22.
    # Let's place them at:
    # (0.2, 0.2), (0.4, 0.2), (0.6, 0.2), (0.8, 0.2), (1.0-0.2, 0.2) -> 5
    # (0.2, 0.4), ... -> 5
    # (0.2, 0.6), ... -> 5
    # (0.2, 0.8), ... -> 5
    # Total 20.
    # 2 left. Place at (0.5, 0.3) and (0.5, 0.7)?
    
    r_center = 0.08
    y_coords = [0.2, 0.4, 0.6, 0.8]
    x_coords = [0.2, 0.4, 0.6, 0.8, 0.8] # Note: 0.8 is repeated? No, 0.2, 0.4, 0.6, 0.8 is 4 points.
    # Let's use 5 points: 0.2, 0.4, 0.6, 0.8, 0.8 is wrong.
    # Let's use 0.2, 0.4, 0.6, 0.8, 0.8? No.
    # Let's use 0.2, 0.4, 0.6, 0.8, 0.8 is not symmetric.
    # Let's use 0.2, 0.4, 0.6, 0.8 and add 0.5?
    
    # Let's just use the existing _initial_model logic but rename it.
    # I will keep _initial_model as is and add _initial_hex and _initial_greedy.
    
    # For _initial_grid, let's do a simple 5x5 grid of 25 circles and one small one.
    # But 5x5 grid of r=0.09 doesn't fit?
    # Distance is 0.2. 2*r = 0.18. Fits.
    # But we need 26.
    # Let's put the 26th at (0.5, 0.5) with r=0.01? It will overlap with (0.5, 0.5) grid point?
    # Grid points are at (i+0.5)/5.
    # i=2, j=2 -> (0.5, 0.5).
    # So (0.5, 0.5) is occupied.
    
    # Let's shift the grid slightly or use a different count.
    # Let's use 4x5 = 20 circles in grid, 4 corners, 2 extra.
    # This is essentially what _initial_model does.
    
    # I will just return the same as _initial_model for now to avoid breaking it,
    # but I really want a distinct hexagonal one.
    
    # Let's implement _initial_hex properly.
    return _initial_model()

def _initial_hex():
    """
    Hexagonal packing initial guess.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Hexagonal packing of 26 circles.
    # Rows of 5, 5, 5, 5, 4, 2? 
    # Or 5, 5, 5, 5, 5, 1?
    # Let's try rows of 5, 5, 5, 5, 4, 2 = 26.
    # Or 5, 5, 5, 5, 5, 1 = 26.
    
    # Let's try to pack them tightly.
    # r = 0.1
    # Horizontal dist = 2*r = 0.2
    # Vertical dist = sqrt(3)*r = 0.1732
    
    r = 0.1
    rows = [5, 5, 5, 5, 4, 2]
    idx = 0
    y = 0.1
    for i, count in enumerate(rows):
        x_start = 0.1
        if i % 2 == 1:
            x_start = 0.1 + r # Shift half step
            
        for j in range(count):
            if idx >= N: break
            pos[idx, 0] = x_start + j * 2 * r
            pos[idx, 1] = y
            radii[idx] = r
            idx += 1
        y += np.sqrt(3) * r
        
    # Adjust positions to fit in [0,1]^2
    # The width used is 5*0.2 = 1.0. So x goes from 0.1 to 0.9. Fits.
    # The height used is 6 rows * 0.1732 = 1.039. Too tall.
    # Need to reduce r or rows.
    
    # Let's try r = 0.09
    # Height = 6 * 0.09 * 1.732 = 0.936. Fits.
    # Width = 5 * 0.09 * 2 = 0.9. Fits.
    
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.09
    rows = [5, 5, 5, 5, 4, 2]
    idx = 0
    y = r
    for i, count in enumerate(rows):
        x_start = r
        if i % 2 == 1:
            x_start = r + r # Shift by r? Hex shift is r * sqrt(3)/2? No, horizontal shift is r.
            # In hex packing, rows are shifted by r.
            pass 
            
        for j in range(count):
            if idx >= N: break
            pos[idx, 0] = x_start + j * 2 * r
            pos[idx, 1] = y
            radii[idx] = r
            idx += 1
        y += np.sqrt(3) * r
        
    # Center the whole structure
    cx = np.mean(pos[:, 0])
    cy = np.mean(pos[:, 1])
    pos[:, 0] += 0.5 - cx
    pos[:, 1] += 0.5 - cy
    
    # Clip to bounds
    pos = np.clip(pos, r, 1.0 - r)
    
    return pos, radii

def _initial_greedy():
    """
    Greedy placement: place circles one by one at the position that allows the largest radius.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # This is slow in Python, but N=26 is small.
    # We will simulate on a grid of candidate points.
    # Or just place them randomly and let the optimizer fix it?
    # Let's try a simple random non-overlapping placement.
    
    rng = np.random.RandomState(123)
    
    for i in range(N):
        # Try to place circle i
        placed = False
        for _ in range(1000):
            # Random position
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            
            # Max radius based on walls
            r_max_wall = min(x, 1-x, y, 1-y)
            
            # Max radius based on existing circles
            r_max_circ = r_max_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                # We want dist >= r_new + r_old
                # r_new <= dist - r_old
                r_possible = dist - radii[j]
                if r_possible < r_max_circ:
                    r_max_circ = r_possible
                    
            if r_max_circ > 0.05: # Minimum radius
                pos[i, 0] = x
                pos[i, 1] = y
                radii[i] = r_max_circ * 0.9 # Shrink slightly to ensure non-overlap
                placed = True
                break
                
        if not placed:
            # Fallback: place in a small spot
            pos[i, 0] = 0.5
            pos[i, 1] = 0.5
            radii[i] = 0.01
            
    return pos, radii
>>>>>>> REPLACE

Now I need to update `construct` to use these multiple initializers.

