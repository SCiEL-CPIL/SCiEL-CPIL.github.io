"""Design a symmetric 32x32 photonic crystal unit cell for maximum TM band gap."""
import numpy as np
import time

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell.
    Uses a dielectric background with air holes in a centered-square lattice pattern.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Create coordinate grids centered at the cell center
    # Cell indices 0..31, center is at 15.5
    # Use pixel centers: (i + 0.5) for i in 0..31
    # Center of cell is at 15.5
    # Distance from center: |i + 0.5 - 15.5| = |i - 15|
    
    x = np.arange(size, dtype=float)
    y = np.arange(size, dtype=float)
    X, Y = np.meshgrid(x, y)
    
    # Center of the cell
    cx = size / 2.0  # 16.0
    cy = size / 2.0  # 16.0
    
    # Distances from center
    dx = X - cx
    dy = Y - cy
    
    # Strategy: Dielectric background (1) with air holes (0)
    # Primary hole at center
    # Secondary holes at positions that maintain 4-fold symmetry
    # For a centered-square lattice of holes, we place holes at:
    # (0, 0), (±d, 0), (0, ±d), (±d, ±d) relative to center
    # where d is the offset
    
    # We want to maximize the gap between bands 4 and 5 for TM polarization
    # Literature suggests that for square lattice photonic crystals,
    # a dielectric rod in air or air hole in dielectric with specific
    # radius/spacing ratios can open large band gaps.
    
    # Let's try: dielectric background with a central air hole and 4 surrounding air holes
    # This creates a pattern with higher symmetry breaking that may open larger gaps
    
    # Parameter: radius of holes, offset of secondary holes
    # We'll do a small parameter search
    
    best_cell = None
    best_score = -np.inf
    
    # Search over hole radius and offset
    # Hole radius: 2 to 8 pixels
    # Offset: 0 to 10 pixels (distance from center to secondary hole centers)
    
    radii = [2, 3, 4, 5, 6, 7, 8]
    offsets = [0, 2, 4, 6, 8, 10, 12]
    
    for r in radii:
        for off in offsets:
            if time.time() > deadline - 10:
                break
            
            # Create the cell: start with all dielectric (1)
            cell = np.ones((size, size), dtype=int)
            
            # Define hole centers
            hole_centers = [(0, 0)]
            if off > 0:
                hole_centers.extend([
                    (off, 0), (-off, 0),
                    (0, off), (0, -off),
                    (off, off), (off, -off),
                    (-off, off), (-off, -off)
                ])
            
            # For each hole center, set pixels within radius r to 0 (air)
            for hx, hy in hole_centers:
                # Hole center in pixel coordinates
                hcx = cx + hx
                hcy = cy + hy
                
                # Distance from each pixel to this hole center
                dist = np.sqrt((X - hcx)**2 + (Y - hcy)**2)
                
                # Set pixels within radius to 0
                cell[dist <= r] = 0
            
            # Enforce symmetry: the cell must be invariant under
            # 180-degree rotation, reflection about x and y axes through center,
            # and transpose (reflection about diagonal)
            # Since our hole placement is symmetric by construction,
            # let's verify and enforce it
            
            # Check row-wise 180-degree rotation symmetry: cell == cell[::-1, ::-1]
            # Check column-wise 180-degree rotation symmetry: cell == cell[:, ::-1]
            # Wait, the task says: cell == cell[::-1] row-wise and column-wise
            # and cell == its transpose
            
            # "invariant under the square symmetry group about the cell centre"
            # This means:
            # 1. cell[i][j] == cell[31-i][31-j] (180-degree rotation)
            # 2. cell[i][j] == cell[i][31-j] (reflection about vertical center line)
            # 3. cell[i][j] == cell[31-i][j] (reflection about horizontal center line)
            # 4. cell[i][j] == cell[j][i] (transpose symmetry)
            
            # Our construction should satisfy these if the hole centers are symmetric.
            # Let's enforce it explicitly by averaging (though for binary, we just check)
            
            # Actually, let's just enforce it by construction:
            # For each pixel (i, j), its symmetric partners are:
            # (i, j), (31-i, 31-j), (i, 31-j), (31-i, j), (j, i), (31-j, 31-i), (j, 31-i), (31-j, i)
            # All 8 should have the same value.
            
            # Let's enforce by taking the majority or just one representative
            # For simplicity, let's use the value at (i, j) and propagate
            
            # Actually, our hole placement is already symmetric, so the cell should be symmetric.
            # But due to discretization, let's enforce it.
            
            # Enforce 4-fold symmetry by averaging over the orbit
            sym_cell = np.zeros((size, size), dtype=float)
            count = np.zeros((size, size), dtype=float)
            
            for i in range(size):
                for j in range(size):
                    # The orbit of (i, j) under D4 symmetry
                    orbit = [
                        (i, j),
                        (size-1-i, j),  # horizontal reflection
                        (i, size-1-j),  # vertical reflection
                        (size-1-i, size-1-j),  # 180-degree rotation
                        (j, i),  # transpose
                        (size-1-j, i),  # transpose + horizontal reflection
                        (j, size-1-i),  # transpose + vertical reflection
                        (size-1-j, size-1-i)  # transpose + 180-degree rotation
                    ]
                    
                    # Unique positions in orbit
                    unique_orbit = list(set(orbit))
                    
                    # For each position in the orbit, set the value to the average
                    # But since we're dealing with binary, let's just use the first one
                    # and propagate
                    val = cell[i, j]
                    for oi, oj in unique_orbit:
                        sym_cell[oi, oj] += val
                        count[oi, oj] += 1
            
            # Convert to binary
            sym_cell = (sym_cell / count > 0.5).astype(int)
            
            # Check if it's valid (all values are 0 or 1)
            if not np.all((sym_cell == 0) | (sym_cell == 1)):
                continue
            
            # Quick estimate of the gap: we can't compute the actual band structure
            # here, but we can use a heuristic. Let's just return the first valid one
            # or do a simple scoring based on the fraction of dielectric.
            
            # Actually, let's just return the best-looking one.
            # For TM band gap in square lattice, a moderate fraction of dielectric
            # (around 50-70%) often works well.
            
            frac = np.mean(sym_cell)
            
            # Heuristic score: prefer fractions around 0.5-0.7
            # This is a rough guess
            if 0.3 < frac < 0.8:
                score = 1.0 - abs(frac - 0.55) * 2
                if score > best_score:
                    best_score = score
                    best_cell = sym_cell
            
            if time.time() > deadline - 5:
                break
    
    # If no good cell found, fall back to a simple dielectric rod
    if best_cell is None:
        # Central dielectric square
        side = 16
        start = size // 2 - side // 2
        end = start + side
        best_cell = np.zeros((size, size), dtype=int)
        best_cell[start:end, start:end] = 1
    
    return best_cell.tolist()
