"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def design(problem):
    """Dual-structure photonic crystal with 4x4 superlattice of air holes in dielectric background."""
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    
    # Deadline guard
    t0 = time.time()
    deadline = 220.0  # Safety margin before 230s kill
    
    # Strategy: Create a 4x4 superlattice pattern within the 32x32 cell
    # This gives 8x8 pixels per sub-cell, allowing varied hole sizes
    # The pattern must be symmetric under D4 (4-fold rotation + reflections)
    
    # Base pattern: dielectric background with air holes
    # Use a 4x4 grid of sub-cells, each 8x8 pixels
    sub_size = size // 4  # 8
    
    # Define hole patterns for each of the 16 sub-cells
    # Due to D4 symmetry, we only need to define 4 unique sub-cells:
    # (0,0), (0,1), (1,0), (1,1) and mirror/rotate for the rest
    # Actually for 4x4, the orbit under D4 has size 1, 4, or 8
    # Let's use a simpler approach: define a base 8x8 pattern and tile it
    
    # Try a pattern with varying hole sizes to create complex Fourier spectrum
    # The key is to have strong components at wavevectors that couple bands 4 and 5
    
    # Design: 4x4 array of circular air holes with modulated radii
    # Radius modulation creates additional diffraction orders
    
    # Define radius for each of the 16 positions in the 4x4 superlattice
    # Using a pattern that breaks translational symmetry while maintaining D4
    radii = [
        [0.35, 0.25, 0.25, 0.35],
        [0.25, 0.40, 0.40, 0.25],
        [0.25, 0.40, 0.40, 0.25],
        [0.35, 0.25, 0.25, 0.35],
    ]
    
    # Build the 32x32 cell
    cell = [[0] * size for _ in range(size)]
    
    for i in range(size):
        for j in range(size):
            # Determine which sub-cell this pixel belongs to
            si = i // sub_size  # 0..3
            sj = j // sub_size  # 0..3
            
            # Local coordinates within sub-cell (0..7)
            li = i % sub_size
            lj = j % sub_size
            
            # Center of this sub-cell in the full cell
            ci = si * sub_size + sub_size / 2.0
            cj = sj * sub_size + sub_size / 2.0
            
            # Pixel center
            px = i + 0.5
            py = j + 0.5
            
            # Distance from sub-cell center
            dx = px - ci
            dy = py - cj
            
            # Normalized distance (0 to ~0.707 for corner)
            dist = (dx * dx + dy * dy) ** 0.5 / sub_size
            
            # If within the hole radius, it's air (0); otherwise dielectric (1)
            r = radii[si][sj]
            if dist < r:
                cell[i][j] = 0  # air
            else:
                cell[i][j] = 1  # dielectric
    
    # Verify symmetry - fix any asymmetries by enforcing D4
    # The pattern should already be symmetric due to the radii array symmetry
    # Let's verify and enforce if needed
    
    # Check row-reversal symmetry
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[size - 1 - i][j]:
                # Average or just take one (they should be same by construction)
                pass
            if cell[i][j] != cell[i][size - 1 - j]:
                pass
            if cell[i][j] != cell[j][i]:
                pass
    
    # Enforce strict symmetry by averaging over D4 orbit
    cell_sym = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            # D4 orbit: (i,j), (size-1-i,j), (i,size-1-j), (size-1-i,size-1-j),
            # (j,i), (size-1-j,i), (j,size-1-i), (size-1-j,size-1-i)
            orbit = [
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, i),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i),
            ]
            val = sum(cell[oi][oj] for oi, oj in orbit) / len(orbit)
            cell_sym[i][j] = 1 if val > 0.5 else 0
    
    cell = [[int(v) for v in row] for row in cell_sym]
    
    # If we have time, try a few alternative patterns
    if time.time() - t0 < deadline - 30:
        # Try alternative: checkerboard-like pattern with larger holes
        alt_radii = [
            [0.45, 0.30, 0.30, 0.45],
            [0.30, 0.50, 0.50, 0.30],
            [0.30, 0.50, 0.50, 0.30],
            [0.45, 0.30, 0.30, 0.45],
        ]
        cell_alt = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                si = i // sub_size
                sj = j // sub_size
                ci = si * sub_size + sub_size / 2.0
                cj = sj * sub_size + sub_size / 2.0
                px = i + 0.5
                py = j + 0.5
                dx = px - ci
                dy = py - cj
                dist = (dx * dx + dy * dy) ** 0.5 / sub_size
                r = alt_radii[si][sj]
                cell_alt[i][j] = 0 if dist < r else 1
        
        # Enforce symmetry
        cell_alt_sym = [[0.0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                orbit = [
                    (i, j), (size - 1 - i, j), (i, size - 1 - j), (size - 1 - i, size - 1 - j),
                    (j, i), (size - 1 - j, i), (j, size - 1 - i), (size - 1 - j, size - 1 - i),
                ]
                val = sum(cell_alt[oi][oj] for oi, oj in orbit) / len(orbit)
                cell_alt_sym[i][j] = 1 if val > 0.5 else 0
        cell_alt = [[int(v) for v in row] for row in cell_alt_sym]
        
        # We can't easily score here, so return the first design
        # But we could try to estimate which is better by counting dielectric fraction
        # Higher contrast patterns often give larger gaps
        frac1 = sum(sum(row) for row in cell) / (size * size)
        frac2 = sum(sum(row) for row in cell_alt) / (size * size)
        # Prefer the one with more dielectric (higher average eps)
        if frac2 > frac1:
            cell = cell_alt
    
    return cell
# EVOLVE-BLOCK-END
