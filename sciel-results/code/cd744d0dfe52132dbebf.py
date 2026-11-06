<<<<<<< SEARCH
    def generate_cell_from_radii(radii):
        """
        Generate a 32x32 cell from a 4x4 grid of circular disks.
        radii: 16 values, each in [0, 0.5] (max radius for 8x8 block is 4 pixels, so 0.5 in normalized units where block is 1x1).
        The blocks are 8x8 pixels.
        """
        cell = np.zeros((size, size), dtype=int)
        block_size = 8
        grid_size = 4
        
        for i in range(grid_size):
            for j in range(grid_size):
                r = radii[i * grid_size + j]
                # Center of the block in pixel coordinates
                cx = i * block_size + block_size / 2.0
                cy = j * block_size + block_size / 2.0
                
                # Local coordinates within the block
                # We iterate over the 8x8 block
                for di in range(block_size):
                    for dj in range(block_size):
                        pi_px = i * block_size + di
                        pj_px = j * block_size + dj
                        # Pixel center relative to block center
                        dx = (di + 0.5) - (block_size / 2.0)
                        dy = (dj + 0.5) - (block_size / 2.0)
                        dist = np.sqrt(dx * dx + dy * dy)
                        # r is in units of block size (0 to 0.5 means 0 to 4 pixels)
                        # dist is in pixels. Max dist in 8x8 block is sqrt(3.5^2 + 3.5^2) ~ 4.95
                        # If r is 0.5 (normalized), radius in pixels is 0.5 * block_size = 4
                        radius_px = r * block_size
                        if dist < radius_px:
                            cell[pi_px, pj_px] = 1
        
        # Enforce symmetry about the center of the 32x32 cell
        # The 4x4 grid is symmetric if radii[i,j] == radii[3-i, 3-j] etc?
        # The cell itself must be symmetric.
        # My generation method produces a cell that is symmetric with respect to the block centers,
        # but the overall 32x32 cell might not be perfectly symmetric if the radii are not symmetric.
        # We enforce symmetry on the final cell.
        cell_rot = cell[::-1, ::-1].copy()
        cell_hflip = cell[:, ::-1].copy()
        cell_vflip = cell[::-1, :].copy()
        result = (cell + cell_rot + cell_hflip + cell_vflip)
        result = (result >= 2).astype(int)
        return result
=======
    def generate_cell_from_params(params):
        """
        Generate a 32x32 cell from a 4x4 grid of superellipses.
        params: 32 values.
        - 16 radii (scale factors) in [0, 1]
        - 16 shape exponents (n) in [1, 8] (1=circle, 2=square-ish, >2=more square)
        """
        cell = np.zeros((size, size), dtype=int)
        block_size = 8
        grid_size = 4
        
        radii = params[:16]
        exponents = params[16:]
        
        for i in range(grid_size):
            for j in range(grid_size):
                r = radii[i * grid_size + j]
                n = exponents[i * grid_size + j]
                
                # Center of the block in pixel coordinates
                # We define the shape in the local coordinate system of the block
                # Local coords x, y in [-0.5, 0.5]
                # Superellipse: |x|/a^n + |y|/b^n <= 1
                # Here a=b=r*0.5 (normalized to block size 1)
                # Actually, let's work in pixel coordinates for precision.
                # Block center is at (i*8 + 3.5, j*8 + 3.5)
                # We check each pixel in the block.
                
                for di in range(block_size):
                    for dj in range(block_size):
                        pi_px = i * block_size + di
                        pj_px = j * block_size + dj
                        
                        # Pixel center relative to block center (in pixels)
                        # Block center is at 3.5, 3.5 within the 8x8 block
                        dx = (di + 0.5) - 3.5
                        dy = (dj + 0.5) - 3.5
                        
                        # Normalize to block size [0, 1]
                        # Max distance from center to corner is sqrt(3.5^2+3.5^2) ~ 4.95
                        # Max distance to edge is 3.5
                        # Let's scale such that r=1.0 fills the block width (diameter 8)
                        # So a = r * 4.0 pixels? No, if r=1, we want it to be big.
                        # Let's define a = r * 3.5 (radius to edge)
                        a = r * 3.5
                        
                        if a < 1e-6:
                            continue
                            
                        # Superellipse equation: (|dx|/a)^n + (|dy|/a)^n <= 1
                        val = (abs(dx) / a) ** n + (abs(dy) / a) ** n
                        if val <= 1.0:
                            cell[pi_px, pj_px] = 1
        
        # Enforce symmetry about the center of the 32x32 cell
        # The 4x4 grid layout itself is symmetric if we enforce params[i,j] == params[3-i, 3-j]
        # But we generate the cell and then enforce pixel symmetry to be safe.
        # The task requires: cell == cell[::-1] (row/col flip) and cell == cell.T
        # This is D4 symmetry.
        cell_rot = cell[::-1, ::-1].copy()
        cell_hflip = cell[:, ::-1].copy()
        cell_vflip = cell[::-1, :].copy()
        # Average or majority vote? Majority vote is better for binary.
        # Sum > 2 means at least 3 of 4 symmetric positions are 1.
        result = (cell + cell_rot + cell_hflip + cell_vflip)
        result = (result >= 2).astype(int)
        return result
>>>>>>> REPLACE

