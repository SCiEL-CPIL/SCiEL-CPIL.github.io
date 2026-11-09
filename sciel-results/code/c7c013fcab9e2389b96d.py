"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def _get_grid_resolution(container):
    """
    Determine a grid resolution that is fine enough to allow packing 
    but coarse enough to be fast.
    We aim for a grid size that is manageable (e.g., < 500k cells).
    """
    W, H, D = container
    # Target max cells ~ 200,000 to 500,000
    target_cells = 300000
    total_vol = W * H * D
    if total_vol <= 0:
        return 1.0
    
    # Scale factor such that (W/scale)*(H/scale)*(D/scale) <= target_cells
    # scale^3 >= total_vol / target_cells
    scale = (total_vol / target_cells) ** (1.0/3.0)
    
    # We want the resolution (step size) to be integer if possible, or close.
    # Let's pick a step size 'step' such that grid dimensions are integers.
    # Grid dims: ceil(W/step), etc.
    
    # Try a few candidate step sizes
    best_step = 1.0
    best_score = float('inf')
    
    # Heuristic: step should be around the scale calculated above
    # But we need integer grid coordinates.
    # Let's just pick a step that makes the grid dimensions reasonable.
    
    # If boxes are small, we need finer grid. If boxes are large, coarser is ok.
    # For now, let's use a fixed step based on container size to keep it simple and robust.
    # If container is 100x100x100, step=5 -> 20x20x20 = 8000 cells. Very fast.
    # If container is 10x10x10, step=1 -> 10x10x10 = 1000 cells.
    
    # Let's aim for grid dimension ~ 30-50 in largest axis.
    max_dim = max(W, H, D)
    target_dim = 40
    step = max_dim / target_dim
    
    # Round step to nearest 0.5 or 1.0 for simplicity? 
    # Actually, let's just use the float step and map box coords to grid indices.
    # However, for "no overlap" in grid, we need integer cells.
    # So we must discretize box dimensions too.
    
    # Let's refine: 
    # We will use a grid where each cell is `step` x `step` x `step`.
    # Box dimensions w, h, d are mapped to gw, gh, gd = ceil(w/step), etc.
    # This might waste space if step is large.
    
    # Alternative: Use a "Skyline" or "Maximal Empty Box" approach? 
    # The prompt asks for a DIFFERENT strategy. Voxel is distinct from Shelf/EPA.
    
    # Let's stick to Voxel but choose step carefully.
    # If step is too large, we can't fit small boxes.
    # Minimum dimension of any box should be >= step? No, we can just snap.
    
    # Let's try to make step a divisor of container dims if possible? 
    # Simpler: Just pick step = 1.0 if container is small (< 50 in all dims).
    # If container is large, use larger step.
    
    if max(W, H, D) <= 100:
        step = 1.0
    elif max(W, H, D) <= 500:
        step = 5.0
    else:
        step = 10.0
        
    # Ensure grid is not huge
    gw = int(np.ceil(W / step))
    gh = int(np.ceil(H / step))
    gd = int(np.ceil(D / step))
    cells = gw * gh * gd
    if cells > 1000000:
        # Increase step
        while cells > 1000000:
            step *= 1.5
            gw = int(np.ceil(W / step))
            gh = int(np.ceil(H / step))
            gd = int(np.ceil(D / step))
            cells = gw * gh * gd
            
    return step

def load(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Sort boxes by volume descending (largest first)
    # This is a standard heuristic for bin packing.
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
    
    # Pre-compute rotations for each box
    box_rots = []
    for idx, (w, h, d) in indexed_boxes:
        rots = list(set(itertools.permutations((w, h, d))))
        # Sort rotations by volume? No, volume is same. 
        # Maybe sort by "flatness" or just keep as is.
        box_rots.append((idx, rots))
        
    # 2. Initialize Grid
    step = _get_grid_resolution(container)
    gw = int(np.ceil(W / step))
    gh = int(np.ceil(H / step))
    gd = int(np.ceil(D / step))
    
    # Occupancy grid: 0 for empty, 1 for occupied
    # We use uint8 for memory efficiency
    grid = np.zeros((gw, gh, gd), dtype=np.uint8)
    
    placements = [None] * len(boxes)
    
    # Helper to check if a box fits at a specific grid position
    # Box grid dims: bw, bh, bd
    # Pos: gx, gy, gz
    def fits_in_grid(bw, bh, bd, gx, gy, gz):
        if gx + bw > gw or gy + bh > gh or gz + bd > gd:
            return False
        # Check if any cell in the sub-cube is occupied
        # grid[gx:gx+bw, gy:gy+bh, gz:gz+bd]
        # Sum of occupied cells should be 0
        return np.sum(grid[gx:gx+bw, gy:gy+bh, gz:gz+bd]) == 0

    # Helper to place box in grid
    def place_in_grid(bw, bh, bd, gx, gy, gz):
        grid[gx:gx+bw, gy:gy+bh, gz:gz+bd] = 1

    # 3. Placement Loop
    for idx, rots in box_rots:
        if time.time() > deadline:
            break
            
        w, h, d = boxes[idx]
        
        # Convert box dims to grid dims
        # We must ensure the box doesn't "overflow" the container in grid space
        # due to rounding up.
        placed = False
        
        # Try each rotation
        for rw, rh, rd in rots:
            bw = int(np.ceil(rw / step))
            bh = int(np.ceil(rh / step))
            bd = int(np.ceil(rd / step))
            
            # Check if box is too big for grid
            if bw > gw or bh > gh or bd > gd:
                continue
                
            # Search for position
            # Strategy: Scan from bottom (gz=0), left (gx=0), front (gy=0)
            # This is "First Fit" in grid space.
            # To make it "Best Fit", we could scan and pick the one with least "waste"
            # but First Fit is faster. Let's stick to First Fit for speed, 
            # but scan in a specific order (z, then x, then y? or z, y, x?).
            # Usually, packing towards the origin (0,0,0) is good.
            
            # We can optimize the search by only checking positions where the 
            # "anchor" (bottom-left-front corner) is in an empty cell.
            
            # Iterate through potential anchor positions
            # To speed up, we can skip rows/cols that are fully occupied?
            # For now, simple triple loop but with early exit.
            
            found_pos = None
            
            # Scan z (height) from 0 to gd-bd
            for gz in range(gd - bd + 1):
                if time.time() > deadline:
                    break
                # Check if any part of this z-slice is free?
                # If the entire slice [0:gw, 0:gh, gz] is full, we can break? 
                # Not necessarily, but if grid[..., gz] is all 1s, we can skip?
                # Actually, if grid[..., gz] has no 0s, we can't place anything starting at gz.
                # But we might place something starting at gz+1.
                
                # Check if there's any empty space in this z-layer for the box footprint?
                # This is complex. Let's just scan x and y.
                
                # Optimization: If the slice grid[:, :, gz] is completely occupied, 
                # we can't place a box with height 1 starting here, but we can place 
                # a box with height > 1? No, the anchor is at gz. 
                # If grid[gx, gy, gz] is 1, we can't place there.
                
                for gx in range(gw - bw + 1):
                    if time.time() > deadline:
                        break
                    for gy in range(gh - bh + 1):
                        if fits_in_grid(bw, bh, bd, gx, gy, gz):
                            found_pos = (gx, gy, gz)
                            break
                    if found_pos:
                        break
                if found_pos:
                    break
                    
            if found_pos:
                gx, gy, gz = found_pos
                place_in_grid(bw, bh, bd, gx, gy, gz)
                # Record placement in real coordinates
                # The box occupies [gx*step, (gx+bw)*step] in x, etc.
                # We should report the actual box dims (rw, rh, rd) and position.
                # Position: x = gx * step, y = gy * step, z = gz * step
                placements[idx] = (0, gx * step, gy * step, gz * step, rw, rh, rd)
                placed = True
                break # Found a rotation that fits, stop trying others for this box
                
        if not placed:
            placements[idx] = (-1, 0, 0, 0, w, h, d)

    # Fill any remaining None with -1 (if loop broke early due to timeout)
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)

    return placements
# EVOLVE-BLOCK-END
