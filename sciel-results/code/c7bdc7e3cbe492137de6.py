"""Skyline / floor-map based 3D container loading."""
import numpy as np
from itertools import product

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Use a 2D height map with coordinate compression.
    # We maintain a grid where each cell (i, j) corresponds to a rectangle in x-z space
    # defined by x_coords[i] to x_coords[i+1] and z_coords[j] to z_coords[j+1].
    # height_map[i][j] stores the minimum y (top surface) of boxes covering that cell.
    # Initially, all heights are 0.
    
    # Initialize coordinate lists
    x_coords = [0, W]
    z_coords = [0, D]
    
    # height_map is a 2D list of lists. Initially 1x1 grid.
    height_map = [[0]]
    
    placed = []  # list of (x, y, z, w, h, d) for reporting, but we mostly use height_map
    
    # Sort boxes by volume descending
    box_indices = list(range(n))
    box_indices.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    final_placements = [None] * n
    
    def update_height_map(x, z, w, h, d, new_top_y):
        """Update the height map for a box placed at (x, z) with dims (w, h, d) and top at new_top_y."""
        nonlocal x_coords, z_coords, height_map
        
        # Find or insert x coordinates
        # We need to split the grid at x and x+w, z and z+d
        # This is complex with a dynamic grid. 
        # Alternative: Since N is small (60), we can just rebuild the grid coordinates and map after each placement?
        # No, that's expensive.
        
        # Let's use a simpler representation for the height map that supports dynamic insertion.
        # Actually, for N=60, we can just maintain the list of placed boxes and compute the "skyline" on the fly for each candidate?
        # The previous approach was O(N^4). 
        # Let's try to optimize the previous approach by pre-computing candidate positions and using vectorized NumPy ops if possible, 
        # or simply optimizing the inner loop.
        
        # Given the 120s limit, the previous O(N^4) approach might be borderline but acceptable if optimized.
        # Let's stick to the list-based approach but optimize the inner loop.
        pass

    # Revert to list-based approach but with optimizations.
    # Optimization 1: Pre-compute candidate xs and zs for each box size? No, they depend on placed boxes.
    # Optimization 2: Use NumPy for the overlap check?
    
    # Let's try a different strategy: 
    # For each box, we want to find the position with minimum y.
    # y(x, z) = max over all placed boxes b that overlap (x,z) of (b.y + b.h).
    # This is a 2D range maximum query problem where the "values" are the tops of boxes, 
    # but the boxes themselves are the rectangles defining the query region.
    
    # Since N is small, we can just iterate.
    
    for idx in box_indices:
        w, h, d = boxes[idx]
        
        # Generate 6 rotations
        rotations = [
            (w, h, d),
            (w, d, h),
            (h, w, d),
            (h, d, w),
            (d, w, h),
            (d, h, w)
        ]
        
        best_placement = None
        best_y = float('inf')
        
        for rw, rh, rd in rotations:
            if rw > W or rh > H or rd > D:
                continue
            
            # Generate candidate x positions: 0 and right edges of placed boxes
            # Also, we should consider left edges? 
            # For "lowest y", if we place at x, the y is determined by boxes overlapping [x, x+rw].
            # The function y(x) is piecewise constant, changing only at boundaries of placed boxes.
            # So checking x in {0} U {b.x, b.x + b.w} is sufficient.
            # However, if we only check right edges, we might miss gaps on the left.
            # Example: Box A at x=10, w=5. Box B w=10.
            # If we only check x=0 and x=15 (A.x+A.w), we might miss placing B at x=5?
            # No, if B is at x=5, it overlaps A? No, A is 10-15, B is 5-15. They overlap.
            # If B is at x=0, it's 0-10. No overlap with A.
            # If we check x=0, y=0. If we check x=15, y=0 (if no other boxes).
            # But what if there is a box C at x=0, w=5, h=10?
            # Then at x=0, y=10. At x=5 (C.x+C.w), y=0.
            # So we must check left edges too? Or right edges of boxes that are to the left?
            # Actually, the critical points for the max height function are the left and right edges of all placed boxes.
            
            xs = {0, W}
            zs = {0, D}
            for (px, py, pz, pw, ph, pd) in placed:
                xs.add(px)
                xs.add(px + pw)
                zs.add(pz)
                zs.add(pz + pd)
            
            # Filter and sort
            xs = sorted([x for x in xs if x + rw <= W])
            zs = sorted([z for z in zs if z + rd <= D])
            
            # If no valid xs or zs, skip
            if not xs or not zs:
                continue
                
            # For each candidate (cx, cz), compute min_y
            # To speed this up, we can break early if min_y already exceeds best_y
            for cx in xs:
                # Optimization: If cx + rw > W, skip (already filtered)
                for cz in zs:
                    min_y = 0
                    # Check all placed boxes
                    for (px, py, pz, pw, ph, pd) in placed:
                        # Quick reject
                        if cx + rw <= px or cx >= px + pw:
                            continue
                        if cz + rd <= pz or cz >= pz + pd:
                            continue
                        # Overlap exists
                        top = py + ph
                        if top > min_y:
                            min_y = top
                            if min_y >= best_y:
                                break
                    
                    if min_y < best_y and min_y + rh <= H:
                        best_y = min_y
                        best_placement = (cx, min_y, cz, rw, rh, rd)
        
        if best_placement is not None:
            cx, cy, cz, rw, rh, rd = best_placement
            placed.append((cx, cy, cz, rw, rh, rd))
            final_placements[idx] = (0, cx, cy, cz, rw, rh, rd)
        else:
            final_placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    return final_placements
