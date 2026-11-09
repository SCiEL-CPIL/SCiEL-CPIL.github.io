import numpy as np

def load(boxes, container):
    W, H, D = container
    
    # Strategy: Shelf Packing
    # We divide the container into shelves along the Z-axis.
    # Each shelf has a height. Within a shelf, we pack boxes into rows along Y.
    # Within a row, we pack boxes along X.
    
    def solve_shelf_packing(sort_key):
        indexed_boxes = list(enumerate(boxes))
        indexed_boxes.sort(key=lambda t: sort_key(t[1]), reverse=True)
        
        results = [None] * len(boxes)
        total_vol = 0
        
        # State for shelf packing
        # current_shelf_z: the z-coordinate of the bottom of the current shelf
        # current_shelf_h: the height of the current shelf (max height of boxes in it)
        # current_row_y: the y-coordinate of the current row in the current shelf
        # current_row_x: the x-coordinate where the next box can start in the current row
        
        # Actually, it's better to track:
        # shelves: list of (z_start, height, rows)
        # But for efficiency, we can just track the current state.
        # Let's use a list of placed boxes to allow for more complex logic if needed,
        # but for basic shelf packing, we can just track the "cursor" positions.
        
        # However, shelf packing usually allows boxes in a shelf to have different heights,
        # but the shelf height is determined by the tallest one.
        # The "next available x" depends on the specific row.
        
        # Let's define a Shelf class or just use variables.
        # Variables:
        # z_pos: start z of current shelf
        # shelf_h: height of current shelf
        # y_pos: start y of current row in current shelf
        # x_pos: start x of current box in current row
        
        # Wait, if we start a new row, x_pos resets to 0.
        # If we start a new shelf, y_pos resets to 0, x_pos resets to 0.
        
        # Initialization
        z_pos = 0
        shelf_h = 0
        y_pos = 0
        x_pos = 0
        
        # We need to handle the case where a box doesn't fit in the current shelf
        # because it's too tall, or doesn't fit in the current row because it's too wide/deep.
        
        for orig_idx, (w_orig, h_orig, d_orig) in indexed_boxes:
            orientations = [
                (w_orig, h_orig, d_orig),
                (w_orig, d_orig, h_orig),
                (h_orig, w_orig, d_orig),
                (h_orig, d_orig, w_orig),
                (d_orig, w_orig, h_orig),
                (d_orig, h_orig, w_orig),
            ]
            
            placed = False
            best_orient = None
            best_pos = None
            
            # Try to place in current shelf/row first, then new row, then new shelf
            # We iterate through orientations. For each orientation, we check if it fits.
            # To be greedy, we might want to pick the orientation that minimizes waste.
            # Simple heuristic: try all orientations, pick the first one that fits
            # in the current context (current row, or new row in same shelf, or new shelf).
            
            # Let's refine the placement logic:
            # 1. Check if it fits in the current row (x_pos + ow <= W, oh <= shelf_h, od <= H - y_pos)
            #    Note: In shelf packing, the "height" of the box in the shelf is the dimension
            #    perpendicular to the shelf floor. If shelf is on XY plane, Z is up.
            #    So box dims are (w, h, d) where w is X, h is Z (height in shelf), d is Y (depth).
            #    Wait, standard shelf packing:
            #    Container: W (x), H (y), D (z).
            #    Shelves are along Z.
            #    So shelf height is in Z direction.
            #    Box orientation (ow, oh, od):
            #    ow -> X
            #    od -> Y
            #    oh -> Z (height in shelf)
            
            # Let's re-map orientations to (width_X, depth_Y, height_Z)
            # Original box: (w, h, d)
            # Orientations for (X, Y, Z):
            # (w, h, d) -> X=w, Y=h, Z=d
            # (w, d, h) -> X=w, Y=d, Z=h
            # (h, w, d) -> X=h, Y=w, Z=d
            # (h, d, w) -> X=h, Y=d, Z=w
            # (d, w, h) -> X=d, Y=w, Z=h
            # (d, h, w) -> X=d, Y=h, Z=w
            
            # We want to minimize the "height" (Z) to keep shelves low, or maybe not?
            # Usually, we want to pack tightly.
            
            # Let's try all 6 orientations.
            # For each orientation (ow, oy, oz):
            #   Check if oz <= D - z_pos (fits in remaining Z)
            #   Check if oy <= H - y_pos (fits in remaining Y for this shelf? No, shelf spans full Y? 
            #   No, shelf packing usually assumes shelves span the full Y width? 
            #   No, "Shelf" in 2D bin packing is a horizontal strip. In 3D, it's a layer.
            #   Standard 3D Shelf Packing:
            #   The container is divided into layers (shelves) along the Z axis.
            #   Each layer has a height.
            #   Within a layer, we perform 2D packing on the W x H face.
            #   This simplifies the problem significantly.
            
            # So, algorithm:
            # 1. Determine shelf height (Z-dim of box).
            # 2. Check if shelf fits in remaining Z.
            # 3. If yes, check if 2D rectangle (ow x oy) fits in the current shelf's 2D layout.
            #    The 2D layout is a W x H area.
            #    We can use a simple 2D shelf packing (rows) inside the layer.
            
            # This is a nested shelf packing: 3D shelves, each containing 2D shelves (rows).
            
            # Let's implement this nested approach.
            # State:
            # z_pos: start of current 3D shelf
            # shelf_h: height of current 3D shelf
            # 
            # Inside the shelf, we have a 2D problem of size W x H.
            # We can track:
            # y_pos: start of current 2D row (shelf) in the 3D shelf
            # row_h: height of current 2D row
            # x_pos: start of current box in the 2D row
            
            # Actually, let's just track the 2D state for the current 3D shelf.
            # If we start a new 3D shelf, we reset the 2D state.
            
            # Let's restart the state tracking for clarity.
            pass # Placeholder for logic below
            
            # We'll iterate orientations.
            # We need to decide where to place it.
            # Candidates:
            # 1. Current 2D row in current 3D shelf.
            # 2. New 2D row in current 3D shelf.
            # 3. New 3D shelf (and new 2D row).
            
            # We want to minimize waste.
            # Heuristic: 
            # - Try to fit in current row (minimizes Y waste).
            # - If not, try new row in current shelf (minimizes Z waste).
            # - If not, try new shelf.
            
            # We need to pick an orientation.
            # Let's try all orientations and pick the one that results in the "best" placement.
            # "Best" could be lowest Z, then lowest Y, then lowest X.
            
            best_z = D + 1
            best_y = H + 1
            best_x = W + 1
            best_ow, best_oy, best_oz = 0, 0, 0
            best_pos_found = False
            
            for ow, oy, oz in orientations:
                if ow > W or oy > H or oz > D:
                    continue
                
                # Option 1: Current Row in Current Shelf
                # Conditions:
                # - oz <= shelf_h (box height must be <= current shelf height? 
                #   No, if we place a box in a shelf, the shelf height becomes max(shelf_h, oz).
                #   But if we are in the middle of a shelf, adding a taller box increases shelf_h.
                #   This is valid.
                # - x_pos + ow <= W
                # - y_pos + oy <= H (Wait, y_pos is the start of the row. The row has a height row_h.
                #   If we place in current row, we need oy <= row_h? 
                #   No, in 2D shelf packing, the row height is determined by the tallest item.
                #   If we add an item with oy > row_h, the row height increases.
                #   This is valid.
                
                # So, for current row:
                if x_pos + ow <= W:
                    # It fits in X.
                    # New row height would be max(row_h, oy)
                    # New shelf height would be max(shelf_h, oz)
                    # This is always a valid placement if it fits in X.
                    # We prefer this if it doesn't increase shelf height too much?
                    # For now, let's just take it if it fits.
                    # But we should check if it's "better" than other options.
                    # Let's record this candidate.
                    cand_z = z_pos
                    cand_y = y_pos
                    cand_x = x_pos
                    if (cand_z, cand_y, cand_x) < (best_z, best_y, best_x):
                        best_z, best_y, best_x = cand_z, cand_y, cand_x
                        best_ow, best_oy, best_oz = ow, oy, oz
                        best_pos_found = True
                        # We don't break immediately because another orientation might be better?
                        # No, if it fits in the current row, it's the best position (lowest Z, Y, X).
                        # So we can break?
                        # Wait, if we break, we might miss an orientation that has a smaller footprint
                        # allowing more boxes later?
                        # Greedy first-fit usually works. Let's stick with first valid in sorted order.
                        # But we are iterating orientations.
                        # If we find a fit in current row, it's the best possible position.
                        # So we can break out of orientation loop?
                        # Yes, because any other position (new row, new shelf) is "worse" in terms of Z/Y.
                        break
            
            if not best_pos_found:
                # Try New Row in Current Shelf
                # Conditions:
                # - oy <= H - y_pos (fits in remaining Y of the 3D shelf? 
                #   No, the 3D shelf spans full H? 
                #   Yes, the 3D shelf is a slab of size W x H x shelf_h.
                #   The 2D packing is on W x H.
                #   So we need oy <= H - y_pos? 
                #   No, y_pos is the Y-coordinate of the current row start.
                #   The current row occupies [y_pos, y_pos + row_h).
                #   A new row would start at y_pos + row_h.
                #   So we need y_pos + row_h + oy <= H.
                
                if y_pos + row_h + oy <= H:
                    # Wait, I haven't defined row_h in the loop yet.
                    # I need to restructure the state variables.
                    
                    pass # This logic is getting messy in the scratchpad.
                    
            # Let's rewrite the whole function cleanly.
            pass
            
        return results, total_vol
        
    # Since the scratchpad logic is getting complex, I will write the final code in the SEARCH/REPLACE block.
    # I will implement a clean Nested Shelf Packing.
    
    # Strategy:
    # 1. Sort boxes.
    # 2. Iterate boxes.
    # 3. For each box, try 6 orientations.
    # 4. For each orientation, try placing in:
    #    a. Current 2D row in current 3D shelf.
    #    b. New 2D row in current 3D shelf.
    #    c. New 3D shelf.
    # 5. Pick the first valid placement (greedy).
    
    # State variables:
    # z_pos: start Z of current 3D shelf
    # shelf_h: height of current 3D shelf
    # y_pos: start Y of current 2D row in current 3D shelf
    # row_h: height of current 2D row
    # x_pos: start X of current box in current 2D row
    
    # Note: When we start a new 2D row, y_pos += row_h, row_h = 0, x_pos = 0.
    # When we start a new 3D shelf, z_pos += shelf_h, shelf_h = 0, y_pos = 0, row_h = 0, x_pos = 0.
    
    # We need to be careful: if shelf_h is 0, it means no boxes placed yet.
    # When placing the first box, we just set the states.
    
    # Let's implement `solve_shelf_packing` properly.
    
    def solve_shelf_packing(sort_key):
        indexed_boxes = list(enumerate(boxes))
        indexed_boxes.sort(key=lambda t: sort_key(t[1]), reverse=True)
        
        results = [None] * len(boxes)
        total_vol = 0
        
        z_pos = 0
        shelf_h = 0
        y_pos = 0
        row_h = 0
        x_pos = 0
        
        for orig_idx, (w_orig, h_orig, d_orig) in indexed_boxes:
            orientations = [
                (w_orig, h_orig, d_orig),
                (w_orig, d_orig, h_orig),
                (h_orig, w_orig, d_orig),
                (h_orig, d_orig, w_orig),
                (d_orig, w_orig, h_orig),
                (d_orig, h_orig, w_orig),
            ]
            
            placed = False
            best_orient = None
            best_pos = None
            
            # Try orientations
            # We want to find the "best" fit.
            # Priority: 
            # 1. Fits in current row (no change to y_pos, row_h, shelf_h, z_pos except x_pos)
            # 2. Fits in new row (change y_pos, row_h, x_pos)
            # 3. Fits in new shelf (change z_pos, shelf_h, y_pos, row_h, x_pos)
            
            # We iterate orientations. For each, we check the 3 options.
            # We pick the first one that works?
            # Or do we pick the one that minimizes (z_pos, y_pos, x_pos)?
            # Since we iterate options in priority order (current row > new row > new shelf),
            # the first valid orientation in the list that fits in the highest priority slot is best.
            # But maybe a different orientation fits in current row, and another fits in new row?
            # We should check all orientations for current row first.
            # If any fits, pick the best one (e.g. smallest x_pos? they all have same x_pos).
            # Actually, x_pos is the same for all boxes in the current row.
            # So if multiple orientations fit in current row, we should pick the one that
            # minimizes the impact on future packing? 
            # Simple heuristic: pick the one with smallest oy (depth) to save Y space?
            # Or smallest oz to save Z space?
            # Let's just pick the first one that fits in current row.
            
            # Step 1: Check if any orientation fits in Current Row
            for ow, oy, oz in orientations:
                if ow > W or oy > H or oz > D:
                    continue
                if x_pos + ow <= W:
                    # Fits in current row
                    # Check if it fits in the shelf height?
                    # No, shelf_h will be updated to max(shelf_h, oz).
                    # Is there a constraint that oz <= shelf_h?
                    # No, shelf_h is the height of the slab. If we put a taller box, the slab gets taller.
                    # This is valid.
                    # So it fits.
                    best_orient = (ow, oy, oz)
                    best_pos = (z_pos, y_pos, x_pos)
                    placed = True
                    break
            
            if not placed:
                # Step 2: Check if any orientation fits in New Row (in current shelf)
                # Condition: y_pos + row_h + oy <= H
                # Note: If shelf_h is 0, we are in a new shelf. 
                # But we handle new shelf in Step 3.
                # So here we assume we are in an existing shelf (shelf_h > 0).
                # Wait, if shelf_h is 0, it's the first box. 
                # The "current row" logic should have caught it if x_pos=0, W>=ow.
                # If it didn't fit in current row, maybe it's too wide?
                # If it's too wide, it won't fit in new row either (same W).
                # So Step 2 is only useful if the box fits in W, but we want to start a new row?
                # No, if it fits in W, it fits in current row (x_pos + ow <= W).
                # Why would we want to start a new row if it fits in current row?
                # We wouldn't. Current row is always better (lower Y).
                # So Step 2 is only reached if it does NOT fit in current row.
                # This means x_pos + ow > W.
                # If x_pos + ow > W, then it won't fit in a new row either?
                # No, new row starts at x_pos=0. So if ow <= W, it fits in a new row.
                # So Step 2 is valid if ow <= W.
                
                for ow, oy, oz in orientations:
                    if ow > W or oy > H or oz > D:
                        continue
                    # Check if it fits in a new row
                    # New row starts at y_new = y_pos + row_h
                    # Need y_new + oy <= H
                    if y_pos + row_h + oy <= H:
                        # It fits in a new row
                        best_orient = (ow, oy, oz)
                        # Position will be (z_pos, y_pos + row_h, 0)
                        best_pos = (z_pos, y_pos + row_h, 0)
                        placed = True
                        break
            
            if not placed:
                # Step 3: Check if any orientation fits in New Shelf
                # Condition: z_pos + shelf_h + oz <= D
                # New shelf starts at z_new = z_pos + shelf_h
                # Inside new shelf, it's a new row at y=0, x=0.
                
                for ow, oy, oz in orientations:
                    if ow > W or oy > H or oz > D:
                        continue
                    if z_pos + shelf_h + oz <= D:
                        # It fits in a new shelf
                        best_orient = (ow, oy, oz)
                        best_pos = (z_pos + shelf_h, 0, 0)
                        placed = True
                        break
            
            if placed:
                ow, oy, oz = best_orient
                z, y, x = best_pos
                
                # Update State
                # Determine which case we are in to update state correctly.
                # We can infer from the position.
                # If z == z_pos and y == y_pos: Current Row
                # If z == z_pos and y == y_pos + row_h: New Row
                # If z == z_pos + shelf_h: New Shelf
                
                if z == z_pos and y == y_pos:
                    # Current Row
                    x_pos += ow
                    row_h = max(row_h, oy)
                    shelf_h = max(shelf_h, oz)
                elif z == z_pos and y == y_pos + row_h:
                    # New Row
                    y_pos += row_h
                    row_h = oy
                    x_pos = ow
                    shelf_h = max(shelf_h, oz)
                elif z == z_pos + shelf_h:
                    # New Shelf
                    z_pos += shelf_h
                    shelf_h = oz
                    y_pos = 0
                    row_h = oy
                    x_pos = ow
                else:
                    # Should not happen
                    pass
                
                results[orig_idx] = (0, x, y, z, ow, oy, oz)
                total_vol += ow * oy * oz
            else:
                results[orig_idx] = (-1, 0, 0, 0, w_orig, h_orig, d_orig)
        
        return results, total_vol

    strategies = [
        lambda b: b[0]*b[1]*b[2],
        lambda b: max(b),
        lambda b: 2*(b[0]*b[1] + b[1]*b[2] + b[0]*b[2]),
        lambda b: b[0]*b[1], # Base area
        lambda b: b[1]*b[2],
        lambda b: b[0]*b[2],
    ]
    
    best_result = None
    best_volume = -1
    
    for sort_key in strategies:
        res, vol = solve_shelf_packing(sort_key)
        if vol > best_volume:
            best_volume = vol
            best_result = res
            
    return best_result
