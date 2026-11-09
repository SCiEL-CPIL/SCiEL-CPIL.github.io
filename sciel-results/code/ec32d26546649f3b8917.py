import time
import random

# EVOLVE-BLOCK-START
def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    # 1. Sort boxes by volume descending to place larger items first
    # We keep track of original indices to map results back
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    placements = [None] * len(boxes)
    
    # Shelves: list of dicts { 'z': height, 'max_h': max height in this shelf, 'occupied': list of (x, y, w, h) }
    # Actually, simpler: for each shelf, we just need to know the current 'front' x and 'top' y?
    # No, shelf packing usually means:
    # A shelf is defined by its bottom z.
    # Within a shelf, we pack boxes side-by-side.
    # To simplify collision checking in a shelf:
    # We can maintain for each shelf a list of placed boxes.
    # To find a spot in a shelf, we check against all existing boxes in that shelf for overlap.
    # Given N=60, checking O(N) boxes per candidate position is fast enough.
    
    shelves = [] # Each shelf: {'z': float, 'max_h': float, 'boxes': list of (x, y, w, h, d)}
    
    def can_place_in_shelf(shelf, w, h, d):
        """
        Try to find a position (x, y) for box (w, h, d) in the given shelf.
        Returns (x, y) if found, else None.
        We scan the shelf for a valid spot.
        Since boxes are placed in a shelf, they share the same z-base (shelf['z']).
        We need to ensure:
        1. x + w <= W
        2. y + h <= D (assuming y is depth, z is height. Wait, standard is x,y,z. 
           Let's align with task: container (W, H, D). 
           Usually W=x, H=y, D=z or W=x, H=z, D=y?
           Task says: (bin, x, y, z, w, h, d).
           Container is (W, H, D).
           Standard interpretation: x in [0, W], y in [0, H], z in [0, D].
           So box dims w, h, d correspond to x, y, z extents.
        """
        # Check vertical fit
        if shelf['z'] + h > D: # Wait, h is y-extent? No.
            # Let's be precise.
            # Box: w (x-extent), h (y-extent), d (z-extent).
            # Shelf is defined by its z-position (bottom).
            # So the box sits at z = shelf['z'].
            # Its top is shelf['z'] + d.
            # It must fit in container height D.
            pass # Vertical check is done outside or here?
        
        # Let's refine the shelf structure.
        # A shelf is a horizontal plane at height z0.
        # Boxes on this shelf have z = z0.
        # Their z-extent is d_box. So they occupy [z0, z0+d_box].
        # For all boxes on a shelf, their bottom z is the same.
        # Therefore, the shelf's "height" is determined by the max d_box in it.
        # The next shelf must start at z0 + max_d_in_shelf.
        
        # To find x, y:
        # We iterate through candidate x, y? No, that's expensive.
        # We can use a grid or just check against existing boxes.
        # With 60 boxes, checking overlap with ~30 boxes in a shelf is cheap.
        
        # Strategy: Try to place at (0,0), then slide x, then slide y?
        # Or simply: Check all existing boxes in the shelf.
        # If no overlap with any, and within bounds, it's valid.
        
        # To find a valid (x,y):
        # We can try x=0, y=0. If blocked, try x=0, y=next_free_y?
        # Simpler: Since N is small, we can just try a few heuristic positions.
        # 1. (0,0)
        # 2. (0, max_y_of_boxes_below? No, shelves are independent in z, but share x,y space?
        # No, shelves are stacked in z. They do NOT share x,y space constraints with each other, 
        # only with the container boundaries.
        # Wait, if shelf 1 is at z=0 and shelf 2 is at z=5, they are in different z-slabs.
        # So x,y placement in shelf 2 is independent of shelf 1's x,y usage, 
        # EXCEPT that shelf 2's z-range must not overlap shelf 1's z-range.
        # Since we define shelves by their bottom z and stack them, z-ranges are disjoint.
        # So within a shelf, we just need to fit in W x H (x-y plane).
        
        # Algorithm for finding (x,y) in a shelf:
        # We maintain a list of boxes in this shelf.
        # We want to find (x,y) such that:
        # 0 <= x, x+w <= W
        # 0 <= y, y+h <= H
        # No overlap with any (x_i, y_i, w_i, h_i) in shelf.
        
        # Heuristic: 
        # 1. Try (0,0).
        # 2. If fail, try (x_i + w_i, y_i) for each box i? (Right of box i)
        # 3. Try (x_i, y_i + h_i) for each box i? (Above box i)
        # 4. Try (x_i + w_i, y_i + h_i)?
        
        # Let's implement a simple "scan" for empty spots.
        # Since W and H can be large, we can't scan every integer.
        # But we can check candidate points derived from existing boxes.
        
        candidate_xs = [0]
        candidate_ys = [0]
        
        for bx, by, bw, bh, bd in shelf['boxes']:
            candidate_xs.append(bx + bw)
            candidate_ys.append(by + bh)
            
        # Also check right edge and top edge?
        # The candidates should cover the "corners" of free space.
        
        best_pos = None
        best_waste = float('inf')
        
        for cx in candidate_xs:
            if cx + w > W:
                continue
            for cy in candidate_ys:
                if cy + h > H:
                    continue
                
                # Check overlap
                overlap = False
                for bx, by, bw, bh, bd in shelf['boxes']:
                    if cx < bx + bw and cx + w > bx and cy < by + bh and cy + h > by:
                        overlap = True
                        break
                if overlap:
                    continue
                
                # Valid spot. Calculate "waste" or "tightness" for Best Fit.
                # Prefer spots that are closer to origin? Or minimize bounding box expansion?
                # Let's prefer lower z (already fixed), then lower x, then lower y?
                # Or minimize max(x+w, y+h)?
                # Let's just pick the first valid one found in a sorted order of candidates?
                # Sorting candidates by x then y ensures we fill from front-left.
                
                if best_pos is None or (cx, cy) < best_pos:
                    best_pos = (cx, cy)
                    
        return best_pos

    for i, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # Mark rest as left out
            for j in range(i, len(boxes)):
                if placements[j] is None:
                    ow, oh, od = boxes[j]
                    placements[j] = (-1, 0, 0, 0, ow, oh, od)
            break
        
        placed = False
        
        # Generate rotations
        # Original: w (x), h (y), d (z)
        rotations = [
            (w, h, d),
            (w, d, h),
            (h, w, d),
            (h, d, w),
            (d, w, h),
            (d, h, w)
        ]
        
        # Remove duplicates
        unique_rotations = list(set(rotations))
        
        # Try to place in existing shelves
        # We want "Best Fit". 
        # Let's evaluate all shelves and all rotations.
        # For each rotation, find if it fits in any shelf.
        # Among all valid (shelf, rotation, pos), pick the one that minimizes "cost".
        # Cost: 
        # 1. If it forces a new shelf, high cost.
        # 2. If it fits in existing shelf, cost is based on how much it extends the shelf's max height?
        #    Or just prefer existing shelves.
        
        best_option = None # (cost, shelf_idx, rot_w, rot_h, rot_d, x, y)
        
        # 1. Check existing shelves
        for s_idx, shelf in enumerate(shelves):
            # Check if box fits vertically in this shelf's z-range?
            # The shelf is at z=shelf['z'].
            # The box will be at z=shelf['z'].
            # It needs shelf['z'] + rot_d <= D.
            # Also, it must not overlap with other shelves in z?
            # Since shelves are stacked, their z-ranges are disjoint by construction.
            # We just need to ensure the box fits in the container height.
            
            for rw, rh, rd in unique_rotations:
                if shelf['z'] + rd > D:
                    continue
                
                pos = can_place_in_shelf(shelf, rw, rh, rd)
                if pos is not None:
                    x, y = pos
                    # Cost: Prefer shelves with lower z? No, z is fixed for shelf.
                    # Prefer shelves that don't increase the shelf's max height too much?
                    # Or just prefer any existing shelf.
                    # Let's use a simple cost: 0 for existing shelf.
                    # Tie-breaker: Smaller z? Or smaller x,y?
                    # Let's say cost = 0.
                    if best_option is None or 0 < best_option[0]: # This logic is wrong, 0 is always min
                        pass
                    
                    # We want the "Best" fit. 
                    # Let's define cost as:
                    # 1. Does it increase the shelf's max_h? If yes, add penalty.
                    # 2. Prefer lower x, y.
                    
                    new_max_h = max(shelf['max_h'], rd)
                    increase = new_max_h - shelf['max_h']
                    
                    # Heuristic cost: increase * 100 + x + y
                    cost = increase * 1000 + x + y
                    
                    if best_option is None or cost < best_option[0]:
                        best_option = (cost, s_idx, rw, rh, rd, x, y)
                        
        # 2. Check if we can start a new shelf
        # Find the highest z where we can start a new shelf.
        # Actually, we should try to start a new shelf at the lowest available z 
        # that is above all existing shelves?
        # Or just above the highest existing shelf?
        # If we start a new shelf, it should be at z = max(z_i + max_h_i) for all shelves i.
        
        if not shelves:
            new_z = 0
        else:
            new_z = max(s['z'] + s['max_h'] for s in shelves)
            
        if new_z + min(unique_rotations, key=lambda r: r[2])[2] <= D: # At least one rotation fits vertically
             # Try to place in a new shelf at new_z
             # We can create a temporary shelf to check x,y fit
             temp_shelf = {'z': new_z, 'max_h': 0, 'boxes': []}
             for rw, rh, rd in unique_rotations:
                 if new_z + rd > D:
                     continue
                 pos = can_place_in_shelf(temp_shelf, rw, rh, rd)
                 if pos is not None:
                     x, y = pos
                     # Cost for new shelf is higher
                     # Penalize creating a new shelf
                     cost = 10000 + x + y 
                     if best_option is None or cost < best_option[0]:
                         best_option = (cost, -1, rw, rh, rd, x, y) # -1 indicates new shelf

        if best_option is not None:
            cost, s_idx, rw, rh, rd, x, y = best_option
            if s_idx == -1:
                # Create new shelf
                new_shelf = {'z': new_z, 'max_h': rd, 'boxes': [(x, y, rw, rh, rd)]}
                shelves.append(new_shelf)
            else:
                shelf = shelves[s_idx]
                shelf['max_h'] = max(shelf['max_h'], rd)
                shelf['boxes'].append((x, y, rw, rh, rd))
            
            placements[i] = (0, x, y, shelves[s_idx]['z'] if s_idx != -1 else new_z, rw, rh, rd)
            placed = True
            
        if not placed:
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    # Fill any remaining None slots (if loop broke early)
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements
# EVOLVE-BLOCK-END
