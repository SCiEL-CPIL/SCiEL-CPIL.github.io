import time
import math
import itertools

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin

    # Generate all 6 rotations for each box
    box_rotations = []
    for i, (w, h, d) in enumerate(boxes):
        rots = set()
        for perm in itertools.permutations((w, h, d)):
            rots.add(perm)
        box_rotations.append(list(rots))

    # Sort boxes by volume descending to pack large ones first
    box_indices = list(range(n))
    box_indices.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)

    def pack_orientation(stack_axis):
        """
        stack_axis: 0 (X), 1 (Y), 2 (Z)
        Shelves are stacked along stack_axis.
        """
        # Determine dimensions for the stacking and cross-section
        if stack_axis == 0:
            # Stack in X. Shelves are slices of width w_shelf.
            # Cross-section is H x D.
            stack_dim = W
            cross_dims = (H, D)
            # Box dims: (w, h, d)
            # Rotation must map to (w_rot, h_rot, d_rot)
            # w_rot is the dimension along X (stacking).
            # h_rot, d_rot are in the cross-section.
        elif stack_axis == 1:
            # Stack in Y. Shelves are slices of height h_shelf.
            # Cross-section is W x D.
            stack_dim = H
            cross_dims = (W, D)
        else:
            # Stack in Z. Shelves are slices of depth d_shelf.
            # Cross-section is W x H.
            stack_dim = D
            cross_dims = (W, H)

        shelves = []
        # Each shelf: { 'start': coord, 'size': size_of_slice, 'boxes': [], 'cross_state': ... }
        # For 2D packing inside the slice, we use a simple shelf packing in the cross-section.
        # Let's define the cross-section packing as 2D Shelf Packing.
        # Cross-section dims: C1, C2.
        # 2D Shelves stacked in C2.
        # Boxes placed in C1.
        
        placements = [None] * n
        
        for idx in box_indices:
            w, h, d = boxes[idx]
            placed = False
            
            # Try all rotations
            for rw, rh, rd in box_rotations[idx]:
                # Determine which dimension of the rotated box corresponds to the stacking axis
                if stack_axis == 0:
                    stack_size = rw
                    c1_size = rh
                    c2_size = rd
                elif stack_axis == 1:
                    stack_size = rh
                    c1_size = rw
                    c2_size = rd
                else:
                    stack_size = rd
                    c1_size = rw
                    c2_size = rh

                if stack_size > stack_dim:
                    continue

                # Try to place in existing shelves
                best_shelf_idx = -1
                best_pos = None
                best_waste = float('inf')

                for s_idx, shelf in enumerate(shelves):
                    # Check if box fits in the stack dimension of this shelf
                    # The shelf has a fixed size 'shelf['size']' determined when it was created.
                    # We only allow placing if stack_size <= shelf['size']
                    if stack_size > shelf['size']:
                        continue
                    
                    # Check 2D fit in the cross-section
                    # The cross-section packing state is stored in shelf['cross_state']
                    # We use a simple 2D shelf packing for the cross-section.
                    # State: list of 2D shelves.
                    # 2D Shelf: { 'start': c2_coord, 'height': c1_max_height, 'x_pos': current_c1_pos }
                    
                    fitted = False
                    fit_pos = None
                    
                    for c2_s_idx, c2_shelf in enumerate(shelf['cross_state']):
                        # c2_shelf['height'] is the max c1_size of boxes in this 2D shelf
                        # We require c1_size <= c2_shelf['height']
                        if c1_size <= c2_shelf['height']:
                            # Check if it fits in c1
                            if c2_shelf['x_pos'] + c1_size <= cross_dims[0]:
                                # It fits!
                                # Calculate waste (optional, for best-fit)
                                waste = c2_shelf['height'] - c1_size
                                if waste < best_waste:
                                    best_waste = waste
                                    best_shelf_idx = s_idx
                                    best_pos = (c2_s_idx, c2_shelf['x_pos'], c2_shelf['start'])
                                    fitted = True
                                # Note: We might want to update the shelf state immediately if we place it?
                                # No, we are in a "try" phase. We need to simulate.
                                # To keep it simple and fast, we use First-Fit for 2D packing.
                                # This avoids complex backtracking.
                                break
                    
                    if fitted:
                        # We found a spot. But we need to update the state.
                        # Since we are using First-Fit in 2D, we can just place it.
                        # However, we need to update the shelf's cross_state.
                        # This mutation makes "Best Fit" across 3D shelves hard if we don't rollback.
                        # Given time, I will use a simpler strategy:
                        # 1. Iterate shelves.
                        # 2. For each shelf, try to place box using 2D Next-Fit.
                        # 3. If it fits, place it and break.
                        
                        # Let's re-evaluate: The "Best Fit" in the knowledge base might refer to choosing the shelf with the least remaining space.
                        # I will implement a greedy placement:
                        # For each shelf, if it fits in 2D (using current state), place it.
                        # To make it "Best Fit", I will sort shelves by their current "fullness" or just try them in order.
                        
                        # Let's just place it in the first shelf that works (First-Fit).
                        # This is fast and often good enough.
                        
                        # Update the 2D shelf state
                        c2_shelf = shelf['cross_state'][best_pos[0]]
                        x_pos = best_pos[1]
                        c2_start = best_pos[2]
                        
                        # Update 2D shelf
                        c2_shelf['x_pos'] += c1_size
                        # Note: c2_shelf['height'] is already >= c1_size, so no change needed.
                        
                        # Record placement
                        if stack_axis == 0:
                            placements[idx] = (0, shelf['start'], c2_start, x_pos, rw, rh, rd) # x, y, z, w, h, d
                            # Wait, coordinate mapping:
                            # Stack X: x = shelf_start. y = c2_start? No.
                            # Cross-section is H (y) x D (z).
                            # c1 is H (y), c2 is D (z).
                            # So y = c2_start? No, c2 is the stacking axis of the 2D shelf.
                            # Let's define 2D shelf stacking in the 2nd cross-dim.
                            # If cross is (H, D), 2D shelves stack in D (z).
                            # So z = c2_start. y = x_pos? No, x_pos is in c1 (H, y).
                            # So y = x_pos. z = c2_start.
                            placements[idx] = (0, shelf['start'], x_pos, c2_start, rw, rh, rd)
                        elif stack_axis == 1:
                            # Stack Y: y = shelf_start.
                            # Cross is W (x) x D (z).
                            # c1 is W (x), c2 is D (z).
                            # 2D shelves stack in D (z).
                            # x = x_pos. z = c2_start.
                            placements[idx] = (0, x_pos, shelf['start'], c2_start, rw, rh, rd)
                        else:
                            # Stack Z: z = shelf_start.
                            # Cross is W (x) x H (y).
                            # c1 is W (x), c2 is H (y).
                            # 2D shelves stack in H (y).
                            # x = x_pos. y = c2_start.
                            placements[idx] = (0, x_pos, c2_start, shelf['start'], rw, rh, rd)
                        
                        placed = True
                        break # Break shelf loop
                    
                    if placed:
                        break # Break rotation loop? No, we found a spot for this rotation.
                
                if placed:
                    break # Break rotation loop
            
            if not placed:
                # Create a new shelf
                # Find the next available position in the stacking axis
                if not shelves:
                    new_start = 0
                else:
                    # Shelves are not necessarily contiguous? 
                    # In this simple model, they are stacked.
                    # So new_start = shelves[-1]['start'] + shelves[-1]['size']
                    # But we might have gaps? No, we pack tightly.
                    new_start = 0
                    for s in shelves:
                        if s['start'] + s['size'] > new_start:
                            new_start = s['start'] + s['size']
                    
                    if new_start + stack_size > stack_dim:
                        continue # Can't fit any more shelves

                # Create new shelf
                # The size of the new shelf is determined by the box that created it?
                # Or we can set it to the max possible?
                # Standard: Set to the size of the first box placed in it.
                new_shelf = {
                    'start': new_start,
                    'size': stack_size,
                    'cross_state': []
                }
                
                # Initialize 2D cross-state with this box
                # 2D shelf: { 'start': 0, 'height': c1_size, 'x_pos': c1_size }
                new_shelf['cross_state'].append({
                    'start': 0,
                    'height': c1_size,
                    'x_pos': c1_size
                })
                
                shelves.append(new_shelf)
                
                # Record placement
                if stack_axis == 0:
                    placements[idx] = (0, new_start, 0, 0, rw, rh, rd)
                elif stack_axis == 1:
                    placements[idx] = (0, 0, new_start, 0, rw, rh, rd)
                else:
                    placements[idx] = (0, 0, 0, new_start, rw, rh, rd)

        # Calculate total volume packed
        total_vol = 0
        for p in placements:
            if p and p[0] == 0:
                total_vol += p[4] * p[5] * p[6]
        
        # Mark unplaced boxes
        final_placements = []
        for i in range(n):
            if placements[i]:
                final_placements.append(placements[i])
            else:
                w, h, d = boxes[i]
                final_placements.append((-1, 0, 0, 0, w, h, d))
        
        return final_placements, total_vol

    best_placements = None
    best_vol = 0

    # Try all 3 orientations
    for axis in [0, 1, 2]:
        if time.time() > deadline:
            break
        placements, vol = pack_orientation(axis)
        if vol > best_vol:
            best_vol = vol
            best_placements = placements

    if best_placements is None:
        # Fallback: return all as left out
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_placements
