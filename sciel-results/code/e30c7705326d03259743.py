import time
import itertools
import numpy as np

def load(boxes, container):
    W, H, D = container
    deadline = time.time() + 110  # safety margin
    
    # Skyline grid: skyline[x][y] = min Z coordinate that is free
    # Using numpy for fast slice max and update
    # If W or H is very large, this might be memory intensive, but for typical problems it's fine.
    # If W*H > 10^6, we might need a sparse approach, but let's assume it's manageable.
    try:
        skyline = np.zeros((W, H), dtype=np.int32)
    except MemoryError:
        # Fallback to sparse or different strategy if too big, but likely won't happen
        skyline = None

    if skyline is None:
        # Fallback to the original O(N) check if grid is too big
        return _load_fallback(boxes, container, deadline)

    # Generate all 6 unique rotations for each box
    def get_rotations(w, h, d):
        return list(set(itertools.permutations((w, h, d))))

    # Sort boxes by volume descending
    box_indices = list(range(len(boxes)))
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in box_indices]
    sorted_indices = sorted(box_indices, key=lambda i: box_volumes[i], reverse=True)

    placements = [None] * len(boxes)
    
    # Candidate coordinates
    x_cands = [0]
    y_cands = [0]
    
    placed_boxes = [] # List of (x, y, z, w, h, d) for potential future use or debugging

    for idx in sorted_indices:
        if time.time() > deadline:
            break
            
        bw, bh, bd = boxes[idx]
        rots = get_rotations(bw, bh, bd)
        
        best_pos = None
        best_score = float('inf')
        best_rot = None
        
        for rw, rh, rd in rots:
            # Iterate over candidate X and Y positions
            # We can prune candidates that are obviously out of bounds
            valid_x = [x for x in x_cands if x + rw <= W]
            valid_y = [y for y in y_cands if y + rh <= H]
            
            for x in valid_x:
                for y in valid_y:
                    # Find the minimum Z this box can be placed at (x, y)
                    # This is the max of the skyline in the footprint
                    z = np.max(skyline[x:x+rw, y:y+rh])
                    
                    if z + rd <= H:
                        # Score: prefer lower z, then lower y, then lower x
                        score = z * W * H + y * W + x
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
                            best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_rot
            placements[idx] = (0, x, y, z, rw, rh, rd)
            
            # Update skyline
            skyline[x:x+rw, y:y+rh] = np.maximum(skyline[x:x+rw, y:y+rh], z + rd)
            
            # Update candidate lists
            new_x = x + rw
            new_y = y + rh
            if new_x not in x_cands:
                x_cands.append(new_x)
                x_cands.sort()
            if new_y not in y_cands:
                y_cands.append(new_y)
                y_cands.sort()
        else:
            placements[idx] = (-1, 0, 0, 0, bw, bh, bd)

    # Fill any remaining None placements
    for i in range(len(boxes)):
        if placements[i] is None:
            bw, bh, bd = boxes[i]
            placements[i] = (-1, 0, 0, 0, bw, bh, bd)
            
    return placements

def _load_fallback(boxes, container, deadline):
    # This is the original logic, just in case numpy grid fails
    W, H, D = container
    def rotations(w, h, d):
        return list(set(itertools.permutations((w, h, d))))
    
    box_indices = list(range(len(boxes)))
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in box_indices]
    sorted_indices = sorted(box_indices, key=lambda i: box_volumes[i], reverse=True)
    
    placements = [None] * len(boxes)
    occupied = []
    
    def can_place(x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for ox, oy, oz, ow, oh, od in occupied:
            if (x < ox + ow and x + w > ox and
                y < oy + oh and y + h > oy and
                z < oz + od and z + d > oz):
                return False
        return True
    
    def find_best_position(w, h, d):
        x_candidates = [0]
        y_candidates = [0]
        z_candidates = [0]
        
        for ox, oy, oz, ow, oh, od in occupied:
            if ox + ow <= W:
                x_candidates.append(ox + ow)
            if oy + oh <= H:
                y_candidates.append(oy + oh)
            if oz + od <= D:
                z_candidates.append(oz + od)
        
        x_candidates = sorted(set(x for x in x_candidates if x + w <= W))
        y_candidates = sorted(set(y for y in y_candidates if y + h <= H))
        z_candidates = sorted(set(z for z in z_candidates if z + d <= D))
        
        best_pos = None
        best_score = float('inf')
        
        for x in x_candidates:
            for y in y_candidates:
                for z in z_candidates:
                    if can_place(x, y, z, w, h, d):
                        score = z * W * H + y * W + x
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    for idx in sorted_indices:
        if time.time() > deadline:
            break
        
        bw, bh, bd = boxes[idx]
        rots = rotations(bw, bh, bd)
        
        placed = False
        for rw, rh, rd in rots:
            pos = find_best_position(rw, rh, rd)
            if pos is not None:
                x, y, z = pos
                placements[idx] = (0, x, y, z, rw, rh, rd)
                occupied.append((x, y, z, rw, rh, rd))
                placed = True
                break
        
        if not placed:
            placements[idx] = (-1, 0, 0, 0, bw, bh, bd)
            
    for i in range(len(boxes)):
        if placements[i] is None:
            bw, bh, bd = boxes[i]
            placements[i] = (-1, 0, 0, 0, bw, bh, bd)
            
    return placements
