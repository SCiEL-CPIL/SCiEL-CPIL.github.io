"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    deadline = time.time() + 110.0  # 10s safety margin before 120s limit

    # Generate all unique orientations for each box
    def get_orientations(w, h, d):
        orients = set()
        for perm in itertools.permutations((w, h, d)):
            orients.add(perm)
        return list(orients)

    # Try to place a box at position (x, y, z) with dimensions (bw, bh, bd)
    # Returns True if it fits without overlapping existing placements
    def can_place(x, y, z, bw, bh, bd, placed):
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        for (px, py, pz, pw, ph, pd) in placed:
            # Check overlap: two boxes overlap if they overlap in all three axes
            if not (x + bw <= px or px + pw <= x):  # x-axis overlap
                if not (y + bh <= py or py + ph <= y):  # y-axis overlap
                    if not (z + bd <= pz or pz + pd <= z):  # z-axis overlap
                        return False
        return True

    # Score a placement: total volume of placed boxes
    def score_placements(placed):
        return sum(pw * ph * pd for (_, _, _, pw, ph, pd) in placed)

    # Shelf-based packing with rotation
    def shelf_pack(sorted_indices, container_dims):
        W, H, D = container_dims
        placed = []  # list of (x, y, z, w, h, d)
        # We'll use a simple shelf approach: process boxes in order, place on shelves
        # Shelves are horizontal planes at different z-levels
        # For each shelf, we place boxes side by side in x, and stack in y within the shelf height
        
        # Actually, let's use a simpler approach: try to place each box at the best position
        # We'll maintain a list of "free spots" or just try all possible positions
        
        # For efficiency, let's use a greedy approach:
        # For each box, try all orientations and all possible (x, y, z) positions
        # and pick the one that minimizes (z, y, x) to pack tightly
        
        for idx in sorted_indices:
            w, h, d = boxes[idx]
            best_pos = None
            best_orient = None
            best_key = None
            
            for orient in get_orientations(w, h, d):
                bw, bh, bd = orient
                # Try to find the lowest, then leftmost, then frontmost position
                # We'll scan z from 0, y from 0, x from 0
                # To be efficient, we'll limit the scan range
                found = False
                for z in range(0, D - bd + 1):
                    if found:
                        break
                    for y in range(0, H - bh + 1):
                        if found:
                            break
                        for x in range(0, W - bw + 1):
                            if can_place(x, y, z, bw, bh, bd, placed):
                                key = (z, y, x)
                                if best_key is None or key < best_key:
                                    best_key = key
                                    best_pos = (x, y, z)
                                    best_orient = orient
                                found = True
                                break
                
                # If we found a position, break early for this orientation
                # Actually, we should check all orientations to find the best overall
                # But for speed, let's just take the first valid one found
                if best_pos is not None:
                    break
            
            if best_pos is not None:
                x, y, z = best_pos
                bw, bh, bd = best_orient
                placed.append((x, y, z, bw, bh, bd))
        
        return placed

    # Better approach: use a priority queue or sorted list of candidate positions
    # Let's use a different strategy: sort boxes by volume descending, then by max dimension
    # and use a more efficient placement search
    
    def pack_boxes(sorted_indices, container_dims):
        W, H, D = container_dims
        placed = []
        
        for idx in sorted_indices:
            w, h, d = boxes[idx]
            best_pos = None
            best_orient = None
            best_key = None
            
            for orient in get_orientations(w, h, d):
                bw, bh, bd = orient
                if bw > W or bh > H or bd > D:
                    continue
                
                # Find the lowest z, then lowest y, then lowest x where it fits
                # We'll scan in order of preference
                for z in range(0, D - bd + 1):
                    for y in range(0, H - bh + 1):
                        for x in range(0, W - bw + 1):
                            if can_place(x, y, z, bw, bh, bd, placed):
                                key = (z, y, x)
                                if best_key is None or key < best_key:
                                    best_key = key
                                    best_pos = (x, y, z)
                                    best_orient = orient
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                
                if best_pos is not None:
                    break
            
            if best_pos is not None:
                x, y, z = best_pos
                bw, bh, bd = best_orient
                placed.append((x, y, z, bw, bh, bd))
        
        return placed

    # Sort boxes by volume descending
    indexed_boxes = [(i, boxes[i]) for i in range(len(boxes))]
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    sorted_indices = [i for i, _ in indexed_boxes]

    # Try the packing
    placed = pack_boxes(sorted_indices, (W, H, D))

    # Build the result
    placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(len(boxes))]
    for (x, y, z, bw, bh, bd) in placed:
        # Find which box this is - we need to track this
        pass

    # The above approach has a bug: we don't track which box is which
    # Let's redo this properly

    def pack_boxes_tracked(sorted_indices, container_dims):
        W, H, D = container_dims
        placed = []  # list of (box_idx, x, y, z, w, h, d)
        
        for idx in sorted_indices:
            w, h, d = boxes[idx]
            best_pos = None
            best_orient = None
            best_key = None
            
            for orient in get_orientations(w, h, d):
                bw, bh, bd = orient
                if bw > W or bh > H or bd > D:
                    continue
                
                for z in range(0, D - bd + 1):
                    for y in range(0, H - bh + 1):
                        for x in range(0, W - bw + 1):
                            if can_place(x, y, z, bw, bh, bd, [(px, py, pz, pw, ph, pd) for (_, px, py, pz, pw, ph, pd) in placed]):
                                key = (z, y, x)
                                if best_key is None or key < best_key:
                                    best_key = key
                                    best_pos = (x, y, z)
                                    best_orient = orient
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                
                if best_pos is not None:
                    break
            
            if best_pos is not None:
                x, y, z = best_pos
                bw, bh, bd = best_orient
                placed.append((idx, x, y, z, bw, bh, bd))
        
        return placed

    placed = pack_boxes_tracked(sorted_indices, (W, H, D))

    # Build the result
    placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(len(boxes))]
    for (idx, x, y, z, bw, bh, bd) in placed:
        placements[idx] = (0, x, y, z, bw, bh, bd)

    return placements
# EVOLVE-BLOCK-END
