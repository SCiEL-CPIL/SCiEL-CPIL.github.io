"""3D Container Loading - Guillotine cutting with best-fit heuristic."""
import time
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    t0 = time.time()
    DEADLINE = 110  # safety margin before 120s limit
    
    n = len(boxes)
    # Sort boxes by volume descending
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    
    placements = [None] * n
    
    def get_orientations(w, h, d):
        """Return unique orientations as (bw, bh, bd) tuples."""
        return list(set(permutations([w, h, d])))
    
    # Each slot: (x, y, z, w, h, d)
    slots = [(0, 0, 0, W, H, D)]
    
    for idx, (orig_idx, (w, h, d)) in indexed_boxes:
        if time.time() - t0 > DEADLINE:
            for j in range(n):
                if placements[j] is None:
                    placements[j] = (-1, 0, 0, 0, boxes[j][0], boxes[j][1], boxes[j][2])
            return placements
        
        placed = False
        orients = get_orientations(w, h, d)
        
        best_slot_idx = -1
        best_orient = None
        best_score = float('inf')
        
        for si, (sx, sy, sz, sw, sh, sd) in enumerate(slots):
            for bw, bh, bd in orients:
                if bw <= sw and bh <= sh and bd <= sd:
                    # Score: minimize z first, then x, then y; also prefer tighter fit
                    # Use weighted sum: z dominates, then x, then y, then wasted space
                    waste = (sw - bw) * sh * sd + sw * (sh - bh) * sd + sw * sh * (sd - bd)
                    score = sz * 1000000 + sx * 1000 + sy * 10 + waste
                    if score < best_score:
                        best_score = score
                        best_slot_idx = si
                        best_orient = (bw, bh, bd)
        
        if best_slot_idx >= 0:
            sx, sy, sz, sw, sh, sd = slots[best_slot_idx]
            bw, bh, bd = best_orient
            
            # Place at bottom-left-front of the slot
            x, y, z = sx, sy, sz
            
            # Remove the slot
            slots.pop(best_slot_idx)
            
            # Add remaining sub-slots
            if sw > bw:
                slots.append((sx + bw, sy, sz, sw - bw, sh, sd))
            if sh > bh:
                slots.append((sx, sy + bh, sz, bw, sh - bh, sd))
            if sd > bd:
                slots.append((sx, sy, sz + bd, bw, bh, sd - bd))
            
            placements[orig_idx] = (0, x, y, z, bw, bh, bd)
            placed = True
        else:
            placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
    
    for j in range(n):
        if placements[j] is None:
            placements[j] = (-1, 0, 0, 0, boxes[j][0], boxes[j][1], boxes[j][2])
    
    return placements
