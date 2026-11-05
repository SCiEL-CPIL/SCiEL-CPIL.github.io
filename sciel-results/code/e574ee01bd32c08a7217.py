"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import permutations

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Generate all 6 possible rotations for each box
    def get_rotations(w, h, d):
        seen = set()
        rots = []
        for perm in permutations([w, h, d]):
            if perm not in seen:
                seen.add(perm)
                rots.append(perm)
        return rots
    
    # Precompute rotations for all boxes
    box_rotations = [get_rotations(w, h, d) for w, h, d in boxes]
    box_volumes = [w * h * d for w, h, d in boxes]
    
    # Helper to check if a box fits in the container
    def fits_in_container(px, py, pz, pw, ph, pd):
        return (px >= 0 and py >= 0 and pz >= 0 and 
                px + pw <= W and py + ph <= H and pz + pd <= D)

    # Helper to check overlap
    def has_overlap(px, py, pz, pw, ph, pd, placed_boxes):
        for (ex, ey, ez, ew, eh, ed) in placed_boxes:
            if (px < ex + ew and ex < px + pw and
                py < ey + eh and ey < py + ph and
                pz < ez + ed and ez < pz + pd):
                return True
        return False

    # Strategy: Best Fit Decreasing with Gravity Heuristic
    # Sort boxes by volume descending
    order = sorted(range(n), key=lambda i: box_volumes[i], reverse=True)
    
    placements = [None] * n
    placed_boxes = []
    total_vol = 0
    
    for idx in order:
        w, h, d = boxes[idx]
        best_pos = None
        best_rot = None
        best_score = float('inf')
        
        for (rw, rh, rd) in box_rotations[idx]:
            # Generate candidate points
            # 1. Origin
            # 2. Faces of existing boxes (Top, Right, Front)
            candidates = set()
            candidates.add((0, 0, 0))
            
            for (ex, ey, ez, ew, eh, ed) in placed_boxes:
                # Place on top
                candidates.add((ex, ey, ez + ed))
                # Place to the right
                candidates.add((ex + ew, ey, ez))
                # Place in front
                candidates.add((ex, ey + eh, ez))
                
            for (px, py, pz) in candidates:
                if not fits_in_container(px, py, pz, rw, rh, rd):
                    continue
                if has_overlap(px, py, pz, rw, rh, rd, placed_boxes):
                    continue
                
                # Gravity Score: Minimize Z, then Y, then X
                # Weights reflect the container dimensions to prioritize filling the "bottom" layer
                score = pz * (W * H) + py * W + px
                
                if score < best_score:
                    best_score = score
                    best_pos = (px, py, pz)
                    best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            px, py, pz = best_pos
            rw, rh, rd = best_rot
            placements[idx] = (0, px, py, pz, rw, rh, rd)
            placed_boxes.append((px, py, pz, rw, rh, rd))
            total_vol += box_volumes[idx]
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
            
    # Fill any remaining None with -1
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements
# EVOLVE-BLOCK-END
