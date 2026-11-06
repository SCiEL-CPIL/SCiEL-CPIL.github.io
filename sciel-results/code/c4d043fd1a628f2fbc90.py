"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 110  # safety margin before 120s limit
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    # Sort indices by volume descending, then by max dimension descending
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: (-(x[1][0]*x[1][1]*x[1][2]), -max(x[1])))
    
    # placed: list of (x, y, z, w, h, d) for successfully placed boxes
    placed = []
    result = [None] * n  # will store (bin, x, y, z, w, h, d) for each original index
    
    def get_orientations(w, h, d):
        """Return unique orientations (w, h, d) that fit in container."""
        seen = set()
        orients = []
        for perm in permutations((w, h, d)):
            ow, oh, od = perm
            if (ow, oh, od) not in seen:
                if ow <= W and oh <= H and od <= D:
                    seen.add((ow, oh, od))
                    orients.append((ow, oh, od))
        return orients
    
    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        """Check if two axis-aligned boxes overlap."""
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)
    
    def find_best_position(ow, oh, od):
        """Find the best (lowest z, then y, then x) position for a box of size (ow, oh, od).
        Returns (x, y, z) or None if no valid position found."""
        # Candidate positions: origin + for each placed box, 3 anchor positions
        candidates = [(0, 0, 0)]
        for (px, py, pz, pw, ph, pd) in placed:
            # Box on top of placed box
            candidates.append((px, py, pz + pd))
            # Box in front of placed box (larger y)
            candidates.append((px, py + ph, pz))
            # Box to the right of placed box (larger x)
            candidates.append((px + pw, py, pz))
        
        best = None
        best_key = None
        for (cx, cy, cz) in candidates:
            # Check bounds
            if cx + ow > W or cy + oh > H or cz + od > D:
                continue
            # Check overlap with all placed boxes
            valid = True
            for (px, py, pz, pw, ph, pd) in placed:
                if overlaps(cx, cy, cz, ow, oh, od, px, py, pz, pw, ph, pd):
                    valid = False
                    break
            if not valid:
                continue
            # BLF: minimize (z, y, x)
            key = (cz, cy, cx)
            if best_key is None or key < best_key:
                best_key = key
                best = (cx, cy, cz)
        
        return best
    
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > DEADLINE:
            break
        
        orients = get_orientations(w, h, d)
        if not orients:
            result[orig_idx] = (-1, 0, 0, 0, w, h, d)
            continue
        
        # Try each orientation, find best position
        best_pos = None
        best_orient = None
        best_key = None
        for (ow, oh, od) in orients:
            pos = find_best_position(ow, oh, od)
            if pos is not None:
                key = (pos[2], pos[1], pos[0])  # (z, y, x)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pos = pos
                    best_orient = (ow, oh, od)
        
        if best_pos is not None:
            bx, by, bz = best_pos
            ow, oh, od = best_orient
            placed.append((bx, by, bz, ow, oh, od))
            result[orig_idx] = (0, bx, by, bz, ow, oh, od)
        else:
            result[orig_idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill in any unprocessed boxes as left out
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    
    return result
# EVOLVE-BLOCK-END
