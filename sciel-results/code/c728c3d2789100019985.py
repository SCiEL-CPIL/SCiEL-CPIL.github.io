"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import permutations

def _get_rotations(w, h, d):
    """Get all unique rotations of a box."""
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    return list(rots)

def _try_place(boxes, container, seed=0):
    """
    Attempt to place boxes into the container using Extreme Point heuristic.
    Returns a list of placements: (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        # Sort rotations: prefer lower height (z-dim) to stack better, then lower depth, then lower width
        rots.sort(key=lambda r: (r[2], r[1], r[0]))
        box_rotations.append(rots)
    
    # Sort boxes by volume (largest first) for better packing
    box_indices = list(range(n))
    box_volumes = [w * h * d for w, h, d in boxes]
    box_indices.sort(key=lambda i: -box_volumes[i])
    
    placed = []  # List of (x, y, z, w, h, d)
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    filled_volume = 0
    
    # Candidate points for placement (Extreme Points)
    # Each point is (x, y, z). We try to place boxes at these points.
    # A point is valid if it is on the "surface" of the placed boxes or the container floor/walls.
    # Simplified: Start with (0,0,0). When a box is placed at (x,y,z) with dims (w,h,d),
    # add (x+w, y, z), (x, y+h, z), (x, y, z+d) to candidates if they are within bounds and not inside existing boxes.
    candidates = [(0, 0, 0)]
    
    def is_inside_existing(px, py, pz, pw, ph, pd):
        for ex, ey, ez, ew, eh, ed in placed:
            if px >= ex and px + pw <= ex + ew and \
               py >= ey and py + ph <= ey + eh and \
               pz >= ez and pz + pd <= ez + ed:
                return True
        return False

    def overlaps_any(px, py, pz, pw, ph, pd):
        for ex, ey, ez, ew, eh, ed in placed:
            if (px < ex + ew and px + pw > ex and
                py < ey + eh and py + ph > ey and
                pz < ez + ed and pz + pd > ez):
                return True
        return False

    def add_candidate(x, y, z):
        if 0 <= x <= W and 0 <= y <= H and 0 <= z <= D:
            # Check if this point is strictly inside an existing box (if so, it's not a surface point)
            # Actually, for extreme point, we just need to check if a box *starting* here overlaps.
            # We add it to the list, and we will filter/sort later or check validity during placement.
            # To keep the list small, we can check if it's covered by existing boxes.
            # A point (x,y,z) is covered if there is a box containing it.
            # But we place boxes *at* the point, so the point is the bottom-left-back corner.
            # We should only keep points that are not inside any existing box.
            inside = False
            for ex, ey, ez, ew, eh, ed in placed:
                if x > ex and x < ex + ew and y > ey and y < ey + eh and z > ez and z < ez + ed:
                    inside = True
                    break
            if not inside:
                if (x, y, z) not in candidates:
                    candidates.append((x, y, z))

    for idx in box_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        rots = box_rotations[idx]
        
        placed_this = False
        
        # Sort candidates by Z (lowest), then Y (lowest), then X (leftmost)
        # This promotes compact packing towards the origin
        # We need to check all candidates, but try the "best" ones first
        # To be safe and fast, we iterate through a sorted copy of candidates
        # Note: candidates list grows. We should probably limit its size or clean it up.
        # For 60 boxes, the number of extreme points is manageable.
        
        # Filter candidates: only keep those that are within container bounds
        valid_candidates = [c for c in candidates if 0 <= c[0] <= W and 0 <= c[1] <= H and 0 <= c[2] <= D]
        # Sort them
        valid_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        
        # To improve performance, we can limit the number of candidates we try per box
        # If we have too many, we might miss a spot. But usually, the "lowest" spots are best.
        # Let's try all valid candidates, but break early if found.
        
        for bw, bh, bd in rots:
            if time.time() > deadline:
                break
            
            # Try each candidate point
            found = False
            # We iterate through sorted candidates
            for cx, cy, cz in valid_candidates:
                # Check bounds
                if cx + bw > W or cy + bh > H or cz + bd > D:
                    continue
                
                # Check overlap
                if not overlaps_any(cx, cy, cz, bw, bh, bd):
                    # Place box
                    placed.append((cx, cy, cz, bw, bh, bd))
                    result[idx] = (0, cx, cy, cz, bw, bh, bd)
                    filled_volume += bw * bh * bd
                    
                    # Generate new candidates
                    add_candidate(cx + bw, cy, cz)
                    add_candidate(cx, cy + bh, cz)
                    add_candidate(cx, cy, cz + bd)
                    
                    # Remove the used candidate? No, other boxes might use it?
                    # No, if a box is placed at (cx,cy,cz), that point is now occupied.
                    # But the *corner* of the new box creates new extreme points.
                    # The point (cx,cy,cz) itself is no longer an "empty" extreme point for a new box to start *at* if it overlaps?
                    # Actually, if a box is placed at P, P is inside the box.
                    # So we should remove P from candidates?
                    # Yes, because any new box starting at P would overlap the existing box (since P is the corner).
                    # Wait, if I place box A at (0,0,0) size (1,1,1).
                    # Candidate (0,0,0) is used.
                    # New candidates: (1,0,0), (0,1,0), (0,0,1).
                    # If I try to place box B at (0,0,0), it overlaps A.
                    # So we should remove (0,0,0) from candidates.
                    if (cx, cy, cz) in candidates:
                        candidates.remove((cx, cy, cz))
                    
                    placed_this = True
                    found = True
                    break
            
            if found:
                break
        
        if not placed_this:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    return result

# Removed broken shelf and feasible strategies to prevent overlaps and save time.
# The robust _try_place is sufficient.

def _evaluate_result(placement, container):
    """Evaluate the fraction of container volume filled."""
    W, H, D = container
    total_volume = W * H * D
    if total_volume == 0:
        return 0.0
    
    filled = 0
    for bin_id, x, y, z, w, h, d in placement:
        if bin_id == 0:
            # Check if placement is valid
            if x >= 0 and y >= 0 and z >= 0 and \
               x + w <= W and y + h <= H and z + d <= D:
                filled += w * h * d
    
    return filled / total_volume

def load(boxes, container):
    """
    Main entry point: place boxes into container.
    Returns list of (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Use the robust Extreme Point strategy
    placement = _try_place(boxes, container)
    
    # Verify validity just in case (though _try_place should be correct)
    # If invalid, return empty/failed placements
    # For now, trust the overlap check in _try_place
    
    return placement
# EVOLVE-BLOCK-END
