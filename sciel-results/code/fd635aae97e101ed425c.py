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
        # Return list of (w, h, d) permutations
        seen = set()
        rots = []
        for perm in permutations([w, h, d]):
            if perm not in seen:
                seen.add(perm)
                rots.append(perm)
        return rots
    
    # Precompute rotations for all boxes
    box_rotations = [get_rotations(w, h, d) for w, h, d in boxes]
    
    # Sort boxes by volume descending (larger first)
    box_volumes = [w * h * d for w, h, d in boxes]
    sorted_indices = sorted(range(n), key=lambda i: box_volumes[i], reverse=True)
    
    # Placement data structure: list of (x, y, z, w, h, d) for placed boxes
    placed = []  # list of tuples (x, y, z, w, h, d)
    
    def can_place(px, py, pz, pw, ph, pd):
        """Check if a box can be placed at (px, py, pz) with dimensions (pw, ph, pd)."""
        # Check bounds
        if px < 0 or py < 0 or pz < 0:
            return False
        if px + pw > W or py + ph > H or pz + pd > D:
            return False
        
        # Check overlap with placed boxes
        for (ex, ey, ez, ew, eh, ed) in placed:
            # Check if boxes overlap
            if (px < ex + ew and ex < px + pw and
                py < ey + eh and ey < py + ph and
                pz < ez + ed and ez < pz + pd):
                return False
        return True
    
    def find_position_for_box(pw, ph, pd):
        """Find a valid position for a box with given dimensions.
        Uses a greedy approach: try to place at the lowest possible position.
        """
        # Candidate positions: 
        # 1. Origin (0,0,0)
        # 2. Along each axis from existing box edges
        candidate_positions = set()
        candidate_positions.add((0, 0, 0))
        
        # Generate candidate positions based on existing placements
        for (ex, ey, ez, ew, eh, ed) in placed:
            # Place to the right of existing box
            candidate_positions.add((ex + ew, ey, ez))
            # Place on top of existing box
            candidate_positions.add((ex, ey + eh, ez))
            # Place behind existing box
            candidate_positions.add((ex, ey, ez + ed))
            # Also try positions aligned with edges
            candidate_positions.add((ex + ew, ey, ez))
            candidate_positions.add((ex, ey + eh, ez))
            candidate_positions.add((ex, ey, ez + ed))
            # Diagonal-ish positions
            candidate_positions.add((ex + ew, ey + eh, ez))
            candidate_positions.add((ex + ew, ey, ez + ed))
            candidate_positions.add((ex, ey + eh, ez + ed))
            candidate_positions.add((ex + ew, ey + eh, ez + ed))
        
        # Also add positions at 0 for each axis
        for (ex, ey, ez, ew, eh, ed) in placed:
            candidate_positions.add((0, ey, ez))
            candidate_positions.add((ex, 0, ez))
            candidate_positions.add((ex, ey, 0))
        
        # Sort candidates by priority: lower z, then lower y, then lower x
        candidates = sorted(candidate_positions, key=lambda p: (p[2], p[1], p[0]))
        
        for (px, py, pz) in candidates:
            if can_place(px, py, pz, pw, ph, pd):
                return (px, py, pz)
        
        return None
    
    # Greedy placement
    placements = [None] * n
    
    for idx in sorted_indices:
        w, h, d = boxes[idx]
        placed_flag = False
        
        # Try all rotations
        for (rw, rh, rd) in box_rotations[idx]:
            pos = find_position_for_box(rw, rh, rd)
            if pos is not None:
                px, py, pz = pos
                placements[idx] = (0, px, py, pz, rw, rh, rd)
                placed.append((px, py, pz, rw, rh, rd))
                placed_flag = True
                break
        
        if not placed_flag:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Ensure all placements are set
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
