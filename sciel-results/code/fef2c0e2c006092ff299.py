"""Skyline / floor-map based 3D container loading."""
import time
import math
import numpy as np
from itertools import product

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 115.0  # safety margin before 120s kill

    # Compute all possible orientations for each box (unique dimensions)
    def get_orientations(w, h, d):
        dims = sorted([w, h, d])
        # Generate all 6 permutations but deduplicate
        seen = set()
        orients = []
        for perm in product(dims, repeat=3):
            if perm in seen:
                continue
            # Actually we need all distinct permutations of the original dims
            pass
        # Simpler: just use the 6 rotations explicitly
        rotations = [
            (w, h, d),
            (w, d, h),
            (h, w, d),
            (h, d, w),
            (d, w, h),
            (d, h, w)
        ]
        unique = []
        seen = set()
        for r in rotations:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        return unique

    # Sort boxes by volume descending
    box_indices = list(range(n))
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    box_indices.sort(key=lambda i: box_volumes[i], reverse=True)

    # Skyline representation: list of (x, y, z, w, h, d) for placed boxes
    # We'll use a list of placed boxes and check against them
    # For efficiency with 60 boxes, we can use a simpler approach
    # Maintain a list of "free" rectangles on the floor, or just check against placed boxes
    
    # Better approach: maintain skyline as list of (x, y, z) anchor points
    # For 3D, we can use a grid-based approach or check against placed boxes
    
    # Let's use a simpler but effective approach:
    # Maintain placed boxes as a list, and for each new box, try all orientations
    # and all potential positions (corners of existing boxes + origin)
    
    placed = []  # list of (x, y, z, w, h, d)
    result = [None] * n
    
    def can_place(x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap with all placed boxes
        for px, py, pz, pw, ph, pd in placed:
            if x < px + pw and x + w > px and \
               y < py + ph and y + h > py and \
               z < pz + pd and z + d > pz:
                return False
        return True

    def get_candidate_positions(placed_boxes, w, h, d):
        """Get candidate (x, y, z) positions to try."""
        candidates = set()
        # Origin
        candidates.add((0, 0, 0))
        # Corners of existing boxes
        for px, py, pz, pw, ph, pd in placed_boxes:
            # Try placing next to each box in each direction
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            # Also try the box's own position (might fit in gaps)
            candidates.add((px, py, pz))
        
        # Filter valid candidates
        valid = []
        for cx, cy, cz in candidates:
            if 0 <= cx < W and 0 <= cy < H and 0 <= cz < D:
                if cx + w <= W and cy + h <= H and cz + d <= D:
                    valid.append((cx, cy, cz))
        return valid

    for idx in box_indices:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        orientations = get_orientations(w, h, d)
        
        best_pos = None
        best_orient = None
        best_score = -1
        
        for ow, oh, od in orientations:
            if ow > W or oh > H or od > D:
                continue
            candidates = get_candidate_positions(placed, ow, oh, od)
            for cx, cy, cz in candidates:
                if can_place(cx, cy, cz, ow, oh, od):
                    # Score: prefer positions that are "lower" and "more left"
                    # to leave room for future boxes
                    score = -(cx + cy + cz)  # Minimize sum of coordinates
                    if score > best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_orient = (ow, oh, od)
        
        if best_pos is not None:
            px, py, pz = best_pos
            pw, ph, pd = best_orient
            placed.append((px, py, pz, pw, ph, pd))
            result[idx] = (0, px, py, pz, pw, ph, pd)
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)

    # Fill in any remaining None entries
    for i in range(n):
        if result[i] is None:
            result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

    return result
