# EVOLVE-BLOCK-START
import numpy as np
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Sort boxes by volume descending, then by max dimension descending
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: (-(x[1][0]*x[1][1]*x[1][2]), -max(x[1])))
    
    placed = []  # List of (x, y, z, w, h, d, box_idx)
    placements = [None] * len(boxes)
    
    # Set of candidate points (x, y, z)
    # A point is a candidate if it is on the "surface" (grounded)
    candidates = set([(0, 0, 0)])
    
    def is_grounded(x, y, z):
        if y == 0:
            return True
        # Check if there is a box directly below this point
        for (ox, oy, oz, ow, oh, od, _) in placed:
            if oy + oh == y:
                if ox <= x < ox + ow and oz <= z < oz + od:
                    return True
        return False

    def is_inside_placed(x, y, z, w, h, d):
        # Check if the box [x, x+w] x [y, y+h] x [z, z+d] overlaps any placed box
        # and is strictly inside or overlapping
        # Actually, we just need to check for overlap for validity
        for (ox, oy, oz, ow, oh, od, _) in placed:
            if not (x + w <= ox or ox + ow <= x or y + h <= oy or oy + oh <= y or z + d <= oz or oz + od <= z):
                return True
        return False

    def is_valid_placement(x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        if not is_grounded(x, y, z):
            return False
        if is_inside_placed(x, y, z, w, h, d):
            return False
        return True

    def update_candidates(new_box):
        nx, ny, nz, nw, nh, nd, _ = new_box
        new_points = [
            (nx, ny, nz),
            (nx + nw, ny, nz),
            (nx, ny + nh, nz),
            (nx + nw, ny + nh, nz),
            (nx, ny, nz + nd),
            (nx + nw, ny, nz + nd),
            (nx, ny + nh, nz + nd),
            (nx + nw, ny + nh, nz + nd)
        ]
        
        # Add valid new points
        for p in new_points:
            px, py, pz = p
            if 0 <= px <= W and 0 <= py <= H and 0 <= pz <= D:
                if is_grounded(px, py, pz):
                    candidates.add(p)
        
        # Remove points that are now inside the new box
        # A point (px, py, pz) is "covered" if it is inside the new box
        # Note: The corner (nx, ny, nz) is the bottom-left-front.
        # Points strictly inside the box volume are no longer surface points.
        # We should remove any candidate that falls within [nx, nx+nw] x [ny, ny+nh] x [nz, nz+nd]
        # However, the faces themselves might still be valid for *other* boxes?
        # If a point is on the face of the new box, it is still a surface point.
        # If a point is strictly inside, it is not.
        # So we remove points where nx < px < nx+nw AND ny < py < ny+nh AND nz < pz < nz+nd.
        # Actually, if a point is on the boundary of the new box, it might be shared with other boxes.
        # It is safer to only remove points that are strictly inside.
        
        to_remove = []
        for p in candidates:
            px, py, pz = p
            if nx < px < nx + nw and ny < py < ny + nh and nz < pz < nz + nd:
                to_remove.append(p)
        
        for p in to_remove:
            candidates.remove(p)

    # Try to place each box
    for box_idx, (w, h, d) in indexed_boxes:
        best_pos = None
        best_rot = None
        best_key = None
        
        # Try all unique permutations of dimensions
        unique_perms = set(permutations([w, h, d]))
        
        for pw, ph, pd in unique_perms:
            for (cx, cy, cz) in list(candidates):
                if is_valid_placement(cx, cy, cz, pw, ph, pd):
                    # Heuristic: Prefer lower y, then lower x, then lower z
                    key = (cy, cx, cz)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = (cx, cy, cz)
                        best_rot = (pw, ph, pd)
        
        if best_pos:
            bx, by, bz = best_pos
            bw, bh, bd = best_rot
            new_box = (bx, by, bz, bw, bh, bd, box_idx)
            placed.append(new_box)
            placements[box_idx] = (0, bx, by, bz, bw, bh, bd)
            update_candidates(new_box)
        else:
            placements[box_idx] = (-1, 0, 0, 0, w, h, d)
            
    # Ensure all placements are set
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements
# EVOLVE-BLOCK-END
