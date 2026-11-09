"""
3D Bin Packing solver using Skyline/Corner-based placement and Best Fit Decreasing.
"""
import time
import heapq

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    container_vol = W * H * D
    
    # Sort boxes by volume descending, then by max dimension descending
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: (x[4], max(x[1], x[2], x[3])), reverse=True)
    
    # Bins state
    # Each bin has a list of placed boxes: (x, y, z, w, h, d)
    # and a set of candidate positions (corners)
    bins = []
    
    def get_corners(placed_boxes, W, H, D):
        """
        Generate candidate positions (corners) for a new box.
        A corner is a point (x, y, z) where a new box can start.
        Initially, (0,0,0).
        For each placed box (bx, by, bz, bw, bh, bd), new corners are:
        (bx + bw, by, bz)  # Right face
        (bx, by + bh, bz)  # Top face
        (bx, by, bz + bd)  # Front face
        We only keep corners that are not dominated by another corner.
        A corner c1 is dominated by c2 if c2.x <= c1.x, c2.y <= c1.y, c2.z <= c1.z.
        """
        corners = set()
        corners.add((0, 0, 0))
        
        for bx, by, bz, bw, bh, bd in placed_boxes:
            # Generate 3 new corners from each box
            c1 = (bx + bw, by, bz)
            c2 = (bx, by + bh, bz)
            c3 = (bx, by, bz + bd)
            
            # Only add if within container bounds
            if c1[0] <= W and c1[1] <= H and c1[2] <= D:
                corners.add(c1)
            if c2[0] <= W and c2[1] <= H and c2[2] <= D:
                corners.add(c2)
            if c3[0] <= W and c3[1] <= H and c3[2] <= D:
                corners.add(c3)
        
        # Remove dominated corners
        # c1 dominates c2 if c1.x <= c2.x and c1.y <= c2.y and c1.z <= c2.z
        # We want to keep the "minimal" corners in the partial order.
        # This is an O(N^2) operation, but N is small (<= 3 * num_boxes_in_bin)
        corner_list = list(corners)
        nondominated = []
        for c in corner_list:
            dominated = False
            for c2 in corner_list:
                if c == c2:
                    continue
                if c2[0] <= c[0] and c2[1] <= c[1] and c2[2] <= c[2]:
                    # c is dominated by c2
                    # If they are identical, we don't care, but if strictly less in any, c is dominated
                    if c2[0] < c[0] or c2[1] < c[1] or c2[2] < c[2]:
                        dominated = True
                        break
            if not dominated:
                nondominated.append(c)
        
        return nondominated

    def is_valid_placement(bin_placed, x, y, z, w, h, d):
        """
        Check if placing a box at (x,y,z) with dims (w,h,d) overlaps with any existing box in the bin.
        """
        for bx, by, bz, bw, bh, bd in bin_placed:
            # Check overlap
            if (x < bx + bw and x + w > bx and
                y < by + bh and y + h > by and
                z < bz + bd and z + d > bz):
                return False
        # Check bounds
        if x + w > W or y + h > H or z + d > D:
            return False
        return True

    def get_best_position(bin_placed, w, h, d):
        """
        Find the best valid position for a box in a bin.
        Returns (x, y, z) or None.
        Heuristic: Minimize (z, y, x) to keep the box as low and compact as possible.
        """
        corners = get_corners(bin_placed, W, H, D)
        best_pos = None
        best_key = None
        
        for x, y, z in corners:
            if is_valid_placement(bin_placed, x, y, z, w, h, d):
                # Priority: Lowest Z, then Lowest Y, then Lowest X
                # This encourages stacking and filling from the bottom
                key = (z, y, x)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pos = (x, y, z)
        
        return best_pos

    # Place each box
    placements = [None] * len(boxes)
    
    for orig_idx, w, h, d, vol in indexed_boxes:
        placed = False
        
        # Check time
        if time.time() > deadline:
            # Fallback: place in new bin at origin
            b = len(bins)
            bins.append([])
            pos = (0, 0, 0)
            bins[b].append((pos[0], pos[1], pos[2], w, h, d))
            placements[orig_idx] = (b, pos[0], pos[1], pos[2], w, h, d)
            continue
        
        # Try existing bins first
        # Best Fit: Try bins in order of "fullness" or "tightness"
        # We'll try all existing bins and pick the one where the box fits most "tightly"
        # Tightness can be measured by the remaining volume in the bin, or the height of the placement.
        # Let's try bins sorted by the number of boxes (more boxes = fuller) or just iterate.
        # For efficiency, we'll just iterate through existing bins and pick the best valid placement
        # based on a global heuristic: minimize (bin_index, z, y, x) ? No, we want to reuse bins.
        
        best_bin_idx = -1
        best_pos = None
        best_score = float('inf')
        
        for b in range(len(bins)):
            if time.time() > deadline:
                break
            pos = get_best_position(bins[b], w, h, d)
            if pos is not None:
                # Score: We prefer bins that are already full.
                # Let's use the height of the placement (z) as a proxy for "how high up" we are placing.
                # Lower z is better.
                # Also, we prefer to fill existing bins, so we add a small penalty for using a new bin (handled by loop order).
                # To encourage filling existing bins, we can score by (z, b) ?
                # Actually, if we just pick the first valid bin, it's First Fit.
                # Best Fit: Pick the bin where the box leaves the least space?
                # Let's stick to a simple heuristic: Pick the bin where the box is placed lowest (smallest z).
                # If tie, pick the smallest bin index.
                score = (pos[2], b) 
                if score < best_score:
                    best_score = score
                    best_bin_idx = b
                    best_pos = pos
        
        if best_bin_idx != -1:
            # Place in the selected bin
            bins[best_bin_idx].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            placements[orig_idx] = (best_bin_idx, best_pos[0], best_pos[1], best_pos[2], w, h, d)
            placed = True
        
        if not placed:
            # Open a new bin
            b = len(bins)
            bins.append([])
            
            # Place at origin (0,0,0) is always valid for a new bin
            pos = (0, 0, 0)
            bins[b].append((pos[0], pos[1], pos[2], w, h, d))
            placements[orig_idx] = (b, pos[0], pos[1], pos[2], w, h, d)
            
            # Check time after placement
            if time.time() > deadline:
                # We can't do more, but we have placed all boxes so far.
                # If there are more boxes, we will hit the deadline in the next iteration.
                pass

    return placements
# EVOLVE-BLOCK-END
