"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pack(boxes, container):
    """
    Place boxes into bins using a Best-Fit Decreasing with 3D corner-based placement.
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 110.0  # safety margin before 120s kill

    # Sort boxes by volume descending for BFD, but keep original indices
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)

    # Each bin will have a list of placed boxes: (x, y, z, w, h, d)
    bins = []  # list of lists
    # Track bounding box of each bin for quick rejection
    bin_bboxes = []  # list of (max_x, max_y, max_z)

    def can_place(bin_placements, x, y, z, w, h, d):
        """Check if a box fits at (x,y,z) in the given bin without overlap and within bounds."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # Check overlap
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True

    def generate_candidates(bin_placements, w, h, d):
        """
        Generate candidate positions from corners of existing boxes.
        Only keep candidates that fit within bounds.
        """
        candidates = set()
        candidates.add((0, 0, 0))
        
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # Generate all 8 corner positions that could serve as lower-left-front
            # of a new box adjacent to this existing box
            for dx in [0, pw]:
                for dy in [0, ph]:
                    for dz in [0, pd]:
                        cx, cy, cz = px + dx, py + dy, pz + dz
                        if cx + w <= W and cy + h <= H and cz + d <= D:
                            candidates.add((cx, cy, cz))
        
        return candidates

    def find_position(bin_placements, w, h, d):
        """
        Find the best position for a box in a bin using corner-based placement.
        Returns (x, y, z) or None if no position found.
        """
        if not bin_placements:
            return (0, 0, 0)
        
        candidates = generate_candidates(bin_placements, w, h, d)
        
        best_pos = None
        best_score = None
        
        for (xc, yc, zc) in candidates:
            if can_place(bin_placements, xc, yc, zc, w, h, d):
                # Score: prefer positions that minimize the resulting bounding box extent
                # Use weighted sum with emphasis on filling compactly
                new_max_x = max(xc + w, max((px + pw) for px, py, pz, pw, ph, pd in bin_placements))
                new_max_y = max(yc + h, max((py + ph) for py, pz, px, pw, ph, pd in bin_placements))
                new_max_z = max(zc + d, max((pz + pd) for pz, px, py, pw, ph, pd in bin_placements))
                
                # Prefer positions that keep the bounding box small
                score = new_max_x * new_max_y * new_max_z
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (xc, yc, zc)
        
        return best_pos

    # Main packing loop
    result = [None] * n  # will hold (bin_id, x, y, z, w, h, d) for each original index
    
    for idx, vol, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break
        
        # Try to place in existing bins, preferring the bin where it fits tightest
        best_bin = -1
        best_pos = None
        best_fit_score = None
        
        for b_idx, bin_placements in enumerate(bins):
            pos = find_position(bin_placements, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Compute the resulting bounding box volume as fit score
                max_x = max(x + w, max((px + pw) for px, py, pz, pw, ph, pd in bin_placements))
                max_y = max(y + h, max((py + ph) for py, pz, px, pw, ph, pd in bin_placements))
                max_z = max(z + d, max((pz + pd) for pz, px, py, pw, ph, pd in bin_placements))
                fit_score = max_x * max_y * max_z
                
                if best_fit_score is None or fit_score < best_fit_score:
                    best_fit_score = fit_score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin >= 0:
            x, y, z = best_pos
            bins[best_bin].append((x, y, z, w, h, d))
            result[idx] = (best_bin, x, y, z, w, h, d)
        else:
            # Create a new bin
            new_bin = [(0, 0, 0, w, h, d)]
            bins.append(new_bin)
            result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

    # If we hit the deadline, place remaining boxes using a simple fallback
    for idx, vol, w, h, d in indexed_boxes:
        if result[idx] is not None:
            continue
        # Find any bin where it can be placed, or create a new one
        placed = False
        for b_idx, bin_placements in enumerate(bins):
            pos = find_position(bin_placements, w, h, d)
            if pos is not None:
                x, y, z = pos
                bins[b_idx].append((x, y, z, w, h, d))
                result[idx] = (b_idx, x, y, z, w, h, d)
                placed = True
                break
        if not placed:
            new_bin = [(0, 0, 0, w, h, d)]
            bins.append(new_bin)
            result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

    return result
# EVOLVE-BLOCK-END
