"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    W, H, D = container
    n = len(boxes)
    
    # Sort boxes by volume descending (best-fit decreasing)
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # Each bin stores a list of placed boxes: (x, y, z, w, h, d) and an occupancy grid
    bins = []
    
    def create_bin():
        return {"boxes": [], "occupied": np.zeros((W, H, D), dtype=bool)}
    
    def find_best_position(bin_data, w, h, d):
        """Find the best valid (x, y, z) position in the bin using bottom-left-far heuristic.
        Returns (x, y, z) or None if no valid position exists."""
        if w > W or h > H or d > D:
            return None
        
        occupied = bin_data["occupied"]
        boxes_list = bin_data["boxes"]
        
        # Generate candidate positions from all combinations of x, y, z boundaries
        x_candidates = {0}
        y_candidates = {0}
        z_candidates = {0}
        
        for (bx, by, bz, bw, bh, bd) in boxes_list:
            if bx + bw < W:
                x_candidates.add(bx + bw)
            if by + bh < H:
                y_candidates.add(by + bh)
            if bz + bd < D:
                z_candidates.add(bz + bd)
        
        # Also add some spread-out candidates to explore the space
        if W > 0:
            x_candidates.add(min(w, W - w) if W - w > 0 else 0)
        if H > 0:
            y_candidates.add(min(h, H - h) if H - h > 0 else 0)
        if D > 0:
            z_candidates.add(min(d, D - d) if D - d > 0 else 0)
        
        best_pos = None
        best_score = float('inf')
        
        for x in sorted(x_candidates):
            if x + w > W:
                continue
            for y in sorted(y_candidates):
                if y + h > H:
                    continue
                for z in sorted(z_candidates):
                    if z + d > D:
                        continue
                    # Check if this position is free using the occupancy grid
                    if not np.any(occupied[x:x+w, y:y+h, z:z+d]):
                        # Score: prefer positions that are lower, then further back, then further right
                        # This is the "bottom-left-far" heuristic
                        score = z * W * H + y * W + x
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    def bin_volume(bin_data):
        return sum(w*h*d for (_, _, _, w, h, d) in bin_data["boxes"])
    
    # Place each box
    placements = [None] * n  # (bin_idx, x, y, z, w, h, d)
    
    for orig_idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break
        
        best_bin = -1
        best_pos = None
        best_score = float('inf')
        
        # Try existing bins
        for bi, bin_data in enumerate(bins):
            pos = find_best_position(bin_data, w, h, d)
            if pos is not None:
                # Score: prefer bin that results in least wasted space
                # Use the position's z+y+x as a proxy for tightness
                score = pos[2] * W * H + pos[1] * W + pos[0]
                if score < best_score:
                    best_score = score
                    best_bin = bi
                    best_pos = pos
        
        if best_bin == -1:
            # Open a new bin
            best_bin = len(bins)
            bins.append(create_bin())
            best_pos = (0, 0, 0)
        
        # Place the box
        x, y, z = best_pos
        bins[best_bin]["boxes"].append((x, y, z, w, h, d))
        bins[best_bin]["occupied"][x:x+w, y:y+h, z:z+d] = True
        placements[orig_idx] = (best_bin, x, y, z, w, h, d)
    
    # Handle any unplaced boxes with a simple shelf fallback
    if any(p is None for p in placements):
        remaining = [i for i in range(n) if placements[i] is None]
        for i in remaining:
            w, h, d = boxes[i]
            placed = False
            for bi, bin_data in enumerate(bins):
                pos = find_best_position(bin_data, w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bin_data["boxes"].append((x, y, z, w, h, d))
                    bin_data["occupied"][x:x+w, y:y+h, z:z+d] = True
                    placements[i] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
            if not placed:
                bi = len(bins)
                bins.append(create_bin())
                bins[bi]["boxes"].append((0, 0, 0, w, h, d))
                bins[bi]["occupied"][0:w, 0:h, 0:d] = True
                placements[i] = (bi, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
