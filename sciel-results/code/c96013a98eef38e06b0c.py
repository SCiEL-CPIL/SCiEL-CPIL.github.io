"""3D Bin Packing with vectorized candidate evaluation, multi-ordering, and local search."""
import time
import numpy as np
import random

def pack(boxes, container):
    """
    Place boxes into bins using FFD with vectorized collision checking.
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 140.0
    container_volume = W * H * D

    # Sort boxes by volume descending for FFD, but keep original indices
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: (-x[1], x[2], x[3], x[4]))

    # Each bin: list of (x, y, z, w, h, d) and a 3D occupancy grid
    bins = []
    grids = []

    def make_grid():
        return np.zeros((H, W, D), dtype=np.uint8)

    def can_place_grid(grid, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        if grid[y:y+h, x:x+w, z:z+d].any():
            return False
        return True

    def place_grid(grid, x, y, z, w, h, d):
        grid[y:y+h, x:x+w, z:z+d] = 1

    def unplace_grid(grid, x, y, z, w, h, d):
        grid[y:y+h, x:x+w, z:z+d] = 0

    def generate_candidates(bin_placements, w, h, d):
        """Generate candidate (x, y, z) positions with more comprehensive coverage."""
        candidates = set()
        candidates.add((0, 0, 0))
        
        # Collect all unique x, y, z coordinates from existing placements
        x_coords = {0}
        y_coords = {0}
        z_coords = {0}
        
        for (px, py, pz, pw, ph, pd) in bin_placements:
            x_coords.add(px)
            x_coords.add(px + pw)
            y_coords.add(py)
            y_coords.add(py + ph)
            z_coords.add(pz)
            z_coords.add(pz + pd)
        
        # Generate all combinations of candidate coordinates
        for xc in x_coords:
            for yc in y_coords:
                for zc in z_coords:
                    if xc + w <= W and yc + h <= H and zc + d <= D:
                        candidates.add((xc, yc, zc))
        
        # Also add positions directly adjacent to each box in all 6 directions
        for (px, py, pz, pw, ph, pd) in bin_placements:
            for dx, dy, dz in [(pw, 0, 0), (-pw, 0, 0), (0, ph, 0), (0, -ph, 0), (0, 0, pd), (0, 0, -pd)]:
                nc_x, nc_y, nc_z = px + dx, py + dy, pz + dz
                if nc_x + w <= W and nc_y + h <= H and nc_z + d <= D and nc_x >= 0 and nc_y >= 0 and nc_z >= 0:
                    candidates.add((nc_x, nc_y, nc_z))
        
        return list(candidates)

    def find_best_position(bin_placements, grid, w, h, d):
        """Find the best position for a box in a bin using weighted scoring."""
        if not bin_placements:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None
        
        candidates = generate_candidates(bin_placements, w, h, d)
        if not candidates:
            return None
        
        # Compute current bounding box of the bin
        max_x = max(px + pw for (px, py, pz, pw, ph, pd) in bin_placements)
        max_y = max(py + ph for (px, py, pz, pw, ph, pd) in bin_placements)
        max_z = max(pz + pd for (px, py, pz, pw, ph, pd) in bin_placements)
        
        best_pos = None
        best_score = float('inf')
        
        for (xc, yc, zc) in candidates:
            if can_place_grid(grid, xc, yc, zc, w, h, d):
                # New bounding box if placed here
                new_max_x = max(xc + w, max_x)
                new_max_y = max(yc + h, max_y)
                new_max_z = max(zc + d, max_z)
                
                # Score: minimize growth in bounding box, with preference for lower/inner positions
                # Weight the growth more heavily than absolute position
                growth_x = max(0, xc + w - max_x)
                growth_y = max(0, yc + h - max_y)
                growth_z = max(0, zc + d - max_z)
                
                # Prefer positions that don't expand the bounding box, and among those, prefer lower/inner
                score = (growth_x * 10 + growth_y * 10 + growth_z * 10 +
                         xc * 0.1 + yc * 0.1 + zc * 0.1)
                
                if score < best_score:
                    best_score = score
                    best_pos = (xc, yc, zc)
        
        return best_pos

    # Main packing loop - FFD
    result = [None] * n
    
    for idx, vol, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break
        
        # Try to place in existing bins
        best_bin = -1
        best_pos = None
        best_fit_score = None
        
        for b_idx, bin_placements in enumerate(bins):
            grid = grids[b_idx]
            pos = find_best_position(bin_placements, grid, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Fit score: prefer positions that minimize the resulting bounding box
                if bin_placements:
                    new_max_x = max(x + w, max(px + pw for (px, py, pz, pw, ph, pd) in bin_placements))
                    new_max_y = max(y + h, max(py + ph for (px, py, pz, pw, ph, pd) in bin_placements))
                    new_max_z = max(z + d, max(pz + pd for (px, py, pz, pw, ph, pd) in bin_placements))
                else:
                    new_max_x, new_max_y, new_max_z = x + w, y + h, z + d
                fit_score = new_max_x + new_max_y + new_max_z
                
                if best_fit_score is None or fit_score < best_fit_score:
                    best_fit_score = fit_score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin >= 0:
            x, y, z = best_pos
            bins[best_bin].append((x, y, z, w, h, d))
            place_grid(grids[best_bin], x, y, z, w, h, d)
            result[idx] = (best_bin, x, y, z, w, h, d)
        else:
            # Create a new bin
            if w <= W and h <= H and d <= D:
                new_grid = make_grid()
                bins.append([(0, 0, 0, w, h, d)])
                grids.append(new_grid)
                place_grid(new_grid, 0, 0, 0, w, h, d)
                result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)
            else:
                # Box doesn't fit in container - place at origin anyway (shouldn't happen)
                new_grid = make_grid()
                bins.append([(0, 0, 0, w, h, d)])
                grids.append(new_grid)
                result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

    # Handle any unplaced boxes (shouldn't happen if container is large enough)
    for idx, vol, w, h, d in indexed_boxes:
        if result[idx] is None:
            for b_idx in range(len(bins)):
                pos = find_best_position(bins[b_idx], grids[b_idx], w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bins[b_idx].append((x, y, z, w, h, d))
                    place_grid(grids[b_idx], x, y, z, w, h, d)
                    result[idx] = (b_idx, x, y, z, w, h, d)
                    break
            else:
                new_grid = make_grid()
                bins.append([(0, 0, 0, w, h, d)])
                grids.append(new_grid)
                result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

    # Local search: try to reduce number of bins and improve last bin occupancy
    if time.time() < deadline - 10:
        changed = True
        iterations = 0
        while changed and time.time() < deadline - 10:
            changed = False
            iterations += 1
            if iterations > 50:  # Safety limit
                break
            for b_idx in range(len(bins) - 1, -1, -1):
                if time.time() > deadline - 10:
                    break
                if not bins[b_idx]:
                    continue
                
                boxes_to_try = list(bins[b_idx])
                for box in boxes_to_try:
                    x, y, z, w, h, d = box
                    moved = False
                    for target_bin in range(b_idx):
                        if time.time() > deadline - 10:
                            break
                        pos = find_best_position(bins[target_bin], grids[target_bin], w, h, d)
                        if pos is not None:
                            unplace_grid(grids[b_idx], x, y, z, w, h, d)
                            bins[b_idx].remove(box)
                            nx, ny, nz = pos
                            bins[target_bin].append((nx, ny, nz, w, h, d))
                            place_grid(grids[target_bin], nx, ny, nz, w, h, d)
                            for i in range(len(result)):
                                bin_id, bx, by, bz, bw, bh, bd = result[i]
                                if bin_id == b_idx and (bx, by, bz, bw, bh, bd) == (x, y, z, w, h, d):
                                    result[i] = (target_bin, nx, ny, nz, w, h, d)
                                    break
                            moved = True
                            changed = True
                            break
                    if not bins[b_idx]:
                        bins.pop(b_idx)
                        grids.pop(b_idx)
                        for i in range(len(result)):
                            bin_id = result[i][0]
                            if bin_id > b_idx:
                                result[i] = (bin_id - 1,) + result[i][1:]
                        break

    # Final local search: try to improve last bin's occupancy by moving boxes from other bins
    if time.time() < deadline - 3 and len(bins) > 1:
        last_bin_idx = len(bins) - 1
        # Try moving boxes from other bins to last bin to improve its fill fraction
        for b_idx in range(last_bin_idx):
            if time.time() > deadline - 3:
                break
            boxes_to_try = list(bins[b_idx])
            for box in boxes_to_try:
                x, y, z, w, h, d = box
                pos = find_best_position(bins[last_bin_idx], grids[last_bin_idx], w, h, d)
                if pos is not None:
                    unplace_grid(grids[b_idx], x, y, z, w, h, d)
                    bins[b_idx].remove(box)
                    nx, ny, nz = pos
                    bins[last_bin_idx].append((nx, ny, nz, w, h, d))
                    place_grid(grids[last_bin_idx], nx, ny, nz, w, h, d)
                    for i in range(len(result)):
                        bin_id, bx, by, bz, bw, bh, bd = result[i]
                        if bin_id == b_idx and (bx, by, bz, bw, bh, bd) == (x, y, z, w, h, d):
                            result[i] = (last_bin_idx, nx, ny, nz, w, h, d)
                            break

    # Second pass: try to eliminate more bins after last-bin optimization
    if time.time() < deadline - 2:
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for b_idx in range(len(bins) - 1, -1, -1):
                if time.time() > deadline - 2:
                    break
                if not bins[b_idx]:
                    continue
                boxes_to_try = list(bins[b_idx])
                for box in boxes_to_try:
                    x, y, z, w, h, d = box
                    moved = False
                    for target_bin in range(b_idx):
                        if time.time() > deadline - 2:
                            break
                        pos = find_best_position(bins[target_bin], grids[target_bin], w, h, d)
                        if pos is not None:
                            unplace_grid(grids[b_idx], x, y, z, w, h, d)
                            bins[b_idx].remove(box)
                            nx, ny, nz = pos
                            bins[target_bin].append((nx, ny, nz, w, h, d))
                            place_grid(grids[target_bin], nx, ny, nz, w, h, d)
                            for i in range(len(result)):
                                bin_id, bx, by, bz, bw, bh, bd = result[i]
                                if bin_id == b_idx and (bx, by, bz, bw, bh, bd) == (x, y, z, w, h, d):
                                    result[i] = (target_bin, nx, ny, nz, w, h, d)
                                    break
                            moved = True
                            changed = True
                            break
                    if not bins[b_idx]:
                        bins.pop(b_idx)
                        grids.pop(b_idx)
                        for i in range(len(result)):
                            bin_id = result[i][0]
                            if bin_id > b_idx:
                                result[i] = (bin_id - 1,) + result[i][1:]
                        break

    return result
