"""3D Bin Packing: Best-fit placement with grid-based occupancy and deadline guard."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Deadline: leave a safety margin before the 120s evaluator kill.
    deadline = time.time() + 110.0

    # Sort boxes by volume descending for better packing, but we must return
    # placements in the original order. We'll track original indices.
    indexed = list(enumerate(boxes))
    indexed.sort(key=lambda t: -(t[1][0] * t[1][1] * t[1][2]))

    # Each bin is represented by a numpy boolean occupancy array of shape (W, H, D)
    # For memory efficiency, if W*H*D is too large, we fall back to a list-of-intervals
    # approach, but for typical 40-box instances with reasonable container sizes,
    # the grid should be fine. Let's check.
    grid_size = W * H * D
    use_grid = grid_size <= 5_000_000  # 5 million booleans = 5 MB per bin, reasonable

    # We'll maintain a list of bins. Each bin has:
    #   - occupied: np.ndarray of shape (W, H, D) if use_grid, else None
    #   - placements: list of (x, y, z, w, h, d) for non-grid fallback
    #   - volume_used: int

    bins = []

    def new_bin():
        if use_grid:
            occ = np.zeros((W, H, D), dtype=bool)
        else:
            occ = None
        bins.append({'occ': occ, 'placements': [], 'vol': 0})
        return bins[-1]

    def can_place_grid(bin_dict, x, y, z, w, h, d):
        """Check if box fits at (x,y,z) in bin using grid."""
        if x + w > W or y + h > H or z + d > D:
            return False
        sub = bin_dict['occ'][x:x+w, y:y+h, z:z+d]
        return not np.any(sub)

    def place_grid(bin_dict, x, y, z, w, h, d):
        bin_dict['occ'][x:x+w, y:y+h, z:z+d] = True
        bin_dict['placements'].append((x, y, z, w, h, d))
        bin_dict['vol'] += w * h * d

    def can_place_list(bin_dict, x, y, z, w, h, d):
        """Check overlap with existing placements (for non-grid mode)."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_dict['placements']:
            # Overlap check in 3D
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True

    def place_list(bin_dict, x, y, z, w, h, d):
        bin_dict['placements'].append((x, y, z, w, h, d))
        bin_dict['vol'] += w * h * d

    # Best-fit: for each box, try to find the position in existing bins that
    # minimizes the "waste" – we use a simple heuristic: try to minimize
    # max(x+w, y+h, z+d) or sum of distances to origin. We'll scan candidate
    # positions. To keep it efficient, we limit the search.

    # For grid mode, we can be smarter: scan all valid positions.
    # But W*H*D can be large. Let's limit: if grid_size is large, use a
    # restricted scan (e.g., only positions where x,y,z are multiples of some
    # step, or just try (0,0,0) and a few other corners).

    # Actually, for 40 boxes, even a full scan of all positions per box per bin
    # might be too slow if W*H*D is large. Let's use a hybrid:
    # - For each bin, try a set of candidate positions: (0,0,0) plus positions
    #   aligned to existing box edges.
    # - If that fails and we have time, do a more thorough search.

    # Simpler approach: use the "bottom-left-farthest" heuristic.
    # For each bin, find the position that minimizes (x, y, z) in a specific order.
    # We'll try all positions but with an early exit.

    # Given the constraints, let's use a practical approach:
    # For each box, for each existing bin (in order of least remaining space first),
    # try to find a valid position. We'll scan x, y, z in a smart order.

    # To keep it fast, we'll limit the number of positions checked per bin.
    # If W*H*D is small enough, we can do a full scan.

    full_scan_limit = 200_000  # max positions to scan per bin per box

    # For each box in sorted order, place it.
    result_placements = [None] * n  # indexed by original index

    for orig_idx, (w, h, d) in indexed:
        if time.time() > deadline:
            # Fallback: place remaining boxes in a new bin with shelf packing
            for remaining_idx, (rw, rh, rd) in indexed:
                if result_placements[remaining_idx] is None:
                    # Use a simple shelf for the rest
                    pass
            # Actually, let's just place all remaining in a simple shelf in a new bin
            break

        placed = False
        # Try existing bins first
        for bin_i, bin_dict in enumerate(bins):
            if use_grid:
                # Scan for best position
                best_pos = None
                best_score = None
                # We want to minimize the "height" used, so prefer low z, then low y, then low x
                # But we also want to fill gaps. Let's use a simple score:
                # score = z * W * H + y * W + x (prefer lower z, then y, then x)
                # Actually, let's just find any valid position, preferring (0,0,0) direction.
                
                # For efficiency, scan z from 0, y from 0, x from 0, and take the first valid.
                # This is "first-fit" in a specific order.
                found = False
                for z in range(D - d + 1):
                    if found:
                        break
                    for y in range(H - h + 1):
                        if found:
                            break
                        for x in range(W - w + 1):
                            if can_place_grid(bin_dict, x, y, z, w, h, d):
                                place_grid(bin_dict, x, y, z, w, h, d)
                                result_placements[orig_idx] = (bin_i, x, y, z, w, h, d)
                                placed = True
                                found = True
                                break
            else:
                # List-based: try (0,0,0) first, then a few other positions
                for z in range(min(D - d + 1, 50)):
                    for y in range(min(H - h + 1, 50)):
                        for x in range(min(W - w + 1, 50)):
                            if can_place_list(bin_dict, x, y, z, w, h, d):
                                place_list(bin_dict, x, y, z, w, h, d)
                                result_placements[orig_idx] = (bin_i, x, y, z, w, h, d)
                                placed = True
                                break
                        if placed:
                            break
                    if placed:
                        break

            if placed:
                break

        if not placed:
            # New bin
            bin_dict = new_bin()
            bin_i = len(bins) - 1
            if use_grid:
                place_grid(bin_dict, 0, 0, 0, w, h, d)
            else:
                place_list(bin_dict, 0, 0, 0, w, h, d)
            result_placements[orig_idx] = (bin_i, 0, 0, 0, w, h, d)

    # If we hit the deadline, place any remaining boxes using a simple shelf
    if any(r is None for r in result_placements):
        # Collect remaining
        remaining = [(i, boxes[i]) for i in range(n) if result_placements[i] is None]
        # Use a new bin with shelf packing
        b = len(bins)
        x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
        for orig_idx, (w, h, d) in remaining:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b += 1
                x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
            result_placements[orig_idx] = (b, x, y, z, w, h, d)
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)

    return result_placements
# EVOLVE-BLOCK-END
