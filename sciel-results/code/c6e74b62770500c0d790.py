"""3D Bin Packing: FDD + best-fit with skyline pruning and multi-pass refinement."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    deadline = time.time() + 110.0  # safety margin

    # Sort boxes by volume descending (FDD heuristic)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)

    # Bins will store list of (x, y, z, w, h, d)
    bins = []
    placement_map = {}  # original index -> (bin_idx, x, y, z)

    # Voxel grid for fast collision checks. Resolution 1 unit.
    # Grid shape: (W, H, D) boolean array per bin.
    bin_grids = []

    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Fallback: place remaining boxes greedily in new bins
            for remaining_orig_idx, (rw, rh, rd) in indexed_boxes:
                if remaining_orig_idx not in placement_map:
                    b_idx = len(bins)
                    bins.append([(0, 0, 0, rw, rh, rd)])
                    placement_map[remaining_orig_idx] = (b_idx, 0, 0, 0)
            break

        placed = False
        # Best-fit: try all existing bins and pick the one with the best position
        best_b_idx = -1
        best_pos = None
        best_score = None
        for b_idx, bin_contents in enumerate(bins):
            pos = _find_position(bins[b_idx], bin_grids[b_idx], W, H, D, w, h, d)
            if pos is not None:
                x, y, z = pos
                score = (z, y, x)
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = pos
                    best_b_idx = b_idx

        if best_b_idx >= 0:
            x, y, z = best_pos
            placement_map[orig_idx] = (best_b_idx, x, y, z)
            bins[best_b_idx].append((x, y, z, w, h, d))
            # Update grid
            grid = bin_grids[best_b_idx]
            grid[x:x+w, y:y+h, z:z+d] = True
            placed = True

        if not placed:
            # Create new bin
            b_idx = len(bins)
            new_grid = np.zeros((W, H, D), dtype=bool)
            new_grid[0:w, 0:h, 0:d] = True
            bins.append([(0, 0, 0, w, h, d)])
            bin_grids.append(new_grid)
            placement_map[orig_idx] = (b_idx, 0, 0, 0)

    # Refinement pass: try to improve by moving boxes to earlier bins or better positions
    if time.time() < deadline - 5.0:
        _refine(bins, bin_grids, placement_map, boxes, W, H, D, deadline)

    # Build result in original order
    result = []
    for i in range(n):
        b_idx, x, y, z = placement_map[i]
        w, h, d = boxes[i]
        result.append((b_idx, x, y, z, w, h, d))

    return result


def _find_position(bin_contents, grid, W, H, D, w, h, d):
    """Find the best position for a box in a bin using a composite score."""
    if not bin_contents:
        if w <= W and h <= H and d <= D:
            return (0, 0, 0)
        return None

    # Candidate x positions: 0 and right edges of existing boxes
    x_candidates = {0}
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        x_candidates.add(bx + bw)
    x_candidates = sorted(x for x in x_candidates if x + w <= W)

    # Candidate y positions: 0 and top edges of existing boxes
    y_candidates = {0}
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        y_candidates.add(by + bh)
    y_candidates = sorted(y for y in y_candidates if y + h <= H)

    # Candidate z positions: 0 and front edges of existing boxes
    z_candidates = {0}
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        z_candidates.add(bz + bd)
    z_candidates = sorted(z for z in z_candidates if z + d <= D)

    for z in z_candidates:
        for y in y_candidates:
            for x in x_candidates:
                # Check if the box fits using the grid
                # grid[x:x+w, y:y+h, z:z+d] should be all False
                if not grid[x:x+w, y:y+h, z:z+d].any():
                    return (x, y, z)

    return None


def _refine(bins, bin_grids, placement_map, boxes, W, H, D, deadline):
    """Try to move boxes to earlier bins or better positions to reduce bin count."""
    # Build reverse map: (bin_idx, x, y, z) -> orig_idx for O(1) lookup
    rev_map = {}
    for idx, (b, x, y, z) in placement_map.items():
        rev_map[(b, x, y, z)] = idx

    max_b = len(bins) - 1
    for b_idx in range(max_b, -1, -1):
        if time.time() > deadline - 3.0:
            break
        if not bins[b_idx]:
            continue

        # Try to move each box in this bin to an earlier bin
        to_move = list(bins[b_idx])
        for box_data in to_move:
            if time.time() > deadline - 3.0:
                break
            bx, by, bz, bw, bh, bd = box_data
            # Find original index via reverse map
            orig_idx = rev_map.get((b_idx, bx, by, bz))
            if orig_idx is None:
                continue

            # Try to place in earlier bins
            moved = False
            for new_b in range(b_idx):
                pos = _find_position(bins[new_b], bin_grids[new_b], W, H, D, bw, bh, bd)
                if pos is not None:
                    x, y, z = pos
                    # Remove from current bin and grid
                    bins[b_idx].remove(box_data)
                    bin_grids[b_idx][bx:bx+bw, by:by+bh, bz:bz+bd] = False
                    # Add to new bin and grid
                    bins[new_b].append((x, y, z, bw, bh, bd))
                    bin_grids[new_b][x:x+bw, y:y+bh, z:z+bd] = True
                    placement_map[orig_idx] = (new_b, x, y, z)
                    rev_map.pop((b_idx, bx, by, bz), None)
                    rev_map[(new_b, x, y, z)] = orig_idx
                    moved = True
                    break
            if moved:
                continue

            # Try to find a better position in the same bin
            bins[b_idx].remove(box_data)
            bin_grids[b_idx][bx:bx+bw, by:by+bh, bz:bz+bd] = False
            pos = _find_position(bins[b_idx], bin_grids[b_idx], W, H, D, bw, bh, bd)
            if pos is not None:
                x, y, z = pos
                bins[b_idx].append((x, y, z, bw, bh, bd))
                bin_grids[b_idx][x:x+bw, y:y+bh, z:z+bd] = True
                placement_map[orig_idx] = (b_idx, x, y, z)
                rev_map.pop((b_idx, bx, by, bz), None)
                rev_map[(b_idx, x, y, z)] = orig_idx
            else:
                bins[b_idx].append(box_data)
                bin_grids[b_idx][bx:bx+bw, by:by+bh, bz:bz+bd] = True
                placement_map[orig_idx] = (b_idx, bx, by, bz)
                rev_map[(b_idx, bx, by, bz)] = orig_idx

    # Remove empty bins and compact labels
    new_bins = []
    new_grids = []
    label_map = {}
    for old_label in range(len(bins)):
        if bins[old_label]:
            label_map[old_label] = len(new_bins)
            new_bins.append(bins[old_label])
            new_grids.append(bin_grids[old_label])
    bins[:] = new_bins
    bin_grids[:] = new_grids
    for idx in placement_map:
        b = placement_map[idx][0]
        if b in label_map:
            placement_map[idx] = (label_map[b],) + placement_map[idx][1:]
# EVOLVE-BLOCK-END
