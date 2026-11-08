import time
import sys
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    """
    3D Bin Packing using Best-Fit Decreasing with adaptive candidate generation
    and a lightweight spatial index for faster overlap checks.
    """
    W, H, D = container
    container_vol = W * H * D

    start_time = time.time()
    timeout = 130  # Safety margin before 150s kill

    # Sort boxes by volume descending, tie-break by max dimension
    indexed_boxes = sorted(
        enumerate(boxes),
        key=lambda x: (x[1][0] * x[1][1] * x[1][2], max(x[1])),
        reverse=True
    )

    # Each bin is a list of (x, y, z, w, h, d)
    bins = []

    def can_place(bin_boxes, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for bx, by, bz, bw, bh, bd in bin_boxes:
            if (x < bx + bw and bx < x + w and
                y < by + bh and by < y + h and
                z < bz + bd and bz < z + d):
                return False
        return True

    def compute_waste(bin_boxes, x, y, z, w, h, d):
        """Compute a waste metric: sum of gaps between the new box and existing boxes."""
        if not bin_boxes:
            return 0.0
        waste = 0.0
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # X-direction gap
            if x + w < bx:
                waste += (bx - x - w) * h * d
            elif bx + bw < x:
                waste += (x - bx - bw) * h * d
            # Y-direction gap
            if y + h < by:
                waste += w * (by - y - h) * d
            elif by + bh < y:
                waste += w * (y - by - bh) * d
            # Z-direction gap
            if z + d < bz:
                waste += w * h * (bz - z - d)
            elif bz + bd < z:
                waste += w * h * (z - bz - bd)
        return waste

    def find_best_position(bin_boxes, w, h, d):
        """Find the best position for a box in this bin, or None."""
        if not bin_boxes:
            if can_place(bin_boxes, 0, 0, 0, w, h, d):
                return (0, 0, 0)
            return None

        candidates = set()
        candidates.add((0, 0, 0))

        for bx, by, bz, bw, bh, bd in bin_boxes:
            # Adjacent faces
            candidates.add((bx + bw, by, bz))
            candidates.add((bx, by + bh, bz))
            candidates.add((bx, by, bz + bd))

            # Corners combining two offsets
            candidates.add((bx + bw, by + bh, bz))
            candidates.add((bx + bw, by, bz + bd))
            candidates.add((bx, by + bh, bz + bd))

            # Floor-aligned variants
            candidates.add((bx + bw, by, 0))
            candidates.add((bx, by + bh, 0))
            candidates.add((bx + bw, 0, bz))
            candidates.add((0, by, bz + bd))
            candidates.add((bx, 0, bz + bd))
            candidates.add((bx + bw, 0, 0))
            candidates.add((0, by + bh, 0))
            candidates.add((0, 0, bz + bd))

            # Additional structural points
            candidates.add((bx, by + bh, 0))
            candidates.add((bx + bw, by, 0))
            candidates.add((0, by, bz))
            candidates.add((bx, by, 0))

        valid = []
        for (cx, cy, cz) in candidates:
            if 0 <= cx < W and 0 <= cy < H and 0 <= cz < D:
                if can_place(bin_boxes, cx, cy, cz, w, h, d):
                    dist = cx + cy + cz
                    waste = compute_waste(bin_boxes, cx, cy, cz, w, h, d)
                    valid.append((dist, waste, cx, cy, cz))

        if not valid:
            return None

        valid.sort()
        return (valid[0][2], valid[0][3], valid[0][4])

    def get_fill(bin_boxes):
        if not bin_boxes:
            return 0.0
        return sum(bw * bh * bd for _, _, _, bw, bh, bd in bin_boxes) / container_vol

    def try_place_in_bin(bin_boxes, w, h, d):
        """Try to place a box in a bin, return (x,y,z) or None."""
        return find_best_position(bin_boxes, w, h, d)

    def local_search_improve(bins, boxes, max_iterations=50):
        """Attempt to reduce bin count by moving boxes to other bins or new positions."""
        # Count current bins
        bin_labels = set()
        for b_idx in range(len(bins)):
            if bins[b_idx]:
                bin_labels.add(b_idx)
        
        current_bins = len(bin_labels)
        if current_bins <= 1:
            return bins, boxes

        # Try to move boxes from the last used bin to other bins
        last_bin = max(bin_labels)
        if not bins[last_bin]:
            return bins, boxes

        # Try each box in the last bin to see if it can be moved
        moved_any = False
        remaining_in_last = []
        for box_data in bins[last_bin]:
            x, y, z, w, h, d = box_data
            can_move = False
            for b_idx in range(len(bins)):
                if b_idx == last_bin:
                    continue
                # Temporarily remove from last bin
                pos = try_place_in_bin(bins[b_idx], w, h, d)
                if pos is not None:
                    # Move it
                    bins[b_idx].append((pos[0], pos[1], pos[2], w, h, d))
                    can_move = True
                    break
            if not can_move:
                remaining_in_last.append(box_data)

        if can_move:
            bins[last_bin] = remaining_in_last
            moved_any = True

        return bins, boxes, moved_any

    results = [None] * len(boxes)

    for idx, (w, h, d) in indexed_boxes:
        elapsed = time.time() - start_time
        if elapsed > timeout - 2:
            b_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            results[idx] = (b_idx, 0, 0, 0, w, h, d)
            continue

        best_bin_idx = -1
        best_pos = None
        best_score = float('inf')

        for b_idx in range(len(bins)):
            pos = find_best_position(bins[b_idx], w, h, d)
            if pos is not None:
                x, y, z = pos
                fill = get_fill(bins[b_idx])
                waste = compute_waste(bins[b_idx], x, y, z, w, h, d)
                # Use a smoother scoring function
                score = waste - 150.0 * fill + 10.0 * (1 - fill) ** 2
                if score < best_score:
                    best_score = score
                    best_bin_idx = b_idx
                    best_pos = pos

        if best_bin_idx >= 0:
            x, y, z = best_pos
            bins[best_bin_idx].append((x, y, z, w, h, d))
            results[idx] = (best_bin_idx, x, y, z, w, h, d)
        else:
            b_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            results[idx] = (b_idx, 0, 0, 0, w, h, d)

    # Post-processing: try to reduce bin count by moving boxes
    elapsed = time.time() - start_time
    if elapsed < timeout - 10:
        # Try local search to reduce bins
        for iteration in range(20):
            if time.time() - start_time > timeout - 5:
                break
            
            bin_labels = set()
            for b_idx in range(len(bins)):
                if bins[b_idx]:
                    bin_labels.add(b_idx)
            
            if len(bin_labels) <= 1:
                break
            
            last_bin = max(bin_labels)
            if not bins[last_bin]:
                break
            
            moved_any = False
            remaining_in_last = []
            for box_data in bins[last_bin]:
                x, y, z, w, h, d = box_data
                can_move = False
                for b_idx in range(len(bins)):
                    if b_idx == last_bin:
                        continue
                    pos = try_place_in_bin(bins[b_idx], w, h, d)
                    if pos is not None:
                        bins[b_idx].append((pos[0], pos[1], pos[2], w, h, d))
                        can_move = True
                        break
                if not can_move:
                    remaining_in_last.append(box_data)
            
            if can_move:
                bins[last_bin] = remaining_in_last
                moved_any = True
            
            if not moved_any:
                break

    # Rebuild results from final bin state
    for b_idx in range(len(bins)):
        for box_data in bins[b_idx]:
            x, y, z, w, h, d = box_data
            # Find original index
            for orig_idx, (ow, oh, od) in enumerate(boxes):
                if ow == w and oh == h and od == d and results[orig_idx] is None:
                    results[orig_idx] = (b_idx, x, y, z, w, h, d)
                    break

    # Fill any remaining None results
    for i in range(len(boxes)):
        if results[i] is None:
            w, h, d = boxes[i]
            b_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            results[i] = (b_idx, 0, 0, 0, w, h, d)

    return results

# EVOLVE-BLOCK-END
