"""3D Bin Packing: BFD with fragmentation-aware bin selection, tightness scoring, and consolidation."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 112.0  # 8s safety margin for 120s limit

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    container_vol = W * H * D

    # Sort boxes by volume descending, then by max dimension for tie-breaking
    indexed_boxes = [(i, w, h, d, w * h * d, max(w, h, d)) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: (-x[4], x[5], x[1], x[2], x[3]))

    # Bins: each bin is a list of placed boxes (x, y, z, w, h, d)
    bins = []

    def can_place(bin_boxes, x, y, z, w, h, d):
        """Check if box fits within container and doesn't overlap with existing boxes."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True

    def position_score(bin_boxes, x, y, z, w, h, d):
        """Score a position: lower is better. Rewards compactness and contact."""
        if not bin_boxes:
            return x + y + z

        # Base: prefer lower coordinates (compact toward origin)
        score = 0.3 * x + 0.3 * y + 0.4 * z

        # Contact/tightness bonus: reward touching existing boxes
        contact = 0
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Check face contact in each dimension
            x_touch = (abs(x - (px + pw)) < 1e-9 or abs((x + w) - px) < 1e-9)
            y_touch = (abs(y - (py + ph)) < 1e-9 or abs((y + h) - py) < 1e-9)
            z_touch = (abs(z - (pz + pd)) < 1e-9 or abs((z + d) - pz) < 1e-9)

            # Only count contact if the boxes actually overlap in the other two dims
            if x_touch:
                if not (y + h <= py or py + ph <= y) and not (z + d <= pz or pz + pd <= z):
                    contact += 1
            if y_touch:
                if not (x + w <= px or px + pw <= x) and not (z + d <= pz or pz + pd <= z):
                    contact += 1
            if z_touch:
                if not (x + w <= px or px + pw <= x) and not (y + h <= py or py + ph <= y):
                    contact += 1

        # More contacts = tighter = lower score
        score -= 0.15 * contact

        return score

    def find_position(bin_boxes, w, h, d):
        """Find the best valid position for a box in a bin. Returns (x,y,z) or None."""
        if not bin_boxes:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None

        # Generate candidate coordinates from box faces
        xs = {0}
        ys = {0}
        zs = {0}

        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if px + pw <= W:
                xs.add(px + pw)
            if py + ph <= H:
                ys.add(py + ph)
            if pz + pd <= D:
                zs.add(pz + pd)

        xs = sorted(xs)
        ys = sorted(ys)
        zs = sorted(zs)

        # Cap candidates for performance
        if len(xs) > 25:
            xs = xs[:25]
        if len(ys) > 25:
            ys = ys[:25]
        if len(zs) > 25:
            zs = zs[:25]

        best_score = float('inf')
        best_pos = None

        for x in xs:
            if x + w > W:
                continue
            for y in ys:
                if y + h > H:
                    continue
                for z in zs:
                    if z + d > D:
                        continue
                    if can_place(bin_boxes, x, y, z, w, h, d):
                        s = position_score(bin_boxes, x, y, z, w, h, d)
                        if s < best_score:
                            best_score = s
                            best_pos = (x, y, z)

        return best_pos

    def bin_utilization(bin_boxes):
        """Volume utilization of a bin (0 to 1)."""
        if not bin_boxes:
            return 0.0
        total_vol = sum(pw * ph * pd for (_, _, _, pw, ph, pd) in bin_boxes)
        return total_vol / container_vol

    def bin_compactness(bin_boxes):
        """
        Estimate how compact the bin's contents are.
        Lower = more compact. Uses bounding box volume / actual volume ratio.
        """
        if len(bin_boxes) <= 1:
            return 0.0

        min_x = min(b[0] for b in bin_boxes)
        min_y = min(b[1] for b in bin_boxes)
        min_z = min(b[2] for b in bin_boxes)
        max_x = max(b[0] + b[3] for b in bin_boxes)
        max_y = max(b[1] + b[4] for b in bin_boxes)
        max_z = max(b[2] + b[5] for b in bin_boxes)

        bbox_vol = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
        actual_vol = sum(b[3] * b[4] * b[5] for b in bin_boxes)

        if actual_vol == 0:
            return 0.0
        # 0 = perfectly compact, higher = more spread out
        return bbox_vol / actual_vol - 1.0

    def place_box(w, h, d):
        """Place a box in the best existing bin or create a new one."""
        best_bin = -1
        best_score = float('inf')
        best_pos = None

        for b_idx, bin_boxes in enumerate(bins):
            pos = find_position(bin_boxes, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Combined score: prefer tighter positions, more compact bins, higher utilization
                pos_s = position_score(bin_boxes, x, y, z, w, h, d)
                util = bin_utilization(bin_boxes)
                compact = bin_compactness(bin_boxes)

                # We want: low pos_s, low compact, high util
                score = pos_s + 0.8 * compact - 0.5 * util

                if score < best_score:
                    best_score = score
                    best_bin = b_idx
                    best_pos = pos

        if best_bin == -1:
            bins.append([])
            best_bin = len(bins) - 1
            best_pos = (0, 0, 0)

        x, y, z = best_pos
        bins[best_bin].append((x, y, z, w, h, d))
        return best_bin, x, y, z

    # Place boxes in sorted order
    placements = [None] * n
    for idx, w, h, d, vol, maxdim in indexed_boxes:
        if time.time() > deadline:
            # Emergency: place remaining boxes in new bins
            for i in range(idx, n):
                _, w2, h2, d2, _, _ = indexed_boxes[i]
                bin_idx = len(bins)
                bins.append([])
                bins[bin_idx].append((0, 0, 0, w2, h2, d2))
                placements[i] = (bin_idx, 0, 0, 0, w2, h2, d2)
            break

        bin_idx, x, y, z = place_box(w, h, d)
        placements[idx] = (bin_idx, x, y, z, w, h, d)

    # Consolidation: try to reduce bin count by moving boxes from smaller bins to larger ones
    if time.time() < deadline - 8.0 and len(bins) > 1:
        consolidation_deadline = deadline - 2.0
        max_bins = len(bins)

        # Sort bins by number of boxes ascending (try to empty smallest bins first)
        bin_order = sorted(range(max_bins), key=lambda i: len(bins[i]))

        for source_bin in bin_order:
            if time.time() > consolidation_deadline:
                break
            if len(bins[source_bin]) == 0:
                continue

            # Try to move all boxes from source_bin to other bins
            all_moved = True
            boxes_in_source = list(bins[source_bin])

            for box_info in boxes_in_source:
                bx, by, bz, bw, bh, bd = box_info
                moved = False

                for target_bin in range(max_bins):
                    if time.time() > consolidation_deadline:
                        all_moved = False
                        break
                    if target_bin == source_bin or len(bins[target_bin]) == 0:
                        continue

                    pos = find_position(bins[target_bin], bw, bh, bd)
                    if pos is not None:
                        px, py, pz = pos
                        bins[target_bin].append((px, py, pz, bw, bh, bd))
                        moved = True
                        break

                if not moved:
                    all_moved = False
                    break

            if all_moved:
                bins[source_bin] = []

        # Remove empty bins and renumber
        new_bins = [b for b in bins if len(b) > 0]
        if len(new_bins) < max_bins:
            bin_remap = {}
            for old_idx, b in enumerate(bins):
                if len(b) > 0:
                    bin_remap[old_idx] = new_bins.index(b)

            for i in range(n):
                if placements[i] is not None:
                    old_bin = placements[i][0]
                    if old_bin in bin_remap:
                        placements[i] = (bin_remap[old_bin],) + placements[i][1:]

            bins = new_bins

    return placements
# EVOLVE-BLOCK-END
