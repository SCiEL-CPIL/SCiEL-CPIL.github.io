"""3D Bin Packing: FFD with corner-point candidate generation."""
import time

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Sort boxes by volume descending (First-Fit Decreasing)
    indexed_boxes = [(w * h * d, i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[0])

    # Bins: each bin is a list of (x, y, z, w, h, d, orig_idx)
    bins = []

    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return (x1 < x2 + w2 and x2 < x1 + w1 and
                y1 < y2 + h2 and y2 < y1 + h1 and
                z1 < z2 + d2 and z2 < z1 + d1)

    def find_best_position(bin_boxes, w, h, d):
        """Find the best (lowest z, then y, then x) valid position for a box in a bin."""
        if not bin_boxes:
            return (0, 0, 0)

        # Generate candidate positions from corners of existing boxes
        candidates = set()
        candidates.add((0, 0, 0))

        for (px, py, pz, pw, ph, pd, _) in bin_boxes:
            # Right of box
            if px + pw <= W:
                candidates.add((px + pw, py, pz))
            # Above box
            if py + ph <= H:
                candidates.add((px, py + ph, pz))
            # In front of box
            if pz + pd <= D:
                candidates.add((px, py, pz + pd))
            # Right + above
            if px + pw <= W and py + ph <= H:
                candidates.add((px + pw, py + ph, pz))
            # Right + in front
            if px + pw <= W and pz + pd <= D:
                candidates.add((px + pw, py, pz + pd))
            # Above + in front
            if py + ph <= H and pz + pd <= D:
                candidates.add((px, py + ph, pz + pd))
            # Right + above + in front
            if px + pw <= W and py + ph <= H and pz + pd <= D:
                candidates.add((px + pw, py + ph, pz + pd))

        # Also try some additional positions: for each box, try placing at
        # (px, py, pz) offset combinations that might fill gaps
        # Try positions aligned with each box's edges
        for (px, py, pz, pw, ph, pd, _) in bin_boxes:
            # Try placing to the right at various y,z
            if px + pw <= W:
                for cy in [0, py, py + ph]:
                    for cz in [0, pz, pz + pd]:
                        if cy + h <= H and cz + d <= D:
                            candidates.add((px + pw, cy, cz))
            # Try placing above at various x,z
            if py + ph <= H:
                for cx in [0, px, px + pw]:
                    for cz in [0, pz, pz + pd]:
                        if cx + w <= W and cz + d <= D:
                            candidates.add((cx, py + ph, cz))
            # Try placing in front at various x,y
            if pz + pd <= D:
                for cx in [0, px, px + pw]:
                    for cy in [0, py, py + ph]:
                        if cx + w <= W and cy + h <= H:
                            candidates.add((cx, cy, pz + pd))

        # Find best valid position
        best_pos = None
        best_key = None

        for (cx, cy, cz) in candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            # Check for overlap
            valid = True
            for (px, py, pz, pw, ph, pd, _) in bin_boxes:
                if overlaps(cx, cy, cz, w, h, d, px, py, pz, pw, ph, pd):
                    valid = False
                    break
            if valid:
                key = (cz, cy, cx)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pos = (cx, cy, cz)

        return best_pos

    # Place each box
    placement_map = [None] * n  # (bin_idx, x, y, z) for each original box

    for vol, orig_idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # Emergency: place in new bin at origin
            bin_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d, orig_idx)])
            placement_map[orig_idx] = (bin_idx, 0, 0, 0)
            continue

        placed = False
        # Try existing bins first
        for b_idx in range(len(bins)):
            if time.time() > deadline:
                break
            pos = find_best_position(bins[b_idx], w, h, d)
            if pos is not None:
                bins[b_idx].append((pos[0], pos[1], pos[2], w, h, d, orig_idx))
                placement_map[orig_idx] = (b_idx, pos[0], pos[1], pos[2])
                placed = True
                break

        if not placed:
            # Open a new bin
            bin_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d, orig_idx)])
            placement_map[orig_idx] = (bin_idx, 0, 0, 0)

    # Build result in original order
    placements = []
    for i in range(n):
        w, h, d = boxes[i]
        b_idx, x, y, z = placement_map[i]
        placements.append((b_idx, x, y, z, w, h, d))

    return placements
# EVOLVE-BLOCK-END
