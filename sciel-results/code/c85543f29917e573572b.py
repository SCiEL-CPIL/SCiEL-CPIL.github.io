"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Sort boxes by volume descending, then by max dimension descending
    indexed = [(i, w, h, d, w * h * d) for i, (w, h, d) in enumerate(boxes)]
    indexed.sort(key=lambda t: (-t[4], -max(t[1], t[2], t[3])))

    # We'll maintain a list of bins. Each bin has a list of placed boxes (x,y,z,w,h,d).
    # For efficiency, we'll use a candidate generation approach.

    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2 and
                z1 < z2 + d2 and z1 + d1 > z2)

    def fits_in_container(x, y, z, w, h, d):
        return x >= 0 and y >= 0 and z >= 0 and x + w <= W and y + h <= H and z + d <= D

    def generate_candidates(placed, w, h, d):
        """Generate candidate positions for a box of size (w,h,d) given placed boxes."""
        if not placed:
            return [(0, 0, 0)]

        candidates = set()
        # Origin
        candidates.add((0, 0, 0))

        # For each placed box, generate positions relative to it
        for (px, py, pz, pw, ph, pd) in placed:
            # x = px + pw (to the right), y and z can be 0 or py or pz
            if px + pw <= W:
                candidates.add((px + pw, 0, 0))
                candidates.add((px + pw, py, 0))
                candidates.add((px + pw, 0, pz))
                candidates.add((px + pw, py, pz))

            # x can be 0 or px, y = py + ph
            if py + ph <= H:
                candidates.add((0, py + ph, 0))
                candidates.add((px, py + ph, 0))
                candidates.add((0, py + ph, pz))
                candidates.add((px, py + ph, pz))

            # x and y can be 0 or px/py, z = pz + pd
            if pz + pd <= D:
                candidates.add((0, 0, pz + pd))
                candidates.add((px, 0, pz + pd))
                candidates.add((0, py, pz + pd))
                candidates.add((px, py, pz + pd))

        # Filter to valid positions within container
        valid = []
        for (cx, cy, cz) in candidates:
            if cx < 0 or cy < 0 or cz < 0:
                continue
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            valid.append((cx, cy, cz))

        return valid

    def is_valid_placement(placed, x, y, z, w, h, d):
        """Check if placing a box at (x,y,z) with size (w,h,d) doesn't overlap with any placed box."""
        for (px, py, pz, pw, ph, pd) in placed:
            if overlaps(x, y, z, w, h, d, px, py, pz, pw, ph, pd):
                return False
        return True

    def placement_cost(x, y, z, w, h, d):
        """Cost function: prefer lower y, then lower z, then lower x (bottom-left-far)."""
        # We want to minimize y first, then z, then x
        return (y, z, x)

    # Best-fit strategy: for each box, try all bins, find best placement
    bins = []  # list of lists of (orig_idx, x,y,z,w,h,d)
    bin_volumes = []  # volume of each bin

    for idx, (orig_idx, w, h, d, vol) in enumerate(indexed):
        if time.time() > deadline:
            break

        best_bin = None
        best_pos = None
        best_cost = None
        best_bin_vol = None

        for bin_idx in range(len(bins)):
            placed = bins[bin_idx]
            # Extract just the geometry for candidate generation
            placed_geom = [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in placed]
            candidates = generate_candidates(placed_geom, w, h, d)

            for (cx, cy, cz) in candidates:
                if not is_valid_placement(placed_geom, cx, cy, cz, w, h, d):
                    continue
                cost = placement_cost(cx, cy, cz, w, h, d)
                # Prefer bins that are already more filled (higher volume)
                # This is a secondary criterion to the placement cost
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_bin = bin_idx
                    best_pos = (cx, cy, cz)
                    best_bin_vol = bin_volumes[bin_idx]
                elif cost == best_cost and bin_volumes[bin_idx] > best_bin_vol:
                    best_bin = bin_idx
                    best_pos = (cx, cy, cz)
                    best_bin_vol = bin_volumes[bin_idx]

        if best_bin is None:
            # Try new bin
            if fits_in_container(0, 0, 0, w, h, d):
                best_bin = len(bins)
                bins.append([])
                bin_volumes.append(0)
                best_pos = (0, 0, 0)
            else:
                # Should not happen if boxes fit in container
                best_bin = len(bins)
                bins.append([])
                bin_volumes.append(0)
                best_pos = (0, 0, 0)

        bins[best_bin].append((orig_idx, best_pos[0], best_pos[1], best_pos[2], w, h, d))
        bin_volumes[best_bin] += vol

    # Handle remaining boxes if deadline hit
    placed_set = set()
    for bin_contents in bins:
        for item in bin_contents:
            placed_set.add(item[0])
    remaining = [item for item in indexed if item[0] not in placed_set]
    for (orig_idx, w, h, d, vol) in remaining:
        if time.time() > deadline:
            # Just place in a new bin at origin (may be invalid but avoid crash)
            bins.append([(orig_idx, 0, 0, 0, w, h, d)])
            bin_volumes.append(vol)
            continue
        # Quick placement: try each existing bin, then new bin
        placed_any = False
        for bin_idx in range(len(bins)):
            placed = bins[bin_idx]
            placed_geom = [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in placed]
            candidates = generate_candidates(placed_geom, w, h, d)
            for (cx, cy, cz) in candidates:
                if is_valid_placement(placed_geom, cx, cy, cz, w, h, d):
                    bins[bin_idx].append((orig_idx, cx, cy, cz, w, h, d))
                    bin_volumes[bin_idx] += vol
                    placed_any = True
                    break
            if placed_any:
                break
        if not placed_any:
            bins.append([(orig_idx, 0, 0, 0, w, h, d)])
            bin_volumes.append(vol)

    # Local optimization: try to move boxes to better positions
    if time.time() < deadline - 15:
        for _ in range(3):  # Multiple passes
            if time.time() > deadline - 10:
                break
            improved = False
            for bi, bin_contents in enumerate(bins):
                if not bin_contents:
                    continue
                # Try to improve each box's position
                for idx in range(len(bin_contents)):
                    if time.time() > deadline - 10:
                        break
                    orig_idx, bx, by, bz, bw, bh, bd = bin_contents[idx]
                    
                    # Remove box temporarily
                    bin_contents.pop(idx)
                    bin_volumes[bi] -= bw * bh * bd
                    placed_geom = [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in bin_contents]
                    new_pos = None
                    best_new_cost = None
                    
                    candidates = generate_candidates(placed_geom, bw, bh, bd)
                    for (cx, cy, cz) in candidates:
                        if is_valid_placement(placed_geom, cx, cy, cz, bw, bh, bd):
                            cost = placement_cost(cx, cy, cz, bw, bh, bd)
                            if best_new_cost is None or cost < best_new_cost:
                                best_new_cost = cost
                                new_pos = (cx, cy, cz)
                    
                    if new_pos is not None:
                        nx, ny, nz = new_pos
                        # Check if new position is better (lower y, z, x)
                        if (ny < by) or (ny == by and nz < bz) or (ny == by and nz == bz and nx < bx):
                            bin_contents.append((orig_idx, nx, ny, nz, bw, bh, bd))
                            bin_volumes[bi] += bw * bh * bd
                            improved = True
                        else:
                            bin_contents.append((orig_idx, bx, by, bz, bw, bh, bd))
                            bin_volumes[bi] += bw * bh * bd
                    else:
                        bin_contents.append((orig_idx, bx, by, bz, bw, bh, bd))
                        bin_volumes[bi] += bw * bh * bd
                    if not improved and time.time() > deadline - 10:
                        break
            if not improved:
                break

    # Build final result
    result = [None] * n
    for bin_idx, bin_contents in enumerate(bins):
        for (orig_idx, x, y, z, w, h, d) in bin_contents:
            result[orig_idx] = (bin_idx, x, y, z, w, h, d)

    # Ensure all are filled
    for i in range(n):
        if result[i] is None:
            result[i] = (0, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

    return result
# EVOLVE-BLOCK-END
