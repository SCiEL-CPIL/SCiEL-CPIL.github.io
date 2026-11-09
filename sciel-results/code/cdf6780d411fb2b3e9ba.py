"""Rewritten 3D bin packing solver: Best-Fit Decreasing with candidate positions and local improvement."""
import time
import math
import random

def pack(boxes, container):
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s kill

    W, H, D = container
    container_vol = W * H * D

    # Sort boxes by volume descending (Best-Fit Decreasing)
    indexed = sorted(enumerate(boxes), key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)

    # Each bin is a list of (x, y, z, w, h, d)
    bins = []

    def can_place(bin_boxes, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # Check overlap: boxes overlap if they intersect in all three axes
            if x < bx + bw and x + w > bx and y < by + bh and y + h > by and z < bz + bd and z + d > bz:
                return False
        return True

    def candidate_positions(bin_boxes, w, h, d):
        # Generate candidate positions: origin and adjacent to existing boxes
        cands = [(0, 0, 0)]
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # Adjacent in +x, +y, +z directions
            cands.append((bx + bw, by, bz))
            cands.append((bx, by + bh, bz))
            cands.append((bx, by, bz + bd))
            # Also try combinations of offsets from different boxes for better gap filling
            # But keep it simple for speed; we can expand later if time allows
        # Filter valid candidates and remove duplicates
        seen = set()
        unique_cands = []
        for cx, cy, cz in cands:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            if (cx, cy, cz) in seen:
                continue
            seen.add((cx, cy, cz))
            if can_place(bin_boxes, cx, cy, cz, w, h, d):
                unique_cands.append((cx, cy, cz))
        return unique_cands

    def score_position(bin_boxes, x, y, z, w, h, d):
        # Best-fit: minimize the sum of distances to existing boxes (tighter packing)
        if not bin_boxes:
            return 0
        # Use a simple heuristic: prefer positions that are close to existing boxes
        # to reduce fragmentation. Use sum of min distances to each box face.
        total = 0
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # Distance in x
            dx = max(0, max(bx - (x + w), x - (bx + bw)))
            dy = max(0, max(by - (y + h), y - (by + bh)))
            dz = max(0, max(bz - (z + d), z - (bz + bd)))
            # If they are adjacent or overlapping in some axis, distance is 0 in that axis
            # We want to minimize the "gap" so prefer tight packing
            total += dx * dx + dy * dy + dz * dz
        return total

    def place_box(box_idx, w, h, d):
        best_bin = -1
        best_pos = None
        best_score = float('inf')

        for bi, bin_boxes in enumerate(bins):
            cands = candidate_positions(bin_boxes, w, h, d)
            for cx, cy, cz in cands:
                sc = score_position(bin_boxes, cx, cy, cz, w, h, d)
                if sc < best_score:
                    best_score = sc
                    best_bin = bi
                    best_pos = (cx, cy, cz)

        if best_bin == -1:
            # Start a new bin
            bins.append([(0, 0, 0, w, h, d)])
            return len(bins) - 1, (0, 0, 0)
        else:
            bins[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            return best_bin, best_pos

    # Place all boxes
    placements = [None] * len(boxes)
    for idx, (orig_idx, (w, h, d)) in enumerate(indexed):
        if time.time() > DEADLINE:
            break
        bi, pos = place_box(orig_idx, w, h, d)
        placements[orig_idx] = (bi, pos[0], pos[1], pos[2], w, h, d)

    # Check if all placed; if not, fill remaining with a simple shelf in a new bin
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            bi = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            placements[i] = (bi, 0, 0, 0, w, h, d)

    # Local improvement: try to move boxes to reduce bin count or improve last-bin fill
    # This is a structural improvement: try moving a box from a higher bin to a lower one
    if time.time() < DEADLINE - 10:
        max_bins = max(p[0] for p in placements) if placements else 0
        for _ in range(50):  # limited improvement passes
            if time.time() > DEADLINE - 5:
                break
            improved = False
            for i in range(len(boxes)):
                bi, x, y, z, w, h, d = placements[i]
                # Try to move to a lower bin
                for target_bin in range(bi - 1, -1, -1):
                    # Remove box from current bin
                    bins[bi].remove((x, y, z, w, h, d))
                    cands = candidate_positions(bins[target_bin], w, h, d)
                    found = False
                    for cx, cy, cz in cands:
                        # Check if moving here is valid and doesn't cause issues
                        # We just need a valid position
                        found = True
                        break
                    if found:
                        bins[target_bin].append((cx, cy, cz, w, h, d))
                        placements[i] = (target_bin, cx, cy, cz, w, h, d)
                        improved = True
                        break
                    else:
                        # Restore
                        bins[bi].append((x, y, z, w, h, d))
                        break
                if improved:
                    break
            if not improved:
                break

    return placements
