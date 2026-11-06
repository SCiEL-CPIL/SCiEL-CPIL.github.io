"""Improved 3D bin packing with Best-Fit Decreasing + space subdivision."""
import time
import sys

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Sort boxes by volume descending (Best-Fit Decreasing)
    indexed_boxes = [(i, w, h, d, w * h * d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[4])

    # Each bin has a list of free spaces (x, y, z, w, h, d)
    # Start with one bin containing the full container as a free space
    bins = []  # list of lists of free spaces
    placements = [None] * n  # (bin_idx, x, y, z, w, h, d)

    def new_bin():
        bins.append([(0, 0, 0, W, H, D)])

    new_bin()

    def split_space(space, box_x, box_y, box_z, box_w, box_h, box_d):
        """Split a free space after placing a box. Returns list of new free spaces."""
        sx, sy, sz, sw, sh, sd = space
        results = []

        # Split along x-axis
        if box_x < sx + sw and sx + box_w < sx + sw:
            # Left part
            if sx + box_w > sx:
                results.append((sx, sy, sz, box_x - sx, sh, sd))
            # Right part
            if sx + sw > sx + box_w:
                results.append((sx + box_w, sy, sz, sx + sw - sx - box_w, sh, sd))

        # Split along y-axis
        if box_y < sy + sh and sy + box_h < sy + sh:
            # Bottom part
            if sy + box_h > sy:
                results.append((sx, sy, sz, sw, box_y - sy, sd))
            # Top part
            if sy + sh > sy + box_h:
                results.append((sx, sy + box_h, sz, sw, sy + sh - sy - box_h, sd))

        # Split along z-axis
        if box_z < sz + sd and sz + box_d < sz + sd:
            # Front part
            if sz + box_d > sz:
                results.append((sx, sy, sz, sw, sh, box_z - sz))
            # Back part
            if sz + sd > sz + box_d:
                results.append((sx, sy, sz + box_d, sw, sh, sz + sd - sz - box_d))

        return results

    def find_best_position(bin_idx, bw, bh, bd):
        """Find the best (lowest, then leftmost, then frontmost) position in the bin."""
        best = None
        best_key = None
        spaces = bins[bin_idx]
        for si, (sx, sy, sz, sw, sh, sd) in enumerate(spaces):
            if bw <= sw and bh <= sh and bd <= sd:
                # The box can fit in this space at position (sx, sy, sz)
                # We prefer the position with smallest z, then y, then x
                key = (sz, sy, sx)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (si, sx, sy, sz)
        return best

    def place_in_bin(bin_idx, box_idx, bw, bh, bd):
        """Try to place a box in a bin. Returns True if placed."""
        result = find_best_position(bin_idx, bw, bh, bd)
        if result is None:
            return False
        si, px, py, pz = result
        # Remove the old space and add split spaces
        old_space = bins[bin_idx][si]
        new_spaces = split_space(old_space, px, py, pz, bw, bh, bd)
        bins[bin_idx][si:si+1] = new_spaces
        placements[box_idx] = (bin_idx, px, py, pz, bw, bh, bd)
        return True

    # Place each box
    for box_idx, w, h, d, vol in indexed_boxes:
        if time.time() > deadline:
            # Fallback: use simple shelf heuristic for remaining boxes
            _fallback_shelf(boxes, container, placements, bins, box_idx, indexed_boxes)
            break

        # Try to place in existing bins, preferring the one with best fit
        best_bin = -1
        best_fit_score = None

        for bi in range(len(bins)):
            if not bins[bi]:
                continue
            result = find_best_position(bi, w, h, d)
            if result is not None:
                si, px, py, pz = result
                # Score: prefer tighter fit (smaller remaining space after placement)
                # Approximate by preferring lower z, then y, then x
                fit_score = (pz, py, px)
                if best_fit_score is None or fit_score < best_fit_score:
                    best_fit_score = fit_score
                    best_bin = bi

        if best_bin >= 0:
            place_in_bin(best_bin, box_idx, w, h, d)
        else:
            # Start a new bin
            new_bin()
            place_in_bin(len(bins) - 1, box_idx, w, h, d)

    # Re-index bin labels to be consecutive starting from 0
    if not placements:
        return []

    # Check if all placements were made
    for i in range(n):
        if placements[i] is None:
            # Should not happen, but handle gracefully
            placements[i] = (0, 0, 0, 0, 0, 0, 0)

    # Remap bin indices to be consecutive
    bin_map = {}
    next_label = 0
    for p in placements:
        if p[0] not in bin_map:
            bin_map[p[0]] = next_label
            next_label += 1

    result = [(bin_map[bi], x, y, z, w, h, d) for (bi, x, y, z, w, h, d) in placements]
    return result


def _fallback_shelf(boxes, container, placements, bins, start_idx, indexed_boxes):
    """Fallback shelf heuristic for remaining boxes if deadline is hit."""
    W, H, D = container
    # Get remaining unplaced boxes
    remaining = []
    for box_idx, w, h, d, vol in indexed_boxes:
        if placements[box_idx] is None:
            remaining.append((box_idx, w, h, d))

    if not remaining:
        return

    # Use the last bin or create a new one
    # Simple shelf packing in a new bin
    bins.append([(0, 0, 0, W, H, D)])
    bin_idx = len(bins) - 1
    x, y, z, row_h, layer_d = 0, 0, 0, 0, 0

    for box_idx, w, h, d in remaining:
        if x + w > W:
            x, y, row_h = 0, y + row_h, 0
        if y + h > H:
            y, z, layer_d = 0, z + layer_d, 0
            x, row_h = 0, 0
        if z + d > D:
            # New bin
            bins.append([(0, 0, 0, W, H, D)])
            bin_idx = len(bins) - 1
            x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
        placements[box_idx] = (bin_idx, x, y, z, w, h, d)
        x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
