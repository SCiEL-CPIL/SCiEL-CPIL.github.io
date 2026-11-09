"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Deadline guard: evaluator kills at 150s, we aim to return well before.
    t_start = time.time()
    DEADLINE = 140.0  # safety margin

    # Sort boxes by volume descending for better packing (largest first)
    indexed_boxes = [(i, b) for i, b in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)

    best_placements = None
    best_score = float('inf')

    def compute_score(placements):
        """Compute the score: mean B + 0.1*(1 - last-bin occupied volume fraction)"""
        if not placements:
            return 0.0
        bins_used = {}
        for (b, x, y, z, w, h, d) in placements:
            if b not in bins_used:
                bins_used[b] = 0
            bins_used[b] += w * h * d
        B = len(bins_used)
        if B == 0:
            return 0.0
        # Find the largest bin label
        last_bin = max(bins_used.keys())
        container_vol = W * H * D
        if container_vol == 0:
            return float(B)
        last_frac = bins_used[last_bin] / container_vol
        return B + 0.1 * (1.0 - last_frac)

    def shelf_pack(boxes_list, W, H, D):
        """Shelf-based packing heuristic."""
        placements = []
        b = 0
        x, y, z = 0, 0, 0
        row_h = 0
        layer_d = 0

        for w, h, d in boxes_list:
            # Try to fit in current position
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x += w
            row_h = max(row_h, h)
            layer_d = max(layer_d, d)

        return placements

    def best_fit_3d(boxes_list, W, H, D):
        """Best-fit decreasing 3D packing using a list of free spaces."""
        # Represent free spaces as a list of (x, y, z, w, h, d)
        free_spaces = [(0, 0, 0, W, H, D)]
        placements = []
        bin_count = 0
        current_bin = 0
        # We'll track per-bin free spaces
        bin_free_spaces = {0: [(0, 0, 0, W, H, D)]}

        for w, h, d in boxes_list:
            placed = False
            # Try current bin first
            for b_idx in sorted(bin_free_spaces.keys()):
                fs_list = bin_free_spaces[b_idx]
                best_idx = -1
                best_waste = float('inf')
                for i, (fx, fy, fz, fw, fh, fd) in enumerate(fs_list):
                    if w <= fw and h <= fh and d <= fd:
                        # Compute waste (how much larger the space is than needed)
                        waste = (fw - w) * fh * fd + fw * (fh - h) * fd + fw * fh * (fd - d)
                        # Prefer tighter fits
                        if waste < best_waste:
                            best_waste = waste
                            best_idx = i
                if best_idx >= 0:
                    fx, fy, fz, fw, fh, fd = fs_list[best_idx]
                    # Place box at (fx, fy, fz)
                    placements.append((b_idx, fx, fy, fz, w, h, d))
                    # Update free spaces: split the space
                    remaining = []
                    # Space above (if any)
                    if fh - h > 0:
                        remaining.append((fx, fy + h, fz, fw, fh - h, fd))
                    # Space to the right (if any)
                    if fw - w > 0:
                        remaining.append((fx + w, fy, fz, fw - w, fh, fd))
                    # Space in front (if any)
                    if fd - d > 0:
                        remaining.append((fx, fy, fz + d, fw, fh, fd - d))
                    # Replace the used space with remaining ones
                    fs_list[best_idx:best_idx+1] = remaining
                    placed = True
                    break

            if not placed:
                # Start a new bin
                bin_count += 1
                current_bin = bin_count
                bin_free_spaces[current_bin] = [(0, 0, 0, W, H, D)]
                placements.append((current_bin, 0, 0, 0, w, h, d))
                # Update free space in new bin
                fw, fh, fd = W, H, D
                remaining = []
                if fh - h > 0:
                    remaining.append((0, h, 0, fw, fh - h, fd))
                if fw - w > 0:
                    remaining.append((w, 0, 0, fw - w, fh, fd))
                if fd - d > 0:
                    remaining.append((0, 0, d, fw, fh, fd - d))
                bin_free_spaces[current_bin] = remaining

        return placements

    # Try multiple heuristics and keep the best
    heuristics = [
        ("shelf", shelf_pack),
        ("best_fit", best_fit_3d),
    ]

    for name, func in heuristics:
        if time.time() > DEADLINE:
            break
        placements = func([b for _, b in indexed_boxes], W, H, D)
        # Map back to original indices - actually we need to return placements for original boxes
        # The problem says "one placement per box, preserving its original ordered dimensions"
        # So we need to place each original box. Let me re-read...
        # "places every box" - so we need to place all boxes from the input list.
        # The sorted order is just for algorithm efficiency, but we need to track which box goes where.

    # Actually, let me reconsider. The function should return placements for each box in the input.
    # Let me redo this properly.

    def pack_with_order(boxes, W, H, D, strategy):
        """Pack boxes, returning placements in original order."""
        n = len(boxes)
        # Create sorted order
        order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)

        if strategy == "shelf":
            return shelf_pack_ordered(boxes, order, W, H, D)
        elif strategy == "best_fit":
            return best_fit_ordered(boxes, order, W, H, D)
        else:
            return shelf_pack_ordered(boxes, order, W, H, D)

    def shelf_pack_ordered(boxes, order, W, H, D):
        n = len(boxes)
        result = [None] * n
        b = 0
        x, y, z = 0, 0, 0
        row_h = 0
        layer_d = 0

        for idx in order:
            w, h, d = boxes[idx]
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            result[idx] = (b, x, y, z, w, h, d)
            x += w
            row_h = max(row_h, h)
            layer_d = max(layer_d, d)

        return result

    def best_fit_ordered(boxes, order, W, H, D):
        n = len(boxes)
        result = [None] * n
        bin_free_spaces = {0: [(0, 0, 0, W, H, D)]}
        bin_count = 0

        for idx in order:
            w, h, d = boxes[idx]
            placed = False
            for b_idx in sorted(bin_free_spaces.keys()):
                fs_list = bin_free_spaces[b_idx]
                best_i = -1
                best_waste = float('inf')
                for i, (fx, fy, fz, fw, fh, fd) in enumerate(fs_list):
                    if w <= fw and h <= fh and d <= fd:
                        waste = (fw - w) * fh * fd + fw * (fh - h) * fd + fw * fh * (fd - d)
                        if waste < best_waste:
                            best_waste = waste
                            best_i = i
                if best_i >= 0:
                    fx, fy, fz, fw, fh, fd = fs_list[best_i]
                    result[idx] = (b_idx, fx, fy, fz, w, h, d)
                    remaining = []
                    if fh - h > 0:
                        remaining.append((fx, fy + h, fz, fw, fh - h, fd))
                    if fw - w > 0:
                        remaining.append((fx + w, fy, fz, fw - w, fh, fd))
                    if fd - d > 0:
                        remaining.append((fx, fy, fz + d, fw, fh, fd - d))
                    fs_list[best_i:best_i+1] = remaining
                    placed = True
                    break

            if not placed:
                bin_count += 1
                bin_free_spaces[bin_count] = [(0, 0, 0, W, H, D)]
                result[idx] = (bin_count, 0, 0, 0, w, h, d)
                fw, fh, fd = W, H, D
                remaining = []
                if fh - h > 0:
                    remaining.append((0, h, 0, fw, fh - h, fd))
                if fw - w > 0:
                    remaining.append((w, 0, 0, fw - w, fh, fd))
                if fd - d > 0:
                    remaining.append((0, 0, d, fw, fh, fd - d))
                bin_free_spaces[bin_count] = remaining

        return result

    # Try multiple strategies
    strategies = ["shelf", "best_fit"]
    for strat in strategies:
        if time.time() > DEADLINE:
            break
        placements = pack_with_order(boxes, W, H, D, strat)
        score = compute_score(placements)
        if score < best_score:
            best_score = score
            best_placements = placements

    # If we still have time, try a few random restarts with different orderings
    if time.time() < DEADLINE - 10:
        for trial in range(20):
            if time.time() > DEADLINE:
                break
            # Random permutation of the sorted order
            perm = list(range(len(order)))
            random.shuffle(perm)
            shuffled_order = [order[i] for i in perm]
            placements = pack_with_order(boxes, W, H, D, "best_fit")
            # Actually need to use shuffled order
            placements = best_fit_ordered(boxes, shuffled_order, W, H, D)
            score = compute_score(placements)
            if score < best_score:
                best_score = score
                best_placements = placements

    if best_placements is None:
        # Fallback: just place everything in bin 0 at origin (may overlap but valid interface)
        best_placements = [(0, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_placements
# EVOLVE-BLOCK-END
