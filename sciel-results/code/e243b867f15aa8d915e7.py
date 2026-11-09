import time
import math
import random
from collections import defaultdict
import numpy as np

# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    W, H, D = container
    n = len(boxes)

    if n == 0:
        return []

    box_volumes = [w * h * d for w, h, d in boxes]
    total_volume = sum(box_volumes)
    container_vol = W * H * D

    # Sort indices by volume descending
    vol_sorted = sorted(range(n), key=lambda i: box_volumes[i], reverse=True)
    max_dim_sorted = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    sum_dim_sorted = sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True)

    orderings = [vol_sorted, max_dim_sorted, sum_dim_sorted]
    for _ in range(3):
        random_order = list(range(n))
        random.shuffle(random_order)
        orderings.append(random_order)

    def compute_score(placements):
        if len(placements) != n:
            return float('inf')
        max_bin = max(p[0] for p in placements)
        bin_labels = set(p[0] for p in placements)
        B = len(bin_labels)
        last_bin_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == max_bin)
        occupied_frac = last_bin_vol / container_vol if container_vol > 0 else 0.0
        return B + 0.1 * (1 - occupied_frac)

    def validate_placement(placements):
        if len(placements) != n:
            return False
        bins = defaultdict(list)
        for i, (b, x, y, z, w, h, d) in enumerate(placements):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            bins[b].append((x, y, z, w, h, d))
        for bin_idx, box_list in bins.items():
            for i in range(len(box_list)):
                for j in range(i + 1, len(box_list)):
                    x1, y1, z1, w1, h1, d1 = box_list[i]
                    x2, y2, z2, w2, h2, d2 = box_list[j]
                    if (x1 < x2 + w2 and x2 < x1 + w1 and
                        y1 < y2 + h2 and y2 < y1 + h1 and
                        z1 < z2 + d2 and z2 < z1 + d1):
                        return False
        return True

    def greedy_place(ordering):
        placements = [None] * n
        bins = []  # list of lists of (x, y, z, w, h, d)

        def find_position_in_bin(w, h, d, bin_boxes):
            # Candidate positions: 0 and edges of existing boxes
            x_cands = sorted({0} | {bx for bx, _, _, _, _, _ in bin_boxes} | {bx + bw for bx, _, _, bw, _, _ in bin_boxes})
            y_cands = sorted({0} | {by for _, by, _, _, _, _ in bin_boxes} | {by + bh for _, by, _, _, bh, _ in bin_boxes})
            z_cands = sorted({0} | {bz for _, _, bz, _, _, _ in bin_boxes} | {bz + bd for _, _, bz, _, _, bd in bin_boxes})

            x_cands = [x for x in x_cands if 0 <= x and x + w <= W]
            y_cands = [y for y in y_cands if 0 <= y and y + h <= H]
            z_cands = [z for z in z_cands if 0 <= z and z + d <= D]

            if not x_cands or not y_cands or not z_cands:
                return None

            best_pos = None
            best_cost = float('inf')

            for z in z_cands:
                for y in y_cands:
                    for x in x_cands:
                        # Check overlap
                        overlaps = False
                        for bx, by, bz, bw, bh, bd in bin_boxes:
                            if x < bx + bw and bx < x + w and \
                               y < by + bh and by < y + h and \
                               z < bz + bd and bz < z + d:
                                overlaps = True
                                break
                        if overlaps:
                            continue
                        # Cost: prefer lower z, then lower y, then lower x
                        cost = z * (H + 1) * (W + 1) + y * (W + 1) + x
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = (x, y, z)
                            if best_pos == (0, 0, 0):
                                return best_pos

            return best_pos

        for box_idx in ordering:
            w, h, d = boxes[box_idx]

            # Try to place in existing bins, prefer bins with more volume used
            placed = False
            bin_indices = list(range(len(bins)))
            # Sort bins by volume used (descending) to try fullest bins first
            bin_indices.sort(key=lambda bi: sum(bw * bh * bd for _, _, _, bw, bh, bd in bins[bi]), reverse=True)

            for bin_i in bin_indices:
                pos = find_position_in_bin(w, h, d, bins[bin_i])
                if pos is not None:
                    x, y, z = pos
                    bins[bin_i].append((x, y, z, w, h, d))
                    placements[box_idx] = (bin_i, x, y, z, w, h, d)
                    placed = True
                    break

            if not placed:
                new_bin_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placements[box_idx] = (new_bin_idx, 0, 0, 0, w, h, d)

        return placements

    def local_search_polish(placements):
        """Try to move boxes from the last bin to earlier bins to reduce B or improve last-bin fill."""
        if time.time() > deadline:
            return placements

        # Build bins from placements
        bins = defaultdict(list)
        for i, (b, x, y, z, w, h, d) in enumerate(placements):
            bins[b].append((i, x, y, z, w, h, d))

        if len(bins) <= 1:
            return placements

        max_bin = max(bins.keys())
        # Get boxes in the last bin sorted by volume descending
        last_bin_boxes = sorted(bins[max_bin], key=lambda item: item[3] * item[4] * item[5], reverse=True)

        new_placements = placements[:]
        moved_count = 0

        for box_idx, x, y, z, w, h, d in last_bin_boxes:
            if time.time() > deadline:
                break

            # Try to move this box to an earlier bin
            for target_bin in sorted(bins.keys()):
                if target_bin == max_bin:
                    continue

                # Try to find a position in target_bin
                target_box_list = [(bx, by, bz, bw, bh, bd) for (_, bx, by, bz, bw, bh, bd) in bins[target_bin]]
                pos = find_position_in_bin(w, h, d, target_box_list)

                if pos is not None:
                    # Move the box
                    # Remove from last bin
                    bins[max_bin] = [item for item in bins[max_bin] if item[0] != box_idx]
                    # Add to target bin
                    new_pos = pos
                    bins[target_bin].append((box_idx, new_pos[0], new_pos[1], new_pos[2], w, h, d))
                    new_placements[box_idx] = (target_bin, new_pos[0], new_pos[1], new_pos[2], w, h, d)
                    moved_count += 1
                    break

        # Remove empty bins and relabel
        if moved_count > 0:
            # Rebuild bins from new_placements
            bins2 = defaultdict(list)
            for i, (b, x, y, z, w, h, d) in enumerate(new_placements):
                bins2[b].append((i, x, y, z, w, h, d))

            # Relabel bins to be consecutive
            old_to_new = {}
            for idx, b in enumerate(sorted(bins2.keys())):
                old_to_new[b] = idx

            for i in range(n):
                b, x, y, z, w, h, d = new_placements[i]
                new_placements[i] = (old_to_new[b], x, y, z, w, h, d)

        return new_placements

    def find_position_in_bin(w, h, d, bin_boxes):
        # Candidate positions: 0 and edges of existing boxes
        x_cands = sorted({0} | {bx for bx, _, _, _, _, _ in bin_boxes} | {bx + bw for bx, _, _, bw, _, _ in bin_boxes})
        y_cands = sorted({0} | {by for _, by, _, _, _, _ in bin_boxes} | {by + bh for _, by, _, _, bh, _ in bin_boxes})
        z_cands = sorted({0} | {bz for _, _, bz, _, _, _ in bin_boxes} | {bz + bd for _, _, bz, _, _, bd in bin_boxes})

        x_cands = [x for x in x_cands if 0 <= x and x + w <= W]
        y_cands = [y for y in y_cands if 0 <= y and y + h <= H]
        z_cands = [z for z in z_cands if 0 <= z and z + d <= D]

        if not x_cands or not y_cands or not z_cands:
            return None

        best_pos = None
        best_cost = float('inf')

        for z in z_cands:
            for y in y_cands:
                for x in x_cands:
                    # Check overlap
                    overlaps = False
                    for bx, by, bz, bw, bh, bd in bin_boxes:
                        if x < bx + bw and bx < x + w and \
                           y < by + bh and by < y + h and \
                           z < bz + bd and bz < z + d:
                            overlaps = True
                            break
                    if overlaps:
                        continue
                    # Cost: prefer lower z, then lower y, then lower x
                    cost = z * (H + 1) * (W + 1) + y * (W + 1) + x
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (x, y, z)
                        if best_pos == (0, 0, 0):
                            return best_pos

        return best_pos

    best_result = None
    best_score = float('inf')

    for order in orderings:
        if time.time() > deadline:
            break

        placements = greedy_place(order)

        if not validate_placement(placements):
            continue

        # Apply local search polish
        polished = local_search_polish(placements)

        if not validate_placement(polished):
            polished = placements

        score = compute_score(polished)

        if score < best_score:
            best_score = score
            best_result = polished[:]

    if best_result is not None:
        return best_result

    # Fallback: one box per bin
    return [(i, 0, 0, 0, w, h, d) for i, (w, h, d) in enumerate(boxes)]

# EVOLVE-BLOCK-END
