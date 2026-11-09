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
    deadline = start_time + 115.0  # Safety margin before 120s limit

    W, H, D = container
    n = len(boxes)

    if n == 0:
        return []

    box_volumes = [w * h * d for w, h, d in boxes]
    total_volume = sum(box_volumes)
    container_vol = W * H * D

    # Sort indices by various heuristics
    vol_sorted = sorted(range(n), key=lambda i: box_volumes[i], reverse=True)
    max_dim_sorted = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    sum_dim_sorted = sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True)
    min_dim_sorted = sorted(range(n), key=lambda i: min(boxes[i]), reverse=True)

    orderings = [vol_sorted, max_dim_sorted, sum_dim_sorted, min_dim_sorted]
    for _ in range(4):
        random_order = list(range(n))
        random.shuffle(random_order)
        orderings.append(random_order)

    def compute_score(placements):
        if len(placements) != n:
            return float('inf')
        bin_labels = set(p[0] for p in placements)
        B = len(bin_labels)
        max_bin = max(p[0] for p in placements)
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

    def find_best_position(w, h, d, bin_boxes):
        """Find the best position for a box in a bin using corner-based candidates."""
        if not bin_boxes:
            return (0, 0, 0)

        # Generate candidate positions from existing box edges
        x_cands = set([0])
        y_cands = set([0])
        z_cands = set([0])

        for bx, by, bz, bw, bh, bd in bin_boxes:
            x_cands.add(bx)
            x_cands.add(bx + bw)
            y_cands.add(by)
            y_cands.add(by + bh)
            z_cands.add(bz)
            z_cands.add(bz + bd)

        # Filter valid candidates
        x_cands = [x for x in x_cands if 0 <= x and x + w <= W]
        y_cands = [y for y in y_cands if 0 <= y and y + h <= H]
        z_cands = [z for z in z_cands if 0 <= z and z + d <= D]

        if not x_cands or not y_cands or not z_cands:
            return None

        best_pos = None
        best_cost = float('inf')

        # Bottom-left-far heuristic: minimize z, then y, then x
        for z in sorted(z_cands):
            for y in sorted(y_cands):
                for x in sorted(x_cands):
                    # Check overlap with all boxes in bin
                    overlaps = False
                    for bx, by, bz, bw, bh, bd in bin_boxes:
                        if (x < bx + bw and bx < x + w and
                            y < by + bh and by < y + h and
                            z < bz + bd and bz < z + d):
                            overlaps = True
                            break
                    if overlaps:
                        continue

                    # Cost: prefer positions that are "tucked in" - low z, low y, low x
                    cost = z * (H + 1) * (W + 1) + y * (W + 1) + x

                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (x, y, z)
                        if best_pos == (0, 0, 0):
                            return best_pos

        return best_pos

    def greedy_place(ordering, bin_selection='fullest'):
        """Greedy placement with configurable bin selection strategy."""
        placements = [None] * n
        bins = []  # list of lists of (x, y, z, w, h, d)
        bin_volumes = []  # track volume used in each bin

        for box_idx in ordering:
            w, h, d = boxes[box_idx]
            placed = False

            if bins:
                if bin_selection == 'fullest':
                    # Try bins with most volume used first
                    bin_indices = sorted(range(len(bins)), key=lambda bi: bin_volumes[bi], reverse=True)
                else:
                    # Try bins with least volume used first (balance)
                    bin_indices = sorted(range(len(bins)), key=lambda bi: bin_volumes[bi])
            else:
                bin_indices = []

            for bin_i in bin_indices:
                pos = find_best_position(w, h, d, bins[bin_i])
                if pos is not None:
                    x, y, z = pos
                    bins[bin_i].append((x, y, z, w, h, d))
                    bin_volumes[bin_i] += w * h * d
                    placements[box_idx] = (bin_i, x, y, z, w, h, d)
                    placed = True
                    break

            if not placed:
                new_bin_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                bin_volumes.append(w * h * d)
                placements[box_idx] = (new_bin_idx, 0, 0, 0, w, h, d)

        return placements

    def local_improve(placements):
        """Try to improve placement by moving boxes to fill bins more tightly."""
        if time.time() > deadline:
            return placements

        # Group boxes by bin
        bins = defaultdict(list)
        for i, (b, x, y, z, w, h, d) in enumerate(placements):
            bins[b].append((i, x, y, z, w, h, d))

        improved = True
        max_iterations = 50
        iteration = 0

        while improved and iteration < max_iterations and time.time() < deadline:
            improved = False
            iteration += 1

            for bin_idx in list(bins.keys()):
                if time.time() > deadline:
                    break

                bin_boxes = bins[bin_idx]
                for i, (box_i, x, y, z, w, h, d) in enumerate(bin_boxes):
                    # Try to remove this box and re-place it in another bin
                    remaining = [b for j, b in enumerate(bin_boxes) if j != i]

                    # Try to place in other bins
                    best_new_bin = None
                    best_new_pos = None
                    best_cost = float('inf')

                    for other_bin in bins:
                        if other_bin == bin_idx:
                            continue
                        other_boxes = [(x2, y2, z2, w2, h2, d2) for (_, x2, y2, z2, w2, h2, d2) in bins[other_bin]]
                        pos = find_best_position(w, h, d, other_boxes)
                        if pos is not None:
                            # Cost: prefer moving to bins that are already being used
                            # and positions that are more compact
                            cost = pos[2] * 100 + pos[1] * 10 + pos[0]
                            if cost < best_cost:
                                best_cost = cost
                                best_new_bin = other_bin
                                best_new_pos = pos

                    if best_new_bin is not None:
                        # Move the box
                        bins[bin_idx].pop(i)
                        if len(bins[bin_idx]) == 0:
                            del bins[bin_idx]
                        bins[best_new_bin].append((box_i, best_new_pos[0], best_new_pos[1], best_new_pos[2], w, h, d))
                        improved = True
                        break

                if improved:
                    break

        # Rebuild placements from bins
        new_placements = [None] * n
        for bin_idx, bin_boxes in bins.items():
            for (box_i, x, y, z, w, h, d) in bin_boxes:
                new_placements[box_i] = (bin_idx, x, y, z, w, h, d)

        # Relabel bins to be consecutive
        if new_placements and all(p is not None for p in new_placements):
            old_to_new = {}
            for idx, b in enumerate(sorted(set(p[0] for p in new_placements))):
                old_to_new[b] = idx
            for i in range(n):
                b, x, y, z, w, h, d = new_placements[i]
                new_placements[i] = (old_to_new[b], x, y, z, w, h, d)
            return new_placements

        return placements

    def last_bin_polish(placements):
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
                pos = find_best_position(w, h, d, target_box_list)

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

    best_result = None
    best_score = float('inf')

    # Try different strategies
    strategies = [
        ('vol_sorted', 'fullest'),
        ('max_dim_sorted', 'fullest'),
        ('sum_dim_sorted', 'fullest'),
        ('min_dim_sorted', 'fullest'),
        ('vol_sorted', 'emptiest'),
        ('max_dim_sorted', 'emptiest'),
    ]

    for order_name, bin_sel in strategies:
        if time.time() > deadline:
            break

        if order_name == 'vol_sorted':
            order = vol_sorted
        elif order_name == 'max_dim_sorted':
            order = max_dim_sorted
        elif order_name == 'sum_dim_sorted':
            order = sum_dim_sorted
        elif order_name == 'min_dim_sorted':
            order = min_dim_sorted
        else:
            continue

        placements = greedy_place(order, bin_sel)

        if not validate_placement(placements):
            continue

        # Apply local search polish
        polished = last_bin_polish(placements)

        if not validate_placement(polished):
            polished = placements

        # Apply further local improvement
        further_polished = local_improve(polished)
        if validate_placement(further_polished):
            polished = further_polished

        score = compute_score(polished)

        if score < best_score:
            best_score = score
            best_result = polished[:]

    # Also try random orderings
    for order in orderings[4:]:
        if time.time() > deadline:
            break

        placements = greedy_place(order, 'fullest')

        if not validate_placement(placements):
            continue

        polished = last_bin_polish(placements)
        if not validate_placement(polished):
            polished = placements

        further_polished = local_improve(polished)
        if validate_placement(further_polished):
            polished = further_polished

        score = compute_score(polished)

        if score < best_score:
            best_score = score
            best_result = polished[:]

    if best_result is not None:
        return best_result

    # Fallback: one box per bin
    return [(i, 0, 0, 0, w, h, d) for i, (w, h, d) in enumerate(boxes)]

# EVOLVE-BLOCK-END
