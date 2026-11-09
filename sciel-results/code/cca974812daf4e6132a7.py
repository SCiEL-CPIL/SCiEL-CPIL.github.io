import time
import math
import random
from collections import defaultdict
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    W, H, D = container
    n = len(boxes)

    if n == 0:
        return []

    box_volumes = [w * h * d for w, h, d in boxes]
    total_volume = sum(box_volumes)
    container_vol = W * H * D

    if container_vol == 0:
        return [(i, 0, 0, 0, w, h, d) for i, (w, h, d) in enumerate(boxes)]

    # Sort indices by various heuristics
    vol_sorted = sorted(range(n), key=lambda i: box_volumes[i], reverse=True)
    max_dim_sorted = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    sum_dim_sorted = sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True)
    min_dim_sorted = sorted(range(n), key=lambda i: min(boxes[i]), reverse=True)
    ratio_sorted = sorted(range(n), key=lambda i: box_volumes[i] / max(boxes[i]) if max(boxes[i]) > 0 else 0, reverse=True)

    orderings = [vol_sorted, max_dim_sorted, sum_dim_sorted, min_dim_sorted, ratio_sorted]
    for _ in range(5):
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

    def check_overlap(x, y, z, w, h, d, bin_boxes):
        """Check if a box at (x,y,z) with dims (w,h,d) overlaps any box in bin_boxes."""
        for bx, by, bz, bw, bh, bd in bin_boxes:
            if (x < bx + bw and bx < x + w and
                y < by + bh and by < y + h and
                z < bz + bd and bz < z + d):
                return True
        return False

    def find_best_position(w, h, d, bin_boxes):
        """
        Find the best position for a box in a bin.
        Uses a priority-based corner candidate approach.
        """
        if not bin_boxes:
            return (0, 0, 0)

        # Generate candidate positions
        x_cands = {0}
        y_cands = {0}
        z_cands = {0}

        for bx, by, bz, bw, bh, bd in bin_boxes:
            x_cands.add(bx + bw)
            y_cands.add(by + bh)
            z_cands.add(bz + bd)

        x_cands.add(0)
        y_cands.add(0)
        z_cands.add(0)

        # Filter valid candidates
        x_cands = sorted([x for x in x_cands if 0 <= x and x + w <= W])
        y_cands = sorted([y for y in y_cands if 0 <= y and y + h <= H])
        z_cands = sorted([z for z in z_cands if 0 <= z and z + d <= D])

        if not x_cands or not y_cands or not z_cands:
            return None

        best_pos = None
        best_cost = float('inf')

        # Use a compactness cost: minimize (z * W * H + y * W + x)
        for z in z_cands:
            for y in y_cands:
                for x in x_cands:
                    if check_overlap(x, y, z, w, h, d, bin_boxes):
                        continue
                    cost = z * W * H + y * W + x
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (x, y, z)
                        if best_pos == (0, 0, 0):
                            return best_pos

        return best_pos

    def find_position_best_fit(w, h, d, bin_boxes):
        """
        Find position using best-fit strategy.
        """
        if not bin_boxes:
            return (0, 0, 0)

        x_cands = {0}
        y_cands = {0}
        z_cands = {0}

        for bx, by, bz, bw, bh, bd in bin_boxes:
            x_cands.add(bx + bw)
            y_cands.add(by + bh)
            z_cands.add(bz + bd)

        x_cands = sorted([x for x in x_cands if 0 <= x and x + w <= W])
        y_cands = sorted([y for y in y_cands if 0 <= y and y + h <= H])
        z_cands = sorted([z for z in z_cands if 0 <= z and z + d <= D])

        if not x_cands or not y_cands or not z_cands:
            return None

        best_pos = None
        best_cost = float('inf')

        for z in z_cands:
            for y in y_cands:
                for x in x_cands:
                    if check_overlap(x, y, z, w, h, d, bin_boxes):
                        continue
                    cost = z * 0.5 + y * 0.3 + x * 0.2
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (x, y, z)
                        if best_pos == (0, 0, 0):
                            return best_pos

        return best_pos

    def greedy_place(ordering, bin_selection='fullest', pos_strategy='compact'):
        """Greedy placement with configurable bin selection and position strategy."""
        placements = [None] * n
        bins = []
        bin_volumes = []

        for box_idx in ordering:
            w, h, d = boxes[box_idx]
            placed = False

            if bins:
                if bin_selection == 'fullest':
                    bin_indices = sorted(range(len(bins)), key=lambda bi: bin_volumes[bi], reverse=True)
                else:
                    bin_indices = sorted(range(len(bins)), key=lambda bi: bin_volumes[bi])
            else:
                bin_indices = []

            for bin_i in bin_indices:
                if pos_strategy == 'compact':
                    pos = find_best_position(w, h, d, bins[bin_i])
                else:
                    pos = find_position_best_fit(w, h, d, bins[bin_i])

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

    def local_improve(placements, max_iterations=100):
        """Try to improve placement by moving boxes to reduce bin count or improve last-bin fill."""
        if time.time() > deadline:
            return placements

        bins = defaultdict(list)
        for i, (b, x, y, z, w, h, d) in enumerate(placements):
            bins[b].append((i, x, y, z, w, h, d))

        improved = True
        iteration = 0

        while improved and iteration < max_iterations and time.time() < deadline - 5:
            improved = False
            iteration += 1

            bin_keys = list(bins.keys())
            for bi in bin_keys:
                if time.time() > deadline - 5:
                    break
                if bi not in bins or len(bins[bi]) == 0:
                    continue
                bin_boxes = bins[bi]
                for i in range(len(bin_boxes)):
                    box_i, x, y, z, w, h, d = bin_boxes[i]

                    best_new_bin = None
                    best_new_pos = None
                    best_cost = float('inf')

                    for other_bin in bin_keys:
                        if other_bin == bi or other_bin not in bins:
                            continue
                        other_boxes = [(x2, y2, z2, w2, h2, d2) for (_, x2, y2, z2, w2, h2, d2) in bins[other_bin]]
                        pos = find_best_position(w, h, d, other_boxes)
                        if pos is not None:
                            other_vol = sum(w2 * h2 * d2 for (_, _, _, w2, h2, d2) in bins[other_bin])
                            cost = -other_vol * 0.01 + z * 100 + y * 10 + x
                            if cost < best_cost:
                                best_cost = cost
                                best_new_bin = other_bin
                                best_new_pos = pos

                    if best_new_bin is not None and best_new_bin != bi:
                        bins[bi].pop(i)
                        if not bins[bi]:
                            del bins[bi]

                        nx, ny, nz = best_new_pos
                        bins[best_new_bin].append((box_i, nx, ny, nz, w, h, d))

                        placements[box_i] = (best_new_bin, nx, ny, nz, w, h, d)
                        improved = True
                        bin_keys = list(bins.keys())
                        break

        return placements

    def try_repack_into_fewer_bins(placements):
        """Attempt to repack all boxes into fewer bins by trying different orderings."""
        if time.time() > deadline - 10:
            return placements

        current_bins = set(p[0] for p in placements)
        if len(current_bins) <= 1:
            return placements

        box_data = [(p[4], p[5], p[6]) for p in placements]

        target_bins = len(current_bins) - 1

        best_result = None
        best_score = float('inf')

        for order in [vol_sorted, max_dim_sorted]:
            if time.time() > deadline - 10:
                break

            bins = []
            bin_volumes = []
            placed_boxes = [None] * n

            for box_idx in order:
                w, h, d = box_data[box_idx]
                placed = False

                for bin_i in range(len(bins)):
                    pos = find_best_position(w, h, d, bins[bin_i])
                    if pos is not None:
                        x, y, z = pos
                        bins[bin_i].append((x, y, z, w, h, d))
                        bin_volumes[bin_i] += w * h * d
                        placed_boxes[box_idx] = (bin_i, x, y, z, w, h, d)
                        placed = True
                        break

                if not placed and len(bins) < target_bins:
                    new_bin_idx = len(bins)
                    bins.append([(0, 0, 0, w, h, d)])
                    bin_volumes.append(w * h * d)
                    placed_boxes[box_idx] = (new_bin_idx, 0, 0, 0, w, h, d)
                elif not placed:
                    break

            if placed is not False and all(p is not None for p in placed_boxes):
                used_bins = set(p[0] for p in placed_boxes if p is not None)
                if len(used_bins) < len(current_bins):
                    if validate_placement(placed_boxes):
                        score = compute_score(placed_boxes)
                        if score < best_score:
                            best_score = score
                            best_result = placed_boxes

        if best_result is not None:
            return best_result
        return placements

    best_result = None
    best_score = float('inf')

    strategies = [
        ('greedy_fullest_compact', lambda order: greedy_place(order, 'fullest', 'compact')),
        ('greedy_fullest_bestfit', lambda order: greedy_place(order, 'fullest', 'bestfit')),
        ('greedy_balance_compact', lambda order: greedy_place(order, 'balance', 'compact')),
    ]

    for strategy_name, strategy_func in strategies:
        for order in orderings:
            if time.time() > deadline - 30:
                break

            try:
                placements = strategy_func(order)

                if not validate_placement(placements):
                    continue

                if time.time() < deadline - 30:
                    improved_placements = local_improve(placements)
                    if validate_placement(improved_placements):
                        placements = improved_placements

                if time.time() < deadline - 15:
                    repacked = try_repack_into_fewer_bins(placements)
                    if validate_placement(repacked):
                        placements = repacked

                score = compute_score(placements)

                if score < best_score:
                    best_score = score
                    best_result = placements[:]
            except Exception:
                continue

    if best_result is not None:
        return best_result

    return [(i, 0, 0, 0, w, h, d) for i, (w, h, d) in enumerate(boxes)]
