import time
import math
import random
import numpy as np
from collections import defaultdict

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 140.0

    W, H, D = container
    n = len(boxes)

    if n == 0:
        return []

    box_volumes = [w * h * d for w, h, d in boxes]
    total_volume = sum(box_volumes)
    container_vol = W * H * D

    if container_vol == 0:
        return [(i, 0, 0, 0, w, h, d) for i, (w, h, d) in enumerate(boxes)]

    # Sort indices by volume descending (Best Fit Decreasing)
    vol_sorted = sorted(range(n), key=lambda i: box_volumes[i], reverse=True)

    # Generate a few different orderings for diversification
    orderings = [vol_sorted]
    for _ in range(7):
        perturbed = vol_sorted[:]
        for _ in range(n // 3):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        orderings.append(perturbed)

    def compute_score(placements):
        if len(placements) != n:
            return float('inf')
        bin_labels = set(p[0] for p in placements)
        B = len(bin_labels)
        if B == 0:
            return 0
        max_bin = max(p[0] for p in placements)
        last_bin_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == max_bin)
        occupied_frac = last_bin_vol / container_vol if container_vol > 0 else 0.0
        return B + 0.1 * (1 - occupied_frac)

    def renumber_bins(placements):
        if len(placements) == 0:
            return placements
        mapping = {}
        next_label = 0
        new_placements = []
        for p in placements:
            old = p[0]
            if old not in mapping:
                mapping[old] = next_label
                next_label += 1
            new_placements.append((mapping[old], p[1], p[2], p[3], p[4], p[5], p[6]))
        return new_placements

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

    class Bin:
        def __init__(self):
            self.boxes = []
            self.volume = 0
            # Use grid if container is small enough
            if W * H * D <= 2000000:
                self.grid = np.zeros((W, H, D), dtype=bool)
            else:
                self.grid = None

        def add_box(self, x, y, z, w, h, d):
            if x + w > W or y + h > H or z + d > D:
                return False
            if self.grid is not None:
                if self.grid[x:x+w, y:y+h, z:z+d].any():
                    return False
            else:
                for bx, by, bz, bw, bh, bd in self.boxes:
                    if (x < bx + bw and bx < x + w and
                        y < by + bh and by < y + h and
                        z < bz + bd and bz < z + d):
                        return False
            self.boxes.append((x, y, z, w, h, d))
            self.volume += w * h * d
            if self.grid is not None:
                self.grid[x:x+w, y:y+h, z:z+d] = True
            return True

        def get_candidate_positions(self, w, h, d):
            """Generate candidate positions from box faces and origin."""
            x_cands = {0}
            y_cands = {0}
            z_cands = {0}
            for bx, by, bz, bw, bh, bd in self.boxes:
                x_cands.add(bx)
                x_cands.add(bx + bw)
                y_cands.add(by)
                y_cands.add(by + bh)
                z_cands.add(bz)
                z_cands.add(bz + bd)
            # Filter to valid ranges
            x_cands = {x for x in x_cands if 0 <= x <= W - w}
            y_cands = {y for y in y_cands if 0 <= y <= H - h}
            z_cands = {z for z in z_cands if 0 <= z <= D - d}
            # Sort to prefer compact positions (lower coordinates)
            x_cands = sorted(x_cands)
            y_cands = sorted(y_cands)
            z_cands = sorted(z_cands)
            # Limit candidates to avoid explosion
            x_cands = x_cands[:15]
            y_cands = y_cands[:15]
            z_cands = z_cands[:15]
            return x_cands, y_cands, z_cands

        def try_place(self, w, h, d):
            """Try to place a box. Returns (x,y,z) or None.
            Uses Best Fit: minimize bounding box expansion, then prefer compact positions."""
            if not self.boxes:
                if w <= W and h <= H and d <= D:
                    return (0, 0, 0)
                return None

            x_cands, y_cands, z_cands = self.get_candidate_positions(w, h, d)
            
            best_pos = None
            best_metric = float('inf')

            for cx in x_cands:
                for cy in y_cands:
                    for cz in z_cands:
                        if cx + w > W or cy + h > H or cz + d > D:
                            continue
                        # Check overlap
                        if self.grid is not None:
                            if self.grid[cx:cx+w, cy:cy+h, cz:cz+d].any():
                                continue
                        else:
                            overlap = False
                            for bx, by, bz, bw, bh, bd in self.boxes:
                                if (cx < bx + bw and bx < cx + w and
                                    cy < by + bh and by < cy + h and
                                    cz < bz + bd and bz < cz + d):
                                    overlap = True
                                    break
                            if overlap:
                                continue

                        # Metric: prefer positions that minimize new bounding box volume
                        # Current bin bounding box
                        max_x = max(b[0] + b[3] for b in self.boxes)
                        max_y = max(b[1] + b[4] for b in self.boxes)
                        max_z = max(b[2] + b[5] for b in self.boxes)
                        
                        new_max_x = max(max_x, cx + w)
                        new_max_y = max(max_y, cy + h)
                        new_max_z = max(max_z, cz + d)
                        
                        # Prefer smaller new bounding box, then closer to origin
                        metric = new_max_x * new_max_y * new_max_z + 0.001 * (cx + cy + cz)
                        
                        if metric < best_metric:
                            best_metric = metric
                            best_pos = (cx, cy, cz)

            return best_pos

    def greedy_pack(ordering):
        bins = []
        placements = [None] * n

        for box_idx in ordering:
            w, h, d = boxes[box_idx]
            placed = False

            best_bin_idx = -1
            best_pos = None
            best_metric = float('inf')

            for i, bin_obj in enumerate(bins):
                pos = bin_obj.try_place(w, h, d)
                if pos is not None:
                    # Prefer bin that results in smallest bounding box expansion
                    # Or simply prefer fullest bin (current volume)
                    # Let's stick with fullest bin for stability, as try_place already optimizes position
                    if best_bin_idx == -1 or bins[i].volume > bins[best_bin_idx].volume:
                        best_bin_idx = i
                        best_pos = pos

            if best_bin_idx != -1:
                x, y, z = best_pos
                bins[best_bin_idx].add_box(x, y, z, w, h, d)
                placements[box_idx] = (best_bin_idx, x, y, z, w, h, d)
            else:
                new_bin = Bin()
                new_bin.add_box(0, 0, 0, w, h, d)
                bins.append(new_bin)
                placements[box_idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

        return placements

    def local_search(placements):
        if time.time() > deadline - 15:
            return placements

        bins_dict = defaultdict(list)
        for i, p in enumerate(placements):
            bins_dict[p[0]].append(i)

        bin_indices = sorted(bins_dict.keys())
        if len(bin_indices) <= 1:
            return placements

        last_bin_idx = bin_indices[-1]
        boxes_to_move = bins_dict[last_bin_idx][:]

        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for box_idx in boxes_to_move[:]:
                if time.time() > deadline - 5:
                    break
                if box_idx not in bins_dict.get(last_bin_idx, []):
                    continue

                p = placements[box_idx]
                w, h, d = p[4], p[5], p[6]

                moved = False
                for other_bin_idx in bin_indices[:-1]:
                    if other_bin_idx not in bins_dict or not bins_dict[other_bin_idx]:
                        continue

                    other_box_indices = bins_dict[other_bin_idx]
                    other_boxes = [(placements[j][1], placements[j][2], placements[j][3], placements[j][4], placements[j][5], placements[j][6]) for j in other_box_indices]

                    # Generate candidates
                    x_cands = {0}
                    y_cands = {0}
                    z_cands = {0}
                    for bx, by, bz, bw, bh, bd in other_boxes:
                        x_cands.add(bx)
                        x_cands.add(bx + bw)
                        y_cands.add(by)
                        y_cands.add(by + bh)
                        z_cands.add(bz)
                        z_cands.add(bz + bd)

                    x_cands = sorted({x for x in x_cands if 0 <= x <= W - w})[:15]
                    y_cands = sorted({y for y in y_cands if 0 <= y <= H - h})[:15]
                    z_cands = sorted({z for z in z_cands if 0 <= z <= D - d})[:15]

                    for cx in x_cands:
                        if moved: break
                        for cy in y_cands:
                            if moved: break
                            for cz in z_cands:
                                if cx + w > W or cy + h > H or cz + d > D:
                                    continue
                                overlap = False
                                for bx, by, bz, bw, bh, bd in other_boxes:
                                    if (cx < bx + bw and bx < cx + w and
                                        cy < by + bh and by < cy + h and
                                        cz < bz + bd and bz < cz + d):
                                        overlap = True
                                        break
                                if not overlap:
                                    placements[box_idx] = (other_bin_idx, cx, cy, cz, w, h, d)
                                    bins_dict[last_bin_idx].remove(box_idx)
                                    if not bins_dict[last_bin_idx]:
                                        del bins_dict[last_bin_idx]
                                        bin_indices = sorted(bins_dict.keys())
                                        if len(bin_indices) <= 1:
                                            return placements
                                        last_bin_idx = bin_indices[-1]
                                        boxes_to_move = bins_dict[last_bin_idx][:]
                                    bins_dict[other_bin_idx].append(box_idx)
                                    moved = True
                                    improved = True
                                    break
                if moved:
                    if box_idx in boxes_to_move:
                        boxes_to_move.remove(box_idx)

        return placements

    best_result = None
    best_score = float('inf')

    for order in orderings:
        if time.time() > deadline - 25:
            break

        try:
            placements = greedy_pack(order)
            if not validate_placement(placements):
                continue

            placements = renumber_bins(placements)
            score = compute_score(placements)

            if time.time() < deadline - 15:
                improved_placements = local_search(placements)
                if validate_placement(improved_placements):
                    improved_placements = renumber_bins(improved_placements)
                    placements = improved_placements
                    score = compute_score(placements)

            if score < best_score:
                best_score = score
                best_result = placements[:]

        except Exception as e:
            continue

    if best_result is not None:
        return best_result

    return [(i, 0, 0, 0, w, h, d) for i, (w, h, d) in enumerate(boxes)]
