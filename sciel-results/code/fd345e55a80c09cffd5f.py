import time
import random
import numpy as np
from collections import defaultdict

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit

    box_dims = [(w, h, d) for w, h, d in boxes]
    volumes = np.array([w * h * d for w, h, d in boxes], dtype=np.float64)
    container_vol = float(W * H * D)
    if container_vol <= 0:
        return [(0, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]

    def is_valid_place(bin_boxes, x, y, z, w, h, d):
        """Check if placing a box at (x,y,z) in bin_boxes is valid."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if not (x + w <= px or x >= px + pw):
                if not (y + h <= py or y >= py + ph):
                    if not (z + d <= pz or z >= pz + pd):
                        return False
        return True

    def get_skyline_positions(bin_boxes, w, h, d):
        """Generate candidate positions based on skyline of existing boxes."""
        cands = [(0, 0, 0)]
        seen = {cands[0]}
        
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # All 8 corners of the box extended
            for dx in [0, pw]:
                for dy in [0, ph]:
                    for dz in [0, pd]:
                        cx, cy, cz = px + dx, py + dy, pz + dz
                        if (cx, cy, cz) not in seen:
                            if cx + w <= W and cy + h <= H and cz + d <= D:
                                seen.add((cx, cy, cz))
                                cands.append((cx, cy, cz))
        
        # Also add positions that align with existing box edges
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Try to fit against each face
            positions = [
                (px + pw, py, pz),
                (px, py + ph, pz),
                (px, py, pz + pd),
            ]
            for pos in positions:
                if pos not in seen:
                    if pos[0] + w <= W and pos[1] + h <= H and pos[2] + d <= D:
                        seen.add(pos)
                        cands.append(pos)
        
        # Sort by (z, y, x) to prefer compact, lower packing
        cands.sort(key=lambda pos: (pos[2], pos[1], pos[0]))
        return cands

    def find_spot(bin_boxes, w, h, d):
        """Find the best valid spot, preferring positions that minimize (z, y, x)."""
        cands = get_skyline_positions(bin_boxes, w, h, d)
        for (x, y, z) in cands:
            if is_valid_place(bin_boxes, x, y, z, w, h, d):
                return (x, y, z)
        return None

    def initial_packing(order):
        """Greedy packing with given order."""
        bins = []  # list of lists of (x,y,z,w,h,d)
        placements = [None] * n
        bin_volumes = []

        for idx in order:
            w, h, d = box_dims[idx]
            best_bin = -1
            best_spot = None
            best_key = None

            for b_idx, bin_boxes in enumerate(bins):
                spot = find_spot(bin_boxes, w, h, d)
                if spot is not None:
                    x, y, z = spot
                    # Prefer bins with more existing volume (consolidation),
                    # then prefer lower z, y, x positions
                    key = (-bin_volumes[b_idx], z, y, x)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_bin = b_idx
                        best_spot = spot

            if best_bin >= 0:
                x, y, z = best_spot
                bins[best_bin].append((x, y, z, w, h, d))
                bin_volumes[best_bin] += w * h * d
                placements[idx] = (best_bin, x, y, z, w, h, d)
            else:
                b_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                bin_volumes.append(w * h * d)
                placements[idx] = (b_idx, 0, 0, 0, w, h, d)

        return placements, bins

    def compute_score(placements):
        """Compute the objective score: B + 0.1 * (1 - last_bin_vol / container_vol)."""
        if not placements:
            return 0.0
        bin_ids = set(p[0] for p in placements if p is not None)
        if not bin_ids:
            return 0.0
        B = len(bin_ids)
        last_bin = max(bin_ids)
        last_bin_vol = 0.0
        for p in placements:
            if p is not None and p[0] == last_bin:
                last_bin_vol += p[4] * p[5] * p[6]
        return B + 0.1 * (1.0 - last_bin_vol / container_vol)

    def get_bin_boxes_from_placements(placements, bin_id):
        """Extract boxes from a given bin from placements."""
        return [p[1:7] for p in placements if p is not None and p[0] == bin_id]

    def try_consolidate_bin(placements, bin_to_remove):
        """Try to move all boxes from bin_to_remove into other bins."""
        boxes_in_bin = [i for i in range(n) if placements[i] is not None and placements[i][0] == bin_to_remove]
        if not boxes_in_bin:
            return placements

        # Sort boxes by volume descending for better packing
        boxes_in_bin.sort(key=lambda i: -volumes[i])

        work_placements = list(placements)
        work_bins = {}
        for b_id in set(p[0] for p in work_placements if p is not None):
            if b_id != bin_to_remove:
                work_bins[b_id] = get_bin_boxes_from_placements(work_placements, b_id)

        for box_idx in boxes_in_bin:
            w, h, d = box_dims[box_idx]
            placed = False
            for b_id in sorted(work_bins.keys()):
                b_boxes = work_bins[b_id]
                spot = find_spot(b_boxes, w, h, d)
                if spot is not None:
                    work_bins[b_id].append((spot[0], spot[1], spot[2], w, h, d))
                    work_placements[box_idx] = (b_id, spot[0], spot[1], spot[2], w, h, d)
                    placed = True
                    break
            if not placed:
                return None

        return work_placements

    def try_move_box(placements, box_idx, target_bin_id):
        """Try to move a single box to a target bin."""
        w, h, d = placements[box_idx][4], placements[box_idx][5], placements[box_idx][6]
        target_boxes = get_bin_boxes_from_placements(placements, target_bin_id)
        spot = find_spot(target_boxes, w, h, d)
        if spot is None:
            return None
        new_placements = list(placements)
        new_placements[box_idx] = (target_bin_id, spot[0], spot[1], spot[2], w, h, d)
        return new_placements

    def local_search(placements, current_score, max_iters=500):
        """Iterative local search to improve the packing."""
        improved = True
        iter_count = 0

        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() > deadline - 5:
                break

            last_bin_id = max(p[0] for p in placements)
            all_bin_ids = sorted(set(p[0] for p in placements))

            # Phase 1: Try to eliminate the last bin
            last_bin_boxes = [i for i in range(n) if placements[i] is not None and placements[i][0] == last_bin_id]
            if len(last_bin_boxes) <= 15:
                new_pl = try_consolidate_bin(placements, last_bin_id)
                if new_pl is not None:
                    new_score = compute_score(new_pl)
                    if new_score < current_score - 1e-12:
                        placements = new_pl
                        current_score = new_score
                        improved = True

            # Phase 2: Try moving individual boxes to consolidate
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                if placements[i] is None:
                    continue
                orig_bin = placements[i][0]
                if orig_bin == last_bin_id:
                    continue

                for tgt_bin in all_bin_ids:
                    if tgt_bin == orig_bin:
                        continue
                    new_pl = try_move_box(placements, i, tgt_bin)
                    if new_pl is not None:
                        new_score = compute_score(new_pl)
                        if new_score < current_score - 1e-12:
                            placements = new_pl
                            current_score = new_score
                            improved = True
                            all_bin_ids = sorted(set(p[0] for p in placements))
                            last_bin_id = max(p[0] for p in placements)
                            break

        return placements, current_score

    def is_valid(placements):
        """Verify the packing is valid."""
        if len(placements) != n:
            return False
        for p in placements:
            if p is None:
                return False
            b, x, y, z, w, h, d = p
            if b < 0 or x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
        # Check overlaps within each bin
        bin_map = defaultdict(list)
        for i, p in enumerate(placements):
            if p is not None:
                bin_map[p[0]].append((i, p))
        for bin_id, box_list in bin_map.items():
            for i in range(len(box_list)):
                for j in range(i + 1, len(box_list)):
                    idx_i, pi = box_list[i]
                    idx_j, pj = box_list[j]
                    xi, yi, zi, wi, hi, di = pi[1], pi[2], pi[3], pi[4], pi[5], pi[6]
                    xj, yj, zj, wj, hj, dj = pj[1], pj[2], pj[3], pj[4], pj[5], pj[6]
                    if (xi < xj + wj and xi + wi > xj and
                        yi < yj + hj and yi + hi > yj and
                        zi < zj + dj and zi + di > zj):
                        return False
        return True

    # ---- Generate multiple diverse initial packings ----
    best_placements = None
    best_score = float('inf')

    # Order 1: Volume descending
    order_vol = list(np.argsort(-volumes))
    if time.time() < deadline - 25:
        pl1, _ = initial_packing(order_vol)
        s1 = compute_score(pl1)
        if s1 < best_score:
            best_score = s1
            best_placements = pl1

    # Order 2: Longest side descending
    order_long = list(np.argsort(-np.array([max(w, h, d) for w, h, d in box_dims])))
    if time.time() < deadline - 25:
        pl2, _ = initial_packing(order_long)
        s2 = compute_score(pl2)
        if s2 < best_score:
            best_score = s2
            best_placements = pl2

    # Order 3: Surface area descending
    order_surf = list(np.argsort(-np.array([2*(w*h + w*d + h*d) for w, h, d in box_dims])))
    if time.time() < deadline - 25:
        pl3, _ = initial_packing(order_surf)
        s3 = compute_score(pl3)
        if s3 < best_score:
            best_score = s3
            best_placements = pl3

    # Order 4: Minimum side descending
    order_min = list(np.argsort(-np.array([min(w, h, d) for w, h, d in box_dims])))
    if time.time() < deadline - 25:
        pl4, _ = initial_packing(order_min)
        s4 = compute_score(pl4)
        if s4 < best_score:
            best_score = s4
            best_placements = pl4

    # Order 5-10: Random perturbations of volume order
    if time.time() < deadline - 35:
        random.seed(42)
        for trial in range(8):
            perturbed = volumes * np.random.uniform(0.7, 1.3, size=n)
            order_rand = list(np.argsort(-perturbed))
            plr, _ = initial_packing(order_rand)
            sr = compute_score(plr)
            if sr < best_score:
                best_score = sr
                best_placements = plr

    # ---- Local search on best solution ----
    if best_placements is not None and time.time() < deadline - 10:
        best_placements, best_score = local_search(best_placements, best_score, max_iters=1000)

    # ---- Try local search on other initial solutions if time permits ----
    if time.time() < deadline - 20:
        other_orders = [order_vol, order_long, order_surf, order_min]
        for order in other_orders:
            if time.time() > deadline - 20:
                break
            pl, _ = initial_packing(order)
            s = compute_score(pl)
            if s < best_score + 0.5:
                pl, s = local_search(pl, s, max_iters=500)
                if s < best_score:
                    best_score = s
                    best_placements = pl

    # ---- Final verification and fallback ----
    if best_placements is None or not is_valid(best_placements):
        # Fallback: Simple shelf packing
        best_placements = []
        b, x, y, z = 0, 0, 0, 0
        current_row_h = 0
        current_layer_d = 0
        for w, h, d in box_dims:
            if x + w > W:
                # New row
                x = 0
                y += current_row_h
                current_row_h = 0
            if y + h > H:
                # New layer
                y = 0
                z += current_layer_d
                x = 0
                current_row_h = 0
                current_layer_d = 0
            if z + d > D:
                # New bin
                b += 1
                x, y, z = 0, 0, 0
                current_row_h = 0
                current_layer_d = 0
            best_placements.append((b, x, y, z, w, h, d))
            x += w
            current_row_h = max(current_row_h, h)
            current_layer_d = max(current_layer_d, d)

    return best_placements
