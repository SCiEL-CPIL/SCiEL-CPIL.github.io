import time
import itertools
import random
import numpy as np

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D

    # Precompute unique rotations for each box
    rotations = []
    for w, h, d in boxes:
        rots = list(set(itertools.permutations((w, h, d))))
        rotations.append(rots)

    def calc_volume(results):
        vol = 0
        for r in results:
            if r[0] == 0:
                vol += r[4] * r[5] * r[6]
        return vol

    def try_placement(ordered_indices, use_dense_candidates=False):
        """
        Attempt to place boxes in the given order.
        If use_dense_candidates is True, generate more candidate positions for better packing.
        Returns a list of placements.
        """
        placed = []  # List of (x, y, z, w, h, d)
        results = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        candidates = {(0, 0, 0)}

        for idx in ordered_indices:
            if time.time() > deadline:
                return results, False

            w, h, d = boxes[idx]
            best_pos = None
            best_rot = None
            best_score = float('inf')

            for rw, rh, rd in rotations[idx]:
                # Check if box fits in container
                if rw > W or rh > H or rd > D:
                    continue

                # Generate candidate positions
                if use_dense_candidates:
                    # Dense candidate generation: combine x, y, z from placed boxes
                    x_opts = {0}
                    y_opts = {0}
                    z_opts = {0}
                    for px, py, pz, pw, ph, pd in placed:
                        x_opts.add(px)
                        x_opts.add(px + pw)
                        y_opts.add(py)
                        y_opts.add(py + ph)
                        z_opts.add(pz)
                        z_opts.add(pz + pd)
                    
                    # Limit the number of candidates to avoid explosion
                    x_opts = sorted(x_opts)[:20]
                    y_opts = sorted(y_opts)[:20]
                    z_opts = sorted(z_opts)[:20]
                    
                    cand_list = [(cx, cy, cz) for cx in x_opts for cy in y_opts for cz in z_opts]
                else:
                    cand_list = list(candidates)

                for cx, cy, cz in cand_list:
                    # Check bounds
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue

                    # Check overlap
                    valid = True
                    for px, py, pz, pw, ph, pd in placed:
                        if (cx < px + pw and px < cx + rw and
                            cy < py + ph and py < cy + rh and
                            cz < pz + pd and pz < cz + rd):
                            valid = False
                            break

                    if valid:
                        # Score: prioritize lower z, then lower y, then lower x
                        # Also add a small penalty for not being adjacent to existing boxes
                        score = cz * 1000000 + cy * 1000 + cx
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_rot = (rw, rh, rd)

            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                placed.append((bx, by, bz, bw, bh, bd))
                results[idx] = (0, bx, by, bz, bw, bh, bd)

                # Generate new candidate points for the sparse set
                # 1. Top
                candidates.add((bx, by, bz + bd))
                # 2. Right
                candidates.add((bx + bw, by, bz))
                # 3. Front
                candidates.add((bx, by + bh, bz))
                # 4. Top-Right
                candidates.add((bx + bw, by, bz + bd))
                # 5. Top-Front
                candidates.add((bx, by + bh, bz + bd))
                # 6. Right-Front
                candidates.add((bx + bw, by + bh, bz))
                # 7. Top-Right-Front
                candidates.add((bx + bw, by + bh, bz + bd))

        return results, True

    def final_insertion_pass(results, placed_list):
        """
        Attempt to insert unplaced boxes into the existing placement.
        Uses dense candidate positions.
        Returns updated results and placed list.
        """
        if time.time() > deadline:
            return results, placed_list

        # Identify unplaced boxes
        unplaced = [i for i in range(n) if results[i][0] == -1]
        if not unplaced:
            return results, placed_list

        # Sort unplaced by volume descending (try to fit big ones first)
        unplaced.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)

        new_placed = list(placed_list)
        new_results = list(results)

        for idx in unplaced:
            if time.time() > deadline:
                break

            w, h, d = boxes[idx]
            best_pos = None
            best_rot = None
            best_score = float('inf')

            for rw, rh, rd in rotations[idx]:
                if rw > W or rh > H or rd > D:
                    continue

                # Generate dense candidates based on current placed boxes
                x_opts = {0}
                y_opts = {0}
                z_opts = {0}
                for px, py, pz, pw, ph, pd in new_placed:
                    x_opts.add(px)
                    x_opts.add(px + pw)
                    y_opts.add(py)
                    y_opts.add(py + ph)
                    z_opts.add(pz)
                    z_opts.add(pz + pd)

                # Limit candidates
                x_opts = sorted(x_opts)[:30]
                y_opts = sorted(y_opts)[:30]
                z_opts = sorted(z_opts)[:30]

                for cx in x_opts:
                    for cy in y_opts:
                        for cz in z_opts:
                            if cx + rw > W or cy + rh > H or cz + rd > D:
                                continue

                            valid = True
                            for px, py, pz, pw, ph, pd in new_placed:
                                if (cx < px + pw and px < cx + rw and
                                    cy < py + ph and py < cy + rh and
                                    cz < pz + pd and pz < cz + rd):
                                    valid = False
                                    break

                            if valid:
                                score = cz * 1000000 + cy * 1000 + cx
                                if score < best_score:
                                    best_score = score
                                    best_pos = (cx, cy, cz)
                                    best_rot = (rw, rh, rd)

            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                new_placed.append((bx, by, bz, bw, bh, bd))
                new_results[idx] = (0, bx, by, bz, bw, bh, bd)

        return new_results, new_placed

    best_results = None
    best_vol = 0

    indices = list(range(n))

    # Strategy 1: Sort by volume (descending)
    sort_keys = [
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2],  # Volume
        lambda i: max(boxes[i]),                            # Max dim
        lambda i: boxes[i][0] * boxes[i][1] + boxes[i][1] * boxes[i][2] + boxes[i][0] * boxes[i][2],  # Surface area
        lambda i: min(boxes[i]),                            # Min dim (ascending)
        lambda i: -min(boxes[i]),                           # Min dim (descending)
    ]

    for sort_key in sort_keys:
        if time.time() > deadline:
            break

        sorted_indices = sorted(indices, key=sort_key, reverse=True)
        results, completed = try_placement(sorted_indices)

        if completed:
            # Try final insertion pass
            placed_list = [(r[1], r[2], r[3], r[4], r[5], r[6]) for r in results if r[0] == 0]
            results2, placed2 = final_insertion_pass(results, placed_list)
            vol = calc_volume(results2)
            if vol > best_vol:
                best_vol = vol
                best_results = results2

    # Strategy 2: Random restarts
    if best_results is not None and time.time() < deadline - 10.0:
        attempts = 0
        while time.time() < deadline - 5.0 and attempts < 30:
            random.shuffle(indices)
            results, completed = try_placement(indices)
            attempts += 1

            if completed:
                # Try final insertion pass
                placed_list = [(r[1], r[2], r[3], r[4], r[5], r[6]) for r in results if r[0] == 0]
                results2, placed2 = final_insertion_pass(results, placed_list)
                vol = calc_volume(results2)
                if vol > best_vol:
                    best_vol = vol
                    best_results = results2

    # Fallback
    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_results
