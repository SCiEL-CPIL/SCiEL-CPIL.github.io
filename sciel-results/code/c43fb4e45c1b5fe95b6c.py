"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import bisect
import heapq

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]

    container_vol = W * H * D
    if container_vol <= 0:
        return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]

    n = len(boxes)

    # Precompute all valid orientations for each box
    box_orientations = []
    for w, h, d in boxes:
        orientations = set()
        for perm in itertools.permutations((w, h, d)):
            orientations.add(perm)
        box_orientations.append(list(orientations))

    def try_placement(box_order, orientation_strategy='best_fit'):
        """
        Try to place boxes in the given order using free placement with candidate generation.
        """
        placements = [None] * n
        placed_count = 0
        total_vol = 0

        placed_boxes = []  # list of (x, y, z, w, h, d)

        def can_place(x, y, z, w, h, d):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            # Check overlap with all placed boxes using early exit
            for (px, py, pz, pw, ph, pd) in placed_boxes:
                # Quick rejection checks
                if x >= px + pw or px >= x + w:
                    continue
                if y >= py + ph or py >= y + h:
                    continue
                if z >= pz + pd or pz >= z + d:
                    continue
                return False
            return True

        def find_position(w, h, d):
            """Find a valid position for a box with dimensions w, h, d."""
            candidates = set()
            candidates.add((0, 0, 0))

            # Generate extreme points (corners) from placed boxes
            # This is O(N) candidates instead of O(N^2)
            for (px, py, pz, pw, ph, pd) in placed_boxes:
                # 8 corners
                for cx in [px, px + pw]:
                    if cx < 0 or cx + w > W: continue
                    for cy in [py, py + ph]:
                        if cy < 0 or cy + h > H: continue
                        for cz in [pz, pz + pd]:
                            if cz < 0 or cz + d > D: continue
                            candidates.add((cx, cy, cz))
                
                # Also consider "top" and "side" alignments for better packing
                if px + pw < W:
                    candidates.add((px + pw, py, pz))
                    candidates.add((px + pw, py, pz + pd))
                if py + ph < H:
                    candidates.add((px, py + ph, pz))
                    candidates.add((px, py + ph, pz + pd))
                if pz + pd < D:
                    candidates.add((px, py, pz + pd))
                    candidates.add((px + pw, py, pz + pd))

            valid_candidates = []
            for (cx, cy, cz) in candidates:
                if can_place(cx, cy, cz, w, h, d):
                    valid_candidates.append((cx, cy, cz))

            if not valid_candidates:
                return None

            # Score: prefer lower z, then y, then x (gravity/compactness)
            valid_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
            return valid_candidates[0]

        for idx in box_order:
            if time.time() > deadline:
                break

            w_orig, h_orig, d_orig = boxes[idx]

            if orientation_strategy == 'best_fit':
                # Try all orientations, pick the one that allows placement with lowest position
                best_orientation = None
                best_pos = None
                best_pos_score = None
                for (ow, oh, od) in box_orientations[idx]:
                    pos = find_position(ow, oh, od)
                    if pos is not None:
                        # Score the position
                        pos_score_val = (pos[2], pos[1], pos[0])
                        if best_pos_score is None or pos_score_val < best_pos_score:
                            best_pos_score = pos_score_val
                            best_pos = pos
                            best_orientation = (ow, oh, od)
                if best_pos is not None:
                    ow, oh, od = best_orientation
                    placed_boxes.append((best_pos[0], best_pos[1], best_pos[2], ow, oh, od))
                    placements[idx] = (0, best_pos[0], best_pos[1], best_pos[2], ow, oh, od)
                    placed_count += 1
                    total_vol += ow * oh * od
            else:
                # Use first orientation
                ow, oh, od = box_orientations[idx][0]
                pos = find_position(ow, oh, od)
                if pos is not None:
                    placed_boxes.append((pos[0], pos[1], pos[2], ow, oh, od))
                    placements[idx] = (0, pos[0], pos[1], pos[2], ow, oh, od)
                    placed_count += 1
                    total_vol += ow * oh * od

        # Fill in unplaced boxes
        for i in range(n):
            if placements[i] is None:
                placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

        return placements, total_vol

    def shelf_packing(box_order, use_rotation=True):
        """
        Shelf-based packing: boxes are placed on horizontal shelves.
        """
        placements = [None] * n
        placed_count = 0
        total_vol = 0

        shelves = []  # list of shelf dicts

        for idx in box_order:
            if time.time() > deadline:
                break

            w_orig, h_orig, d_orig = boxes[idx]

            if not use_rotation:
                orientations = [(w_orig, h_orig, d_orig)]
            else:
                orientations = box_orientations[idx]

            best_shelf_idx = None
            best_pos = None
            best_orientation = None
            best_score = None

            for (ow, oh, od) in orientations:
                # Try to place on an existing shelf
                for s_idx, s in enumerate(shelves):
                    # Shelf height must accommodate this box
                    if oh <= s['h'] and s['x'] + ow <= W and od <= D:
                        pos = (s['x'], 0, s['z'])
                        # Score: prefer lower z, then lower x
                        score = (s['z'], s['x'])
                        if best_score is None or score < best_score:
                            best_score = score
                            best_shelf_idx = s_idx
                            best_pos = pos
                            best_orientation = (ow, oh, od)

                # Try to create a new shelf
                if oh <= H:
                    # Find the minimum z where a new shelf of height oh can fit
                    min_z = 0
                    for s in shelves:
                        if s['z'] + s['h'] > min_z:
                            min_z = s['z'] + s['h']

                    if min_z + oh <= H:
                        pos = (0, 0, min_z)
                        score = (min_z, 0)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_shelf_idx = -1  # signal to create new shelf
                            best_pos = pos
                            best_orientation = (ow, oh, od)

            if best_shelf_idx is not None and best_pos is not None:
                ow, oh, od = best_orientation
                if best_shelf_idx >= 0:
                    # Place on existing shelf
                    shelf = shelves[best_shelf_idx]
                    shelf['x'] += ow
                else:
                    # Create new shelf
                    new_shelf = {'z': best_pos[2], 'h': oh, 'x': ow, 'boxes': []}
                    shelves.append(new_shelf)

                placements[idx] = (0, best_pos[0], best_pos[1], best_pos[2], ow, oh, od)
                placed_count += 1
                total_vol += ow * oh * od

        for i in range(n):
            if placements[i] is None:
                placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

        return placements, total_vol

    def layer_packing(box_order, use_rotation=True):
        """
        Layer-based packing similar to the inspiration.
        """
        placements = [None] * n
        placed_count = 0
        total_vol = 0

        x, y, z, row_h, layer_d = 0, 0, 0, 0, 0

        for idx in box_order:
            if time.time() > deadline:
                break

            w_orig, h_orig, d_orig = boxes[idx]

            if not use_rotation:
                orientations = [(w_orig, h_orig, d_orig)]
            else:
                orientations = box_orientations[idx]

            best_orientation = None
            best_x, best_y, best_z = None, None, None
            best_row_h, best_layer_d = None, None
            best_score = None

            for (ow, oh, od) in orientations:
                # Simulate placement
                sx, sy, sz = x, y, z
                srow_h, slayer_d = row_h, layer_d

                if sx + ow > W:
                    sx, sy, srow_h = 0, sy + srow_h, 0
                if sy + oh > H:
                    sy, sz, slayer_d, sx, srow_h = 0, sz + slayer_d, 0, 0, 0
                if sz + od > D:
                    continue  # Doesn't fit

                # This orientation fits. Score it.
                score = (sz, sy, sx)
                if best_score is None or score < best_score:
                    best_score = score
                    best_orientation = (ow, oh, od)
                    best_x, best_y, best_z = sx, sy, sz
                    best_row_h, best_layer_d = srow_h, slayer_d

            if best_orientation is not None:
                ow, oh, od = best_orientation
                placements[idx] = (0, best_x, best_y, best_z, ow, oh, od)
                placed_count += 1
                total_vol += ow * oh * od

                # Update state
                x = best_x + ow
                y = best_y
                z = best_z
                row_h = max(best_row_h, oh)
                layer_d = max(best_layer_d, od)

        for i in range(n):
            if placements[i] is None:
                placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

        return placements, total_vol

    # max_cubes_packing removed due to inefficiency

    # Define sorting strategies
    strategies = []
    
    # 1. Volume Descending
    strategies.append(sorted(range(n), key=lambda i: -(boxes[i][0] * boxes[i][1] * boxes[i][2])))
    # 2. Max Dimension Descending
    strategies.append(sorted(range(n), key=lambda i: -max(boxes[i])))
    # 3. Min Dimension Ascending
    strategies.append(sorted(range(n), key=lambda i: min(boxes[i])))
    # 4. Max Dimension Ascending
    strategies.append(sorted(range(n), key=lambda i: max(boxes[i])))
    # 5. Volume Ascending
    strategies.append(sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2]))
    # 6. Surface Area Descending
    strategies.append(sorted(range(n), key=lambda i: -(2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]))))
    # 7. Aspect Ratio (Max/Min) Descending
    strategies.append(sorted(range(n), key=lambda i: -(max(boxes[i]) / max(1, min(boxes[i])))))
    # 8. Aspect Ratio (Max/Min) Ascending
    strategies.append(sorted(range(n), key=lambda i: (max(boxes[i]) / max(1, min(boxes[i])))))
    # 9. Second-largest dimension descending
    strategies.append(sorted(range(n), key=lambda i: -sorted(boxes[i], reverse=True)[1]))
    # 10. Perimeter descending (2*(w+h+d))
    strategies.append(sorted(range(n), key=lambda i: -(2*(boxes[i][0]+boxes[i][1]+boxes[i][2]))))
    # 11. Volume / Max Dimension (compactness)
    strategies.append(sorted(range(n), key=lambda i: -(boxes[i][0] * boxes[i][1] * boxes[i][2] / max(1, max(boxes[i])))))
    # 12. Min Dimension Descending
    strategies.append(sorted(range(n), key=lambda i: -min(boxes[i])))

    best_placements = None
    best_vol = 0

    # Run deterministic strategies with Free Placement
    for order in strategies:
        if time.time() > deadline:
            break
        result = try_placement(order, 'best_fit')
        if result:
            _, vol = result
            if vol > best_vol:
                best_vol = vol
                best_placements = result[0]

    # Run Shelf Packing with top strategies
    for order in strategies[:6]:
        if time.time() > deadline:
            break
        result = shelf_packing(order, True)
        if result:
            _, vol = result
            if vol > best_vol:
                best_vol = vol
                best_placements = result[0]

    # Run Layer Packing with top strategies
    for order in strategies[:6]:
        if time.time() > deadline:
            break
        result = layer_packing(order, True)
        if result:
            _, vol = result
            if vol > best_vol:
                best_vol = vol
                best_placements = result[0]

    # Run Max Cubes Packing removed

    # Local search: try to improve the best placement by moving boxes
    def local_search(placements, placed_boxes, total_vol, max_iterations=50):
        """Try to improve placement by moving boxes to better positions."""
        if time.time() > deadline:
            return placements, placed_boxes, total_vol
        
        improved = True
        iteration = 0
        while improved and iteration < max_iterations and time.time() < deadline - 5.0:
            improved = False
            iteration += 1
            
            # Try to move each placed box to a better position
            for i in range(len(placed_boxes) - 1, -1, -1):
                if time.time() > deadline:
                    break
                
                px, py, pz, pw, ph, pd = placed_boxes[i]
                
                # Find which box index this is
                box_idx = None
                for idx in range(n):
                    if placements[idx] is not None and placements[idx][0] == 0:
                        if (placements[idx][1], placements[idx][2], placements[idx][3]) == (px, py, pz):
                            box_idx = idx
                            break
                
                if box_idx is None:
                    continue
                
                # Try to find a better position for this box
                best_new_pos = None
                best_score = (pz, py, px)
                
                # Generate candidate positions
                candidates = set()
                candidates.add((0, 0, 0))
                
                for (qx, qy, qz, qw, qh, qd) in placed_boxes:
                    if (qx, qy, qz, qw, qh, qd) == (px, py, pz, pw, ph, pd):
                        continue
                    for cx in [qx, qx + qw]:
                        for cy in [qy, qy + qh]:
                            for cz in [qz, qz + qd]:
                                if 0 <= cx < W and 0 <= cy < H and 0 <= cz < D:
                                    candidates.add((cx, cy, cz))
                
                # Try each candidate position
                for (cx, cy, cz) in candidates:
                    # Temporarily remove this box
                    placed_boxes[i] = None
                    placed_boxes_filtered = [b for b in placed_boxes if b is not None]
                    
                    # Check if new position is valid
                    valid = True
                    if cx + pw > W or cy + ph > H or cz + pd > D:
                        valid = False
                    else:
                        for (qx, qy, qz, qw, qh, qd) in placed_boxes_filtered:
                            if not (cx >= qx + qw or qx >= cx + pw):
                                if not (cy >= qy + qh or qy >= cy + ph):
                                    if not (cz >= qz + qd or qz >= cz + pd):
                                        valid = False
                                        break
                    
                    # Restore box
                    placed_boxes[i] = (px, py, pz, pw, ph, pd)
                    
                    if valid and (cz, cy, cx) < best_score:
                        best_score = (cz, cy, cx)
                        best_new_pos = (cx, cy, cz)
                
                # If we found a better position, move the box
                if best_new_pos is not None:
                    placed_boxes[i] = (best_new_pos[0], best_new_pos[1], best_new_pos[2], pw, ph, pd)
                    placements[box_idx] = (0, best_new_pos[0], best_new_pos[1], best_new_pos[2], pw, ph, pd)
                    improved = True
        
        return placements, placed_boxes, total_vol

    # Apply local search to the best placement found so far
    if best_placements is not None and time.time() < deadline - 10.0:
        # Reconstruct placed_boxes from best_placements
        placed_boxes = []
        for i in range(n):
            if best_placements[i][0] == 0:
                _, x, y, z, w, h, d = best_placements[i]
                placed_boxes.append((x, y, z, w, h, d))
        
        best_placements, placed_boxes, best_vol = local_search(best_placements, placed_boxes, best_vol)

    # Randomized Restart with Free Placement
    rng = random.Random(42)
    random_trials = 0
    max_random_trials = 50 # Reduced count, each trial is faster now

    while time.time() < deadline - 10.0 and random_trials < max_random_trials:
        # Mix of fully random and partially sorted orders
        if random_trials % 2 == 0:
            order = list(range(n))
            rng.shuffle(order)
        else:
            # Start with large boxes, then randomize the rest
            order = sorted(range(n), key=lambda i: -(boxes[i][0] * boxes[i][1] * boxes[i][2]))
            split = n // 2
            tail = order[split:]
            rng.shuffle(tail)
            order = order[:split] + tail

        result = try_placement(order, 'best_fit')
        if result:
            _, vol = result
            if vol > best_vol:
                best_vol = vol
                best_placements = result[0]
        random_trials += 1

    # Final local search on the best result
    if best_placements is not None and time.time() < deadline - 5.0:
        # Reconstruct placed_boxes from best_placements
        placed_boxes = []
        for i in range(n):
            if best_placements[i][0] == 0:
                _, x, y, z, w, h, d = best_placements[i]
                placed_boxes.append((x, y, z, w, h, d))
        
        # Limit local search iterations to ensure we don't timeout
        best_placements, placed_boxes, best_vol = local_search(best_placements, placed_boxes, best_vol, max_iterations=20)

    if best_placements is None:
        return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]

    return best_placements

# EVOLVE-BLOCK-END
