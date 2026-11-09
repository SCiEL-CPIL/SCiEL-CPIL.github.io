"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]

    container_vol = W * H * D
    if container_vol <= 0:
        return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]

    n = len(boxes)
    if n == 0:
        return []

    # Precompute orientations
    box_orientations = []
    for w, h, d in boxes:
        orientations = set()
        for perm in itertools.permutations((w, h, d)):
            orientations.add(perm)
        box_orientations.append(list(orientations))

    # Sorting strategies
    strategies = [
        lambda i: -(boxes[i][0] * boxes[i][1] * boxes[i][2]),  # Volume desc
        lambda i: -max(boxes[i]),                              # Max dim desc
        lambda i: min(boxes[i]),                               # Min dim asc
        lambda i: boxes[i][0] * boxes[i][1],                   # Area desc
        lambda i: -min(boxes[i]),                              # Min dim desc
        lambda i: boxes[i][0] + boxes[i][1] + boxes[i][2],     # Sum asc
        lambda i: max(boxes[i]) / min(boxes[i]) if min(boxes[i]) > 0 else 0, # Aspect ratio
    ]

    def try_placement(box_order, seed=0):
        """
        Place boxes using extreme point candidate generation and a refined scoring function.
        """
        placements = [None] * n
        placed_boxes = []  # (x, y, z, w, h, d)
        total_vol = 0
        
        # Track bounding box of placed items to penalize expansion
        max_x = 0
        max_y = 0
        max_z = 0

        def can_place(x, y, z, w, h, d):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            for (px, py, pz, pw, ph, pd) in placed_boxes:
                if x < px + pw and x + w > px:
                    if y < py + ph and y + h > py:
                        if z < pz + pd and z + d > pz:
                            return False
            return True

        def generate_candidates(w, h, d):
            """Generate extreme point candidates."""
            candidates = set()
            candidates.add((0, 0, 0))
            
            for (px, py, pz, pw, ph, pd) in placed_boxes:
                # 6 faces
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px, py, pz + pd))
                if px > 0: candidates.add((px - w, py, pz))
                if py > 0: candidates.add((px, py - h, pz))
                if pz > 0: candidates.add((px, py, pz - d))
                
                # 12 edges (corners of the box)
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
                
                # Additional gap-filling candidates
                if px > 0:
                    candidates.add((px - w, py + ph, pz))
                    candidates.add((px - w, py, pz + pd))
                if py > 0:
                    candidates.add((px + pw, py - h, pz))
                    candidates.add((px, py - h, pz + pd))
                if pz > 0:
                    candidates.add((px + pw, py, pz - d))
                    candidates.add((px, py + ph, pz - d))

            valid = []
            for (cx, cy, cz) in candidates:
                if cx < 0 or cy < 0 or cz < 0:
                    continue
                if cx + w > W or cy + h > H or cz + d > D:
                    continue
                if can_place(cx, cy, cz, w, h, d):
                    valid.append((cx, cy, cz))
            
            return valid

        def score_position(pos, w, h, d):
            """
            Score a position. Lower is better.
            1. Compactness: Minimize x+y+z
            2. Bounding Box Penalty: Minimize expansion of max_x, max_y, max_z
            """
            x, y, z = pos
            
            # 1. Compactness score
            compact_score = x + y + z
            
            # 2. Bounding Box Expansion
            new_max_x = max(max_x, x + w)
            new_max_y = max(max_y, y + h)
            new_max_z = max(max_z, z + d)
            
            # Penalize the increase in the bounding box volume
            # Or simply the sum of increases
            increase_x = max(0, new_max_x - max_x)
            increase_y = max(0, new_max_y - max_y)
            increase_z = max(0, new_max_z - max_z)
            
            bbox_penalty = increase_x + increase_y + increase_z
            
            # Combine: Prioritize compactness, but heavily penalize bounding box growth
            # The weight for bbox_penalty can be tuned. 
            # A large bbox means more wasted space in the corners.
            total_score = compact_score + 0.5 * bbox_penalty
            
            return total_score

        for idx in box_order:
            if time.time() > deadline:
                break

            w_orig, h_orig, d_orig = boxes[idx]
            best_orientation = None
            best_pos = None
            best_score = float('inf')

            for (ow, oh, od) in box_orientations[idx]:
                candidates = generate_candidates(ow, oh, od)
                if not candidates:
                    continue
                
                for pos in candidates:
                    s = score_position(pos, ow, oh, od)
                    if s < best_score:
                        best_score = s
                        best_pos = pos
                        best_orientation = (ow, oh, od)
                
                # Early break if we found a "perfect" fit at origin? No, keep checking other orientations for better fit?
                # Actually, if we found a position, other orientations might offer a lower score. 
                # So we must check all orientations for the best score.

            if best_pos is not None:
                ow, oh, od = best_orientation
                placed_boxes.append((best_pos[0], best_pos[1], best_pos[2], ow, oh, od))
                placements[idx] = (0, best_pos[0], best_pos[1], best_pos[2], ow, oh, od)
                total_vol += ow * oh * od
                
                # Update bounding box
                max_x = max(max_x, best_pos[0] + ow)
                max_y = max(max_y, best_pos[1] + oh)
                max_z = max(max_z, best_pos[2] + od)

        # Fill in unplaced boxes
        for i in range(n):
            if placements[i] is None:
                placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

        return placements, total_vol

    def shelf_packing(box_order, use_rotation=True):
        """Shelf-based packing as a fallback/alternative."""
        placements = [None] * n
        total_vol = 0
        shelves = []  # (z, height, current_x)

        for idx in box_order:
            if time.time() > deadline:
                break

            w_orig, h_orig, d_orig = boxes[idx]
            orientations = box_orientations[idx] if use_rotation else [(w_orig, h_orig, d_orig)]

            best_shelf_idx = None
            best_pos = None
            best_orientation = None
            best_score = None

            for (ow, oh, od) in orientations:
                # Try existing shelves
                for si, (sz, sh, sx) in enumerate(shelves):
                    if sh == oh and sx + ow <= W and od <= D:
                        pos = (sx, 0, sz)
                        pos_score = (sz, sx)
                        if best_score is None or pos_score < best_score:
                            best_score = pos_score
                            best_shelf_idx = si
                            best_pos = pos
                            best_orientation = (ow, oh, od)
                
                # Try new shelf
                if oh <= H:
                    min_z = 0
                    for (sz, sh, sx) in shelves:
                        if sz + sh > min_z:
                            min_z = sz + sh
                    if min_z + oh <= H:
                        pos = (0, 0, min_z)
                        pos_score = (min_z, 0)
                        if best_score is None or pos_score < best_score:
                            best_score = pos_score
                            best_shelf_idx = len(shelves)
                            best_pos = pos
                            best_orientation = (ow, oh, od)

            if best_pos is not None and best_orientation is not None:
                ow, oh, od = best_orientation
                if best_shelf_idx < len(shelves):
                    sz, sh, sx = shelves[best_shelf_idx]
                    shelves[best_shelf_idx] = (sz, sh, sx + ow)
                else:
                    shelves.append((best_pos[2], oh, ow))
                
                placements[idx] = (0, best_pos[0], best_pos[1], best_pos[2], ow, oh, od)
                total_vol += ow * oh * od

        for i in range(n):
            if placements[i] is None:
                placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

        return placements, total_vol

    best_placements = None
    best_vol = 0

    # 1. Deterministic Strategies
    for s_idx, strategy in enumerate(strategies):
        if time.time() > deadline:
            break
        box_order = sorted(range(n), key=strategy)
        result = try_placement(box_order, seed=s_idx)
        if result:
            vol = result[1]
            if vol > best_vol:
                best_vol = vol
                best_placements = result[0]

    # 2. Shelf Packing
    for s_idx, strategy in enumerate(strategies[:3]):
        if time.time() > deadline:
            break
        box_order = sorted(range(n), key=strategy)
        result = shelf_packing(box_order, True)
        if result:
            vol = result[1]
            if vol > best_vol:
                best_vol = vol
                best_placements = result[0]

    # 3. Random Restarts
    rng = random.Random(42)
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # Estimate time per restart
        # We'll just run as many as we can
        restarts = 0
        while time.time() < deadline - 2.0:
            order = list(range(n))
            # Mix of sorting and shuffling
            r = rng.random()
            if r < 0.3:
                # Volume weighted random sort
                weights = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
                # Simple random shuffle with bias
                rng.shuffle(order)
            elif r < 0.6:
                # Random shuffle
                rng.shuffle(order)
            else:
                # Sorted by random key
                rng.shuffle(order)
                order.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2] * rng.random())
            
            result = try_placement(order, seed=restarts)
            if result:
                vol = result[1]
                if vol > best_vol:
                    best_vol = vol
                    best_placements = result[0]
            restarts += 1

    if best_placements is None:
        return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]

    return best_placements

# EVOLVE-BLOCK-END
