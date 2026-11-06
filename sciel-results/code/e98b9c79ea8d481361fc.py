"""Multi-restart local search with interval-based spatial index for 3D packing."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # Precompute orientations and volumes
    all_orients = []
    all_vols = []
    for i in range(n):
        w, h, d = boxes[i]
        orients = list(set(itertools.permutations((w, h, d))))
        all_orients.append(orients)
        all_vols.append(w * h * d)

    # Spatial Index using Interval Trees (simplified via sorted lists + binary search)
    # We maintain separate lists of placed boxes sorted by x, y, z start positions.
    # For a query box (x, y, z, w, h, d), a placed box (px, py, pz, pw, ph, pd) overlaps
    # if NOT (x+w <= px OR px+pw <= x OR y+h <= py OR py+ph <= y OR z+d <= pz OR pz+pd <= z)
    # This is equivalent to:
    # px < x+w AND px+pw > x AND py < y+h AND py+ph > y AND pz < z+d AND pz+pd > z

    # To speed up, we can filter by one dimension first.
    # Let's use a simple approach: maintain a list of placed boxes.
    # For 60 boxes, O(N) overlap check is ~3600 ops per box placement.
    # 60 boxes * 3600 = 216k ops per full pack.
    # 100 restarts * 216k = 21.6M ops. Python can do ~10-50M simple ops/sec.
    # So 100 restarts might take 0.5 - 2 seconds. We can afford many more.

    def try_pack(order):
        placed = []  # List of (x, y, z, w, h, d)
        result = [(-1, 0, 0, 0, 0, 0, 0)] * n
        filled_vol = 0

        for idx in order:
            w, h, d = boxes[idx]
            orients = all_orients[idx]
            
            # Candidate positions: (0,0,0) and faces of existing boxes
            # To reduce candidates, we only check a subset or use a smarter anchor generation
            # Standard BLF (Bottom-Left-Far) or similar heuristics
            anchors = set()
            anchors.add((0, 0, 0))
            for (px, py, pz, pw, ph, pd) in placed:
                anchors.add((px + pw, py, pz))
                anchors.add((px, py + ph, pz))
                anchors.add((px, py, pz + pd))
                # Corners
                anchors.add((px + pw, py + ph, pz))
                anchors.add((px + pw, py, pz + pd))
                anchors.add((px, py + ph, pz + pd))
                anchors.add((px + pw, py + ph, pz + pd))
            
            # Filter anchors that are inside container
            valid_anchors = []
            for (ax, ay, az) in anchors:
                if ax < W and ay < H and az < D:
                    valid_anchors.append((ax, ay, az))
            
            # Sort anchors by heuristic: bottom (z), then left (x), then front (y)
            # This encourages filling from bottom-up, left-right
            valid_anchors.sort(key=lambda a: (a[2], a[0], a[1]))

            best_pos = None
            best_orient = None
            best_score = -1

            for (ax, ay, az) in valid_anchors:
                for (bw, bh, bd) in orients:
                    if ax + bw > W or ay + bh > H or az + bd > D:
                        continue
                    
                    # Check overlap
                    overlap = False
                    for (px, py, pz, pw, ph, pd) in placed:
                        # Separation check
                        if (ax + bw <= px or px + pw <= ax or
                            ay + bh <= py or py + ph <= ay or
                            az + bd <= pz or pz + pd <= az):
                            continue
                        else:
                            overlap = True
                            break
                    
                    if not overlap:
                        # Heuristic score: prefer lower z, then lower x, then lower y
                        # This is consistent with the sort, but we break ties by orientation size?
                        # Let's just take the first valid one found in the sorted order.
                        # However, different orientations might fit in the same spot.
                        # We want to maximize volume utilization locally?
                        # Actually, just taking the first valid anchor/orient in sorted order is a standard greedy.
                        # To improve, we can score the placement.
                        # Score = - (az * 1000 + ax * 100 + ay)  # Prefer lower coords
                        # But since anchors are sorted, the first one is the "best" coordinate-wise.
                        # We should pick the orientation that leaves the most space? Hard to know.
                        # Let's stick to first valid.
                        best_pos = (ax, ay, az)
                        best_orient = (bw, bh, bd)
                        break
                if best_pos is not None:
                    break

            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                placed.append((bx, by, bz, bw, bh, bd))
                result[idx] = (0, bx, by, bz, bw, bh, bd)
                filled_vol += bw * bh * bd
            else:
                result[idx] = (-1, 0, 0, 0, w, h, d)

        return filled_vol, result

    # Generate initial orders
    indexed_boxes = list(range(n))
    
    # Strategy 1: Volume Descending
    order_vol = sorted(indexed_boxes, key=lambda i: all_vols[i], reverse=True)
    
    # Strategy 2: Max Dimension Descending
    order_max = sorted(indexed_boxes, key=lambda i: max(boxes[i]), reverse=True)
    
    # Strategy 3: Aspect Ratio Descending
    order_aspect = sorted(indexed_boxes, key=lambda i: max(boxes[i]) / min(boxes[i]) if min(boxes[i]) > 0 else 0, reverse=True)
    
    # Strategy 4: Surface Area Descending
    order_surf = sorted(indexed_boxes, key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    
    best_vol = -1
    best_result = None

    # Try deterministic strategies first
    for order in [order_vol, order_max, order_aspect, order_surf]:
        if time.time() > deadline:
            break
        vol, res = try_pack(order)
        if vol > best_vol:
            best_vol = vol
            best_result = res

    # Randomized restarts
    # We can also try to "kick" boxes that were left out or swap them?
    # For now, just random shuffles of the sorted lists.
    
    random.seed(42)
    
    # Estimate time per pack
    # If we have time left, do random restarts.
    # Let's run as many as possible within the deadline.
    
    while time.time() < deadline - 5.0:  # 5s safety margin for final return
        # Create a randomized order
        # Mix of volume sort and random
        # 50% chance pure random, 50% chance volume-sorted with random swaps
        r = random.random()
        if r < 0.5:
            order = indexed_boxes[:]
            random.shuffle(order)
        else:
            order = order_vol[:]
            # Randomly swap some elements
            for _ in range(n // 2):
                i, j = random.sample(range(n), 2)
                order[i], order[j] = order[j], order[i]
        
        vol, res = try_pack(order)
        if vol > best_vol:
            best_vol = vol
            best_result = res
            
        # Optional: Local Search / Improvement
        # If we found a good result, we could try to insert the left-out boxes into the remaining space?
        # The try_pack function already tries to place all boxes in the order.
        # Boxes left out are those that didn't fit.
        # A more advanced strategy: After a pack, look at the "holes" and see if any left-out box fits.
        # But try_pack is greedy.
        
        # To speed up random restarts, we can skip if the volume is clearly worse?
        # No, we don't know until we finish.

    if best_result is None:
        # Fallback: simple shelf packing
        placements = []
        x, y, z = 0, 0, 0
        row_h = 0
        layer_d = 0
        for w, h, d in boxes:
            if x + w > W:
                x = 0
                y += row_h
                row_h = 0
            if y + h > H:
                y = 0
                z += layer_d
                layer_d = 0
                x = 0
                row_h = 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x += w
            row_h = max(row_h, h)
            layer_d = max(layer_d, d)
        best_result = placements

    return best_result
# EVOLVE-BLOCK-END
