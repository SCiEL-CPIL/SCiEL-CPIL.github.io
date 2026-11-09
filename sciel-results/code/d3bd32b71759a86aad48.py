import time
import random
import numpy as np
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin

    # Precompute unique orientations for each box
    box_orients = []
    for b in boxes:
        orients = set(permutations(b))
        box_orients.append(list(orients))

    best_vol = 0
    best_placements = None

    def solve(order):
        """
        Attempts to pack boxes in the given order.
        Returns (placements_list, total_volume).
        """
        grid = np.zeros((W, H, D), dtype=bool)
        placed = []  # List of (x, y, z, w, h, d)
        box_placements = [None] * n
        vol = 0
        
        for idx in order:
            w, h, d = boxes[idx]
            placed_box = False
            
            # Try all orientations
            for rw, rh, rd in box_orients[idx]:
                # Generate candidate positions
                # 1. Origin
                candidates = [(0, 0, 0)]
                
                # 2. Corners of all placed boxes
                for (px, py, pz, pw, ph, pd) in placed:
                    # 7 corners relative to the placed box
                    candidates.append((px + pw, py, pz))
                    candidates.append((px, py + ph, pz))
                    candidates.append((px, py, pz + pd))
                    candidates.append((px + pw, py + ph, pz))
                    candidates.append((px + pw, py, pz + pd))
                    candidates.append((px, py + ph, pz + pd))
                    candidates.append((px + pw, py + ph, pz + pd))
                
                # Filter and sort candidates
                # Prefer positions that are lower (smaller z, then y, then x) 
                # or compact (smaller sum). 
                # Standard "First Fit" uses (x, y, z) lexicographical order.
                # Let's try (x, y, z) to pack against walls.
                valid_candidates = []
                seen = set()
                for cx, cy, cz in candidates:
                    if 0 <= cx and cx + rw <= W and \
                       0 <= cy and cy + rh <= H and \
                       0 <= cz and cz + rd <= D:
                        if (cx, cy, cz) not in seen:
                            seen.add((cx, cy, cz))
                            valid_candidates.append((cx, cy, cz))
                
                # Sort by x, then y, then z (First Fit)
                valid_candidates.sort(key=lambda p: (p[0], p[1], p[2]))
                
                # Try to place
                for cx, cy, cz in valid_candidates:
                    # Check overlap using numpy grid
                    if not grid[cx:cx+rw, cy:cy+rh, cz:cz+rd].any():
                        # Place the box
                        grid[cx:cx+rw, cy:cy+rh, cz:cz+rd] = True
                        placed.append((cx, cy, cz, rw, rh, rd))
                        box_placements[idx] = (cx, cy, cz, rw, rh, rd)
                        vol += rw * rh * rd
                        placed_box = True
                        break
                
                if placed_box:
                    break
            
            if not placed_box:
                box_placements[idx] = None

        # Build result
        res = []
        for i in range(n):
            if box_placements[i] is not None:
                x, y, z, rw, rh, rd = box_placements[i]
                res.append((0, x, y, z, rw, rh, rd))
            else:
                w, h, d = boxes[i]
                res.append((-1, 0, 0, 0, w, h, d))
        
        return res, vol

    # Deterministic Orders
    # 1. Volume Desc
    order_vol = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    # 2. Max Dim Desc
    order_max = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    # 3. Min Dim Desc
    order_min = sorted(range(n), key=lambda i: min(boxes[i]), reverse=True)
    # 4. Surface Area Desc
    order_surf = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    # 5. Sum Dim Desc
    order_sum = sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True)

    orders = [order_vol, order_max, order_min, order_surf, order_sum]

    for order in orders:
        if time.time() > deadline:
            break
        res, vol = solve(order)
        if vol > best_vol:
            best_vol = vol
            best_placements = res

    # Random Restarts
    rng = random.Random(42)
    # We can also randomize the candidate sorting strategy slightly? 
    # Or just shuffle the box order.
    while time.time() < deadline:
        order = list(range(n))
        rng.shuffle(order)
        res, vol = solve(order)
        if vol > best_vol:
            best_vol = vol
            best_placements = res

    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        
    return best_placements
