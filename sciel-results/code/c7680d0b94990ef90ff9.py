"""
Strategy: Skyline-based 3D packing with multi-heuristic ordering and local search.
Uses a 2D height-map (skyline) representation for fast overlap checks and placement,
combined with multiple greedy ordering strategies and a limited local search phase.
"""
import time
import itertools
import random
import math

# EVOLVE-BLOCK-START

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 115.0  # 5s safety margin for 120s limit

    # Precompute orientations for each box
    all_orients = []
    for i in range(n):
        w, h, d = boxes[i]
        orients = set(itertools.permutations((w, h, d)))
        all_orients.append(list(orients))

    # Precompute volumes
    volumes = [w * h * d for w, h, d in boxes]
    container_vol = W * H * D

    # Skyline-based packing: maintain a 2D height map (x, y) -> z_height
    # We use a discrete grid for the height map.
    # To keep it efficient, we use a dict-based sparse representation.

    def pack_with_skyline(order):
        """Pack boxes in the given order using a 2D skyline height map."""
        # height_map: dict from (x, y) to z_height
        # For efficiency, we'll use a list of lists if the grid is small enough,
        # otherwise a dict. Given container sizes are likely moderate, we'll use a grid.
        
        # Determine grid resolution: use integer coordinates
        # If W, H are large, we need to be careful. But for 60 boxes, likely fine.
        # Let's use a grid of size (W+1) x (H+1) if feasible, else dict.
        
        # For safety, let's use a dict-based approach that's O(k) per placement where k is number of occupied cells.
        # Actually, a better approach: for each candidate (x, y), we need the minimum z such that the box fits.
        # The skyline at (x, y) is the max z of any box covering (x, y).
        
        # Let's use a grid if W*H is not too large (say < 100000)
        use_grid = (W + 1) * (H + 1) < 100000
        
        if use_grid:
            height_grid = [[0] * (H + 1) for _ in range(W + 1)]
        else:
            height_grid = None
            height_dict = {}  # (x, y) -> z

        placed = []
        result = [None] * n
        filled = 0

        for idx in order:
            w, h, d = boxes[idx]
            best_pos = None
            best_orient = None
            best_cost = float('inf')

            for bw, bh, bd in all_orients[idx]:
                if bw > W or bh > H or bd > D:
                    continue

                # Try to find the best (x, y) position for this orientation
                # We want to minimize x + y + z (BLF heuristic)
                # z is determined by the skyline at the footprint of the box

                # For each possible (x, y), compute the required z
                # This is expensive if we iterate all (x, y). 
                # Optimization: only consider anchor points derived from placed boxes.
                
                # Generate candidate (x, y) positions
                candidates = set()
                candidates.add((0, 0))
                for (px, py, pz, pw, ph, pd) in placed:
                    candidates.add((px + pw, py))
                    candidates.add((px, py + ph))
                    candidates.add((px + pw, py + ph))
                    # Also try positions just before edges
                    if px > 0:
                        candidates.add((max(0, px - bw), py))
                        candidates.add((px, max(0, py - bh)))
                    
                for cx, cy in candidates:
                    if cx + bw > W or cy + bh > H:
                        continue
                    
                    # Compute the required z: max z-height over the footprint
                    if use_grid:
                        min_z = 0
                        for gx in range(cx, cx + bw):
                            for gy in range(cy, cy + bh):
                                if height_grid[gx][gy] > min_z:
                                    min_z = height_grid[gx][gy]
                    else:
                        min_z = 0
                        for gx in range(cx, cx + bw):
                            for gy in range(cy, cy + bh):
                                key = (gx, gy)
                                if key in height_dict and height_dict[key] > min_z:
                                    min_z = height_dict[key]
                    
                    cz = min_z
                    if cz + bd > D:
                        continue

                    # Valid placement! Compute cost
                    cost = cx + cy + cz
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (cx, cy, cz)
                        best_orient = (bw, bh, bd)

                if best_cost == 0:
                    break

            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                placed.append((bx, by, bz, bw, bh, bd))
                result[idx] = (0, bx, by, bz, bw, bh, bd)
                filled += bw * bh * bd

                # Update height map
                if use_grid:
                    for gx in range(bx, bx + bw):
                        for gy in range(by, by + bh):
                            new_h = bz + bd
                            if new_h > height_grid[gx][gy]:
                                height_grid[gx][gy] = new_h
                else:
                    for gx in range(bx, bx + bw):
                        for gy in range(by, by + bh):
                            key = (gx, gy)
                            new_h = bz + bd
                            if new_h > height_dict.get(key, 0):
                                height_dict[key] = new_h
            else:
                result[idx] = (-1, 0, 0, 0, w, h, d)

        return filled, result

    # Multiple initial ordering strategies
    strategies = []

    # 1. Volume descending
    strategies.append(sorted(range(n), key=lambda i: volumes[i], reverse=True))

    # 2. Volume ascending
    strategies.append(sorted(range(n), key=lambda i: volumes[i]))

    # 3. Max dimension descending
    strategies.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))

    # 4. Random
    for _ in range(3):
        s = list(range(n))
        random.shuffle(s)
        strategies.append(s)

    # Try each strategy and keep the best
    best_filled = 0
    best_result = None
    best_order = None

    for order in strategies:
        if time.time() > deadline - 30.0:
            break
        filled, result = pack_with_skyline(order)
        if filled > best_filled:
            best_filled = filled
            best_result = result
            best_order = order[:]

    # Local search: swap two boxes in the best order
    if time.time() < deadline - 20.0 and best_order is not None:
        current_order = best_order[:]
        current_filled = best_filled
        T = 500.0
        cooling = 0.99
        min_T = 1.0
        steps = 0
        max_steps = 5000

        while T > min_T and time.time() < deadline - 10.0 and steps < max_steps:
            i, j = random.sample(range(n), 2)
            current_order[i], current_order[j] = current_order[j], current_order[i]

            new_filled, new_result = pack_with_skyline(current_order)
            delta = new_filled - current_filled

            if delta > 0 or random.random() < math.exp(delta / T):
                current_filled = new_filled
                if current_filled > best_filled:
                    best_filled = current_filled
                    best_result = new_result
            else:
                current_order[i], current_order[j] = current_order[j], current_order[i]

            T *= cooling
            steps += 1

    # Fallback
    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_result

# EVOLVE-BLOCK-END
