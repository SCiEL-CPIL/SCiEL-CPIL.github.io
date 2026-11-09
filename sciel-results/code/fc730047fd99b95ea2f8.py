"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from itertools import permutations

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 145.0  # seconds, safety margin before 150s kill

    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D

    # Initial placement list
    placements = [(-1, 0, 0, 0, 0, 0, 0)] * n

    def can_place(x, y, z, w, h, d, placed_list):
        """Check if box at (x,y,z) with dims (w,h,d) doesn't overlap with any placed box."""
        for px, py, pz, pw, ph, pd in placed_list:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True

    def get_orientations(w, h, d):
        """Return all unique orientations as (ow, oh, od) tuples."""
        orients = set(permutations((w, h, d)))
        return list(orients)

    def get_candidate_positions(placed, ow, oh, od, W, H, D):
        """Generate candidate positions: corners of placed boxes + origin."""
        positions = set()
        positions.add((0, 0, 0))
        
        xs = set([0, W])
        ys = set([0, H])
        zs = set([0, D])
        
        for px, py, pz, pw, ph, pd in placed:
            xs.add(px)
            xs.add(px + pw)
            ys.add(py)
            ys.add(py + ph)
            zs.add(pz)
            zs.add(pz + pd)
        
        # Only generate positions that are "interesting" - near existing boxes
        for px, py, pz, pw, ph, pd in placed:
            # Right face
            nx = px + pw
            if nx + ow <= W:
                for ny in [py, py + ph]:
                    for nz in [pz, pz + pd]:
                        if ny + oh <= H and nz + od <= D:
                            positions.add((nx, ny, nz))
            # Top face
            ny = py + ph
            if ny + oh <= H:
                for nx in [px, px + pw]:
                    for nz in [pz, pz + pd]:
                        if nx + ow <= W and nz + od <= D:
                            positions.add((nx, ny, nz))
            # Front face
            nz = pz + pd
            if nz + od <= D:
                for nx in [px, px + pw]:
                    for ny in [py, py + ph]:
                        if nx + ow <= W and ny + oh <= H:
                            positions.add((nx, ny, nz))
        
        # Also add some grid-based positions for better coverage
        if len(placed) < 50:
            # Add positions along axes at box boundaries
            for px, py, pz, pw, ph, pd in placed:
                if px + pw + ow <= W:
                    positions.add((px + pw, 0, 0))
                if py + ph + oh <= H:
                    positions.add((0, py + ph, 0))
                if pz + pd + od <= D:
                    positions.add((0, 0, pz + pd))
        
        return list(positions)

    def greedy_fill(boxes_sorted, container, max_time=30.0):
        """Greedy first-fit with multiple orientation tries."""
        W, H, D = container
        placed = []
        result = {}
        deadline = start_time + max_time
        
        for idx in boxes_sorted:
            if time.time() > deadline:
                break
            
            w, h, d = boxes[idx]
            orientations = get_orientations(w, h, d)
            best_orient = None
            best_pos = None
            best_score = float('inf')
            
            for ow, oh, od in orientations:
                if ow > W or oh > H or od > D:
                    continue
                
                positions_to_try = get_candidate_positions(placed, ow, oh, od, W, H, D)
                
                for (tx, ty, tz) in positions_to_try:
                    if tx + ow > W or ty + oh > H or tz + od > D:
                        continue
                    if can_place(tx, ty, tz, ow, oh, od, placed):
                        # Score: prefer positions that minimize wasted space
                        # Use a combination of height, then x, then z
                        score = ty * 10000 + tx * 100 + tz
                        if score < best_score:
                            best_score = score
                            best_orient = (ow, oh, od)
                            best_pos = (tx, ty, tz)
            
            if best_pos is not None:
                placed.append((best_pos[0], best_pos[1], best_pos[2], best_orient[0], best_orient[1], best_orient[2]))
                result[idx] = (best_pos[0], best_pos[1], best_pos[2], best_orient[0], best_orient[1], best_orient[2])
        
        return result, placed

    def compute_fill(result):
        """Compute total volume of placed boxes as fraction of container."""
        total = 0
        for idx, (x, y, z, w, h, d) in result.items():
            total += w * h * d
        return total / container_vol

    # Multiple sorting strategies
    def sort_by_volume(boxes):
        return sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    def sort_by_max_dim(boxes):
        return sorted(range(len(boxes)), key=lambda i: max(boxes[i]), reverse=True)
    
    def sort_by_min_dim(boxes):
        return sorted(range(len(boxes)), key=lambda i: min(boxes[i]), reverse=True)
    
    def sort_by_surface_area(boxes):
        return sorted(range(len(boxes)), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    
    def sort_by_volume_max_ratio(boxes):
        return sorted(range(len(boxes)), key=lambda i: (boxes[i][0]*boxes[i][1]*boxes[i][2]) / max(boxes[i]) if max(boxes[i]) > 0 else 0, reverse=True)

    best_result = {}
    best_fill = 0.0

    strategies = [
        sort_by_volume,
        sort_by_max_dim,
        sort_by_min_dim,
        sort_by_surface_area,
        sort_by_volume_max_ratio,
    ]

    # Run each strategy with time budget
    time_per_strategy = (DEADLINE - (time.time() - start_time)) / len(strategies)
    time_per_strategy = max(time_per_strategy, 10.0)  # at least 10s per strategy
    time_per_strategy = min(time_per_strategy, 40.0)  # at most 40s per strategy

    for strat in strategies:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        boxes_sorted = strat(boxes)
        remaining_time = DEADLINE - 5 - elapsed
        alloc_time = min(time_per_strategy, remaining_time)
        
        result, placed = greedy_fill(boxes_sorted, container, max_time=alloc_time)
        
        fill = compute_fill(result)
        if fill > best_fill:
            best_fill = fill
            best_result = dict(result)
    
    # Second pass: try to place remaining boxes in the best solution
    if best_result and time.time() - start_time < DEADLINE - 10:
        placed = [(x, y, z, w, h, d) for (x, y, z, w, h, d) in best_result.values()]
        unplaced = [i for i in range(n) if i not in best_result]
        
        # Sort unplaced by volume descending for better packing
        unplaced.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        
        for idx in unplaced:
            if time.time() - start_time > DEADLINE - 2:
                break
            w, h, d = boxes[idx]
            orientations = get_orientations(w, h, d)
            
            best_orient = None
            best_pos = None
            best_score = float('inf')
            
            for ow, oh, od in orientations:
                if ow > W or oh > H or od > D:
                    continue
                
                positions_to_try = get_candidate_positions(placed, ow, oh, od, W, H, D)
                
                for (tx, ty, tz) in positions_to_try:
                    if tx + ow > W or ty + oh > H or tz + od > D:
                        continue
                    if can_place(tx, ty, tz, ow, oh, od, placed):
                        score = ty * 10000 + tx * 100 + tz
                        if score < best_score:
                            best_score = score
                            best_orient = (ow, oh, od)
                            best_pos = (tx, ty, tz)
            
            if best_pos is not None:
                best_result[idx] = (best_pos[0], best_pos[1], best_pos[2], best_orient[0], best_orient[1], best_orient[2])
                placed.append((best_pos[0], best_pos[1], best_pos[2], best_orient[0], best_orient[1], best_orient[2]))
    
    # Build final placements list
    for i in range(n):
        if i in best_result:
            x, y, z, w, h, d = best_result[i]
            placements[i] = (0, x, y, z, w, h, d)
        else:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)

    return placements
# EVOLVE-BLOCK-END
