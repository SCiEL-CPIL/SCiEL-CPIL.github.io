"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 118.0  # seconds, safety margin before 120s kill

    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D

    # Initial placement list
    placements = [(-1, 0, 0, 0, 0, 0, 0)] * n

    def get_orientations(w, h, d):
        """Generate all 6 unique orientations."""
        orientations = []
        seen = set()
        for i in range(3):
            for j in range(2):
                if j == 0:
                    o = (w, h, d)
                else:
                    o = (w, d, h)
                # Actually need proper permutation
                pass
        
        # Proper 6 orientations
        perms = [
            (w, h, d), (w, d, h),
            (h, w, d), (h, d, w),
            (d, w, h), (d, h, w),
        ]
        for p in perms:
            if p not in seen:
                seen.add(p)
                orientations.append(p)
        return orientations

    def can_place_fast(x, y, z, w, h, d, placed):
        """Check if box at (x,y,z) with dims (w,h,d) doesn't overlap with any placed box."""
        for px, py, pz, pw, ph, pd in placed:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True

    def find_best_position(placed, ow, oh, od, W, H, D):
        """Find the best (lowest, leftmost, frontmost) position for a box with given orientation."""
        best_pos = None
        best_score = float('inf')
        
        # Candidate positions: origin and offsets from existing boxes
        candidates = set()
        candidates.add((0, 0, 0))
        
        for px, py, pz, pw, ph, pd in placed:
            # Right
            nx = px + pw
            if nx + ow <= W:
                candidates.add((nx, py, pz))
                candidates.add((nx, py + ph, pz))
                candidates.add((nx, py, pz + pd))
                candidates.add((nx, py + ph, pz + pd))
            # Above
            ny = py + ph
            if ny + oh <= H:
                candidates.add((px, ny, pz))
                candidates.add((px + pw, ny, pz))
                candidates.add((px, ny, pz + pd))
                candidates.add((px + pw, ny, pz + pd))
            # Front
            nz = pz + pd
            if nz + od <= D:
                candidates.add((px, py, nz))
                candidates.add((px + pw, py, nz))
                candidates.add((px, py + ph, nz))
                candidates.add((px + pw, py + ph, nz))
        
        for (tx, ty, tz) in candidates:
            if tx + ow > W or ty + oh > H or tz + od > D:
                continue
            if can_place_fast(tx, ty, tz, ow, oh, od, placed):
                # Score: prefer lower y (height), then lower z, then lower x
                # Actually prefer compact: minimize max extent
                score = ty * 10000 + tz * 100 + tx
                if score < best_score:
                    best_score = score
                    best_pos = (tx, ty, tz)
        
        return best_pos

    def greedy_fill(boxes_sorted, container, time_limit=None):
        """Greedy first-fit decreasing with multiple orientation tries."""
        W, H, D = container
        placed = []  # list of (x, y, z, w, h, d)
        result = {}  # box_idx -> (x, y, z, w, h, d)
        
        for idx in boxes_sorted:
            if time_limit is not None and time.time() - time_limit > 0:
                break
            
            w, h, d = boxes[idx]
            orientations = get_orientations(w, h, d)
            
            best_orient = None
            best_pos = None
            best_score = float('inf')
            
            for ow, oh, od in orientations:
                if ow > W or oh > H or od > D:
                    continue
                
                pos = find_best_position(placed, ow, oh, od, W, H, D)
                if pos is not None:
                    # Score: prefer lower positions
                    score = pos[1] * 10000 + pos[2] * 100 + pos[0]
                    if score < best_score:
                        best_score = score
                        best_orient = (ow, oh, od)
                        best_pos = pos
            
            if best_pos is not None:
                placed.append((best_pos[0], best_pos[1], best_pos[2], best_orient[0], best_orient[1], best_orient[2]))
                result[idx] = (best_pos[0], best_pos[1], best_pos[2], best_orient[0], best_orient[1], best_orient[2])
        
        return result, placed

    def compute_fill(result):
        """Compute total volume of placed boxes."""
        total = 0
        for idx, (x, y, z, w, h, d) in result.items():
            total += w * h * d
        return total / container_vol

    # Multiple sorting strategies
    def sort_by_volume(boxes):
        return sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    def sort_by_max_dim(boxes):
        return sorted(range(boxes), key=lambda i: max(boxes[i]), reverse=True)
    
    def sort_by_surface_area(boxes):
        return sorted(range(len(boxes)), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    
    def sort_by_volume_to_max_ratio(boxes):
        return sorted(range(len(boxes)), key=lambda i: (boxes[i][0]*boxes[i][1]*boxes[i][2]) / max(boxes[i]) if max(boxes[i]) > 0 else 0, reverse=True)
    
    def sort_by_min_dim(boxes):
        return sorted(range(len(boxes)), key=lambda i: min(boxes[i]), reverse=True)

    strategies = [
        sort_by_volume,
        sort_by_max_dim,
        sort_by_surface_area,
        sort_by_volume_to_max_ratio,
        sort_by_min_dim,
    ]

    best_result = {}
    best_fill = 0.0

    # Run multiple strategies
    for strat in strategies:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        boxes_sorted = strat(boxes)
        result, placed = greedy_fill(boxes_sorted, container)
        
        fill = compute_fill(result)
        if fill > best_fill:
            best_fill = fill
            best_result = result
    
    # Second pass: try to place remaining boxes
    if best_result and time.time() - start_time < DEADLINE - 10:
        placed = [(x, y, z, w, h, d) for (x, y, z, w, h, d) in best_result.values()]
        
        unplaced = [i for i in range(n) if i not in best_result]
        
        # Sort unplaced by volume descending
        unplaced.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        
        for idx in unplaced:
            if time.time() - start_time > DEADLINE - 5:
                break
            w, h, d = boxes[idx]
            orientations = get_orientations(w, h, d)
            
            best_orient = None
            best_pos = None
            best_score = float('inf')
            
            for ow, oh, od in orientations:
                if ow > W or oh > H or od > D:
                    continue
                
                pos = find_best_position(placed, ow, oh, od, W, H, D)
                if pos is not None:
                    score = pos[1] * 10000 + pos[2] * 100 + pos[0]
                    if score < best_score:
                        best_score = score
                        best_orient = (ow, oh, od)
                        best_pos = pos
            
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
