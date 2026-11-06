"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Helper to check if a box fits at a point
    def fits(x, y, z, w, h, d, occupied):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap with occupied boxes
        for (ox, oy, oz, ow, oh, od) in occupied:
            # If not disjoint in any dimension, they overlap
            if not (x + w <= ox or ox + ow <= x or 
                    y + h <= oy or oy + oh <= y or 
                    z + d <= oz or oz + od <= z):
                return False
        return True

    def update_extreme_points(points, x, y, z, w, h, d, occupied):
        # Generate new extreme points from the placed box
        # The 6 corners of the new box
        new_points = [
            (x + w, y, z),
            (x, y + h, z),
            (x, y, z + d),
            (x + w, y + h, z),
            (x + w, y, z + d),
            (x, y + h, z + d),
            (x + w, y + h, z + d) # Top-back-right is also a potential start for boxes above/behind
        ]
        
        for p in new_points:
            px, py, pz = p
            # Only add if it's within container bounds and not inside an occupied box
            if 0 <= px <= W and 0 <= py <= H and 0 <= pz <= D:
                # Check if this point is "free" (not inside any occupied box)
                is_free = True
                for (ox, oy, oz, ow, oh, od) in occupied:
                    if ox <= px < ox + ow and oy <= py < oy + oh and oz <= pz < oz + od:
                        is_free = False
                        break
                if is_free:
                    points.add(p)

    def pack_with_order(sorted_indices, boxes, container):
        W, H, D = container
        occupied = []
        points = {(0, 0, 0)}
        placements = [None] * len(boxes)
        
        for idx in sorted_indices:
            w, h, d = boxes[idx]
            best_placement = None
            best_key = None
            
            # Try all permutations of dimensions
            for perm in set(permutations([w, h, d])):
                pw, ph, pd = perm
                # Try all extreme points
                for (px, py, pz) in points:
                    if fits(px, py, pz, pw, ph, pd, occupied):
                        # Heuristic: Prefer lowest z, then lowest y, then lowest x
                        key = (pz, py, px)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_placement = (px, py, pz, pw, ph, pd)
            
            if best_placement:
                px, py, pz, pw, ph, pd = best_placement
                occupied.append((px, py, pz, pw, ph, pd))
                placements[idx] = (0, px, py, pz, pw, ph, pd)
                update_extreme_points(points, px, py, pz, pw, ph, pd, occupied)
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
        
        return placements

    # Define different sorting strategies
    n = len(boxes)
    indices = list(range(n))
    
    strategies = []
    
    # 1. Volume descending
    strategies.append(sorted(indices, key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2])))
    
    # 2. Max dimension descending
    strategies.append(sorted(indices, key=lambda i: -max(boxes[i])))
    
    # 3. Sum of dimensions descending
    strategies.append(sorted(indices, key=lambda i: -(boxes[i][0]+boxes[i][1]+boxes[i][2])))
    
    # 4. Volume / MaxDim (aspect ratio proxy)
    strategies.append(sorted(indices, key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2] / max(boxes[i]))))
    
    best_placements = None
    best_volume = 0
    
    for strat in strategies:
        placements = pack_with_order(strat, boxes, container)
        vol = 0
        for p in placements:
            if p[0] == 0:
                vol += p[4] * p[5] * p[6]
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            
    # Fallback if all failed (should not happen)
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_placements
# EVOLVE-BLOCK-END
