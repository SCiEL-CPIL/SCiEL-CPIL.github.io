"""3D Container Loading with Extreme Points + Local Search."""
import time
import random
import itertools
from collections import defaultdict

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    def get_orientations(w, h, d):
        """Get all 6 unique orientations of a box."""
        o = set()
        for perm in itertools.permutations((w, h, d)):
            o.add(perm)
        return list(o)
    
    def pack_with_order(order, boxes, container, max_extreme_points=500):
        """Pack boxes in given order using extreme points heuristic."""
        W, H, D = container
        n = len(boxes)
        placements = [None] * n
        occupied = []  # list of (x, y, z, w, h, d) for placed boxes
        extreme_points = [(0, 0, 0)]
        
        for idx in order:
            w, h, d = boxes[idx]
            orientations = get_orientations(w, h, d)
            
            best_pos = None
            best_score = None
            best_orient = None
            
            for (ow, oh, od) in orientations:
                # Check if orientation fits in container
                if ow > W or oh > H or od > D:
                    continue
                
                # Try each extreme point
                for (ex, ey, ez) in extreme_points:
                    # Check bounds
                    if ex + ow > W or ey + oh > H or ez + od > D:
                        continue
                    
                    # Check overlap with all placed boxes
                    overlap = False
                    for (px, py, pz, pw, ph, pd) in occupied:
                        if (ex < px + pw and ex + ow > px and
                            ey < py + ph and ey + oh > py and
                            ez < pz + pd and ez + od > pz):
                            overlap = True
                            break
                    if overlap:
                        continue
                    
                    # Score: prefer lower z, then lower y, then lower x (compactness)
                    score = (ez, ey, ex)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (ex, ey, ez)
                        best_orient = (ow, oh, od)
            
            if best_pos is not None:
                ex, ey, ez = best_pos
                ow, oh, od = best_orient
                placements[idx] = (0, ex, ey, ez, ow, oh, od)
                occupied.append((ex, ey, ez, ow, oh, od))
                
                # Generate new extreme points
                new_points = [
                    (ex + ow, ey, ez),
                    (ex, ey + oh, ez),
                    (ex, ey, ez + od),
                    (ex + ow, ey + oh, ez),
                    (ex + ow, ey, ez + od),
                    (ex, ey + oh, ez + od),
                ]
                for np in new_points:
                    if np not in extreme_points:
                        extreme_points.append(np)
                
                # Limit extreme points to avoid slowdown
                if len(extreme_points) > max_extreme_points:
                    # Keep the most promising ones (lowest z, then y, then x)
                    extreme_points.sort(key=lambda p: (p[2], p[1], p[0]))
                    extreme_points = extreme_points[:max_extreme_points]
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
        
        return placements
    
    def calc_volume(placements):
        """Calculate total volume of placed boxes."""
        vol = 0
        for p in placements:
            if p[0] == 0:
                _, _, _, _, w, h, d = p
                vol += w * h * d
        return vol
    
    # Precompute box volumes for sorting
    box_volumes = [w * h * d for w, h, d in boxes]
    
    # Generate initial orderings using different heuristics
    orderings = []
    
    # 1. Sort by volume descending (FFD)
    order1 = sorted(range(n), key=lambda i: box_volumes[i], reverse=True)
    orderings.append(order1)
    
    # 2. Sort by surface area descending
    surface_areas = [2*(w*h + w*d + h*d) for w, h, d in boxes]
    order2 = sorted(range(n), key=lambda i: surface_areas[i], reverse=True)
    orderings.append(order2)
    
    # 3. Sort by max edge descending
    max_edges = [max(w, h, d) for w, h, d in boxes]
    order3 = sorted(range(n), key=lambda i: max_edges[i], reverse=True)
    orderings.append(order3)
    
    # 4. Original order
    orderings.append(list(range(n)))
    
    # Evaluate initial packings
    best_placements = None
    best_volume = 0
    
    for order in orderings:
        if time.time() > deadline:
            break
        placements = pack_with_order(order, boxes, container)
        vol = calc_volume(placements)
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
    
    # Local search: try swapping adjacent boxes in the ordering
    if best_placements is not None:
        current_order = orderings[0]  # Start from FFD order
        current_volume = best_volume
        improved = True
        max_swaps = 5000
        
        while improved and time.time() < deadline:
            improved = False
            for i in range(n - 1):
                if time.time() > deadline:
                    break
                # Try swapping i and i+1
                new_order = current_order.copy()
                new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
                placements = pack_with_order(new_order, boxes, container)
                vol = calc_volume(placements)
                if vol > current_volume:
                    current_volume = vol
                    current_order = new_order
                    best_volume = vol
                    best_placements = placements
                    improved = True
                max_swaps -= 1
                if max_swaps <= 0:
                    break
            if max_swaps <= 0:
                break
    
    # If we haven't found any placement, use the best from initial heuristics
    if best_placements is None:
        best_placements = pack_with_order(orderings[0], boxes, container)
    
    return best_placements
