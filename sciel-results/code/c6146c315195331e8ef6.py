import time
import itertools
import numpy as np

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Handle edge cases
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 40s margin for safety
    
    # 1. Preprocess boxes: Sort by volume descending (FFD)
    # We need to keep track of original indices to return results in original order
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # Store placements in original index order
    # placement[i] = (bin, x, y, z, w, h, d)
    # bin=0 if placed, -1 if not
    placements = [None] * n
    
    # 2. Initialize Extreme Points
    # Each point is (x, y, z)
    extreme_points = [(0, 0, 0)]
    
    # List of placed boxes: (x, y, z, w, h, d)
    placed_boxes = []
    
    # Helper to check if a point is dominated (inside or on boundary of a placed box)
    # Actually, we just need to check if a candidate position (px, py, pz) with dims (ow, oh, od)
    # is valid.
    
    # Optimization: Use a set for fast duplicate checking of extreme points
    ep_set = set(extreme_points)
    
    # 3. Iterate through sorted boxes
    for idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # If we run out of time, mark remaining as not placed
            for j in range(len(indexed_boxes)):
                orig_idx = indexed_boxes[j][0]
                if placements[orig_idx] is None:
                    ow, oh, od = boxes[orig_idx]
                    placements[orig_idx] = (-1, 0, 0, 0, ow, oh, od)
            break
            
        # Get unique orientations for this box
        # Dimensions are (w, h, d). Permutations:
        orients = list(set(itertools.permutations([w, h, d])))
        
        best_placement = None
        best_score = float('inf')
        
        # Sort extreme points to prioritize "lower" and "front" positions
        # This helps fill the container from the bottom-up, front-to-back
        # Sorting by (x, y, z) ensures we try the most "compact" positions first
        # which generally leads to better packing by keeping the "skyline" low.
        # However, sorting every time is O(E log E). E is small (< 200), so it's fine.
        # We can also limit the number of points we consider if it gets too large,
        # but for 60 boxes, it should stay manageable.
        
        # To improve performance, we can pre-filter extreme points that are clearly
        # too high or too far back? No, just sorting is a good heuristic.
        
        # Let's iterate through sorted points
        sorted_eps = sorted(extreme_points)
        
        for px, py, pz in sorted_eps:
            if time.time() > deadline:
                break
                
            for ow, oh, od in orients:
                # Check bounds
                if px + ow > W or py + oh > H or pz + od > D:
                    continue
                
                # Check overlap with all placed boxes
                valid = True
                for bx, by, bz, bw, bh, bd in placed_boxes:
                    # Check if (px, py, pz, ow, oh, od) overlaps (bx, by, bz, bw, bh, bd)
                    if (px < bx + bw and bx < px + ow and
                        py < by + bh and by < py + oh and
                        pz < bz + bd and bz < pz + od):
                        valid = False
                        break
                
                if valid:
                    # Found a valid placement
                    # Score it. 
                    # Heuristic: Minimize the "distance" from the origin.
                    # This encourages filling the container from the corner.
                    # A better heuristic might be to minimize the increase in the bounding box
                    # of all placed items, but that's expensive.
                    # Let's stick to minimizing x+y+z for now, but maybe weight it?
                    # If we just use x+y+z, it might ignore that a box placed at (0,0,10)
                    # is worse than (0,0,0) even if x+y+z is similar? No, (0,0,0) is 0, (0,0,10) is 10.
                    # So it works.
                    
                    # Alternative score: Minimize the "wasted" space above the box?
                    # Or maximize the "support"?
                    # Let's try a combination: Minimize (x + y + z) but also prefer
                    # orientations that align with the container axes if possible?
                    # The set of orientations already handles this.
                    
                    score = px + py + pz
                    
                    # Tie-breaker: If scores are equal, prefer the one with smaller volume?
                    # Or larger? Smaller volume might allow better packing of remaining?
                    # Actually, we are placing the current box. The volume is fixed.
                    # So tie-breaker could be based on the "tightness" of the fit?
                    # Let's just use the first one found with the minimum score.
                    
                    if score < best_score:
                        best_score = score
                        best_placement = (px, py, pz, ow, oh, od)
                        
        if best_placement is not None:
            px, py, pz, ow, oh, od = best_placement
            placements[idx] = (0, px, py, pz, ow, oh, od)
            placed_boxes.append((px, py, pz, ow, oh, od))
            
            # Generate new extreme points
            # The new box occupies [px, px+ow] x [py, py+oh] x [pz, pz+od]
            # New potential positions for next boxes are the corners of this box that are "exposed"
            new_points = [
                (px + ow, py, pz),
                (px, py + oh, pz),
                (px, py, pz + od)
            ]
            
            # Add new points to extreme_points, avoiding duplicates
            for p in new_points:
                if p not in ep_set:
                    ep_set.add(p)
                    extreme_points.append(p)
        else:
            # Box could not be placed
            placements[idx] = (-1, 0, 0, 0, w, h, d)
            
    # Fill any remaining None (if deadline hit)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements
