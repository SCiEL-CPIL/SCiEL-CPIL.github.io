"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    # 120s limit, keep 10s margin for safety/overhead
    deadline = start_time + 110.0
    
    # Preprocess: store box dimensions and original indices
    box_data = []
    for i, (w, h, d) in enumerate(boxes):
        box_data.append((i, w, h, d))
        
    # Initial result: all boxes left out
    final_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    def get_rotations(w, h, d):
        # Generate unique rotations
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        return list(rots)

    def can_place(x, y, z, bw, bh, bd, placed):
        """Check if a box fits at (x,y,z) without overlapping any placed box."""
        for (px, py, pz, pw, ph, pd) in placed:
            # If they don't separate in any axis, they overlap
            if not (x + bw <= px or px + pw <= x or 
                    y + bh <= py or py + ph <= y or 
                    z + bd <= pz or pz + pd <= z):
                return False
        return True

    def find_best_fit(bw, bh, bd, placed):
        """
        Finds the best position (x, y, z) for a box of size (bw, bh, bd)
        using a Best-Fit heuristic.
        Returns (x, y, z) or None if no position found.
        
        Heuristic: 
        1. Generate candidate coordinates from 0 and right faces of existing boxes.
        2. Filter valid non-overlapping positions.
        3. Score positions by 'tightness' (sum of distances to walls/neighbors) 
           or simply by (z, x, -y) to favor bottom-left-far.
           For better packing, we often want to minimize z (bottom), then x (left), 
           then maximize y (far) to push against the back.
           However, 'Best Fit' usually implies minimizing the leftover space in the 
           specific 'slot' or gap. A common proxy is minimizing (x+y+z) or 
           maximizing the 'support' (distance to floor/walls).
           
           Let's use a score that favors:
           1. Lowest Z (Bottom)
           2. Leftmost X
           3. Farthest Y (against back wall)
           This is the standard BLF (Bottom-Left-Far) heuristic.
        """
        # Candidate x-coordinates: 0 and right faces of all placed boxes
        x_candidates = set([0])
        y_candidates = set([0])
        z_candidates = set([0])
        
        for (px, py, pz, pw, ph, pd) in placed:
            x_candidates.add(px + pw)
            y_candidates.add(py + ph)
            z_candidates.add(pz + pd)
            
        # Filter candidates that fit within container bounds
        x_candidates = [x for x in x_candidates if x + bw <= W]
        y_candidates = [y for y in y_candidates if y + bh <= H]
        z_candidates = [z for z in z_candidates if z + bd <= D]
        
        if not x_candidates or not y_candidates or not z_candidates:
            return None

        best_pos = None
        best_score = None
        best_coord = None
        
        # To improve packing, we can also consider "snap" points slightly offset? 
        # No, for axis-aligned boxes, faces are the only relevant coordinates for 
        # optimal packing in a greedy sense.
        
        for z in sorted(z_candidates):
            for x in sorted(x_candidates):
                for y in y_candidates:
                    # Check overlap
                    if can_place(x, y, z, bw, bh, bd, placed):
                        # Score: Minimize z, then Minimize x, then Maximize y
                        # We want the "lowest" z, then "leftmost" x, then "farthest" y
                        # Score tuple: (z, x, -y) -> min
                        score = (z, x, -y)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_coord = (x, y, z)
                            
        return best_coord

    def run_packing_pass(unplaced_indices, placed, final_placements, box_data, deadline):
        """
        Attempts to place as many boxes as possible from unplaced_indices.
        Returns list of indices that were successfully placed in this pass.
        """
        placed_now = []
        
        # Sort unplaced by volume descending to place big ones first in each pass
        # This helps fill large gaps early.
        sorted_unplaced = sorted(unplaced_indices, key=lambda i: box_data[i][1]*box_data[i][2]*box_data[i][3], reverse=True)
        
        for idx in sorted_unplaced:
            if time.time() > deadline:
                break
                
            if idx in placed_now:
                continue
                
            i, w, h, d = box_data[idx]
            rots = get_rotations(w, h, d)
            placed_this = False
            
            for rw, rh, rd in rots:
                pos = find_best_fit(rw, rh, rd, placed)
                if pos is not None:
                    x, y, z = pos
                    placed.append((x, y, z, rw, rh, rd))
                    final_placements[idx] = (0, x, y, z, rw, rh, rd)
                    placed_now.append(idx)
                    placed_this = True
                    break
                    
        return placed_now

    # Strategy: Multi-pass
    # Pass 1: Place all boxes that can be placed in the empty container (or partially filled).
    # Pass 2: Try to place remaining boxes. As the container gets filled with irregular shapes,
    #         new gaps might form that are suitable for smaller boxes.
    
    unplaced = list(range(n))
    
    # We run multiple passes until no more boxes can be placed or time runs out.
    # Each pass re-evaluates all unplaced boxes.
    # This is more expensive than a single pass but allows "filling in" gaps.
    
    max_passes = 10
    for p in range(max_passes):
        if time.time() > deadline:
            break
        if not unplaced:
            break
            
        placed_in_pass = run_packing_pass(unplaced, placed, final_placements, box_data, deadline)
        
        # Remove placed boxes from unplaced list
        if placed_in_pass:
            unplaced = [i for i in unplaced if i not in placed_in_pass]
        else:
            # If no box could be placed in this pass, further passes are unlikely to help
            # unless we change the strategy (e.g., shaking/relaxing), but we are using a deterministic greedy.
            # So we break.
            break

    return final_placements
# EVOLVE-BLOCK-END
