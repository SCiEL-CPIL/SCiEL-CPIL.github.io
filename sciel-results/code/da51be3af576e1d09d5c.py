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
    # 150s limit, keep 10s margin for safety/overhead
    deadline = start_time + 140.0
    
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

    def can_place(placed, x, y, z, bw, bh, bd):
        """Check if a box of size (bw, bh, bd) at (x, y, z) overlaps with any placed box."""
        for (px, py, pz, pw, ph, pd) in placed:
            if not (x + bw <= px or px + pw <= x or 
                    y + bh <= py or py + ph <= y or 
                    z + bd <= pz or pz + pd <= z):
                return False
        return True

    def find_best_fit(placed, bw, bh, bd):
        """
        Find the 'best fit' position for a box.
        Best fit is defined as the position that minimizes the sum of distances
        to the closest placed box on each axis (encouraging tight packing).
        Returns (x, y, z) or None.
        """
        if bw > W or bh > H or bd > D:
            return None

        # Candidate coordinates are 0 and the max extent of placed boxes
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

        best_score = None
        best_coord = None
        
        for z in z_candidates:
            for y in y_candidates:
                for x in x_candidates:
                    if can_place(placed, x, y, z, bw, bh, bd):
                        # Score: Prefer positions that are "lower" (z), "left" (x), "back" (y)
                        # This prioritizes filling the bottom layer first.
                        score = (z, x, y)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_coord = (x, y, z)
                            
        return best_coord

    # --- Main Strategy: Multi-pass Best Fit with Randomized Restart ---
    
    def run_packing_pass(boxes_to_place, placed, final_placements, deadline):
        """
        Attempt to place boxes from boxes_to_place into the container.
        Returns the number of newly placed boxes.
        """
        placed_count = 0
        # Sort boxes by volume descending for this pass
        boxes_to_place = sorted(boxes_to_place, key=lambda b: b[1]*b[2]*b[3], reverse=True)
        
        for idx, w, h, d in boxes_to_place:
            if time.time() > deadline:
                break
                
            rots = get_rotations(w, h, d)
            placed_this = False
            
            # Try each rotation
            for rw, rh, rd in rots:
                pos = find_best_fit(placed, rw, rh, rd)
                if pos is not None:
                    x, y, z = pos
                    placed.append((x, y, z, rw, rh, rd))
                    final_placements[idx] = (0, x, y, z, rw, rh, rd)
                    placed_this = True
                    break
                    
            if not placed_this:
                # If we couldn't place it now, maybe we can later? 
                # In a simple pass, we skip it. 
                pass

        return placed_count

    # We will do multiple passes. In each pass, we try to place any box that fits.
    # We sort by volume descending initially, but in subsequent passes, 
    # we might try smaller boxes first to fill gaps, or stick to volume to keep big items prominent.
    # Let's stick to Volume Descending for stability, but the "Best Fit" position selection
    # is the key differentiator from "First Fit".
    
    def attempt_packing(seed):
        """Run a full packing attempt with a given seed."""
        random.seed(seed)
        placed = []
        # Reset final_placements for this attempt
        attempt_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        unplaced_indices = list(range(n))
        
        max_passes = 5
        for p in range(max_passes):
            if time.time() > deadline:
                break
            
            # Create a list of (index, w, h, d) for unplaced boxes
            candidates = []
            for i in unplaced_indices:
                w, h, d = boxes[i]
                candidates.append((i, w, h, d))
                
            if not candidates:
                break
                
            # Sort by volume descending
            candidates.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
            
            newly_placed = 0
            still_unplaced = []
            
            for idx, w, h, d in candidates:
                if time.time() > deadline:
                    break
                    
                rots = get_rotations(w, h, d)
                placed_this = False
                
                for rw, rh, rd in rots:
                    pos = find_best_fit(placed, rw, rh, rd)
                    if pos is not None:
                        x, y, z = pos
                        placed.append((x, y, z, rw, rh, rd))
                        attempt_placements[idx] = (0, x, y, z, rw, rh, rd)
                        placed_this = True
                        newly_placed += 1
                        break
                        
                if not placed_this:
                    still_unplaced.append(idx)
                    
            # If no new boxes were placed in this pass, further passes won't help
            if newly_placed == 0:
                break
                
            # Update unplaced list for next pass
            unplaced_indices = still_unplaced
            
        # Calculate total volume placed
        total_vol = sum(p[3]*p[4]*p[5] for p in placed)
        return total_vol, attempt_placements

    # Run multiple attempts with different seeds to find the best packing
    # The current program uses a deterministic seed (or none). We will try a few random seeds.
    # However, the current program is deterministic. Let's try a few variations.
    
    best_vol = -1
    best_placements = None
    
    # Try a few seeds
    seeds = [0, 1, 2, 3, 4]
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        vol, placements = attempt_packing(seed)
        
        if vol > best_vol:
            best_vol = vol
            best_placements = placements
            
    # If no placements were found (should not happen if container is valid), return default
    if best_placements is None:
        return final_placements
        
    return best_placements
# EVOLVE-BLOCK-END
