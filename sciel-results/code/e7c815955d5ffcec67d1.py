import time
import math
import random
from typing import List, Tuple

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard: Evaluator kills at 150s, but task says 120s. 
    # Use 115s to be safe.
    deadline = time.time() + 115.0
    
    box_vols = [w * h * d for w, h, d in boxes]
    total_vol = sum(box_vols)
    container_vol = W * H * D
    min_bins = max(1, math.ceil(total_vol / container_vol))
    
    # Sort indices by volume descending (Largest First)
    indices = list(range(n))
    indices_sorted = sorted(indices, key=lambda i: box_vols[i], reverse=True)
    
    def compute_score(placements, W, H, D):
        if not placements:
            return 0.0
        bins_used = set(p[0] for p in placements)
        B = len(bins_used)
        last_bin = max(bins_used)
        last_bin_vol = sum(p[3] * p[4] * p[5] for p in placements if p[0] == last_bin)
        return B + 0.1 * (1 - last_bin_vol / container_vol)

    def find_positions(existing_items, w, h, d, W, H, D):
        """
        Generate candidate positions for a box (w,h,d) in a bin with existing_items.
        Returns a list of (x, y, z) positions.
        """
        candidates = set()
        # Start with origin
        candidates.add((0, 0, 0))
        
        # For each existing item, add positions relative to its corners
        for (ex, ey, ez, ew, eh, ed) in existing_items:
            # 1. To the right (x+ew)
            # 2. Above (y+eh)
            # 3. In front (z+ed)
            # We consider all combinations of these offsets to cover corners
            
            # Right
            candidates.add((ex + ew, ey, ez))
            # Above
            candidates.add((ex, ey + eh, ez))
            # Front
            candidates.add((ex, ey, ez + ed))
            
            # Right + Above
            candidates.add((ex + ew, ey + eh, ez))
            # Right + Front
            candidates.add((ex + ew, ey, ez + ed))
            # Above + Front
            candidates.add((ex, ey + eh, ez + ed))
            
            # Right + Above + Front (Far corner)
            candidates.add((ex + ew, ey + eh, ez + ed))
            
            # Also consider "sliding" along axes if possible? 
            # For simplicity and speed, corner-based is standard for BFD.
            
        # Filter candidates that fit in the container
        valid_candidates = []
        for (cx, cy, cz) in candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            
            # Check for overlap
            overlap = False
            for (ex, ey, ez, ew, eh, ed) in existing_items:
                if (cx < ex + ew and ex < cx + w and
                    cy < ey + eh and ey < cy + h and
                    cz < ez + ed and ez < cz + d):
                    overlap = True
                    break
            if not overlap:
                valid_candidates.append((cx, cy, cz))
        
        return valid_candidates

    def evaluate_position(existing_items, w, h, d, pos, W, H, D):
        """
        Score a position. Lower is better.
        Heuristic: 
        1. Prefer positions that are "lower" (smaller z, then y, then x).
        2. Prefer positions that maximize contact with existing boxes (minimize distance to nearest box).
        """
        cx, cy, cz = pos
        
        # Primary: Minimize z, then y, then x (pack to bottom-left-back)
        # This tends to create solid blocks at the bottom.
        score = cz * 10000 + cy * 100 + cx
        
        # Secondary: Maximize "support" (touching existing boxes)
        # If a box is floating, it's bad.
        # Check if it touches any existing box or the floor/walls
        touches = 0
        if cz == 0: touches += 1 # Floor
        if cy == 0: touches += 1 # Back wall
        if cx == 0: touches += 1 # Left wall
        
        for (ex, ey, ez, ew, eh, ed) in existing_items:
            # Check face contact
            if cz + d == ez: touches += 1 # On top of
            if cz == ez + ed: touches += 1 # Behind
            if cy + h == ey: touches += 1 # On side (y+)
            if cy == ey + eh: touches += 1 # Behind (y-)
            if cx + w == ex: touches += 1 # On side (x+)
            if cx == ex + ew: touches += 1 # Behind (x-)
            
            # Break early if we have enough touches
            if touches > 3: break
            
        # Penalize floating boxes (fewer touches)
        score -= touches * 100000
        
        return score

    def solve(ordering):
        """
        Run the packing algorithm with a specific ordering of box indices.
        Returns a list of placements or None if it fails (shouldn't happen with infinite bins, but for safety).
        """
        state = {} # bin_id -> list of (x,y,z,w,h,d)
        placements = [None] * len(ordering)
        
        for i, box_idx in enumerate(ordering):
            if time.time() > deadline:
                return None
            
            w, h, d = boxes[box_idx]
            
            best_bin = None
            best_pos = None
            best_score = float('inf')
            
            # Try existing bins
            for bin_id in sorted(state.keys()):
                pos_list = find_positions(state[bin_id], w, h, d, W, H, D)
                if not pos_list:
                    continue
                
                # Evaluate all valid positions in this bin
                # Optimization: If we have many positions, we might not want to evaluate all if time is tight.
                # But usually < 50 positions per bin.
                for pos in pos_list:
                    s = evaluate_position(state[bin_id], w, h, d, pos, W, H, D)
                    if s < best_score:
                        best_score = s
                        best_bin = bin_id
                        best_pos = pos
                        
            # If no existing bin works, open a new one
            if best_bin is None:
                new_bin_id = max(state.keys()) + 1 if state else 0
                best_bin = new_bin_id
                best_pos = (0, 0, 0)
                # Update state for the new bin immediately for consistency in logic flow, 
                # though we just need to record it.
            
            # Place the box
            if best_bin not in state:
                state[best_bin] = []
            state[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            
            placements[i] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)
            
        # Reorder placements to match original box index order
        final_placements = [None] * n
        for i, box_idx in enumerate(ordering):
            final_placements[box_idx] = placements[i]
            
        return final_placements

    best_result = None
    best_score = float('inf')
    
    # Strategy 1: Volume Descending (Standard BFD)
    result = solve(indices_sorted)
    if result:
        s = compute_score(result, W, H, D)
        if s < best_score:
            best_score = s
            best_result = result
    
    # Strategy 2: Volume Descending, but tie-break by max dimension
    # This can help align boxes better
    indices_sorted2 = sorted(indices, key=lambda i: (box_vols[i], max(boxes[i])), reverse=True)
    if time.time() < deadline:
        result = solve(indices_sorted2)
        if result:
            s = compute_score(result, W, H, D)
            if s < best_score:
                best_score = s
                best_result = result

    # Strategy 3: Randomized Restart with small perturbation
    # If we have time, try a few random shuffles of the last few items or slight reordering
    # This helps escape local minima where a specific order blocks a better fit.
    if time.time() < deadline - 10: # Leave 10s for safety
        attempts = 0
        while time.time() < deadline - 5 and attempts < 5:
            # Create a slightly perturbed ordering
            # Swap a few random adjacent items in the sorted list
            perturbed = indices_sorted[:]
            # Randomly swap 2-5 pairs
            for _ in range(random.randint(2, 5)):
                i = random.randint(0, len(perturbed) - 2)
                perturbed[i], perturbed[i+1] = perturbed[i+1], perturbed[i]
            
            result = solve(perturbed)
            if result:
                s = compute_score(result, W, H, D)
                if s < best_score:
                    best_score = s
                    best_result = result
            attempts += 1

    # Fallback if something went wrong (should not happen)
    if best_result is None:
        # Simple shelf as absolute last resort
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_result = placements

    return best_result
